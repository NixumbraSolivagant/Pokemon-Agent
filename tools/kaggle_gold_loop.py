from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tools.great_tusk_candidates import build_candidates, inspect_package


DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"
DEFAULT_TEAM_ID = 16451965
DEFAULT_STATE = Path("outputs/kaggle_gold/state.json")
SECRET_PATTERNS = (
    re.compile(rb'"key"\s*:\s*"[^"\r\n]+"', re.IGNORECASE),
    re.compile(rb"(?:password|passwd|access[_-]?token)\s*[:=]\s*\S+", re.IGNORECASE),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


@dataclass(slots=True)
class CandidateRecord:
    candidate_id: str
    name: str
    package: str
    package_sha256: str
    hypothesis: str
    parent_champion: str | None = None
    deck_sha256: str | None = None
    source_sha256: str | None = None
    gate_results: dict[str, Any] = field(default_factory=dict)
    submission_ids: list[int] = field(default_factory=list)
    public_score: float | None = None
    non_self_episodes: int = 0
    status: str = "built"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": 1,
                "competition": DEFAULT_COMPETITION,
                "team_id": DEFAULT_TEAM_ID,
                "champion_id": None,
                "candidates": {},
                "active_submissions": [],
                "submission_limits": {},
                "events": [],
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def upsert(self, record: CandidateRecord) -> CandidateRecord:
        existing = self.data["candidates"].get(record.candidate_id)
        payload = asdict(record)
        if existing:
            payload["created_at"] = existing.get("created_at", record.created_at)
            payload["submission_ids"] = sorted(
                set(existing.get("submission_ids", [])) | set(record.submission_ids)
            )
            merged_gates = dict(existing.get("gate_results", {}))
            merged_gates.update(record.gate_results)
            payload["gate_results"] = merged_gates
        payload["updated_at"] = utc_now()
        self.data["candidates"][record.candidate_id] = payload
        self.save()
        return CandidateRecord(**payload)

    def record_event(self, event: str, **details: Any) -> None:
        self.data["events"].append({"time": utc_now(), "event": event, **details})
        self.save()

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        try:
            return self.data["candidates"][candidate_id]
        except KeyError as exc:
            raise KeyError(f"unknown candidate: {candidate_id}") from exc


class KaggleCli:
    def __init__(self, executable: Path, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run):
        self.executable = executable
        self.runner = runner

    def _run(self, *args: str, expect_json: bool = False) -> Any:
        command = [str(self.executable), *args]
        completed = self.runner(command, check=True, capture_output=True, text=True)
        output = completed.stdout.strip()
        if not expect_json:
            return output
        return json.loads(output or "null")

    def team_submissions(self, team_id: int) -> list[dict[str, Any]]:
        value = self._run("competitions", "team-submissions", str(team_id), "--format", "json", "-q", expect_json=True)
        return list(value or [])

    def submission_limits(self, competition: str) -> dict[str, Any]:
        value = self._run("competitions", "submission-limits", competition, "--json", "-q", expect_json=True)
        return dict(value or {})

    def episodes(self, submission_id: int) -> list[dict[str, Any]]:
        value = self._run("competitions", "episodes", str(submission_id), "--format", "json", "-q", expect_json=True)
        return list(value or [])

    def submit(self, competition: str, package: Path, message: str) -> int:
        output = self._run("competitions", "submit", competition, "-f", str(package), "-m", message, "-q")
        matches = re.findall(r"\b(\d{7,})\b", output)
        if not matches:
            raise RuntimeError(f"could not parse submission id from Kaggle output: {output!r}")
        return int(matches[-1])


def assert_no_secret_leak(path: Path) -> None:
    payloads: list[tuple[str, bytes]] = []
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.size > 8 * 1024 * 1024:
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    payloads.append((member.name, handle.read()))
    else:
        payloads.append((path.name, path.read_bytes()))
    for name, data in payloads:
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise ValueError(f"possible credential embedded in {path}:{name}")


def candidate_id(name: str, package_sha256: str) -> str:
    return f"{name}:{package_sha256[:12]}"


def register_candidate(registry: Registry, package: Path, name: str, hypothesis: str, parent: str | None) -> CandidateRecord:
    assert_no_secret_leak(package)
    package_info = inspect_package(package)
    record = CandidateRecord(
        candidate_id=candidate_id(name, package_info["sha256"]),
        name=name,
        package=str(package),
        package_sha256=package_info["sha256"],
        hypothesis=hypothesis,
        parent_champion=parent,
        deck_sha256=package_info["deck_sha256"],
        source_sha256=package_info["main_sha256"],
    )
    return registry.upsert(record)


def remaining_submissions(limits: dict[str, Any]) -> int:
    for key in (
        "numAllowedNow",
        "remaining",
        "remainingSubmissions",
        "remainingDailySubmissions",
        "remainingDailyAllowance",
    ):
        value = limits.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    daily = limits.get("daily")
    if isinstance(daily, dict):
        for key in ("remaining", "remainingSubmissions"):
            value = daily.get(key)
            if isinstance(value, (int, float)):
                return int(value)
    raise ValueError(f"unrecognized submission limit payload: {limits}")


def plan_dual_slot(
    champion_id: str,
    challenger_id: str,
    remaining: int,
    active_submissions: list[dict[str, Any]] | None = None,
    champion_submission_ids: set[int] | None = None,
) -> list[str]:
    if champion_id == challenger_id:
        raise ValueError("challenger must differ from champion")
    if remaining < 1:
        raise RuntimeError("submission cycle has no remaining submissions")
    active_ids = {
        int(item.get("id", item.get("ref", -1)))
        for item in active_submissions or []
        if item.get("id", item.get("ref")) not in (None, "")
    }
    champion_ids = champion_submission_ids or set()
    if active_ids.intersection(champion_ids):
        return [challenger_id]
    if remaining < 2:
        raise RuntimeError("champion is not active and one remaining submission cannot preserve it")
    return [champion_id, challenger_id]


def sync_registry(registry: Registry, kaggle: KaggleCli) -> dict[str, Any]:
    competition = str(registry.data.get("competition", DEFAULT_COMPETITION))
    team_id = int(registry.data.get("team_id", DEFAULT_TEAM_ID))
    active = kaggle.team_submissions(team_id)
    limits = kaggle.submission_limits(competition)
    registry.data["active_submissions"] = active
    registry.data["submission_limits"] = limits
    by_submission = {
        int(submission_id): candidate
        for candidate in registry.data["candidates"].values()
        for submission_id in candidate.get("submission_ids", [])
    }
    for submission in active:
        submission_id = int(submission.get("id", submission.get("ref", -1)))
        candidate = by_submission.get(submission_id)
        if candidate is None:
            continue
        score = submission.get("publicScore")
        candidate["public_score"] = float(score) if score not in (None, "") else None
        candidate["updated_at"] = utc_now()
    registry.record_event("sync", active_count=len(active))
    return {"active_submissions": active, "submission_limits": limits}


def execute_cycle(
    registry: Registry,
    kaggle: KaggleCli,
    challenger_id: str,
    execute: bool,
) -> list[dict[str, Any]]:
    champion_id = registry.data.get("champion_id")
    if not champion_id:
        raise RuntimeError("registry has no champion_id")
    if execute and hasattr(kaggle, "team_submissions"):
        sync_registry(registry, kaggle)
    limits = registry.data.get("submission_limits") or kaggle.submission_limits(registry.data["competition"])
    champion_record = registry.candidate(champion_id)
    order = plan_dual_slot(
        champion_id,
        challenger_id,
        remaining_submissions(limits),
        active_submissions=registry.data.get("active_submissions", []),
        champion_submission_ids={int(value) for value in champion_record.get("submission_ids", [])},
    )
    actions: list[dict[str, Any]] = []
    roles = ("challenger",) if order == [challenger_id] else ("champion-preserve", "challenger")
    for role, current_id in zip(roles, order):
        record = registry.candidate(current_id)
        package = Path(record["package"])
        assert_no_secret_leak(package)
        message = f"gold-loop {role} {current_id}"
        action = {"role": role, "candidate_id": current_id, "package": str(package), "message": message}
        if execute:
            submission_id = kaggle.submit(registry.data["competition"], package, message)
            action["submission_id"] = submission_id
            record["submission_ids"] = sorted(set(record.get("submission_ids", [])) | {submission_id})
            record["status"] = "submitted"
            record["updated_at"] = utc_now()
            registry.save()
        actions.append(action)
    registry.record_event("submission_cycle", execute=execute, actions=actions)
    return actions


def command_build(args: argparse.Namespace, registry: Registry) -> dict[str, Any]:
    records = build_candidates(args.manifest, args.out)
    registered = []
    for row in records:
        registered.append(
            asdict(
                register_candidate(
                    registry,
                    Path(row["submission"]),
                    row["name"],
                    row["hypothesis"],
                    row["parent"],
                )
            )
        )
    return {"registered": registered}


def monitor_candidate(registry: Registry, kaggle: KaggleCli, current_id: str) -> dict[str, Any]:
    record = registry.candidate(current_id)
    if not record.get("submission_ids"):
        raise RuntimeError(f"candidate has no submissions: {current_id}")
    submission_id = int(record["submission_ids"][-1])
    episodes = kaggle.episodes(submission_id)
    non_self = 0
    for episode in episodes:
        agents = episode.get("agents") or []
        opponent_ids = {
            int(agent.get("submissionId", -1))
            for agent in agents
            if int(agent.get("submissionId", -1)) != submission_id
        }
        if opponent_ids:
            non_self += 1
    record["non_self_episodes"] = non_self
    record["updated_at"] = utc_now()
    sync_registry(registry, kaggle)
    registry.record_event("monitor", candidate_id=current_id, submission_id=submission_id, non_self_episodes=non_self)
    return {
        "candidate_id": current_id,
        "submission_id": submission_id,
        "episodes": len(episodes),
        "non_self_episodes": non_self,
        "public_score": record.get("public_score"),
    }


def promote_candidate(registry: Registry, current_id: str, min_episodes: int, min_score: float | None) -> dict[str, Any]:
    record = registry.candidate(current_id)
    if int(record.get("non_self_episodes", 0)) < min_episodes:
        raise RuntimeError(f"promotion requires {min_episodes} non-self episodes")
    score = record.get("public_score")
    if min_score is not None and (score is None or float(score) < min_score):
        raise RuntimeError(f"promotion requires public score >= {min_score}")
    catastrophic = record.get("gate_results", {}).get("catastrophic_regression")
    if catastrophic:
        raise RuntimeError("candidate has a catastrophic regression gate")
    registry.data["champion_id"] = current_id
    record["status"] = "champion"
    record["updated_at"] = utc_now()
    registry.record_event("promote", candidate_id=current_id, public_score=score)
    return {"champion_id": current_id, "public_score": score}


def ingest_gate(registry: Registry, current_id: str, gate_name: str, ranking: Path) -> dict[str, Any]:
    record = registry.candidate(current_id)
    rows = json.loads(ranking.read_text(encoding="utf-8"))
    row = next((value for value in rows if value.get("name") == record["name"]), None)
    if row is None:
        raise ValueError(f"{record['name']} not found in {ranking}")
    games = int(row.get("games", 0))
    no_results = int(row.get("no_results", 0))
    worst_rate = float(row.get("worst_rate", 0.0))
    gate = {
        "report": str(ranking),
        "games": games,
        "wins": int(row.get("wins", 0)),
        "losses": int(row.get("losses", 0)),
        "no_results": no_results,
        "mean_rate": float(row.get("mean_rate", 0.0)),
        "worst_rate": worst_rate,
        "worst_lower": float(row.get("worst_lower", 0.0)),
        "robust_score": float(row.get("robust_score", 0.0)),
        "passed": games > 0 and no_results / games <= 0.02 and worst_rate >= 0.40,
        "recorded_at": utc_now(),
    }
    record.setdefault("gate_results", {})[gate_name] = gate
    record["gate_results"]["catastrophic_regression"] = not gate["passed"]
    record["updated_at"] = utc_now()
    registry.record_event("ingest_gate", candidate_id=current_id, gate_name=gate_name, passed=gate["passed"])
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private Kaggle champion/challenger control loop.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--kaggle", type=Path, default=Path(".venv-kaggle-cli/bin/kaggle"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--manifest", type=Path, default=Path("configs/great_tusk_gold_private.json"))
    build_parser.add_argument("--out", type=Path, default=Path("outputs/great_tusk_gold"))

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("package", type=Path)
    register_parser.add_argument("--name", required=True)
    register_parser.add_argument("--hypothesis", required=True)
    register_parser.add_argument("--parent")

    subparsers.add_parser("sync")

    champion_parser = subparsers.add_parser("set-champion")
    champion_parser.add_argument("candidate_id")

    cycle_parser = subparsers.add_parser("submit-cycle")
    cycle_parser.add_argument("challenger_id")
    cycle_parser.add_argument("--execute", action="store_true")

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("candidate_id")

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("candidate_id")
    promote_parser.add_argument("--min-episodes", type=int, default=40)
    promote_parser.add_argument("--min-score", type=float)

    gate_parser = subparsers.add_parser("ingest-gate")
    gate_parser.add_argument("candidate_id")
    gate_parser.add_argument("--name", required=True)
    gate_parser.add_argument("--ranking", type=Path, required=True)

    args = parser.parse_args(argv)
    registry = Registry(args.state)
    if args.command == "build":
        result = command_build(args, registry)
    elif args.command == "register":
        result = asdict(register_candidate(registry, args.package, args.name, args.hypothesis, args.parent))
    elif args.command == "sync":
        result = sync_registry(registry, KaggleCli(args.kaggle))
    elif args.command == "set-champion":
        registry.candidate(args.candidate_id)
        registry.data["champion_id"] = args.candidate_id
        registry.record_event("set_champion", candidate_id=args.candidate_id)
        result = {"champion_id": args.candidate_id}
    elif args.command == "submit-cycle":
        result = {"actions": execute_cycle(registry, KaggleCli(args.kaggle), args.challenger_id, args.execute)}
    elif args.command == "monitor":
        result = monitor_candidate(registry, KaggleCli(args.kaggle), args.candidate_id)
    elif args.command == "promote":
        result = promote_candidate(registry, args.candidate_id, args.min_episodes, args.min_score)
    elif args.command == "ingest-gate":
        result = ingest_gate(registry, args.candidate_id, args.name, args.ranking)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
