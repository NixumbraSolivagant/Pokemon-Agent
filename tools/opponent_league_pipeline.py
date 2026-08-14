from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import math
import shutil
import shlex
import subprocess
import tempfile
import threading
import time
import zipfile
import statistics
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tools.behavior_clone_audit import audit
from tools.build_meta_submission import build as build_submission
from tools.replay_imitation import train


COMPETITION = "pokemon-tcg-ai-battle"
SCHEMA_VERSION = 2
BAND_QUOTAS = ((1, 100, 100), (101, 500, 150), (501, 1500, 120), (1501, 3000, 80), (3001, 10000, 50))
EPISODE_CAPS = ((1, 20, 80), (21, 100, 60), (101, 500, 40), (501, 1500, 30), (1501, 10000, 20))
QUALITY_GATES = {"semantic": 0.88, "ability": 0.85, "attack": 0.90, "setup": 0.90}
TURN_QUALITY_GATES = {"macro": 0.80, "ability": 0.85, "attack": 0.85, "terminal": 0.70, "jaccard": 0.75}
CLONE_ALGORITHM_VERSION = 6
CLONE_PROFILES = (
    ("balanced", {"n_estimators": 900, "learning_rate": 0.03, "num_leaves": 63, "head_num_leaves": 31, "min_child_samples": 32, "head_n_estimators": 450, "reg_lambda": 8.0, "eval_metric": "NDCG:top=3"}),
    ("deep", {"n_estimators": 1200, "learning_rate": 0.025, "num_leaves": 63, "head_num_leaves": 63, "min_child_samples": 28, "head_n_estimators": 600, "reg_lambda": 10.0, "eval_metric": "NDCG:top=3"}),
    ("softmax", {"n_estimators": 900, "learning_rate": 0.03, "num_leaves": 63, "head_num_leaves": 31, "min_child_samples": 32, "head_n_estimators": 450, "reg_lambda": 8.0, "loss_function": "QuerySoftMax", "eval_metric": "NDCG:top=3"}),
    ("softmax_deep", {"n_estimators": 1200, "learning_rate": 0.025, "num_leaves": 63, "head_num_leaves": 63, "min_child_samples": 28, "head_n_estimators": 600, "reg_lambda": 10.0, "loss_function": "QuerySoftMax", "eval_metric": "NDCG:top=3"}),
)
AUTHENTICATION_ERROR = "authentication_required"


class RequestPacer:
    def __init__(self, interval: float, jitter: float, seed: int = 20260804):
        self.interval = max(0.0, interval)
        self.jitter = max(0.0, jitter)
        self.random = random.Random(seed)
        self.lock = threading.Lock()
        self.next_request = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_request)
            self.next_request = scheduled + self.interval + self.random.uniform(0.0, self.jitter)
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)

    def cooldown(self, seconds: float) -> None:
        with self.lock:
            self.next_request = max(self.next_request, time.monotonic() + max(0.0, seconds))


def retry_after_seconds(detail: str, fallback: float) -> float:
    match = re.search(r"retry[- ]after[^0-9]*(\d+(?:\.\d+)?)", detail, re.IGNORECASE)
    return float(match.group(1)) if match else fallback


def is_rate_limit_error(detail: str) -> bool:
    lowered = detail.lower()
    return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def run_json(command: list[str]) -> Any:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout or "null")


def kaggle_command(value: str) -> list[str]:
    return shlex.split(value.replace("~", str(Path.home()), 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_band(rank: int) -> str:
    if rank <= 100:
        return "top100"
    if rank <= 1500:
        return "mid"
    return "long_tail"


def deterministic_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    if count <= 1:
        return rows[:count]
    indexes = {round(index * (len(rows) - 1) / (count - 1)) for index in range(count)}
    return [rows[index] for index in sorted(indexes)]


def select_teams(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for lower, upper, count in BAND_QUOTAS:
        band = [row for row in rows if lower <= int(row["rank"]) <= upper]
        selected.extend(deterministic_sample(band, count))
    return selected


def episode_cap(rank: int) -> int:
    for lower, upper, cap in EPISODE_CAPS:
        if lower <= rank <= upper:
            return cap
    return 30


def select_episodes(rows: list[dict[str, Any]], cap: int) -> list[int]:
    valid = [
        row for row in rows
        if str(row.get("state", "")).endswith("COMPLETED")
        and str(row.get("type", "")).endswith("PUBLIC")
    ]
    valid.sort(key=lambda row: str(row.get("createTime", "")), reverse=True)
    if len(valid) <= cap:
        return [int(row["id"]) for row in valid]
    recent_count = cap // 2
    recent = valid[:recent_count]
    older = deterministic_sample(valid[recent_count:], cap - recent_count)
    return [int(row["id"]) for row in [*recent, *older]]


def load_leaderboard(kaggle: str, competition: str, workspace: Path, csv_path: Path | None) -> list[dict[str, Any]]:
    if csv_path is None:
        download_dir = workspace / "leaderboard_download"
        shutil.rmtree(download_dir, ignore_errors=True)
        download_dir.mkdir(parents=True)
        subprocess.run([*kaggle_command(kaggle), "competitions", "leaderboard", competition, "--download", "-p", str(download_dir), "-q"], check=True)
        archive = next(download_dir.glob("*.zip"))
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(download_dir)
        csv_path = next(download_dir.glob("*.csv"))
    with csv_path.open(encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    return [
        {
            "rank": int(row["Rank"]),
            "team_id": int(row["TeamId"]),
            "team_name": row["TeamName"],
            "score": float(row["Score"]),
            "last_submission_date": row.get("LastSubmissionDate", ""),
        }
        for row in raw
        if row.get("Rank") and row.get("Score")
    ]


def snapshot(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    leaderboard = load_leaderboard(args.kaggle, args.competition, args.out, args.leaderboard_csv)
    teams = select_teams(leaderboard)
    sources: list[dict[str, Any]] = []
    for team_number, team in enumerate(teams, 1):
        submissions = run_json([*kaggle_command(args.kaggle), "competitions", "team-submissions", str(team["team_id"]), "--format", "json", "-q"])
        submissions = sorted(submissions or [], key=lambda row: float(row.get("publicScore") or -1), reverse=True)
        source_limit = 2 if team["rank"] <= 20 else 1
        for submission in submissions[:source_limit]:
            submission_id = int(submission["id"])
            episodes = run_json([*kaggle_command(args.kaggle), "competitions", "episodes", str(submission_id), "--format", "json", "-q"])
            episode_ids = select_episodes(episodes or [], episode_cap(team["rank"]))
            if not episode_ids:
                continue
            source_id = f"team_{team['team_id']}_submission_{submission_id}"
            sources.append({
                **team,
                "source_id": source_id,
                "submission_id": submission_id,
                "submission_score": float(submission.get("publicScore") or team["score"]),
                "submission_date": submission.get("dateSubmitted", ""),
                "rank_band": rank_band(team["rank"]),
                "episode_ids": episode_ids,
                "replay_path": f"replays/{source_id}",
            })
        print(f"[snapshot] {team_number}/{len(teams)} team={team['team_id']} sources={len(sources)}", flush=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "competition": args.competition,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampling": {"bands": BAND_QUOTAS, "episode_caps": EPISODE_CAPS},
        "leaderboard_size": len(leaderboard),
        "teams_selected": len(teams),
        "sources": sources,
    }
    write_json(args.out / "snapshot.json", payload)
    print(json.dumps({"teams": len(teams), "sources": len(sources), "episodes": sum(len(row["episode_ids"]) for row in sources)}, indent=2))


def download_episode(
    kaggle: str,
    key: str,
    episode_id: int,
    staging: Path,
    destination: Path,
    pacer: RequestPacer | None = None,
    max_retries: int = 5,
    max_backoff: float = 300.0,
) -> tuple[str, str | None]:
    target = destination / f"episode-{episode_id}-replay.json.gz"
    if target.exists():
        return key, None
    work = staging / str(episode_id)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        last_error = "replay download failed"
        attempt = 0
        while True:
            if pacer is not None:
                pacer.wait()
            completed = subprocess.run(
                [*kaggle_command(kaggle), "competitions", "replay", str(episode_id), "-p", str(work), "-q"],
                check=False,
                capture_output=True,
                text=True,
            )
            detail = " | ".join(
                line.strip()
                for line in (completed.stderr or completed.stdout).splitlines()
                if line.strip()
            )
            if completed.returncode != 0:
                last_error = detail or f"kaggle replay exited {completed.returncode}"
            else:
                replay = next(work.glob("*replay.json"), None)
                if replay is not None:
                    destination.mkdir(parents=True, exist_ok=True)
                    with replay.open("rb") as source, gzip.open(target, "wb", compresslevel=6) as output:
                        shutil.copyfileobj(source, output)
                    return key, None
                last_error = "kaggle replay command produced no replay.json"
            if "Authentication required" in last_error:
                return key, f"{AUTHENTICATION_ERROR}: {last_error}"
            if is_rate_limit_error(last_error):
                delay = min(max_backoff, retry_after_seconds(last_error, 30.0 * (2 ** min(attempt, 8))))
                if pacer is not None:
                    pacer.cooldown(delay)
                else:
                    time.sleep(delay)
                attempt += 1
                continue
            if attempt >= max_retries:
                return key, last_error
            delay = min(max_backoff, 5.0 * (2 ** min(attempt, 8)))
            if pacer is not None:
                pacer.cooldown(delay)
            else:
                time.sleep(delay)
            attempt += 1
    except Exception as exc:
        return key, f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def sync_remote(local_root: Path, remote: str, remote_root: str, port: int) -> None:
    subprocess.run([
        "sshpass", "-e", "rsync", "-az", "--no-owner", "--no-group", "--omit-dir-times",
        "-e", f"ssh -p {port} -o StrictHostKeyChecking=accept-new",
        f"{local_root}/", f"{remote}:{remote_root}/",
    ], check=True)


def purge_local_replays(root: Path) -> None:
    replay_root = root / "replays"
    if replay_root.exists():
        shutil.rmtree(replay_root)


def collect(args: argparse.Namespace) -> None:
    root = args.root
    snapshot_data = read_json(root / "snapshot.json")
    if not snapshot_data:
        raise FileNotFoundError(root / "snapshot.json")
    staging = args.staging
    staging.mkdir(parents=True, exist_ok=True)
    state_path = root / "collect_state.json"
    state = read_json(state_path, {"completed": [], "failures": {}})
    completed = {str(value) for value in state.get("completed", [])}
    failures = dict(state.get("failures", {}))
    pending: list[tuple[str, int, Path]] = []
    for source in snapshot_data["sources"]:
        destination = root / source["replay_path"]
        for episode_id in source["episode_ids"]:
            key = f"{source['source_id']}:{int(episode_id)}"
            if key not in completed:
                pending.append((key, int(episode_id), destination))
    pacer = RequestPacer(args.request_interval, args.request_jitter, args.seed)
    print(
        f"[collect] workers={args.workers} interval={args.request_interval:.2f}s "
        f"jitter={args.request_jitter:.2f}s max_retries={args.max_retries}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_episode,
                args.kaggle,
                key,
                episode_id,
                staging,
                destination,
                pacer,
                args.max_retries,
                args.max_backoff,
            ): key
            for key, episode_id, destination in pending
        }
        authentication_error = None
        for number, future in enumerate(as_completed(futures), 1):
            key, error = future.result()
            if error:
                failures[key] = error
            else:
                completed.add(key)
                failures.pop(key, None)
            if number % args.checkpoint_every == 0 or number == len(futures):
                write_json(state_path, {"completed": sorted(completed), "failures": failures})
                if args.remote and args.remote_root:
                    sync_remote(root, args.remote, args.remote_root, args.ssh_port)
                    purge_local_replays(root)
            print(f"[collect] {number}/{len(futures)} completed={len(completed)} failures={len(failures)}", flush=True)
            if error and error.startswith(f"{AUTHENTICATION_ERROR}:"):
                authentication_error = error
                write_json(state_path, {"completed": sorted(completed), "failures": failures})
                for pending_future in futures:
                    pending_future.cancel()
                break
    if authentication_error:
        raise RuntimeError(authentication_error)
    write_json(root / "collection_complete.json", {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episodes_collected": len(completed),
        "failures": len(failures),
    })
    if args.remote and args.remote_root:
        sync_remote(root, args.remote, args.remote_root, args.ssh_port)
        purge_local_replays(root)


def wilson_lower(successes: float, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    spread = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * total)) / total)
    return (center - spread) / denominator


def quality_gate(result: dict[str, Any]) -> dict[str, Any]:
    groups = result.get("by_group", {})
    ability_row = groups.get("ability", {})
    setup_row = groups.get("SETUP", groups.get("setup", {}))
    attack_rows = [row for name, row in groups.items() if name.startswith("attack:")]
    attack_counts_available = any("decisions" in row for row in attack_rows)
    attack_decisions = sum(int(row.get("decisions", 0)) for row in attack_rows)
    attack_semantic = sum(float(row.get("semantic", 0)) for row in attack_rows)
    attack_rate = (
        attack_semantic / max(1, attack_decisions)
        if attack_counts_available
        else sum(float(row.get("semantic_rate", 0.0)) for row in attack_rows) / max(1, len(attack_rows))
    )
    ability_decisions = int(ability_row.get("decisions", 0))
    ability_semantic = float(ability_row.get("semantic", 0))
    setup_decisions = int(setup_row.get("decisions", 0))
    setup_semantic = float(setup_row.get("semantic", 0))
    turn = result.get("turn_fidelity") or {}
    main_rates = [
        float(groups[name].get("semantic_rate", 0.0))
        for name in ("play", "attach", "evolve", "ability", "retreat", "end")
        if name in groups
    ]
    if attack_counts_available:
        main_rates.append(attack_rate)
    fallback_macro = sum(main_rates) / max(1, len(main_rates))
    terminal_decisions = attack_decisions + int(groups.get("end", {}).get("decisions", 0))
    terminal_semantic = attack_semantic + float(groups.get("end", {}).get("semantic", 0))
    turn_macro = float(turn.get("main_macro_recall", fallback_macro))
    turn_attack = float(turn.get("attack_recall", attack_rate))
    turn_ability = float(turn.get("ability_recall", ability_row.get("semantic_rate", 1.0)))
    turn_terminal = float(turn.get("terminal_recall", terminal_semantic / max(1, terminal_decisions)))
    turn_jaccard = float(turn.get("action_jaccard_median", result.get("semantic_rate", 0.0)))
    stop_balanced = float(turn.get("stop_balanced_accuracy", 1.0))
    strength = result.get("strength_gate") or {"passed": True, "status": "not_configured"}
    robustness = result.get("training_robustness") or {}
    folds = robustness.get("temporal_folds") or []
    fold_attack = [float((row.get("by_type", {}).get("attack") or {}).get("recall", 1.0)) for row in folds]
    fold_terminal = []
    for row in folds:
        by_type = row.get("by_type", {})
        present = [
            float((by_type.get(name) or {}).get("recall", 1.0))
            for name in ("attack", "end")
            if int((by_type.get(name) or {}).get("decisions", 0)) >= 10
        ]
        if present:
            fold_terminal.append(min(present))
    checks = {
        "turn_macro": turn_macro >= TURN_QUALITY_GATES["macro"],
        "ability": turn_ability >= TURN_QUALITY_GATES["ability"],
        "ability_confidence": ability_decisions < 20 or wilson_lower(ability_semantic, ability_decisions) >= 0.75,
        "attack_coverage": not attack_counts_available or attack_decisions >= 20,
        "attack": turn_attack >= TURN_QUALITY_GATES["attack"],
        "attack_confidence": not attack_counts_available or wilson_lower(attack_semantic, attack_decisions) >= 0.80,
        "terminal": terminal_decisions < 20 or turn_terminal >= TURN_QUALITY_GATES["terminal"],
        "turn_jaccard": turn_jaccard >= TURN_QUALITY_GATES["jaccard"],
        "stop_balanced": stop_balanced >= 0.85,
        "temporal_attack": not fold_attack or min(fold_attack) >= 0.75,
        "temporal_terminal": not fold_terminal or min(fold_terminal) >= 0.55,
        "setup": float(setup_row.get("semantic_rate", 1.0)) >= QUALITY_GATES["setup"],
        "setup_confidence": setup_decisions < 20 or wilson_lower(setup_semantic, setup_decisions) >= 0.80,
        "strength": bool(strength.get("passed", True)),
        "errors": int(result.get("errors", 0)) == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "semantic": float(result.get("semantic_rate", 0.0)),
            "legacy_semantic_passed": float(result.get("semantic_rate", 0.0)) >= QUALITY_GATES["semantic"],
            "turn_macro": turn_macro,
            "turn_jaccard": turn_jaccard,
            "ability": float(ability_row.get("semantic_rate", 1.0)),
            "ability_decisions": ability_decisions,
            "ability_lower": wilson_lower(ability_semantic, ability_decisions) if ability_decisions else 1.0,
            "attack": turn_attack,
            "attack_decisions": attack_decisions,
            "attack_lower": wilson_lower(attack_semantic, attack_decisions),
            "setup": float(setup_row.get("semantic_rate", 1.0)),
            "setup_decisions": setup_decisions,
            "setup_lower": wilson_lower(setup_semantic, setup_decisions) if setup_decisions else 1.0,
            "terminal": turn_terminal,
            "terminal_decisions": terminal_decisions,
            "stop_balanced": stop_balanced,
            "temporal_attack_min": min(fold_attack) if fold_attack else 1.0,
            "temporal_terminal_min": min(fold_terminal) if fold_terminal else 1.0,
            "strength_status": strength.get("status", "configured"),
            "errors": int(result.get("errors", 0)),
        },
        "failed": [name for name, passed in checks.items() if not passed],
    }


def quality_pass(result: dict[str, Any]) -> bool:
    return bool(quality_gate(result)["passed"])


def evaluate_strength(
    candidate: Path,
    baseline: Path,
    games: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    from local_eval.evaluator import run_match, submission_name
    from local_eval.models import EvalConfig
    from tools.closed_loop_clone_audit import summarize_closed_loop

    name = submission_name(candidate)
    report = run_match(
        candidate,
        baseline,
        games,
        EvalConfig(
            profile="kaggle",
            seed=seed,
            workers=max(1, workers),
            record_mode="training",
            record_focus=name,
            progress=False,
        ),
        Path.cwd(),
    )
    row = next(stats for stats in report.standings if stats.name == name)
    score = (row.wins + 0.5 * row.draws) / max(1, row.games)
    failures = row.crashes + row.timeouts + row.invalids + row.no_results
    lower = wilson_lower(row.wins + 0.5 * row.draws, row.games)
    closed_loop = summarize_closed_loop(report, name)
    return {
        "status": "configured",
        "passed": failures == 0 and score >= 0.55 and lower >= 0.48,
        "games": row.games,
        "wins": row.wins,
        "losses": row.losses,
        "draws": row.draws,
        "score": score,
        "wilson_lower": lower,
        "failures": failures,
        "baseline": str(baseline),
        "closed_loop": closed_loop,
    }


def behavior_signature(result: dict[str, Any]) -> str:
    groups = result.get("by_group", {})
    payload = {
        "semantic": round(float(result.get("semantic_rate", 0.0)), 2),
        "groups": {
            name: round(float(row.get("semantic_rate", 0.0)), 2)
            for name, row in sorted(groups.items())
            if int(row.get("decisions", 0)) >= 3
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_source(
    root: Path,
    source: dict[str, Any],
    number: int,
    args: argparse.Namespace,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_type = str(getattr(args, "task_type", "GPU"))
    gpu_devices = str(getattr(args, "gpu_devices", "0") or "")
    source_id = source["source_id"]
    replay_dir = root / source["replay_path"]
    replay_count = len(list(replay_dir.glob("*replay.json*")))
    row = {
        **source,
        "replay_count": replay_count,
        "status": "rejected",
        "clone_algorithm_version": CLONE_ALGORITHM_VERSION,
    }
    if replay_count < args.min_episodes:
        row["rejection_reason"] = "insufficient_episodes"
        return row
    force_rebuild = bool(getattr(args, "force_rebuild", False))
    stale_version = int((previous or {}).get("clone_algorithm_version", 0) or 0) != CLONE_ALGORITHM_VERSION
    technical_retry = bool(previous) and str(previous.get("rejection_reason", "")) not in {
        "insufficient_episodes", "quality_gate",
    }
    if previous and (
        force_rebuild or stale_version or replay_count > int(previous.get("replay_count", 0)) or technical_retry
    ):
        for old_path in (
            *(root / "artifacts").glob(f"{source_id}*.json"),
            *(root / "packages").glob(f"{source_id}*.tar.gz"),
            *(root / "manifests").glob(f"{source_id}*.json"),
        ):
            old_path.unlink(missing_ok=True)
        for parts_dir in (root / "artifacts").glob(f"{source_id}*_parts"):
            shutil.rmtree(parts_dir, ignore_errors=True)
    attempts = []
    best: tuple[tuple[float, ...], dict[str, Any]] | None = None
    opponent_only_fast = bool(getattr(args, "opponent_only_fast", False))
    profiles = CLONE_PROFILES[:1] if opponent_only_fast else CLONE_PROFILES
    for profile_index, (profile, profile_params) in enumerate(profiles):
        manifest = root / "manifests" / f"{source_id}.{profile}.json"
        artifact = root / "artifacts" / f"{source_id}.{profile}.json"
        package = root / "packages" / f"{source_id}.{profile}.tar.gz"
        source_rows = [{"name": source_id, "path": str(replay_dir), "target_name": source["team_name"], "split": "train", "weight": 1.0}]
        required_team = str(getattr(args, "required_team", "") or "").casefold()
        history_replays = getattr(args, "required_history_replays", None)
        if history_replays and str(source["team_name"]).casefold() == required_team and Path(history_replays).exists():
            source_rows.append({
                "name": f"{source_id}_history",
                "path": str(history_replays),
                "target_name": source["team_name"],
                "split": "train",
                "weight": float(getattr(args, "required_history_weight", 0.5)),
            })
        majkel = str(source.get("team_name", "")).casefold() == "majkel1337"
        manifest_data = {
            "family": "generic_replay",
            "clone_algorithm_version": CLONE_ALGORITHM_VERSION,
            "seed": args.seed + number * 10 + profile_index,
            **profile_params,
            "value_n_estimators": int(profile_params.get("head_n_estimators", 600)),
            "n_jobs": args.model_jobs,
            "parse_workers": 4,
            "win_weight": 1.0,
            "loss_weight": 0.35,
            "task_type": task_type,
            "devices": gpu_devices if task_type.upper() == "GPU" else "",
            "enable_main_type_heads": True,
            "enable_main_router": True,
            "router_mode": "regime_soft_v2" if majkel else "soft_confidence",
            "router_schema_version": 2 if majkel else 1,
            "router_feature_mode": "compact_v1",
            "stop_balance_max_weight": 1.0,
            "stop_no_attack_continue_weight": 1.0,
            "stop_attack_ready_continue_weight": 1.5,
            "stop_ko_ready_continue_weight": 2.0,
            "stop_weight_cap": 2.70,
            "stop_ensemble_size": 3,
            "stop_oof_folds": 5 if majkel else 0,
            "stop_oof_false_stop_multiplier": 1.35,
            "stop_run_ablations": majkel,
            "stop_gate_continue": 0.92 if majkel else 0.90,
            "stop_gate_terminal": 0.75 if majkel else 0.70,
            "stop_gate_balanced": 0.87 if majkel else 0.85,
            "stop_gate_attack_continue": 0.89 if majkel else 0.0,
            "stop_gate_ko_continue": 0.85 if majkel else 0.0,
            "stop_gate_turn_8_continue": 0.85 if majkel else 0.0,
            "stop_require_test_gate": majkel,
            "split_seed": 20260813 if majkel else args.seed + number * 10 + profile_index,
            "low_level_shared_only": True,
            "type_balance_max_weight": 6.0,
            "router_num_leaves": 31 if profile in {"balanced", "softmax"} else 63,
            "router_n_estimators": 450 if profile in {"balanced", "softmax"} else 600,
            "router_min_child_samples": 32,
            "router_l2_leaf_reg": 10.0,
            "cache_dir": str(root / "cache" / source_id),
            "sources": source_rows,
            "require_same_deck_sources": len(source_rows) > 1,
        }
        if opponent_only_fast:
            manifest_data.update(
                {
                    "fast_finalize": True,
                    "parse_workers": 1,
                    "n_estimators": 96,
                    "head_n_estimators": 64,
                    "value_n_estimators": 64,
                    "minimum_head_decisions": 1_000_000_000,
                    "enable_main_type_heads": False,
                    "enable_main_router": False,
                    "stop_ensemble_size": 1,
                }
            )
        shared_init_artifact = getattr(args, "shared_init_artifact", None)
        if shared_init_artifact:
            manifest_data["shared_init_artifact"] = str(shared_init_artifact)
            manifest_data["training_stage"] = "source_finetune"
        write_json(manifest, manifest_data)
        try:
            artifact_data = train(manifest, artifact)
            decisions = int(
                artifact_data["split_stats"]["train_decisions"]
                + artifact_data["split_stats"]["validation_decisions"]
                + artifact_data["split_stats"]["test_decisions"]
            )
            if decisions < args.min_decisions:
                raise ValueError("insufficient_decisions")
            build_submission(artifact, package, args.runtime, args.cg_dir, "clone_fidelity")
            if opponent_only_fast:
                candidate = {
                    **row,
                    "status": "opponent_ready",
                    "profile": profile,
                    "artifact": str(artifact),
                    "package": str(package),
                    "package_sha256": sha256_file(package),
                    "deck_sha256": artifact_data["deck_sha256"],
                    "behavior_signature": sha256_file(package),
                    "decisions": decisions,
                    "opponent_only_fast": True,
                }
                attempts.append({"profile": profile, "artifact": str(artifact), "opponent_only_fast": True})
                best = ((1.0, float(decisions), 0.0, 0.0), candidate)
                break
            test_ids = {str(value).split(":")[-1] for value in artifact_data["split_stats"].get("test_episodes", [])}
            audit_result = audit(package, replay_dir, source["team_name"], test_ids or None)
            audit_result["training_robustness"] = artifact_data.get("hard_main_metrics", {})
            strength_baseline = getattr(args, "strength_baseline", None)
            if strength_baseline:
                audit_result["strength_gate"] = evaluate_strength(
                    package,
                    Path(strength_baseline),
                    int(getattr(args, "strength_games", 256)),
                    int(getattr(args, "strength_workers", 16)),
                    int(args.seed) + number * 100 + profile_index,
                )
            gate = quality_gate(audit_result)
            audit_path = root / "audits" / f"{source_id}.{profile}.json"
            write_json(audit_path, {**audit_result, "quality_gate": gate})
            candidate = {
                **row,
                "status": "qualified" if gate["passed"] else "rejected",
                "profile": profile,
                "artifact": str(artifact),
                "package": str(package),
                "package_sha256": sha256_file(package),
                "deck_sha256": artifact_data["deck_sha256"],
                "behavior_signature": behavior_signature(audit_result),
                "decisions": decisions,
                "audit": audit_result,
                "quality_gate": gate,
            }
            if not gate["passed"]:
                candidate["rejection_reason"] = "quality_gate"
            attempts.append({"profile": profile, "quality_gate": gate, "artifact": str(artifact)})
            metrics = gate["metrics"]
            score = (
                float(gate["passed"]),
                metrics["semantic"],
                metrics["attack"],
                metrics["ability"],
            )
            if best is None or score > best[0]:
                best = (score, candidate)
            if gate["passed"]:
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[profile-error] source={source_id} profile={profile} error={error}", flush=True)
            attempts.append({"profile": profile, "error": error})
    if best is None:
        row["rejection_reason"] = attempts[-1].get("error", "all_profiles_failed") if attempts else "all_profiles_failed"
        row["profile_attempts"] = attempts
        return row
    best[1]["profile_attempts"] = attempts
    return best[1]


def ensure_shared_pretrain(
    root: Path,
    available: list[tuple[int, dict[str, Any], dict[str, Any] | None]],
    targets: list[tuple[int, dict[str, Any], dict[str, Any] | None]],
    args: argparse.Namespace,
) -> Path | None:
    if bool(getattr(args, "disable_shared_pretrain", False)):
        return None
    artifact = root / "artifacts" / f"shared_pretrain_v{CLONE_ALGORITHM_VERSION}.json"
    force = bool(getattr(args, "force_shared_pretrain", False))
    if artifact.exists() and not force:
        data = read_json(artifact, {})
        if int(data.get("version", 0)) >= 6 and int(data.get("clone_algorithm_version", 0)) == CLONE_ALGORITHM_VERSION:
            return artifact
    if artifact.exists():
        artifact.unlink()
    shutil.rmtree(root / "artifacts" / f"shared_pretrain_v{CLONE_ALGORITHM_VERSION}_parts", ignore_errors=True)
    target_ids = {source["source_id"] for _, source, _ in targets}
    candidates = [item for item in available if item[1]["source_id"] not in target_ids]
    limit = max(0, int(getattr(args, "shared_pretrain_sources", 24)))
    if len(candidates) < 2 or limit == 0:
        return None
    candidates.sort(key=lambda item: (int(item[1].get("rank", 999999)), item[1]["source_id"]))
    if len(candidates) > limit:
        indices = deterministic_sample([{"index": index} for index in range(len(candidates))], limit)
        candidates = [candidates[int(row["index"])] for row in indices]
    manifest = root / "manifests" / f"shared_pretrain_v{CLONE_ALGORITHM_VERSION}.json"
    write_json(manifest, {
        "family": "generic_replay",
        "clone_algorithm_version": CLONE_ALGORITHM_VERSION,
        "training_stage": "shared_pretrain",
        "seed": int(args.seed) + 900000,
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "head_num_leaves": 63,
        "min_child_samples": 20,
        "head_n_estimators": 450,
        "value_n_estimators": 450,
        "n_jobs": int(args.model_jobs),
        "parse_workers": 4,
        "task_type": str(getattr(args, "task_type", "GPU")),
        "devices": str(getattr(args, "gpu_devices", "0") or ""),
        "eval_metric": "NDCG:top=3",
        "enable_main_type_heads": True,
        "type_balance_max_weight": 6.0,
        "cache_dir": str(root / "cache" / f"shared_pretrain_v{CLONE_ALGORITHM_VERSION}"),
        "sources": [
            {
                "name": source["source_id"],
                "path": str(root / source["replay_path"]),
                "target_name": source["team_name"],
                "split": "train",
            }
            for _, source, _ in candidates
        ],
    })
    print(f"[shared] training sources={len(candidates)} artifact={artifact}", flush=True)
    train(manifest, artifact)
    print(f"[shared] ready artifact={artifact}", flush=True)
    return artifact


def build_sources(args: argparse.Namespace) -> None:
    root = args.root
    snapshot_data = read_json(root / "snapshot.json")
    registry = read_json(root / "registry.json", {"schema_version": SCHEMA_VERSION, "entries": []})
    existing = {entry["source_id"]: entry for entry in registry["entries"]}
    sources = snapshot_data["sources"]
    build_workers = max(1, int(args.build_workers))
    task_type = str(getattr(args, "task_type", "GPU"))
    gpu_devices = str(getattr(args, "gpu_devices", "0") or "")
    ready: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []
    for number, source in enumerate(sources, 1):
        source_id = source["source_id"]
        replay_dir = root / source["replay_path"]
        replay_count = len(list(replay_dir.glob("*replay.json*")))
        previous = existing.get(source_id)
        same_version = int((previous or {}).get("clone_algorithm_version", 0) or 0) == CLONE_ALGORITHM_VERSION
        force_rebuild = bool(getattr(args, "force_rebuild", False))
        if previous and previous.get("status") == "qualified" and same_version and not force_rebuild:
            continue
        if replay_count < args.min_episodes:
            existing[source_id] = {
                **source,
                "replay_count": replay_count,
                "status": "rejected",
                "rejection_reason": "insufficient_episodes",
            }
            continue
        stable_rejection = previous and str(previous.get("rejection_reason", "")) in {
            "insufficient_episodes",
            "quality_gate",
        }
        if stable_rejection and same_version and not force_rebuild and replay_count <= int(previous.get("replay_count", 0)):
            continue
        ready.append((number, source, previous))
    all_ready = list(ready)
    pilot_size = max(0, int(getattr(args, "pilot_size", 0)))
    if pilot_size > 0 and len(ready) > pilot_size:
        required_team = str(getattr(args, "required_team", "") or "").casefold()
        required = [item for item in ready if str(item[1].get("team_name", "")).casefold() == required_team]
        bands = [[], [], [], [], []]
        for item in ready:
            if item in required:
                continue
            rank = int(item[1].get("rank", 999999))
            band = 0 if rank <= 20 else 1 if rank <= 100 else 2 if rank <= 500 else 3 if rank <= 1500 else 4
            bands[band].append(item)
        selected = required[:1]
        while len(selected) < pilot_size and any(bands):
            for band in bands:
                if band and len(selected) < pilot_size:
                    selected.append(band.pop(0))
        ready = selected
    pilot_ids = [source["source_id"] for _, source, _ in ready] if pilot_size > 0 else []
    if pilot_ids:
        write_json(root / "pilot_selection.json", {
            "clone_algorithm_version": CLONE_ALGORITHM_VERSION,
            "source_ids": pilot_ids,
        })
    registry["entries"] = sorted(existing.values(), key=lambda item: (int(item.get("rank", 999999)), item["source_id"]))
    write_json(root / "registry.json", registry)
    print(
        f"[build] workers={build_workers} model_jobs={args.model_jobs} "
        f"task_type={task_type} devices={gpu_devices or 'cpu'} "
        f"sources={len(sources)} ready={len(ready)}",
        flush=True,
    )
    if not ready:
        return
    args.shared_init_artifact = ensure_shared_pretrain(root, all_ready, ready, args)
    executor_class = ProcessPoolExecutor if bool(getattr(args, "opponent_only_fast", False)) and task_type == "CPU" else ThreadPoolExecutor
    with executor_class(max_workers=build_workers) as executor:
        futures = {
            executor.submit(build_source, root, source, number, args, previous): (number, source["source_id"])
            for number, source, previous in ready
        }
        for completed, future in enumerate(as_completed(futures), 1):
            number, source_id = futures[future]
            row = future.result()
            existing[source_id] = row
            registry["entries"] = sorted(existing.values(), key=lambda item: (int(item.get("rank", 999999)), item["source_id"]))
            write_json(root / "registry.json", registry)
            print(
                f"[build] {completed}/{len(ready)} source_index={number} "
                f"{source_id} status={row['status']}",
                flush=True,
            )
    if pilot_ids:
        rows = [existing[source_id] for source_id in pilot_ids if source_id in existing]
        qualified = [row for row in rows if row.get("status") == "qualified"]
        semantic_values = [float((row.get("quality_gate") or {}).get("metrics", {}).get("semantic", 0.0)) for row in rows]
        turn_macro_values = [float((row.get("quality_gate") or {}).get("metrics", {}).get("turn_macro", 0.0)) for row in rows]
        attack_values = [float((row.get("quality_gate") or {}).get("metrics", {}).get("attack", 0.0)) for row in rows]
        required_team = str(getattr(args, "required_team", "") or "")
        required_rows = [row for row in rows if str(row.get("team_name", "")).casefold() == required_team.casefold()]
        report = {
            "clone_algorithm_version": CLONE_ALGORITHM_VERSION,
            "selected": len(rows),
            "qualified": len(qualified),
            "required_qualified": min(len(rows), int(getattr(args, "pilot_min_qualified", 7))),
            "semantic_median": statistics.median(semantic_values) if semantic_values else 0.0,
            "turn_macro_median": statistics.median(turn_macro_values) if turn_macro_values else 0.0,
            "attack_median": statistics.median(attack_values) if attack_values else 0.0,
            "required_team": required_team or None,
            "required_team_qualified": not required_team or bool(required_rows and required_rows[0].get("status") == "qualified"),
            "source_ids": pilot_ids,
        }
        report["passed"] = (
            report["qualified"] >= report["required_qualified"]
            and report["turn_macro_median"] >= float(getattr(args, "pilot_min_turn_macro_median", 0.80))
            and report["attack_median"] >= float(getattr(args, "pilot_min_attack_median", 0.85))
            and report["required_team_qualified"]
        )
        write_json(root / "pilot_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if not report["passed"]:
            raise RuntimeError(f"opponent clone pilot failed: {report}")


def category(entry: dict[str, Any]) -> str:
    if entry.get("source_type") in {"existing", "specialist", "historical", "generated"}:
        return "specialist"
    return "top100" if int(entry.get("rank", 999999)) <= 100 else "mid"


def quality_score(entry: dict[str, Any]) -> float:
    audit_data = entry.get("audit", {})
    gate_metrics = (entry.get("quality_gate") or {}).get("metrics", {})
    strength = (audit_data.get("strength_gate") or {}).get("score", 0.5)
    rank = int(entry.get("rank", 3000))
    return (
        0.30 * float(gate_metrics.get("turn_macro", audit_data.get("semantic_rate", 0.0)))
        + 0.25 * float(gate_metrics.get("attack", 0.0))
        + 0.20 * float(strength)
        + 0.15 * max(0.0, 1.0 - rank / 3000.0)
        + 0.10 * min(1.0, int(entry.get("decisions", 0)) / 5000.0)
    )


def freeze(args: argparse.Namespace) -> None:
    root = args.root
    registry = read_json(root / "registry.json", {"entries": []})
    entries = [entry for entry in registry["entries"] if entry.get("status") == "qualified" and Path(entry["package"]).exists()]
    for path in args.existing:
        entries.append({
            "source_id": f"existing_{path.stem}", "source_type": "existing", "rank": 999999,
            "package": str(path), "package_sha256": sha256_file(path), "deck_sha256": sha256_file(path),
            "decisions": 5000, "audit": {"semantic_rate": 1.0}, "status": "qualified",
        })
    for path in args.generated:
        entries.append({
            "source_id": f"generated_{path.stem}", "source_type": "generated", "rank": 999999,
            "package": str(path), "package_sha256": sha256_file(path), "deck_sha256": sha256_file(path),
            "decisions": 5000, "audit": {"semantic_rate": 1.0}, "status": "qualified",
        })
    entries.sort(key=quality_score, reverse=True)
    deduped: list[dict[str, Any]] = []
    deck_counts: dict[str, int] = {}
    behavior_seen: set[str] = set()
    team_seen: set[str] = set()
    for entry in entries:
        deck_hash = str(entry.get("deck_sha256", ""))
        signature = str(entry.get("behavior_signature", entry.get("package_sha256", "")))
        team_key = str(entry.get("team_id", entry.get("source_id", "")))
        cluster_key = f"{deck_hash}:{signature}"
        if cluster_key in behavior_seen or team_key in team_seen:
            continue
        if deck_counts.get(deck_hash, 0) >= 1:
            continue
        behavior_seen.add(cluster_key)
        team_seen.add(team_key)
        deck_counts[deck_hash] = deck_counts.get(deck_hash, 0) + 1
        deduped.append(entry)
    target_counts = {"top100": round(args.target_size * 0.60), "mid": round(args.target_size * 0.25)}
    target_counts["specialist"] = args.target_size - target_counts["top100"] - target_counts["mid"]
    selected: list[dict[str, Any]] = []
    for name, count in target_counts.items():
        selected.extend([entry for entry in deduped if category(entry) == name][:count])
    selected_ids = {entry["source_id"] for entry in selected}
    selected.extend(entry for entry in deduped if entry["source_id"] not in selected_ids)
    if len(selected) < args.minimum_size:
        raise RuntimeError(f"only {len(selected)} qualified diverse opponents; minimum is {args.minimum_size}")
    selected = selected[:args.target_size]
    split_sizes = {"train": round(len(selected) * 0.60), "dev": round(len(selected) * 0.20)}
    split_sizes["holdout"] = len(selected) - split_sizes["train"] - split_sizes["dev"]
    buckets = {"train": [], "dev": [], "holdout": []}
    by_category = {name: [entry for entry in selected if category(entry) == name] for name in ("top100", "mid", "specialist")}
    for rows in by_category.values():
        for index, entry in enumerate(rows):
            fraction = index / max(1, len(rows))
            split = "train" if fraction < 0.60 else "dev" if fraction < 0.80 else "holdout"
            buckets[split].append(entry)
    for split, rows in buckets.items():
        manifest_rows = [{
            "name": entry["source_id"], "path": entry["package"], "weight": 1.0,
            "category": category(entry), "lineage_id": entry["source_id"],
            "deck_sha256": entry.get("deck_sha256", ""),
        } for entry in rows]
        write_json(root / "pool" / f"{split}.json", {"schema_version": SCHEMA_VERSION, "split": split, "opponents": manifest_rows})
    coverage = {
        "selected": len(selected),
        "unique_teams": len({str(entry.get("team_id", entry["source_id"])) for entry in selected}),
        "unique_decks": len({str(entry.get("deck_sha256", "")) for entry in selected}),
        "unique_behaviors": len({str(entry.get("behavior_signature", entry.get("package_sha256", ""))) for entry in selected}),
        "top100": sum(category(entry) == "top100" for entry in selected),
        "generated": sum(entry.get("source_type") == "generated" for entry in selected),
    }
    requirements = {
        "unique_decks": args.minimum_unique_decks,
        "unique_behaviors": args.minimum_unique_behaviors,
        "top100": args.minimum_top100,
    }
    missing = {name: (coverage[name], minimum) for name, minimum in requirements.items() if coverage[name] < minimum}
    if missing:
        raise RuntimeError(f"opponent coverage below minimum: {missing}")
    write_json(root / "pool" / "coverage.json", coverage)
    write_json(root / "pool" / "summary.json", {"target": args.target_size, "selected": len(selected), "categories": target_counts, "splits": {key: len(value) for key, value in buckets.items()}, "coverage": coverage})
    print(json.dumps(read_json(root / "pool" / "summary.json"), indent=2))


def status(args: argparse.Namespace) -> None:
    snapshot_data = read_json(args.root / "snapshot.json", {"sources": []})
    state = read_json(args.root / "collect_state.json", {"completed": [], "failures": {}})
    registry = read_json(args.root / "registry.json", {"entries": []})
    counts: dict[str, int] = {}
    for entry in registry["entries"]:
        counts[entry.get("status", "unknown")] = counts.get(entry.get("status", "unknown"), 0) + 1
    print(json.dumps({
        "sources": len(snapshot_data["sources"]),
        "episodes_planned": sum(len(row["episode_ids"]) for row in snapshot_data["sources"]),
        "episodes_collected": len(state.get("completed", [])),
        "download_failures": len(state.get("failures", {})),
        "registry": counts,
        "pool": read_json(args.root / "pool" / "summary.json", {}),
    }, indent=2))


def update(args: argparse.Namespace) -> None:
    if (args.root / "snapshot.json").exists():
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        snapshot_dir = args.root / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.root / "snapshot.json", snapshot_dir / f"snapshot_{stamp}.json")
    args.out = args.root
    snapshot(args)
    collect(args)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build and maintain a leakage-safe Kaggle opponent league.")
    sub = result.add_subparsers(dest="command", required=True)
    snapshot_parser = sub.add_parser("snapshot")
    snapshot_parser.add_argument("--out", type=Path, required=True)
    snapshot_parser.add_argument("--competition", default=COMPETITION)
    snapshot_parser.add_argument("--kaggle", default=str(Path.home() / ".local/bin/kaggle"))
    snapshot_parser.add_argument("--leaderboard-csv", type=Path)
    snapshot_parser.set_defaults(func=snapshot)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--root", type=Path, required=True)
    collect_parser.add_argument("--kaggle", default=str(Path.home() / ".local/bin/kaggle"))
    collect_parser.add_argument("--staging", type=Path, default=Path("/tmp/opponent-league-staging"))
    collect_parser.add_argument("--workers", type=int, default=2)
    collect_parser.add_argument("--checkpoint-every", type=int, default=50)
    collect_parser.add_argument("--request-interval", type=float, default=1.55)
    collect_parser.add_argument("--request-jitter", type=float, default=0.0)
    collect_parser.add_argument("--max-retries", type=int, default=5)
    collect_parser.add_argument("--max-backoff", type=float, default=300.0)
    collect_parser.add_argument("--seed", type=int, default=20260804)
    collect_parser.add_argument("--remote")
    collect_parser.add_argument("--remote-root")
    collect_parser.add_argument("--ssh-port", type=int, default=22)
    collect_parser.set_defaults(func=collect)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--root", type=Path, required=True)
    build_parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    build_parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    build_parser.add_argument("--min-episodes", type=int, default=30)
    build_parser.add_argument("--min-decisions", type=int, default=1000)
    build_parser.add_argument("--build-workers", type=int, default=4)
    build_parser.add_argument("--model-jobs", type=int, default=12)
    build_parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    build_parser.add_argument("--gpu-devices", default="0")
    build_parser.add_argument("--seed", type=int, default=20260804)
    build_parser.add_argument("--force-rebuild", action="store_true")
    build_parser.add_argument("--opponent-only-fast", action="store_true")
    build_parser.add_argument("--disable-shared-pretrain", action="store_true")
    build_parser.add_argument("--force-shared-pretrain", action="store_true")
    build_parser.add_argument("--shared-pretrain-sources", type=int, default=24)
    build_parser.add_argument("--pilot-size", type=int, default=0)
    build_parser.add_argument("--pilot-min-qualified", type=int, default=7)
    build_parser.add_argument("--pilot-min-turn-macro-median", type=float, default=0.80)
    build_parser.add_argument("--pilot-min-attack-median", type=float, default=0.85)
    build_parser.add_argument("--required-team", default="Majkel1337")
    build_parser.add_argument("--required-history-replays", type=Path)
    build_parser.add_argument("--required-history-weight", type=float, default=0.5)
    build_parser.add_argument("--strength-baseline", type=Path)
    build_parser.add_argument("--strength-games", type=int, default=256)
    build_parser.add_argument("--strength-workers", type=int, default=16)
    build_parser.set_defaults(func=build_sources)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--target-size", type=int, default=120)
    freeze_parser.add_argument("--minimum-size", type=int, default=80)
    freeze_parser.add_argument("--existing", nargs="*", type=Path, default=[])
    freeze_parser.add_argument("--generated", nargs="*", type=Path, default=[])
    freeze_parser.add_argument("--minimum-unique-decks", type=int, default=60)
    freeze_parser.add_argument("--minimum-unique-behaviors", type=int, default=100)
    freeze_parser.add_argument("--minimum-top100", type=int, default=50)
    freeze_parser.set_defaults(func=freeze)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--root", type=Path, required=True)
    status_parser.set_defaults(func=status)
    update_parser = sub.add_parser("update")
    update_parser.add_argument("--root", type=Path, required=True)
    update_parser.add_argument("--competition", default=COMPETITION)
    update_parser.add_argument("--kaggle", required=True)
    update_parser.add_argument("--leaderboard-csv", type=Path)
    update_parser.add_argument("--staging", type=Path, default=Path("/tmp/opponent-league-staging"))
    update_parser.add_argument("--workers", type=int, default=2)
    update_parser.add_argument("--checkpoint-every", type=int, default=50)
    update_parser.add_argument("--request-interval", type=float, default=3.0)
    update_parser.add_argument("--request-jitter", type=float, default=2.0)
    update_parser.add_argument("--max-retries", type=int, default=5)
    update_parser.add_argument("--max-backoff", type=float, default=300.0)
    update_parser.add_argument("--seed", type=int, default=20260804)
    update_parser.add_argument("--remote")
    update_parser.add_argument("--remote-root")
    update_parser.add_argument("--ssh-port", type=int, default=22)
    update_parser.set_defaults(func=update)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
