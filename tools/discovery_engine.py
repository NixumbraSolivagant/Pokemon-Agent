from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file
from local_eval.evaluator import run_candidate_pool, save_report, submission_name
from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.build_models import BuildConfig
from tools.build_submission import build_submission
from tools.discovery_feedback import aggregate_feedback, write_pressure_artifacts
from tools.discovery_space import generate_discovery_configs, read_build_metadata
from tools.export_kaggle_submission import export_kaggle_submission
from tools.gold_gate import classify_candidate as classify_gold_candidate
from tools.loss_mining import digest_loss_files, digest_scenarios, load_scenarios as load_loss_scenarios, mine_losses
from tools.opponent_models import OpponentArchive, generate_counter_opponents
from tools.policy_genome import StrategyArchive, StrategyGenome, compile_opponent_genome
from tools.psro import psro_bonus_names, write_psro
from tools.racing import rank_for_racing, report_racing, robust_stats, wilson_lower_bound
from tools.reference_pool import DEFAULT_REFERENCE_PATHS
from tools.scenario_eval import run_scenarios


DEFAULT_OUT = Path("outputs/discovery_gold")
DEFAULT_INCUMBENT = Path("outputs/submissions/champion_latest.tar.gz")
DEFAULT_PROMOTE = Path("outputs/submissions/champion_discovery.tar.gz")
DEFAULT_SUBMISSION = Path("outputs/submissions/submission_discovery.tar.gz")
DEFAULT_POOL = DEFAULT_REFERENCE_PATHS
MAX_DISCOVERY_WORKERS = 30
REQUIRED_ANCHOR_NAMES = (
    "i-have-one-rear-card",
    "submission_820",
    "pokemon-ai-battle-best-ptcg-advanced",
    "multiply-agent-best-940-lb",
)


@dataclass(slots=True)
class Stage:
    name: str
    games_per_pair: int
    candidate_limit: int
    pool_limit: int
    record_mode: str
    finalists: int


@dataclass(slots=True)
class DiscoveryProfile:
    name: str
    generations: int
    population: int
    workers: int
    max_actions: int
    run_timeout_s: float
    stage_a: Stage
    stage_b: Stage
    stage_c: Stage
    stage_d: Stage
    manual_slots: int
    max_no_result_rate: float
    min_holdout_score_delta: float
    psro_slots: int


@dataclass(slots=True)
class CandidateRecord:
    name: str
    family: str
    origin: str
    internal_tarball: str
    eval_tarball: str
    notes: str
    sha256: str
    config: dict[str, Any]


def profile_from_name(name: str) -> DiscoveryProfile:
    if name == "smoke_discovery":
        return DiscoveryProfile(
            name=name,
            generations=1,
            population=10,
            workers=2,
            max_actions=260,
            run_timeout_s=90.0,
            stage_a=Stage("stage_a", 1, 10, 3, "none", 5),
            stage_b=Stage("stage_b", 1, 5, 3, "none", 3),
            stage_c=Stage("stage_c", 1, 3, 3, "losses", 2),
            stage_d=Stage("stage_d_holdout", 1, 2, 4, "losses", 2),
            manual_slots=3,
            max_no_result_rate=0.03,
            min_holdout_score_delta=-25.0,
            psro_slots=1,
        )
    if name == "a800_discovery":
        return DiscoveryProfile(
            name=name,
            generations=24,
            population=144,
            workers=0,
            max_actions=1000,
            run_timeout_s=180.0,
            stage_a=Stage("stage_a", 4, 96, 8, "none", 32),
            stage_b=Stage("stage_b", 16, 32, 8, "sample", 12),
            stage_c=Stage("stage_c_confirm", 64, 12, 10, "sample", 6),
            stage_d=Stage("stage_d_holdout", 512, 4, 12, "sample", 4),
            manual_slots=8,
            max_no_result_rate=0.01,
            min_holdout_score_delta=8.0,
            psro_slots=4,
        )
    if name != "a800_turbo_discovery":
        raise ValueError(
            f"Unknown profile {name!r}; expected smoke_discovery, a800_discovery, or a800_turbo_discovery"
        )
    return DiscoveryProfile(
        name=name,
        generations=24,
        population=144,
        workers=0,
        max_actions=1000,
        run_timeout_s=180.0,
        stage_a=Stage("stage_a", 4, 144, 10, "none", 48),
        stage_b=Stage("stage_b", 16, 48, 10, "sample", 16),
        stage_c=Stage("stage_c_confirm", 64, 16, 12, "sample", 6),
        stage_d=Stage("stage_d_holdout", 512, 4, 14, "sample", 4),
        manual_slots=10,
        max_no_result_rate=0.01,
        min_holdout_score_delta=8.0,
        psro_slots=6,
    )


def resolve_workers(args: argparse.Namespace, profile: DiscoveryProfile) -> int:
    requested = args.workers if args.workers is not None else profile.workers
    if requested and requested > 0:
        return min(MAX_DISCOVERY_WORKERS, requested)
    cpu_count = os.cpu_count() or max(1, profile.workers)
    workers = max(1, cpu_count - max(0, args.cpu_headroom))
    if args.max_workers and args.max_workers > 0:
        workers = min(workers, args.max_workers)
    return min(MAX_DISCOVERY_WORKERS, workers)


def resolve_build_workers(args: argparse.Namespace, profile: DiscoveryProfile) -> int:
    requested = getattr(args, "build_workers", None)
    if requested is not None:
        return max(1, requested)
    return max(1, min(MAX_DISCOVERY_WORKERS, resolve_workers(args, profile)))


def config_to_dict(cfg: BuildConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["base"] = str(cfg.base)
    data["out"] = str(cfg.out)
    data["runtime_source"] = str(cfg.runtime_source)
    data["runtime_cg_dir"] = str(cfg.runtime_cg_dir)
    return data


def candidate_generation_summary(configs: list[BuildConfig], requested_population: int, records_built: int | None = None) -> dict[str, Any]:
    origins: dict[str, int] = {}
    families: dict[str, int] = {}
    deck_overrides = 0
    opponent_models: dict[str, int] = {}
    policy_variants: dict[str, int] = {}
    lineages: dict[str, int] = {}
    deck_hashes: set[str] = set()
    for cfg in configs:
        origins[cfg.origin or "unknown"] = origins.get(cfg.origin or "unknown", 0) + 1
        families[cfg.family or "unknown"] = families.get(cfg.family or "unknown", 0) + 1
        opponent_models[cfg.opponent_model or "perfect"] = opponent_models.get(cfg.opponent_model or "perfect", 0) + 1
        policy_variants[cfg.policy_variant or "default"] = policy_variants.get(cfg.policy_variant or "default", 0) + 1
        if cfg.deck_override:
            deck_overrides += 1
            deck_hashes.add(hashlib.sha256(",".join(map(str, sorted(cfg.deck_override))).encode("utf-8")).hexdigest())
        lineage = str(cfg.strategy_genome.get("lineage", "unknown")) if cfg.strategy_genome else "unknown"
        lineages[lineage] = lineages.get(lineage, 0) + 1
    summary = {
        "population_requested": requested_population,
        "configs_generated": len(configs),
        "records_built": records_built,
        "underfilled": len(configs) < requested_population,
        "underfill_ratio": (len(configs) / requested_population) if requested_population else 1.0,
        "deck_overrides": deck_overrides,
        "unique_decks": len(deck_hashes),
        "lineage_counts": dict(sorted(lineages.items(), key=lambda kv: (-kv[1], kv[0]))),
        "origin_counts": dict(sorted(origins.items(), key=lambda kv: (-kv[1], kv[0]))),
        "family_counts": dict(sorted(families.items(), key=lambda kv: (-kv[1], kv[0]))),
        "opponent_model_counts": dict(sorted(opponent_models.items(), key=lambda kv: (-kv[1], kv[0]))),
        "policy_variant_counts": dict(sorted(policy_variants.items(), key=lambda kv: (-kv[1], kv[0]))[:24]),
    }
    if records_built is not None:
        summary["build_success_rate"] = records_built / max(1, len(configs))
    return summary


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def unique_existing(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            digest = sha256_file(path)
        except Exception:
            continue
        if digest in seen:
            continue
        seen.add(digest)
        out.append(path.resolve())
    return out


def clean_stats(stats: AgentStats, max_no_result_rate: float) -> bool:
    return (
        stats.crashes == 0
        and stats.timeouts == 0
        and stats.invalids == 0
        and stats.no_results / max(1, stats.games) <= max_no_result_rate
    )


def _progress_overwrite(mode: str) -> bool:
    mode = (mode or "auto").lower()
    return mode == "overwrite" or (mode == "auto" and os.isatty(1))


def _write_progress_file(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(line + "\n", encoding="utf-8")
    tmp.replace(path)


def _print_progress(
    label: str,
    phase: str,
    done: int,
    total: int,
    extra: str = "",
    mode: str = "auto",
    force: bool = False,
    progress_file: Path | None = None,
) -> None:
    total = max(0, total)
    ratio = (done / total) if total else 1.0
    width = 28
    filled = min(width, max(0, int(ratio * width)))
    bar = "#" * filled + "-" * (width - filled)
    line = f"[{label}] {phase} [{bar}] {done}/{total} {ratio * 100:5.1f}%"
    if extra:
        line += f" | {extra}"
    if (mode or "auto").lower() == "file":
        if progress_file is not None:
            _write_progress_file(progress_file, line)
        return
    if _progress_overwrite(mode):
        end = "\n" if force else ""
        print("\r\033[K" + line, end=end, flush=True)
    else:
        print(line, flush=True)


def build_records(
    configs: list[BuildConfig],
    eval_dir: Path,
    strip_search_wrapper: bool,
    build_workers: int = 1,
    progress: bool = False,
    progress_label: str = "build",
    progress_interval_s: float = 5.0,
    progress_mode: str = "auto",
    progress_file: Path | None = None,
    failures_out: Path | None = None,
) -> list[CandidateRecord]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    total = len(configs)
    last_progress = 0.0
    if progress:
        _print_progress(
            progress_label,
            "build",
            0,
            total,
            "starting candidate tarball export",
            mode=progress_mode,
            progress_file=progress_file,
        )

    def build_one(index: int, cfg: BuildConfig) -> tuple[int, CandidateRecord | None, dict[str, Any] | None]:
        try:
            built = build_submission(cfg)
            eval_tarball = eval_dir / built.name
            export_kaggle_submission(
                built,
                eval_tarball,
                strip_search_wrapper=strip_search_wrapper,
                exclude_internal_files=False,
            )
            return (
                index,
                CandidateRecord(
                    name=cfg.name,
                    family=cfg.family,
                    origin=cfg.origin,
                    internal_tarball=str(built.resolve()),
                    eval_tarball=str(eval_tarball.resolve()),
                    notes=cfg.notes,
                    sha256=sha256_file(built),
                    config=config_to_dict(cfg),
                ),
                None,
            )
        except Exception as exc:
            return index, None, {"skipped": cfg.name, "reason": f"{type(exc).__name__}: {exc}"}

    records_by_index: dict[int, CandidateRecord] = {}
    failures: list[dict[str, Any]] = []
    completed = 0
    workers = max(1, int(build_workers))
    if workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(build_one, index, cfg) for index, cfg in enumerate(configs, start=1)]
            for future in as_completed(futures):
                index, record, failure = future.result()
                completed += 1
                if record is not None:
                    records_by_index[index] = record
                if failure is not None:
                    failures.append(failure)
                    print(json.dumps(failure))
                now = time.monotonic()
                if progress and (completed == total or now - last_progress >= max(0.25, progress_interval_s)):
                    last_progress = now
                    name = configs[index - 1].name if 0 <= index - 1 < len(configs) else ""
                    _print_progress(
                        progress_label,
                        "build",
                        completed,
                        total,
                        f"built={len(records_by_index)} failed={len(failures)} last={name}",
                        mode=progress_mode,
                        force=completed == total,
                        progress_file=progress_file,
                    )
    else:
        for index, cfg in enumerate(configs, start=1):
            _, record, failure = build_one(index, cfg)
            completed += 1
            if record is not None:
                records_by_index[index] = record
            if failure is not None:
                failures.append(failure)
                print(json.dumps(failure))
            now = time.monotonic()
            if progress and (index == total or now - last_progress >= max(0.25, progress_interval_s)):
                last_progress = now
                _print_progress(
                    progress_label,
                    "build",
                    index,
                    total,
                    f"built={len(records_by_index)} failed={len(failures)} last={cfg.name}",
                    mode=progress_mode,
                    force=index == total,
                    progress_file=progress_file,
                )
    if failures_out is not None:
        failures_out.parent.mkdir(parents=True, exist_ok=True)
        failures_out.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return [records_by_index[index] for index in sorted(records_by_index)]


def candidate_by_name(records: list[CandidateRecord]) -> dict[str, CandidateRecord]:
    return {record.name: record for record in records}


def stats_by_name(report: MatchReport) -> dict[str, AgentStats]:
    return {stats.name: stats for stats in report.standings}


def score_delta(report: MatchReport, candidate: str, incumbent: str) -> tuple[float, int, int]:
    stats = stats_by_name(report)
    cand = stats[candidate]
    inc = stats[incumbent]
    rec = cand.opponents.get(incumbent, {"wins": 0, "losses": 0, "draws": 0})
    return cand.kaggle_score_estimate - inc.kaggle_score_estimate, rec["wins"], rec["losses"]


def no_result_rate(row: dict[str, Any]) -> float:
    return float(row.get("no_results") or 0) / max(1.0, float(row.get("games") or 0))


def inspect_submission(path: Path) -> dict[str, Any]:
    required = {"main.py", "deck.csv", "cg/game.py", "cg/api.py"}
    result = {
        "path": str(path),
        "exists": path.exists(),
        "required_ok": False,
        "missing": sorted(required),
        "wrapper_count": 0,
        "last_alias_ok": False,
        "error": "",
    }
    if not path.exists():
        result["error"] = "submission file does not exist"
        return result
    try:
        with tarfile.open(path, "r:gz") as tar:
            names = set(tar.getnames())
            result["missing"] = sorted(required - names)
            result["required_ok"] = not result["missing"]
            payload = tar.extractfile("main.py")
            main = payload.read().decode("utf-8") if payload else ""
        marker = "# --- Champion Great Tusk search wrapper injected by tools.build_submission ---"
        result["wrapper_count"] = main.count(marker)
        search_idx = main.rfind("def _gt_search_action")
        alias_idx = main.rfind("kaggle_agent = agent")
        result["last_alias_ok"] = search_idx < 0 or alias_idx > search_idx
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def classify_final(
    final: dict[str, Any],
    submission_check: dict[str, Any] | None = None,
    min_delta: float = 8.0,
    max_no_result_rate: float = 0.01,
    near_clean_no_result_rate: float = 0.02,
) -> dict[str, Any]:
    reasons: list[str] = []
    status = str(final.get("status", ""))
    delta = float(final.get("score_delta_vs_incumbent") or 0.0)
    h2h_wins = int(final.get("h2h_wins") or 0)
    h2h_losses = int(final.get("h2h_losses") or 0)
    h2h_wilson = final.get("h2h_wilson_low")
    robust_margin = final.get("robust_score_margin")
    stage_rows = [row for row in final.get("stage_d_top", []) if isinstance(row, dict)]
    best_row = next((row for row in stage_rows if row.get("name") == final.get("name") or row.get("name") == final.get("best")), {})
    nr_rate = no_result_rate(best_row) if best_row else 0.0
    package_ok = True
    if submission_check is not None:
        package_ok = bool(submission_check.get("required_ok")) and bool(submission_check.get("last_alias_ok")) and not submission_check.get("error")
        if not package_ok:
            reasons.append("submission package check failed")
    if status == "failed":
        return {"champion_class": "failed", "decision": "rerun_required", "submit_ready": False, "reasons": ["run failed", *reasons]}
    if status != "promoted":
        return {"champion_class": "portfolio_candidate", "decision": "hold_portfolio", "submit_ready": False, "reasons": ["not promoted", *reasons]}
    if h2h_wilson is None and delta < min_delta:
        reasons.append(f"score_delta {delta:.3f} below {min_delta:.3f}")
    if h2h_wilson is not None and float(h2h_wilson) < 0.50:
        reasons.append(f"H2H Wilson lower bound {float(h2h_wilson):.3f} below 0.500")
    if robust_margin is not None and float(robust_margin) < 0.0:
        reasons.append(f"anchor robust score margin {float(robust_margin):.3f} below 0.000")
    if h2h_wins < h2h_losses:
        reasons.append(f"H2H losing vs incumbent: {h2h_wins}-{h2h_losses}")
    if nr_rate > max_no_result_rate:
        reasons.append(f"no_result_rate {nr_rate:.4f} above {max_no_result_rate:.4f}")
    if nr_rate > max_no_result_rate and nr_rate <= near_clean_no_result_rate:
        return {"champion_class": "unstable_candidate", "decision": "hold_portfolio", "submit_ready": False, "reasons": reasons}
    if reasons:
        return {"champion_class": "portfolio_candidate", "decision": "hold_portfolio", "submit_ready": False, "reasons": reasons}
    if not package_ok:
        return {"champion_class": "portfolio_candidate", "decision": "hold_portfolio", "submit_ready": False, "reasons": reasons}
    return {"champion_class": "strict_champion", "decision": "gold_gate_required", "submit_ready": False, "reasons": ["strict incumbent gate passed; gold gate still required"]}


def write_decision_brief(out: Path, final: dict[str, Any], decision: dict[str, Any], submission_check: dict[str, Any] | None = None) -> None:
    lines = [
        "# Discovery Decision Brief",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- champion_class: `{decision.get('champion_class')}`",
        f"- submit_ready: `{decision.get('submit_ready')}`",
        f"- status: `{final.get('status')}`",
        f"- candidate: `{final.get('name') or final.get('best', '')}`",
        f"- score_delta_vs_incumbent: `{final.get('score_delta_vs_incumbent')}`",
        f"- h2h: `{final.get('h2h_wins', 0)}-{final.get('h2h_losses', 0)}`",
        f"- score_warning: `local_trueskill_score is not Kaggle leaderboard score`",
        "",
        "## Reasons",
    ]
    for reason in decision.get("reasons", []):
        lines.append(f"- {reason}")
    if submission_check is not None:
        lines.extend(
            [
                "",
                "## Submission Check",
                f"- required_ok: `{submission_check.get('required_ok')}`",
                f"- missing: `{submission_check.get('missing')}`",
                f"- wrapper_count: `{submission_check.get('wrapper_count')}`",
                f"- last_alias_ok: `{submission_check.get('last_alias_ok')}`",
                f"- error: `{submission_check.get('error')}`",
            ]
        )
    gold_gate = final.get("gold_gate") or {}
    if isinstance(gold_gate, dict) and gold_gate:
        lines.extend(
            [
                "",
                "## Gold Gate",
                f"- decision: `{gold_gate.get('decision')}`",
                f"- gold_gate_passed: `{gold_gate.get('gold_gate_passed')}`",
                f"- target_lb_score: `{gold_gate.get('target_lb_score')}`",
                f"- known_lb_score: `{gold_gate.get('known_lb_score')}`",
                f"- submission_sha256: `{gold_gate.get('submission_sha256')}`",
            ]
        )
        for reason in gold_gate.get("reasons", []):
            lines.append(f"- {reason}")
        for gate in gold_gate.get("matchup_gates", [])[:12]:
            if isinstance(gate, dict):
                lines.append(
                    f"- vs `{gate.get('opponent')}` win_rate=`{gate.get('win_rate')}` "
                    f"target=`{gate.get('target')}` passed=`{gate.get('passed')}`"
                )
    feedback = final.get("generation_feedback") or {}
    if isinstance(feedback, dict):
        lines.extend(
            [
                "",
                "## Feedback Pressure",
                f"- pressure_count: `{feedback.get('pressure_count')}`",
                f"- top_kinds: `{feedback.get('top_kinds')}`",
                f"- out: `{feedback.get('out')}`",
            ]
        )
    microburst = final.get("microburst") or {}
    if isinstance(microburst, dict):
        lines.extend(
            [
                "",
                "## Microburst",
                f"- built: `{microburst.get('built')}`",
                f"- selected: `{microburst.get('selected')}`",
                f"- feedback: `{microburst.get('feedback')}`",
            ]
        )
    generation = final.get("candidate_generation") or {}
    if isinstance(generation, dict):
        lines.extend(
            [
                "",
                "## Candidate Generation",
                f"- requested: `{generation.get('population_requested')}`",
                f"- generated: `{generation.get('configs_generated')}`",
                f"- built: `{generation.get('records_built')}`",
                f"- underfilled: `{generation.get('underfilled')}`",
                f"- origin_counts: `{generation.get('origin_counts')}`",
            ]
        )
    lines.append("")
    lines.append("## Stage D Top")
    for row in final.get("stage_d_top", [])[:10]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('name')}` local_trueskill_score=`{row.get('kaggle_score_estimate')}` "
            f"wl=`{row.get('wins')}-{row.get('losses')}` no_results=`{row.get('no_results')}`"
        )
    lines.append("")
    lines.append("## Manual Candidates")
    for row in final.get("manual_candidates", []):
        if isinstance(row, dict):
            lines.append(f"- `{row.get('name')}` role=`{row.get('role', row.get('origin', ''))}` path=`{row.get('tarball')}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_run_phase(state: dict[str, Any], out: Path, generation: int, phase: str, detail: str = "") -> None:
    state["active_generation"] = generation
    state["active_phase"] = phase
    state["active_detail"] = detail
    state["updated_at"] = int(time.time())
    write_json(out / "state.json", state)


def latest_scenario_path(state: dict[str, Any]) -> Path | None:
    for row in reversed(state.get("history", [])):
        final = row.get("final") or {}
        path = ((final.get("scenarios") or {}).get("out"))
        if path and Path(path).exists() and Path(path).stat().st_size > 2:
            return Path(path)
    return None


def scenario_path_if_useful(path: Path) -> Path | None:
    if path.exists() and path.stat().st_size > 2:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, list) and data:
            return path
        if isinstance(data, dict) and data.get("scenarios"):
            return path
    return None


def feedback_path_for_run(args: argparse.Namespace) -> Path | None:
    if getattr(args, "feedback_mode", "closed_loop") == "off":
        return None
    explicit = getattr(args, "feedback_path", None)
    if explicit:
        return explicit
    return args.out / "generation_feedback.json"


def _record_value(record: CandidateRecord | dict[str, Any], key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def resolve_feedback_target_paths(feedback: dict[str, Any], records: list[CandidateRecord | dict[str, Any]], limit: int = 16) -> list[Path]:
    pressure_items = [item for item in feedback.get("pressure_items", []) if isinstance(item, dict)]
    if not pressure_items:
        return []

    by_name: dict[str, Path] = {}
    by_family: dict[str, list[Path]] = {}
    for record in records:
        name = str(_record_value(record, "name") or "")
        family = str(_record_value(record, "family") or "")
        tarball = (
            _record_value(record, "internal_tarball")
            or _record_value(record, "tarball")
            or _record_value(record, "eval_tarball")
        )
        if not name or not tarball:
            continue
        path = Path(str(tarball))
        if not path.exists():
            continue
        resolved = path.resolve()
        by_name[name] = resolved
        if family:
            by_family.setdefault(family, []).append(resolved)

    out: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path in seen or len(out) >= limit:
            return
        seen.add(path)
        out.append(path)

    for item in pressure_items:
        target = str(item.get("target_candidate") or "")
        if target in by_name:
            add(by_name[target])
        family = str(item.get("target_family") or "")
        if family and family != "unknown":
            for path in by_family.get(family, [])[:2]:
                add(path)
        if len(out) >= limit:
            break
    return out


def history_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in reversed(state.get("history", [])):
        final = row.get("final") or {}
        records.extend(record for record in final.get("records", []) if isinstance(record, dict))
    return records


def opponent_pressure_from_feedback(feedback: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    motif_keys = {
        "switch_pivot": "switch_frequency",
        "resource_safety": "resource_denial",
        "hand_disruption": "resource_denial",
        "defensive_tools": "bench_pressure",
        "prize_race": "prize_race_bias",
        "fast_ko": "tempo",
    }
    for item in feedback.get("pressure_items", []):
        weight = max(0.0, float(item.get("severity", 0.0))) * max(0.0, float(item.get("confidence", 0.0))) * max(0.0, float(item.get("budget_weight", 1.0)))
        for motif in item.get("recommended_motifs", []):
            key = motif_keys.get(str(motif))
            if key:
                totals[key] = totals.get(key, 0.0) + weight
        opponent = str(item.get("target_opponent") or "").lower()
        if "mill" in opponent:
            totals["self_mill"] = totals.get("self_mill", 0.0) + weight
        if "lucario" in opponent or "ko" in opponent:
            totals["aggression"] = totals.get("aggression", 0.0) + weight
    scale = max(totals.values(), default=0.0)
    return {key: value / scale for key, value in totals.items()} if scale > 0 else {}


def build_generation_feedback(
    args: argparse.Namespace,
    gen_dir: Path,
    generation: int,
    incumbent_name: str,
    population: int,
    stage_names: list[str],
    scenario_paths: list[Path],
    out: Path,
    microburst: bool = False,
    previous_feedback: Path | None = None,
) -> dict[str, Any]:
    if getattr(args, "feedback_mode", "closed_loop") == "off":
        return {"version": 2, "generation": generation, "mode": "off", "pressure_items": []}
    audit_only_holdout = getattr(args, "holdout_feedback_mode", "audit_only") == "audit_only"
    reports = {
        stage: gen_dir / stage / "report.json"
        for stage in stage_names
        if (gen_dir / stage / "report.json").exists() and not (audit_only_holdout and "holdout" in stage)
    }
    psro_paths = [
        gen_dir / stage / "psro.json"
        for stage in stage_names
        if (gen_dir / stage / "psro.json").exists() and not (audit_only_holdout and "holdout" in stage)
    ]
    scenario_inputs = [
        path
        for path in scenario_paths
        if path.exists() and not (audit_only_holdout and path.name == "scenarios.json")
    ]
    feedback = aggregate_feedback(
        out=out,
        generation=generation,
        stage_reports=reports,
        scenario_paths=scenario_inputs,
        psro_paths=psro_paths,
        build_failures=gen_dir / "build_failures.json",
        incumbent_name=incumbent_name,
        population=population,
        holdout_feedback_mode=getattr(args, "holdout_feedback_mode", "audit_only"),
        microburst=microburst,
        previous_feedback=previous_feedback,
    )
    write_pressure_artifacts(out.parent, feedback)
    return feedback


def write_pool_manifest(out: Path, incumbent: Path, core_pool: list[Path], hof_paths: list[Path], holdout_count: int) -> dict[str, Any]:
    entries: list[dict[str, Any]] = [{"role": "incumbent", "path": str(incumbent)}]
    seen = {str(incumbent)}
    holdout_slots = min(max(1, len(core_pool) // 3), max(0, holdout_count), len(core_pool))
    holdout_paths = {str(path) for path in core_pool[-holdout_slots:]} if holdout_slots else set()
    for path in core_pool:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        role = "holdout" if key in holdout_paths else "core"
        entries.append({"role": role, "path": key})
    for path in hof_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        entries.append({"role": "hof", "path": key})
    manifest = {
        "version": 1,
        "policy": "generation may use incumbent/core/hof; holdout is audit signal and is not used for card-level mutation.",
        "entries": entries,
        "role_counts": {
            role: sum(1 for entry in entries if entry.get("role") == role)
            for role in sorted({str(entry.get("role")) for entry in entries})
        },
    }
    write_json(out, manifest)
    return manifest


def pool_paths_by_role(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    roles: dict[str, list[Path]] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "core")
        path = Path(str(entry.get("path") or ""))
        if path.exists():
            roles.setdefault(role, []).append(path)
    return roles


def evaluation_pools(
    pools: dict[str, list[Path]],
    counter_opponents: list[Path],
) -> tuple[list[Path], list[Path]]:
    incumbent = pools.get("incumbent", [])
    discovery = unique_existing(
        [*incumbent, *counter_opponents, *pools.get("core", []), *pools.get("hof", [])]
    )
    holdout = unique_existing(
        [*incumbent, *pools.get("holdout", []), *pools.get("core", []), *pools.get("hof", [])]
    )
    return discovery, holdout


def select_stage_opponents(pool: list[Path], stage: Stage, incumbent_path: Path | None = None) -> list[Path]:
    existing = unique_existing(pool)
    if stage.pool_limit <= 0:
        return []
    incumbent_resolved = incumbent_path.resolve() if incumbent_path and incumbent_path.exists() else None
    incumbent: list[Path] = []
    required: list[Path] = []
    counters: list[Path] = []
    remaining: list[Path] = []
    for path in existing:
        name = submission_name(path)
        if incumbent_resolved is not None and path.resolve() == incumbent_resolved:
            incumbent.append(path)
        elif name in REQUIRED_ANCHOR_NAMES:
            required.append(path)
        elif name.startswith("counter_"):
            counters.append(path)
        else:
            remaining.append(path)
    missing = [name for name in REQUIRED_ANCHOR_NAMES if name not in {submission_name(path) for path in required}]
    if incumbent_resolved is not None and not incumbent:
        missing.insert(0, "incumbent")
    if missing:
        raise ValueError(f"Missing required evaluation anchors: {', '.join(missing)}")
    fixed = [*incumbent[:1], *sorted(required, key=lambda path: REQUIRED_ANCHOR_NAMES.index(submission_name(path)))]
    if len(fixed) > stage.pool_limit:
        raise ValueError(f"Stage {stage.name} pool_limit={stage.pool_limit} cannot fit required anchors")
    counter_quota = 4 if stage.pool_limit >= 12 else 3
    selected = [*fixed, *counters[: min(counter_quota, stage.pool_limit - len(fixed))]]
    for path in remaining:
        if len(selected) >= stage.pool_limit:
            break
        selected.append(path)
    return selected


def run_scenario_gate(
    records: list[CandidateRecord],
    names: list[str],
    scenario_path: Path | None,
    opponent: Path,
    out: Path,
    args: argparse.Namespace,
    incumbent_name: str,
) -> tuple[list[str], dict[str, Any]]:
    if not getattr(args, "scenario_gate", True) or scenario_path is None or not scenario_path.exists():
        return names, {"status": "skipped", "reason": "no scenario input"}
    by_name = candidate_by_name(records)
    selected = [name for name in names[: max(1, args.scenario_gate_candidates)] if name in by_name]
    if incumbent_name in by_name and incumbent_name not in selected:
        selected = [incumbent_name, *selected]
    candidate_paths = [Path(by_name[name].eval_tarball) for name in selected]
    if not candidate_paths:
        return names, {"status": "skipped", "reason": "no selected candidates"}
    rows = run_scenarios(
        scenario_path,
        candidate_paths,
        opponent,
        out / "scenario_gate.json",
        limit=args.scenario_gate_limit,
        takeover_actions=args.scenario_takeover_actions,
        seed=args.seed + 909,
    )
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        data = row.to_dict()
        by_candidate.setdefault(data["candidate"], []).append(data)
    summary: dict[str, Any] = {"status": "ok", "scenario_path": str(scenario_path), "candidates": {}}
    incumbent_delta = 0.0
    for name, items in by_candidate.items():
        reached = [item for item in items if item.get("reached")]
        reach_rate = len(reached) / max(1, len(items))
        avg_delta = sum(float(item.get("delta") or 0.0) for item in reached) / max(1, len(reached))
        summary["candidates"][name] = {"rows": len(items), "reach_rate": reach_rate, "avg_delta": avg_delta}
        if name == incumbent_name:
            incumbent_delta = avg_delta
    penalized = {
        name
        for name, item in summary["candidates"].items()
        if name != incumbent_name
        and item["reach_rate"] >= 0.60
        and item["avg_delta"] < incumbent_delta - args.scenario_penalty_margin
    }
    adjusted = [name for name in names if name not in penalized] + [name for name in names if name in penalized]
    summary["penalized"] = sorted(penalized)
    (out / "scenario_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return adjusted, summary


def select_next_names(
    report: MatchReport,
    records: list[CandidateRecord],
    stage: Stage,
    profile: DiscoveryProfile,
    incumbent_name: str,
) -> list[str]:
    candidate_names = {record.name for record in records}
    rows = [
        stats
        for stats in report.standings
        if stats.name in candidate_names and clean_stats(stats, profile.max_no_result_rate)
    ]
    rows.sort(key=lambda s: (s.kaggle_score_estimate, s.wins - s.losses, s.wins, -s.losses), reverse=True)
    names: list[str] = rank_for_racing(report, candidate_names, max(1, stage.finalists), profile.max_no_result_rate)
    for name in psro_bonus_names(report, candidate_names, profile.psro_slots):
        if name not in names:
            names.append(name)
    by_record = candidate_by_name(records)
    used_families = {by_record[name].family for name in names if name in by_record}
    for stats in rows:
        record = by_record.get(stats.name)
        if record is None or record.family in used_families:
            continue
        names.append(stats.name)
        used_families.add(record.family)
        if len(used_families) >= 3:
            break
    if incumbent_name in candidate_names:
        names = [incumbent_name] + [name for name in names if name != incumbent_name]
    return names[: max(1, stage.candidate_limit)]


def anchor_only_report(report: MatchReport, candidate_names: set[str]) -> MatchReport:
    standings: list[AgentStats] = []
    for stats in report.standings:
        if stats.name not in candidate_names:
            standings.append(stats)
            continue
        opponents = {
            name: dict(record)
            for name, record in stats.opponents.items()
            if name not in candidate_names
        }
        wins = sum(record["wins"] for record in opponents.values())
        losses = sum(record["losses"] for record in opponents.values())
        draws = sum(record["draws"] for record in opponents.values())
        standings.append(
            AgentStats(
                name=stats.name,
                tarball=stats.tarball,
                sha256=stats.sha256,
                games=wins + losses + draws,
                wins=wins,
                losses=losses,
                draws=draws,
                crashes=stats.crashes,
                timeouts=stats.timeouts,
                invalids=stats.invalids,
                no_results=0,
                mu=stats.mu,
                sigma=stats.sigma,
                kaggle_score_estimate=stats.kaggle_score_estimate,
                opponents=opponents,
            )
        )
    metadata = dict(report.metadata)
    metadata["primary_score"] = "anchor_only"
    return MatchReport(report.config, report.submissions, report.games, standings, metadata)


def eval_stage(
    records: list[CandidateRecord],
    selected_names: list[str],
    pool: list[Path],
    out_dir: Path,
    stage: Stage,
    profile: DiscoveryProfile,
    args: argparse.Namespace,
    seed_offset: int,
    incumbent_name: str,
) -> tuple[MatchReport, list[str]]:
    by_name = candidate_by_name(records)
    ordered_names = [name for name in selected_names if name in by_name and name != incumbent_name]
    if incumbent_name in by_name:
        ordered_names = [incumbent_name] + ordered_names
    selected_records = [by_name[name] for name in ordered_names[: stage.candidate_limit]]
    candidate_tarballs = unique_existing([Path(record.eval_tarball) for record in selected_records])
    incumbent_path = Path(by_name[incumbent_name].eval_tarball) if incumbent_name in by_name else None
    opponent_tarballs = select_stage_opponents(pool, stage, incumbent_path)
    cfg = EvalConfig(
        seed=args.seed + seed_offset,
        workers=resolve_workers(args, profile),
        max_actions=args.max_actions or profile.max_actions,
        run_timeout_s=args.run_timeout or profile.run_timeout_s,
        record_mode=stage.record_mode,
        record_sample_rate=max(args.record_sample_rate, 0.05) if "holdout" in stage.name else args.record_sample_rate,
        record_gzip=True,
        progress=args.progress,
        progress_label=stage.name,
        progress_interval_s=args.progress_interval,
        progress_mode=args.progress_mode,
        progress_file=str(out_dir / "progress.txt") if args.progress_mode == "file" else "",
        archive_cache_dir=str(args.out / ".submission_cache"),
        max_in_flight=resolve_workers(args, profile) * 2,
    )
    report = run_candidate_pool(
        candidate_tarballs,
        opponent_tarballs,
        stage.games_per_pair,
        cfg,
        Path.cwd(),
        peer_span=max(0, len(candidate_tarballs) - 1) if "holdout" in stage.name else 1,
    )
    save_report(report, out_dir / stage.name)
    write_psro(report, out_dir / stage.name / "psro.json", [r.name for r in selected_records])
    primary_report = anchor_only_report(report, {record.name for record in selected_records})
    next_names = select_next_names(primary_report, selected_records, stage, profile, incumbent_name)
    return report, next_names


def export_candidate(record: CandidateRecord, promote: Path, submission_out: Path, strip_search_wrapper: bool) -> dict[str, Any]:
    promote.parent.mkdir(parents=True, exist_ok=True)
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.internal_tarball, promote)
    export_kaggle_submission(Path(record.internal_tarball), submission_out, strip_search_wrapper=strip_search_wrapper)
    return {
        "name": record.name,
        "family": record.family,
        "origin": record.origin,
        "promote": str(promote),
        "submission": str(submission_out),
        "submission_sha256": sha256_file(submission_out),
        "internal_sha256": record.sha256,
        "notes": record.notes,
    }


def export_manual(
    records: list[CandidateRecord],
    names: list[str],
    out_dir: Path,
    slots: int,
    strip_search_wrapper: bool,
    roles: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    by_name = candidate_by_name(records)
    manual_dir = out_dir / "manual_candidates"
    manual_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for name in names:
        if len(out) >= slots:
            break
        record = by_name.get(name)
        if record is None:
            continue
        role = (roles or {}).get(record.name, "backup")
        target = manual_dir / f"{role}_{record.name}.tar.gz"
        export_kaggle_submission(Path(record.internal_tarball), target, strip_search_wrapper=strip_search_wrapper)
        out.append({"name": record.name, "family": record.family, "origin": record.origin, "role": role, "tarball": str(target)})
    return out


def incumbent_path_from_state(state: dict[str, Any], default: Path) -> Path:
    final = state.get("final") or {}
    promoted = final.get("promote")
    if promoted and Path(promoted).exists():
        return Path(promoted)
    incumbent = state.get("incumbent")
    if incumbent and Path(str(incumbent)).exists():
        return Path(str(incumbent))
    return default


def hof_paths_from_state(state: dict[str, Any], limit: int = 16) -> list[Path]:
    out: list[Path] = []
    for row in reversed(state.get("history", [])):
        final = row.get("final") or {}
        path = final.get("promote")
        if path and Path(path).exists():
            out.append(Path(path))
        for manual in final.get("manual_candidates", []):
            mpath = manual.get("tarball")
            if mpath and Path(mpath).exists():
                out.append(Path(mpath))
        if len(out) >= limit:
            break
    return list(reversed(out[:limit]))


def update_strategy_archive(
    archive: StrategyArchive,
    records: list[CandidateRecord],
    report: MatchReport,
    names: list[str],
) -> int:
    by_record = candidate_by_name(records)
    stats = stats_by_name(report)
    added = 0
    for name in names:
        record = by_record.get(name)
        row = stats.get(name)
        if record is None or row is None:
            continue
        genome_data = dict(record.config.get("strategy_genome") or {})
        if not genome_data:
            continue
        try:
            genome = StrategyGenome.from_dict(genome_data)
        except (KeyError, TypeError, ValueError):
            continue
        if archive.add(genome, float(row.kaggle_score_estimate)):
            added += 1
    return added


def run_generation(state: dict[str, Any], args: argparse.Namespace, profile: DiscoveryProfile, generation: int) -> dict[str, Any]:
    gen_dir = args.out / f"generation_{generation:03d}"
    variants_dir = gen_dir / "variants"
    incumbent_tarball = incumbent_path_from_state(state, args.incumbent)
    hof_paths = hof_paths_from_state(state)
    digest_path = args.loss_digest if args.loss_digest else args.out / "loss_digest.json"
    feedback_path = feedback_path_for_run(args)
    previous_feedback = read_json(feedback_path) if feedback_path and feedback_path.exists() else {}
    target_paths = resolve_feedback_target_paths(previous_feedback, history_records(state))
    requested_population = args.population or profile.population
    strategy_archive_path = args.out / "strategy_archive.json"
    strategy_archive = StrategyArchive.load(strategy_archive_path)
    set_run_phase(state, args.out, generation, "build", "generating candidate configs")
    configs = generate_discovery_configs(
        incumbent_tarball=incumbent_tarball,
        out_dir=variants_dir,
        population=requested_population,
        generation=generation,
        seed=args.seed,
        hof_paths=hof_paths,
        include_portfolio=args.include_portfolio,
        loss_digest_path=digest_path if digest_path.exists() else None,
        feedback_path=feedback_path if feedback_path and feedback_path.exists() else None,
        target_paths=target_paths,
        min_diversity_distance=args.min_diversity_distance,
        parents=strategy_archive.elites(),
    )
    opponent_archive_path = args.out / "opponent_archive.json"
    opponent_archive = OpponentArchive.load(opponent_archive_path)
    counter_genomes = generate_counter_opponents(
        count=max(6, min(16, requested_population // 8)),
        seed=args.seed + generation * 3571,
        parents=opponent_archive.genomes(),
        pressure=opponent_pressure_from_feedback(previous_feedback),
        generation=generation,
    )
    for genome in counter_genomes:
        opponent_archive.add(genome)
    opponent_archive.compact()
    opponent_archive.save(opponent_archive_path)
    opponent_base = BuildConfig(name="counter_runtime", origin="coevolved_opponent")
    opponent_configs = [compile_opponent_genome(genome, opponent_base, gen_dir / "opponent_variants") for genome in counter_genomes]
    generation_summary = candidate_generation_summary(configs, requested_population)
    write_json(gen_dir / "candidate_generation.json", generation_summary)
    if generation_summary["underfilled"]:
        print(
            json.dumps(
                {
                    "warning": "candidate_population_underfilled",
                    "generation": generation,
                    "population_requested": requested_population,
                    "configs_generated": len(configs),
                    "underfill_ratio": generation_summary["underfill_ratio"],
                }
            ),
            flush=True,
        )
    records = build_records(
        configs,
        gen_dir / "eval_variants",
        args.strip_search_wrapper,
        build_workers=resolve_build_workers(args, profile),
        progress=args.progress,
        progress_label=f"generation_{generation:03d}",
        progress_interval_s=args.progress_interval,
        progress_mode=args.progress_mode,
        progress_file=gen_dir / "progress.txt",
        failures_out=gen_dir / "build_failures.json",
    )
    if not records:
        raise RuntimeError("No candidates built")
    opponent_records = build_records(
        opponent_configs,
        gen_dir / "opponent_eval_variants",
        args.strip_search_wrapper,
        build_workers=resolve_build_workers(args, profile),
        progress=args.progress,
        progress_label=f"generation_{generation:03d}_opponents",
        progress_interval_s=args.progress_interval,
        progress_mode=args.progress_mode,
        progress_file=gen_dir / "opponent_progress.txt",
        failures_out=gen_dir / "opponent_build_failures.json",
    )
    write_json(
        gen_dir / "opponent_manifest.json",
        {
            "generated": [genome.to_dict() for genome in counter_genomes],
            "built": [asdict(record) for record in opponent_records],
            "archive": str(opponent_archive_path),
        },
    )
    generation_summary = candidate_generation_summary(configs, requested_population, len(records))
    write_json(gen_dir / "candidate_generation.json", generation_summary)
    incumbent_name = submission_name(incumbent_tarball)
    all_names = [record.name for record in records]
    pool_manifest = write_pool_manifest(gen_dir / "pool_manifest.json", incumbent_tarball, unique_existing(args.pool), hof_paths, profile.stage_d.pool_limit)
    pools = pool_paths_by_role(pool_manifest)
    discovery_pool, holdout_pool = evaluation_pools(
        pools,
        [Path(record.eval_tarball) for record in opponent_records],
    )

    set_run_phase(state, args.out, generation, profile.stage_a.name, "running Stage A ladder")
    report_a, names_a = eval_stage(records, all_names, discovery_pool, gen_dir, profile.stage_a, profile, args, generation * 10000 + 101, incumbent_name)
    state["stage_a_top"] = names_a
    write_json(args.out / "state.json", state)

    set_run_phase(state, args.out, generation, profile.stage_b.name, "running Stage B ladder")
    report_b, names_b = eval_stage(records, names_a, discovery_pool, gen_dir, profile.stage_b, profile, args, generation * 10000 + 202, incumbent_name)
    stage_b_scenarios = gen_dir / "stage_b_scenarios.json"
    mine_losses(
        [gen_dir / profile.stage_b.name / "game_records"],
        stage_b_scenarios,
        focus="",
        limit=max(8, args.scenario_limit // 2),
        max_prefix_actions=args.scenario_max_prefix_actions,
        max_trigger_turn=args.scenario_max_trigger_turn,
        min_drop=args.scenario_min_drop,
    )
    microburst_feedback_path = gen_dir / "microburst" / "generation_feedback.json"
    microburst_feedback = build_generation_feedback(
        args,
        gen_dir,
        generation,
        incumbent_name,
        min(args.microburst_max, max(16, int((args.population or profile.population) * args.microburst_fraction))),
        [profile.stage_a.name, profile.stage_b.name],
        [stage_b_scenarios],
        microburst_feedback_path,
        microburst=True,
        previous_feedback=feedback_path if feedback_path and feedback_path.exists() else None,
    )
    microburst_target_paths = resolve_feedback_target_paths(microburst_feedback, records)
    microburst_records: list[CandidateRecord] = []
    microburst_names: list[str] = []
    if (
        getattr(args, "feedback_mode", "closed_loop") == "closed_loop"
        and microburst_feedback.get("pressure_items")
        and args.microburst_fraction > 0
    ):
        set_run_phase(state, args.out, generation, "microburst_build", "building pressure-directed Stage B repair candidates")
        micro_generation = generation * 1000 + 501
        micro_population = min(args.microburst_max, max(16, int((args.population or profile.population) * args.microburst_fraction)))
        micro_configs = generate_discovery_configs(
            incumbent_tarball=incumbent_tarball,
            out_dir=gen_dir / "microburst" / "variants",
            population=micro_population,
            generation=micro_generation,
            seed=args.seed + 1701,
            hof_paths=hof_paths,
            include_portfolio=False,
            feedback_path=microburst_feedback_path,
            target_paths=microburst_target_paths,
            pressure_only=True,
            min_diversity_distance=args.min_diversity_distance,
        )
        microburst_records = build_records(
            micro_configs,
            gen_dir / "microburst" / "eval_variants",
            args.strip_search_wrapper,
            build_workers=resolve_build_workers(args, profile),
            progress=args.progress,
            progress_label=f"generation_{generation:03d}_microburst",
            progress_interval_s=args.progress_interval,
            progress_mode=args.progress_mode,
            progress_file=gen_dir / "microburst" / "progress.txt",
            failures_out=gen_dir / "microburst" / "build_failures.json",
        )
        if microburst_records:
            micro_stage = Stage(
                "stage_b_microburst",
                max(1, profile.stage_b.games_per_pair // 2),
                min(len(microburst_records), max(4, args.microburst_max)),
                profile.stage_b.pool_limit,
                "sample",
                min(profile.stage_b.finalists, max(2, len(microburst_records) // 3)),
            )
            set_run_phase(state, args.out, generation, micro_stage.name, "running microburst Stage B ladder")
            _, micro_names = eval_stage(
                microburst_records,
                [record.name for record in microburst_records],
                discovery_pool,
                gen_dir / "microburst",
                micro_stage,
                profile,
                args,
                generation * 10000 + 252,
                incumbent_name,
            )
            microburst_names = [name for name in micro_names if name != micro_incumbent]
            records.extend([record for record in microburst_records if record.name != micro_incumbent])
            names_b = [*names_b, *[name for name in microburst_names if name not in names_b]]
    names_b, scenario_gate = run_scenario_gate(
        records,
        names_b,
        scenario_path_if_useful(stage_b_scenarios) or latest_scenario_path(state),
        incumbent_tarball,
        gen_dir,
        args,
        incumbent_name,
    )
    state["stage_b_top"] = names_b
    state["scenario_gate"] = scenario_gate
    state["microburst"] = {"built": len(microburst_records), "selected": microburst_names[:12], "feedback": str(microburst_feedback_path)}
    write_json(args.out / "state.json", state)

    set_run_phase(state, args.out, generation, profile.stage_c.name, "running Stage C confirmation")
    report_c, names_c = eval_stage(records, names_b, discovery_pool, gen_dir, profile.stage_c, profile, args, generation * 10000 + 303, incumbent_name)
    stage_c_scenarios = gen_dir / "stage_c_scenarios.json"
    mine_losses(
        [gen_dir / profile.stage_c.name / "game_records"],
        stage_c_scenarios,
        focus="",
        limit=args.scenario_limit,
        max_prefix_actions=args.scenario_max_prefix_actions,
        max_trigger_turn=args.scenario_max_trigger_turn,
        min_drop=args.scenario_min_drop,
    )
    names_c, stage_c_scenario_gate = run_scenario_gate(
        records,
        names_c,
        scenario_path_if_useful(stage_c_scenarios),
        incumbent_tarball,
        gen_dir / "stage_c_scenario_gate",
        args,
        incumbent_name,
    )
    state["stage_c_top"] = names_c
    state["stage_c_scenario_gate"] = stage_c_scenario_gate
    write_json(args.out / "state.json", state)

    set_run_phase(state, args.out, generation, profile.stage_d.name, "running Stage D holdout")
    report_d, names_d = eval_stage(records, names_c, holdout_pool, gen_dir, profile.stage_d, profile, args, generation * 10000 + 404, incumbent_name)
    set_run_phase(state, args.out, generation, "finalize", "classifying holdout result")
    stats = stats_by_name(report_d)
    by_name = candidate_by_name(records)
    clean_names = [name for name in names_d if name in stats and clean_stats(stats[name], profile.max_no_result_rate)]
    primary_report_d = anchor_only_report(report_d, set(by_name))
    racing = report_racing(primary_report_d, clean_names)
    roles = dict(racing.get("roles") or {})
    clean_names = [name for name in racing.get("ranked", []) if name in clean_names]
    archive_added = update_strategy_archive(strategy_archive, records, report_d, clean_names)
    strategy_archive.save(strategy_archive_path)
    write_json(gen_dir / "racing_report.json", racing)
    write_json(gen_dir / "robustness_report.json", racing)
    if not clean_names:
        final = {"status": "failed", "reason": "no clean holdout finalists", "stage_d_top": names_d}
    else:
        best_name = clean_names[0]
        delta = 0.0
        h2h_wins = 0
        h2h_losses = 0
        if incumbent_name in stats and best_name != incumbent_name:
            delta, h2h_wins, h2h_losses = score_delta(report_d, best_name, incumbent_name)
        h2h_draws = 0
        h2h_wilson = 0.0
        robust_margin = float("-inf")
        if best_name in stats and incumbent_name in stats and best_name != incumbent_name:
            h2h = stats[best_name].opponents.get(incumbent_name, {"wins": 0, "losses": 0, "draws": 0})
            h2h_draws = int(h2h.get("draws", 0))
            h2h_wilson = wilson_lower_bound(h2h_wins, h2h_losses, h2h_draws)
            primary_stats = stats_by_name(primary_report_d)
            robust_margin = robust_stats(primary_stats[best_name])["robust_score"] - robust_stats(primary_stats[incumbent_name])["robust_score"]
        strict_ok = (
            best_name != incumbent_name
            and incumbent_name in stats
            and h2h_wilson >= 0.50
            and robust_margin >= 0.0
        )
        role_order = {"generalist": 0, "anti_fast_ko": 1, "anti_control": 2}
        manual_names = sorted(
            [name for name in clean_names if name != incumbent_name],
            key=lambda name: (role_order.get(roles.get(name, "backup"), 9), clean_names.index(name)),
        )
        if strict_ok:
            final = export_candidate(by_name[best_name], args.promote, args.submission_out, args.strip_search_wrapper)
            final.update({
                "status": "promoted",
                "score_delta_vs_incumbent": delta,
                "h2h_wins": h2h_wins,
                "h2h_losses": h2h_losses,
                "h2h_draws": h2h_draws,
                "h2h_wilson_low": h2h_wilson,
                "robust_score_margin": robust_margin,
            })
            state["incumbent"] = final["promote"]
        else:
            gate_reasons = []
            threshold = args.min_holdout_score_delta if args.min_holdout_score_delta is not None else profile.min_holdout_score_delta
            if best_name == incumbent_name:
                gate_reasons.append("best candidate is the incumbent")
            if incumbent_name not in stats:
                gate_reasons.append("incumbent missing from Stage D report")
            if incumbent_name in stats and delta < threshold:
                gate_reasons.append("score_delta below threshold")
            if incumbent_name in stats and h2h_wins < h2h_losses:
                gate_reasons.append(f"H2H losing vs incumbent: {h2h_wins}-{h2h_losses}")
            if incumbent_name in stats and h2h_wilson < 0.50:
                gate_reasons.append(f"H2H Wilson lower bound below 0.500: {h2h_wilson:.3f}")
            if incumbent_name in stats and robust_margin < 0.0:
                gate_reasons.append(f"anchor robust score margin below 0.000: {robust_margin:.3f}")
            final = {
                "status": "held",
                "reason": "best holdout candidate did not clear strict incumbent gate",
                "best": best_name,
                "score_delta_vs_incumbent": delta,
                "h2h_wins": h2h_wins,
                "h2h_losses": h2h_losses,
                "h2h_draws": h2h_draws,
                "h2h_wilson_low": h2h_wilson,
                "robust_score_margin": robust_margin,
                "incumbent": str(incumbent_tarball),
                "gate_reasons": gate_reasons,
            }
        final["manual_candidates"] = export_manual(
            records,
            manual_names,
            gen_dir,
            args.manual_slots if args.manual_slots is not None else profile.manual_slots,
            args.strip_search_wrapper,
            roles,
        )
        final["stage_d_top"] = [
            stats[name].to_dict()
            for name in clean_names
            if name in stats
        ]
        write_json(
            gen_dir / "candidate_manifest.json",
            {
                "generation": generation,
                "candidates": final["manual_candidates"],
                "roles": roles,
                "reference_free_runtime": True,
            },
        )

    record_dirs = [gen_dir / profile.stage_c.name / "game_records", gen_dir / profile.stage_d.name / "game_records"]
    scenario_out = gen_dir / "scenarios.json"
    scenarios = mine_losses(
        record_dirs,
        scenario_out,
        focus="",
        limit=args.scenario_limit,
        max_prefix_actions=args.scenario_max_prefix_actions,
        max_trigger_turn=args.scenario_max_trigger_turn,
        min_drop=args.scenario_min_drop,
    )
    write_json(gen_dir / "failure_scenarios.json", [scenario.to_dict() for scenario in scenarios])
    psro_path = gen_dir / profile.stage_d.name / "psro.json"
    if psro_path.exists():
        write_json(gen_dir / "matchup_matrix.json", read_json(psro_path))
    loss_digest = digest_scenarios([scenario.to_dict() for scenario in scenarios], gen_dir / "loss_digest.json")
    training_scenarios = [*load_loss_scenarios(stage_b_scenarios), *load_loss_scenarios(stage_c_scenarios)]
    if getattr(args, "holdout_feedback_mode", "audit_only") == "weak_signal":
        training_scenarios.extend(scenario.to_dict() for scenario in scenarios)
    digest_scenarios(training_scenarios, args.out / "loss_digest.json")
    next_feedback_path = gen_dir / "generation_feedback.json"
    generation_feedback = build_generation_feedback(
        args,
        gen_dir,
        generation,
        incumbent_name,
        args.population or profile.population,
        [profile.stage_a.name, profile.stage_b.name, profile.stage_c.name, profile.stage_d.name],
        [stage_b_scenarios, stage_c_scenarios, scenario_out],
        next_feedback_path,
        microburst=False,
        previous_feedback=feedback_path if feedback_path and feedback_path.exists() else None,
    )
    if feedback_path is not None:
        write_json(feedback_path, generation_feedback)
    final["scenarios"] = {"out": str(scenario_out), "count": len(scenarios)}
    final["loss_digest"] = {"out": str(gen_dir / "loss_digest.json"), **loss_digest}
    final["generation_feedback"] = {
        "out": str(next_feedback_path),
        "root": str(feedback_path) if feedback_path is not None else "",
        "pressure_count": len(generation_feedback.get("pressure_items", [])),
        "top_kinds": (generation_feedback.get("summary") or {}).get("top_kinds", {}),
    }
    final["scenario_gate"] = scenario_gate
    final["stage_c_scenario_gate"] = stage_c_scenario_gate
    final["microburst"] = state.get("microburst", {})
    final["pool_manifest"] = {"out": str(gen_dir / "pool_manifest.json"), "role_counts": pool_manifest.get("role_counts", {})}
    final["candidate_generation"] = {**generation_summary, "out": str(gen_dir / "candidate_generation.json")}
    final["strategy_archive"] = {
        "out": str(strategy_archive_path),
        "parents_loaded": len(strategy_archive.elites()) - archive_added,
        "entries_added": archive_added,
        "entries": len(strategy_archive.elites()),
    }
    final["reports"] = {
        "stage_a": str(gen_dir / profile.stage_a.name / "report.json"),
        "stage_b": str(gen_dir / profile.stage_b.name / "report.json"),
        "stage_c": str(gen_dir / profile.stage_c.name / "report.json"),
        "stage_d": str(gen_dir / profile.stage_d.name / "report.json"),
    }
    final["records"] = [asdict(record) for record in records]
    submission_check = inspect_submission(args.submission_out) if Path(args.submission_out).exists() else None
    incumbent_decision = classify_final(
        final,
        submission_check,
        min_delta=args.min_holdout_score_delta if args.min_holdout_score_delta is not None else profile.min_holdout_score_delta,
        max_no_result_rate=profile.max_no_result_rate,
        near_clean_no_result_rate=args.near_clean_no_result_rate,
    )
    final["incumbent_gate"] = incumbent_decision
    decision = incumbent_decision
    if args.gold_gate and final.get("status") == "promoted" and final.get("name"):
        gold_decision = classify_gold_candidate(
            report_d,
            str(final["name"]),
            incumbent_name=incumbent_name,
            submission_sha256=str(final.get("submission_sha256") or ""),
            max_no_result_rate=profile.max_no_result_rate,
            min_matchup_games=profile.stage_d.games_per_pair,
        )
        final["gold_gate"] = gold_decision
        decision = {
            "champion_class": gold_decision["decision"],
            "decision": gold_decision["decision"],
            "submit_ready": gold_decision["submit_ready"],
            "reasons": gold_decision["reasons"],
        }
    final.update(decision)
    write_json(gen_dir / "decision.json", {"final": final, "submission_check": submission_check, "decision": decision})
    write_decision_brief(gen_dir / "decision_brief.md", final, decision, submission_check)
    write_decision_brief(args.out / "decision_brief.md", final, decision, submission_check)
    state["generation"] = generation
    state["final"] = final
    state.setdefault("history", []).append({"generation": generation, "final": final})
    write_json(args.out / "state.json", state)
    write_json(args.out / "final_report.json", final)
    return state


def run_engine(args: argparse.Namespace) -> dict[str, Any]:
    profile = profile_from_name(args.profile)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "state.json"
    state = read_json(state_path) if args.resume else {}
    state.setdefault("profile", asdict(profile))
    state["effective_workers"] = resolve_workers(args, profile)
    state["effective_build_workers"] = resolve_build_workers(args, profile)
    state.setdefault("incumbent", str(args.incumbent))
    generations = args.generations if args.generations is not None else profile.generations
    start = int(state.get("generation", 0)) + 1
    for generation in range(start, start + generations):
        state = run_generation(state, args, profile, generation)
        print(json.dumps({"generation": generation, "final": state.get("final", {})}, indent=2))
    return state.get("final", {})


def status(args: argparse.Namespace) -> dict[str, Any]:
    state = read_json(args.out / "state.json")
    final = read_json(args.out / "final_report.json")
    return {
        "out": str(args.out),
        "generation": state.get("generation"),
        "active_generation": state.get("active_generation"),
        "active_phase": state.get("active_phase"),
        "active_detail": state.get("active_detail"),
        "incumbent": state.get("incumbent"),
        "effective_workers": state.get("effective_workers"),
        "effective_build_workers": state.get("effective_build_workers"),
        "scenario_gate": state.get("scenario_gate"),
        "stage_c_scenario_gate": state.get("stage_c_scenario_gate"),
        "microburst": state.get("microburst"),
        "stage_a_top": state.get("stage_a_top", [])[:8],
        "stage_b_top": state.get("stage_b_top", [])[:8],
        "stage_c_top": state.get("stage_c_top", [])[:8],
        "final": final,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    final = read_json(args.out / "final_report.json")
    submission_path = args.submission or Path(final.get("submission") or "")
    submission_check = inspect_submission(submission_path) if submission_path else None
    decision = classify_final(
        final,
        submission_check,
        min_delta=args.min_holdout_score_delta,
        max_no_result_rate=args.max_no_result_rate,
        near_clean_no_result_rate=args.near_clean_no_result_rate,
    )
    result = {
        "out": str(args.out),
        "submission": str(submission_path),
        "decision": decision,
        "submission_check": submission_check,
        "final": {
            key: final.get(key)
            for key in (
                "status",
                "name",
                "best",
                "score_delta_vs_incumbent",
                "h2h_wins",
                "h2h_losses",
                "manual_candidates",
            )
        },
    }
    write_json(args.out / "decision.json", result)
    write_decision_brief(args.out / "decision_brief.md", final, decision, submission_check)
    return result


def digest_losses_command(args: argparse.Namespace) -> dict[str, Any]:
    scenario_paths = list(args.out.glob("generation_*/scenarios.json"))
    digest = digest_loss_files(scenario_paths, args.out / "loss_digest.json")
    return {"out": str(args.out / "loss_digest.json"), **digest}


def export_final(args: argparse.Namespace) -> dict[str, Any]:
    final = read_json(args.out / "final_report.json")
    best_name = final.get("name")
    records = [CandidateRecord(**row) for row in final.get("records", [])]
    by_name = candidate_by_name(records)
    if best_name not in by_name:
        raise SystemExit(f"No promoted final candidate in {args.out / 'final_report.json'}")
    return export_candidate(by_name[best_name], args.promote, args.submission_out, args.strip_search_wrapper)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=["a800_discovery", "a800_turbo_discovery", "smoke_discovery"], default="a800_discovery")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--promote", type=Path, default=DEFAULT_PROMOTE)
    parser.add_argument("--submission-out", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--pool", nargs="*", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--workers", type=int, help="Concurrent games. Use 0 for os.cpu_count() minus --cpu-headroom.")
    parser.add_argument("--build-workers", type=int, help="Concurrent candidate build/export workers. Defaults to min(32, effective workers).")
    parser.add_argument("--cpu-headroom", type=int, default=2, help="CPU cores to leave idle when --workers is 0 or omitted by an auto-worker profile.")
    parser.add_argument("--max-workers", type=int, help="Upper bound for auto worker resolution.")
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--run-timeout", type=float)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--record-sample-rate", type=float, default=0.02)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--progress-mode", choices=["auto", "line", "overwrite", "file"], default="file")
    parser.add_argument("--manual-slots", type=int)
    parser.add_argument("--scenario-limit", type=int, default=48)
    parser.add_argument("--scenario-min-drop", type=float, default=250.0)
    parser.add_argument("--scenario-max-prefix-actions", type=int, default=140)
    parser.add_argument("--scenario-max-trigger-turn", type=int, default=18)
    parser.add_argument("--scenario-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scenario-gate-candidates", type=int, default=16)
    parser.add_argument("--scenario-gate-limit", type=int, default=24)
    parser.add_argument("--scenario-takeover-actions", type=int, default=80)
    parser.add_argument("--scenario-penalty-margin", type=float, default=180.0)
    parser.add_argument("--loss-digest", type=Path)
    parser.add_argument("--feedback-mode", choices=["off", "next_gen", "closed_loop"], default="closed_loop")
    parser.add_argument("--feedback-path", type=Path)
    parser.add_argument("--microburst-fraction", type=float, default=0.15)
    parser.add_argument("--microburst-max", type=int, default=48)
    parser.add_argument("--holdout-feedback-mode", choices=["audit_only", "weak_signal"], default="audit_only")
    parser.add_argument("--min-diversity-distance", type=float, default=0.08)
    parser.add_argument("--near-clean-no-result-rate", type=float, default=0.02)
    parser.add_argument("--min-holdout-score-delta", type=float)
    parser.add_argument("--gold-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip-search-wrapper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-portfolio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discovery-first optimizer for Pokemon TCG AI Battle submissions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the staged discovery loop.")
    add_run_args(p_run)

    p_status = sub.add_parser("status", help="Print current discovery state.")
    p_status.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p_audit = sub.add_parser("audit", help="Classify final discovery output and write decision_brief.md.")
    p_audit.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_audit.add_argument("--submission", type=Path)
    p_audit.add_argument("--min-holdout-score-delta", type=float, default=8.0)
    p_audit.add_argument("--max-no-result-rate", type=float, default=0.01)
    p_audit.add_argument("--near-clean-no-result-rate", type=float, default=0.02)

    p_mine = sub.add_parser("mine-losses", help="Mine collapse scenarios from game_records.")
    p_mine.add_argument("inputs", nargs="+", type=Path)
    p_mine.add_argument("--out", type=Path, default=DEFAULT_OUT / "scenarios.json")
    p_mine.add_argument("--focus", default="")
    p_mine.add_argument("--limit", type=int, default=64)
    p_mine.add_argument("--max-prefix-actions", type=int, default=140)
    p_mine.add_argument("--max-trigger-turn", type=int, default=16)
    p_mine.add_argument("--min-drop", type=float, default=250.0)

    p_digest = sub.add_parser("digest-losses", help="Build loss_digest.json from generation scenarios.")
    p_digest.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p_scenario = sub.add_parser("run-scenarios", help="Replay mined scenarios against candidates.")
    p_scenario.add_argument("--scenarios", type=Path, required=True)
    p_scenario.add_argument("--candidate", nargs="+", type=Path, required=True)
    p_scenario.add_argument("--opponent", type=Path, required=True)
    p_scenario.add_argument("--out", type=Path, default=DEFAULT_OUT / "scenario_eval.json")
    p_scenario.add_argument("--limit", type=int, default=24)
    p_scenario.add_argument("--takeover-actions", type=int, default=80)
    p_scenario.add_argument("--seed", type=int, default=20260718)

    p_export = sub.add_parser("export-final", help="Re-export the promoted discovery final.")
    p_export.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_export.add_argument("--promote", type=Path, default=DEFAULT_PROMOTE)
    p_export.add_argument("--submission-out", type=Path, default=DEFAULT_SUBMISSION)
    p_export.add_argument("--strip-search-wrapper", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        result = run_engine(args)
    elif args.cmd == "status":
        result = status(args)
    elif args.cmd == "audit":
        result = audit(args)
    elif args.cmd == "mine-losses":
        scenarios = mine_losses(args.inputs, args.out, args.focus, args.limit, args.max_prefix_actions, args.max_trigger_turn, args.min_drop)
        result = {"out": str(args.out), "scenarios": len(scenarios)}
    elif args.cmd == "digest-losses":
        result = digest_losses_command(args)
    elif args.cmd == "run-scenarios":
        rows = run_scenarios(args.scenarios, args.candidate, args.opponent, args.out, args.limit, args.takeover_actions, args.seed)
        result = {"out": str(args.out), "rows": len(rows)}
    elif args.cmd == "export-final":
        result = export_final(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
