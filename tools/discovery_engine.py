from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file
from local_eval.evaluator import run_ladder, save_report
from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.build_submission import BuildConfig, build_submission
from tools.discovery_space import generate_discovery_configs, read_build_metadata
from tools.export_kaggle_submission import export_kaggle_submission
from tools.loss_mining import mine_losses
from tools.psro import psro_bonus_names, write_psro
from tools.scenario_eval import run_scenarios


DEFAULT_OUT = Path("outputs/discovery_gold")
DEFAULT_INCUMBENT = Path("outputs/submissions/champion_latest.tar.gz")
DEFAULT_PROMOTE = Path("outputs/submissions/champion_discovery.tar.gz")
DEFAULT_SUBMISSION = Path("outputs/submissions/submission_discovery.tar.gz")
DEFAULT_POOL = [
    Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
    Path("outputs/reference_submissions/pokemon-steel.tar.gz"),
    Path("outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz"),
    Path("outputs/reference_submissions/improved-probabilistic-agent.tar.gz"),
    Path("outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz"),
    Path("outputs/reference_submissions/multiply-agent-best-940-lb.tar.gz"),
    Path("outputs/reference_submissions/pokemon-ai-battle-best-ptcg-advanced.tar.gz"),
    Path("基准/submission_sorce_700.tar.gz"),
]


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
            stage_a=Stage("stage_a", 2, 144, 8, "none", 36),
            stage_b=Stage("stage_b", 24, 40, 8, "sample", 16),
            stage_c=Stage("stage_c_confirm", 80, 18, 10, "losses", 8),
            stage_d=Stage("stage_d_holdout", 160, 8, 12, "losses", 5),
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
        population=224,
        workers=0,
        max_actions=1000,
        run_timeout_s=180.0,
        stage_a=Stage("stage_a", 2, 224, 10, "none", 56),
        stage_b=Stage("stage_b", 32, 72, 10, "sample", 24),
        stage_c=Stage("stage_c_confirm", 96, 28, 12, "losses", 10),
        stage_d=Stage("stage_d_holdout", 192, 10, 14, "losses", 6),
        manual_slots=10,
        max_no_result_rate=0.01,
        min_holdout_score_delta=8.0,
        psro_slots=6,
    )


def resolve_workers(args: argparse.Namespace, profile: DiscoveryProfile) -> int:
    requested = args.workers if args.workers is not None else profile.workers
    if requested and requested > 0:
        return requested
    cpu_count = os.cpu_count() or max(1, profile.workers)
    workers = max(1, cpu_count - max(0, args.cpu_headroom))
    if args.max_workers and args.max_workers > 0:
        workers = min(workers, args.max_workers)
    return workers


def config_to_dict(cfg: BuildConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["base"] = str(cfg.base)
    data["out"] = str(cfg.out)
    return data


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
    progress: bool = False,
    progress_label: str = "build",
    progress_interval_s: float = 5.0,
    progress_mode: str = "auto",
    progress_file: Path | None = None,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
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
    for index, cfg in enumerate(configs, start=1):
        try:
            built = build_submission(cfg)
            eval_tarball = eval_dir / built.name
            export_kaggle_submission(
                built,
                eval_tarball,
                strip_search_wrapper=strip_search_wrapper,
                exclude_internal_files=False,
            )
            records.append(
                CandidateRecord(
                    name=cfg.name,
                    family=cfg.family,
                    origin=cfg.origin,
                    internal_tarball=str(built.resolve()),
                    eval_tarball=str(eval_tarball.resolve()),
                    notes=cfg.notes,
                    sha256=sha256_file(built),
                    config=config_to_dict(cfg),
                )
            )
        except Exception as exc:
            print(json.dumps({"skipped": cfg.name, "reason": f"{type(exc).__name__}: {exc}"}))
        now = time.monotonic()
        if progress and (index == total or now - last_progress >= max(0.25, progress_interval_s)):
            last_progress = now
            _print_progress(
                progress_label,
                "build",
                index,
                total,
                f"built={len(records)} last={cfg.name}",
                mode=progress_mode,
                force=index == total,
                progress_file=progress_file,
            )
    return records


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
    rows.sort(
        key=lambda s: (
            s.kaggle_score_estimate,
            s.wins - s.losses,
            s.wins,
            -s.losses,
        ),
        reverse=True,
    )
    names: list[str] = []
    for stats in rows[: max(1, stage.finalists)]:
        if stats.name not in names:
            names.append(stats.name)
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
    tarballs = unique_existing([Path(r.eval_tarball) for r in selected_records] + pool[: stage.pool_limit])
    cfg = EvalConfig(
        seed=args.seed + seed_offset,
        workers=resolve_workers(args, profile),
        max_actions=args.max_actions or profile.max_actions,
        run_timeout_s=args.run_timeout or profile.run_timeout_s,
        record_mode=stage.record_mode,
        record_sample_rate=args.record_sample_rate,
        record_gzip=True,
        progress=args.progress,
        progress_label=stage.name,
        progress_interval_s=args.progress_interval,
        progress_mode=args.progress_mode,
        progress_file=str(out_dir / "progress.txt") if args.progress_mode == "file" else "",
    )
    report = run_ladder(tarballs, stage.games_per_pair, cfg, Path.cwd())
    save_report(report, out_dir / stage.name)
    write_psro(report, out_dir / stage.name / "psro.json", [r.name for r in selected_records])
    next_names = select_next_names(report, selected_records, stage, profile, incumbent_name)
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


def export_manual(records: list[CandidateRecord], names: list[str], out_dir: Path, slots: int, strip_search_wrapper: bool) -> list[dict[str, Any]]:
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
        target = manual_dir / f"{len(out) + 1:02d}_{record.name}.tar.gz"
        export_kaggle_submission(Path(record.internal_tarball), target, strip_search_wrapper=strip_search_wrapper)
        out.append({"name": record.name, "family": record.family, "origin": record.origin, "tarball": str(target)})
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


def run_generation(state: dict[str, Any], args: argparse.Namespace, profile: DiscoveryProfile, generation: int) -> dict[str, Any]:
    gen_dir = args.out / f"generation_{generation:03d}"
    variants_dir = gen_dir / "variants"
    incumbent_tarball = incumbent_path_from_state(state, args.incumbent)
    hof_paths = hof_paths_from_state(state)
    configs = generate_discovery_configs(
        incumbent_tarball=incumbent_tarball,
        out_dir=variants_dir,
        population=args.population or profile.population,
        generation=generation,
        seed=args.seed,
        hof_paths=hof_paths,
        include_portfolio=args.include_portfolio,
    )
    records = build_records(
        configs,
        gen_dir / "eval_variants",
        args.strip_search_wrapper,
        progress=args.progress,
        progress_label=f"generation_{generation:03d}",
        progress_interval_s=args.progress_interval,
        progress_mode=args.progress_mode,
        progress_file=gen_dir / "progress.txt",
    )
    if not records:
        raise RuntimeError("No candidates built")
    incumbent_name = f"d{generation:03d}_incumbent"
    all_names = [record.name for record in records]
    pool = unique_existing([*args.pool, *hof_paths])

    report_a, names_a = eval_stage(records, all_names, pool, gen_dir, profile.stage_a, profile, args, generation * 10000 + 101, incumbent_name)
    state["stage_a_top"] = names_a
    write_json(args.out / "state.json", state)

    report_b, names_b = eval_stage(records, names_a, pool, gen_dir, profile.stage_b, profile, args, generation * 10000 + 202, incumbent_name)
    state["stage_b_top"] = names_b
    write_json(args.out / "state.json", state)

    report_c, names_c = eval_stage(records, names_b, pool, gen_dir, profile.stage_c, profile, args, generation * 10000 + 303, incumbent_name)
    state["stage_c_top"] = names_c
    write_json(args.out / "state.json", state)

    report_d, names_d = eval_stage(records, names_c, pool, gen_dir, profile.stage_d, profile, args, generation * 10000 + 404, incumbent_name)
    stats = stats_by_name(report_d)
    by_name = candidate_by_name(records)
    clean_names = [name for name in names_d if name in stats and clean_stats(stats[name], profile.max_no_result_rate)]
    clean_names.sort(key=lambda name: (stats[name].kaggle_score_estimate, stats[name].wins - stats[name].losses), reverse=True)
    if not clean_names:
        final = {"status": "failed", "reason": "no clean holdout finalists", "stage_d_top": names_d}
    else:
        best_name = clean_names[0]
        delta = 0.0
        h2h_wins = 0
        h2h_losses = 0
        if incumbent_name in stats and best_name != incumbent_name:
            delta, h2h_wins, h2h_losses = score_delta(report_d, best_name, incumbent_name)
        strict_ok = best_name != incumbent_name and (
            incumbent_name not in stats
            or (delta >= args.min_holdout_score_delta if args.min_holdout_score_delta is not None else profile.min_holdout_score_delta)
        )
        manual_names = [name for name in clean_names if name != incumbent_name]
        if strict_ok:
            final = export_candidate(by_name[best_name], args.promote, args.submission_out, args.strip_search_wrapper)
            final.update({"status": "promoted", "score_delta_vs_incumbent": delta, "h2h_wins": h2h_wins, "h2h_losses": h2h_losses})
            state["incumbent"] = final["promote"]
        else:
            final = {
                "status": "held",
                "reason": "best holdout candidate did not clear strict incumbent gate",
                "best": best_name,
                "score_delta_vs_incumbent": delta,
                "h2h_wins": h2h_wins,
                "h2h_losses": h2h_losses,
                "incumbent": str(incumbent_tarball),
            }
        final["manual_candidates"] = export_manual(
            records,
            manual_names,
            gen_dir,
            args.manual_slots if args.manual_slots is not None else profile.manual_slots,
            args.strip_search_wrapper,
        )
        final["stage_d_top"] = [
            stats[name].to_dict()
            for name in clean_names
            if name in stats
        ]

    record_dirs = [gen_dir / profile.stage_c.name / "game_records", gen_dir / profile.stage_d.name / "game_records"]
    scenario_out = gen_dir / "scenarios.json"
    scenarios = mine_losses(record_dirs, scenario_out, focus="", limit=args.scenario_limit)
    final["scenarios"] = {"out": str(scenario_out), "count": len(scenarios)}
    final["reports"] = {
        "stage_a": str(gen_dir / profile.stage_a.name / "report.json"),
        "stage_b": str(gen_dir / profile.stage_b.name / "report.json"),
        "stage_c": str(gen_dir / profile.stage_c.name / "report.json"),
        "stage_d": str(gen_dir / profile.stage_d.name / "report.json"),
    }
    final["records"] = [asdict(record) for record in records]
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
        "incumbent": state.get("incumbent"),
        "effective_workers": state.get("effective_workers"),
        "stage_a_top": state.get("stage_a_top", [])[:8],
        "stage_b_top": state.get("stage_b_top", [])[:8],
        "stage_c_top": state.get("stage_c_top", [])[:8],
        "final": final,
    }


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
    parser.add_argument("--min-holdout-score-delta", type=float)
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

    p_mine = sub.add_parser("mine-losses", help="Mine collapse scenarios from game_records.")
    p_mine.add_argument("inputs", nargs="+", type=Path)
    p_mine.add_argument("--out", type=Path, default=DEFAULT_OUT / "scenarios.json")
    p_mine.add_argument("--focus", default="")
    p_mine.add_argument("--limit", type=int, default=64)
    p_mine.add_argument("--max-prefix-actions", type=int, default=140)
    p_mine.add_argument("--max-trigger-turn", type=int, default=16)
    p_mine.add_argument("--min-drop", type=float, default=450.0)

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
    elif args.cmd == "mine-losses":
        scenarios = mine_losses(args.inputs, args.out, args.focus, args.limit, args.max_prefix_actions, args.max_trigger_turn, args.min_drop)
        result = {"out": str(args.out), "scenarios": len(scenarios)}
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
