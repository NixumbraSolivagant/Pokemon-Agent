from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file
from local_eval.evaluator import run_ladder, save_report
from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.build_submission import BuildConfig, build_submission
from tools.export_kaggle_submission import export_kaggle_submission
from tools.prior_deck_space import (
    REFERENCE_BASES,
    great_tusk_prior_configs,
    metal_prior_configs,
    portfolio_seed_configs,
)


DEFAULT_OUT = Path("outputs/gold_factory")
DEFAULT_PROMOTE = Path("outputs/submissions/gold_factory_champion.tar.gz")
DEFAULT_SUBMISSION = Path("outputs/submissions/submission.tar.gz")
DEFAULT_POOL = [
    Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
    Path("outputs/reference_submissions/pokemon-steel.tar.gz"),
    Path("outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz"),
    Path("outputs/reference_submissions/improved-probabilistic-agent.tar.gz"),
    Path("outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz"),
    Path("基准/submission_sorce_700.tar.gz"),
]


@dataclass(slots=True)
class CandidateRecord:
    name: str
    family: str
    internal_tarball: str
    eval_tarball: str
    notes: str = ""
    sha256: str = ""


@dataclass(slots=True)
class StagePlan:
    name: str
    games_per_pair: int
    candidate_limit: int
    pool_limit: int
    record_mode: str


@dataclass(slots=True)
class GoldProfile:
    name: str
    population: int
    stage_a: StagePlan
    stage_b: StagePlan
    stage_c: StagePlan
    finalists: int
    manual_slots: int
    workers: int
    max_actions: int
    run_timeout_s: float
    max_no_result_rate: float


def profile_from_name(name: str) -> GoldProfile:
    if name == "smoke":
        return GoldProfile(
            name=name,
            population=12,
            stage_a=StagePlan("stage_a", 1, 12, 4, "none"),
            stage_b=StagePlan("stage_b", 2, 6, 4, "none"),
            stage_c=StagePlan("stage_c", 4, 3, 4, "losses"),
            finalists=3,
            manual_slots=3,
            workers=4,
            max_actions=300,
            run_timeout_s=90.0,
            max_no_result_rate=0.02,
        )
    if name != "a800_gold":
        raise ValueError(f"Unknown profile {name!r}; expected a800_gold or smoke")
    return GoldProfile(
        name=name,
        population=96,
        stage_a=StagePlan("stage_a", 3, 96, 8, "none"),
        stage_b=StagePlan("stage_b", 40, 28, 8, "losses"),
        stage_c=StagePlan("stage_c", 160, 8, 8, "losses"),
        finalists=5,
        manual_slots=5,
        workers=36,
        max_actions=1000,
        run_timeout_s=180.0,
        max_no_result_rate=0.01,
    )


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
    seen_sha: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            digest = sha256_file(path)
        except Exception:
            continue
        if digest in seen_sha:
            continue
        seen_sha.add(digest)
        out.append(path.resolve())
    return out


def clean_stats(stats: AgentStats, max_no_result_rate: float) -> bool:
    return (
        stats.crashes == 0
        and stats.timeouts == 0
        and stats.invalids == 0
        and stats.no_results / max(1, stats.games) <= max_no_result_rate
    )


def top_clean_candidates(
    report: MatchReport,
    candidate_names: set[str],
    limit: int,
    max_no_result_rate: float,
) -> list[AgentStats]:
    rows = [
        row
        for row in report.standings
        if row.name in candidate_names and clean_stats(row, max_no_result_rate)
    ]
    rows.sort(
        key=lambda s: (
            s.kaggle_score_estimate,
            s.wins - s.losses,
            s.wins,
            -s.no_results,
        ),
        reverse=True,
    )
    return rows[: max(1, limit)]


def config_to_record(cfg: BuildConfig, eval_tarball: Path) -> CandidateRecord:
    return CandidateRecord(
        name=cfg.name,
        family=cfg.family,
        internal_tarball=str(cfg.out.resolve()),
        eval_tarball=str(eval_tarball.resolve()),
        notes=cfg.notes,
        sha256=sha256_file(cfg.out),
    )


def seed_configs(out_dir: Path) -> list[BuildConfig]:
    configs: list[BuildConfig] = []
    for family, base in REFERENCE_BASES.items():
        if not base.exists():
            continue
        configs.append(
            BuildConfig(
                name=f"seed_{family}",
                family=family,
                base=base,
                out=out_dir / f"seed_{family}.tar.gz",
                injection="none" if family != "great_tusk" else "great_tusk",
                enable_search=family == "great_tusk",
                deck_files=("deck.csv", "lucario_deck.csv") if family == "lucario" else ("deck.csv",),
                notes=f"First-class seed candidate from {base.name}.",
            )
        )
    return configs


def generate_candidate_configs(out_dir: Path, population: int) -> list[BuildConfig]:
    configs: list[BuildConfig] = []
    configs.extend(seed_configs(out_dir))

    gt_base = REFERENCE_BASES["great_tusk"]
    gt_incumbent = BuildConfig(
        name="gt_anchor",
        family="great_tusk",
        base=gt_base,
        out=out_dir / "gt_anchor.tar.gz",
        injection="great_tusk",
        enable_search=True,
        notes="Great Tusk anchor for broad deck/search exploration.",
    )
    configs.extend(great_tusk_prior_configs(gt_incumbent, 1, out_dir, max(12, population // 3)))
    configs.extend(metal_prior_configs(1, out_dir, max(12, population // 3)))
    configs.extend(portfolio_seed_configs(1, out_dir))

    # Lucario variants need both deck files mutated because the reference main.py
    # reads lucario_deck.csv directly.
    lucario = REFERENCE_BASES.get("lucario")
    if lucario and lucario.exists():
        lucario_swaps = [
            ("lucario_boss_pressure", [(1182, 1213)]),
            ("lucario_switch_density", [(1123, 1213)]),
            ("lucario_energy_density", [(6, 1213)]),
            ("lucario_hero_cape", [(1159, 1152)]),
            ("lucario_lillie_trim", [(1182, 1227)]),
        ]
        for name, swaps in lucario_swaps:
            configs.append(
                BuildConfig(
                    name=f"g001_{name}",
                    family="lucario",
                    base=lucario,
                    out=out_dir / f"g001_{name}.tar.gz",
                    injection="none",
                    enable_search=False,
                    deck_swaps=swaps,
                    deck_files=("deck.csv", "lucario_deck.csv"),
                    notes=f"Lucario broad-search variant: {name}.",
                )
            )

    seen: set[str] = set()
    unique: list[BuildConfig] = []
    for cfg in configs:
        if cfg.name in seen:
            continue
        seen.add(cfg.name)
        unique.append(cfg)
        if len(unique) >= population:
            break
    return unique


def build_candidates(
    configs: list[BuildConfig],
    eval_dir: Path,
    strip_search_wrapper: bool,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    eval_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        try:
            built = build_submission(cfg)
            eval_tarball = eval_dir / built.name
            export_kaggle_submission(
                built,
                eval_tarball,
                strip_search_wrapper=strip_search_wrapper,
                exclude_internal_files=False,
            )
            records.append(config_to_record(cfg, eval_tarball))
        except Exception as exc:
            print(json.dumps({"skipped": cfg.name, "reason": f"{type(exc).__name__}: {exc}"}))
    return records


def stage_eval_config(args: argparse.Namespace, profile: GoldProfile, stage: StagePlan, seed_offset: int) -> EvalConfig:
    return EvalConfig(
        seed=args.seed + seed_offset,
        workers=args.workers or profile.workers,
        max_actions=args.max_actions or profile.max_actions,
        run_timeout_s=args.run_timeout or profile.run_timeout_s,
        record_mode=stage.record_mode,
        record_sample_rate=args.record_sample_rate,
        record_gzip=True,
    )


def run_stage(
    records: list[CandidateRecord],
    selected_names: set[str] | list[str],
    pool: list[Path],
    out_dir: Path,
    stage: StagePlan,
    args: argparse.Namespace,
    profile: GoldProfile,
    seed_offset: int,
) -> tuple[MatchReport, list[AgentStats]]:
    if isinstance(selected_names, set):
        selected_records = [r for r in records if r.name in selected_names][: stage.candidate_limit]
    else:
        by_name = candidate_by_name(records)
        selected_records = [by_name[name] for name in selected_names if name in by_name][: stage.candidate_limit]
    tarballs = unique_existing(
        [Path(r.eval_tarball) for r in selected_records]
        + pool[: stage.pool_limit]
    )
    cfg = stage_eval_config(args, profile, stage, seed_offset)
    report = run_ladder(tarballs, stage.games_per_pair, cfg, Path.cwd())
    save_report(report, out_dir / stage.name)
    top = top_clean_candidates(
        report,
        {r.name for r in selected_records},
        args.finalists or profile.finalists,
        profile.max_no_result_rate,
    )
    return report, top


def candidate_by_name(records: list[CandidateRecord]) -> dict[str, CandidateRecord]:
    return {record.name: record for record in records}


def export_final(best: CandidateRecord, promote: Path, submission_out: Path, strip_search_wrapper: bool) -> None:
    promote.parent.mkdir(parents=True, exist_ok=True)
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best.internal_tarball, promote)
    export_kaggle_submission(
        promote,
        submission_out,
        strip_search_wrapper=strip_search_wrapper,
    )


def export_manual_candidates(
    records_by_name: dict[str, CandidateRecord],
    finalists: list[AgentStats],
    out_dir: Path,
    slots: int,
    strip_search_wrapper: bool,
) -> list[tuple[CandidateRecord, Path]]:
    out: list[tuple[CandidateRecord, Path]] = []
    export_dir = out_dir / "manual_candidates"
    export_dir.mkdir(parents=True, exist_ok=True)
    for stats in finalists[: max(0, slots)]:
        record = records_by_name.get(stats.name)
        if record is None:
            continue
        target = export_dir / f"{len(out) + 1:02d}_{record.name}.tar.gz"
        export_kaggle_submission(
            Path(record.internal_tarball),
            target,
            strip_search_wrapper=strip_search_wrapper,
        )
        out.append((record, target))
    return out


def run_factory(args: argparse.Namespace) -> dict[str, Any]:
    profile = profile_from_name(args.profile)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = read_json(state_path) if args.resume else {}

    if "records" not in state:
        configs = generate_candidate_configs(out_dir / "variants", args.population or profile.population)
        records = build_candidates(configs, out_dir / "eval_variants", args.strip_search_wrapper)
        state["records"] = [asdict(record) for record in records]
        state["profile"] = asdict(profile)
        write_json(state_path, state)
    else:
        records = [CandidateRecord(**row) for row in state["records"]]

    pool = unique_existing(args.pool)
    all_candidate_names = {record.name for record in records}
    report_a, top_a = run_stage(
        records,
        all_candidate_names,
        pool,
        out_dir,
        profile.stage_a,
        args,
        profile,
        101,
    )
    state["stage_a_top"] = [row.to_dict() for row in top_a]
    write_json(state_path, state)

    report_b, top_b = run_stage(
        records,
        [row.name for row in top_a],
        pool,
        out_dir,
        profile.stage_b,
        args,
        profile,
        202,
    )
    state["stage_b_top"] = [row.to_dict() for row in top_b]
    write_json(state_path, state)

    report_c, top_c = run_stage(
        records,
        [row.name for row in top_b],
        pool,
        out_dir,
        profile.stage_c,
        args,
        profile,
        303,
    )
    state["stage_c_top"] = [row.to_dict() for row in top_c]

    if not top_c:
        state["final"] = {"status": "failed", "reason": "no clean finalists"}
        write_json(state_path, state)
        return state

    records_by_name = candidate_by_name(records)
    best = records_by_name[top_c[0].name]
    export_final(best, args.promote, args.submission_out, args.strip_search_wrapper)
    manual_candidates = export_manual_candidates(
        records_by_name,
        top_c,
        out_dir,
        args.manual_slots if args.manual_slots is not None else profile.manual_slots,
        args.strip_search_wrapper,
    )
    state["final"] = {
        "status": "ok",
        "best": asdict(best),
        "promote": str(args.promote),
        "submission": str(args.submission_out),
        "submission_sha256": sha256_file(args.submission_out),
        "manual_candidates": [{"name": r.name, "tarball": str(p)} for r, p in manual_candidates],
        "stage_c_standings": [s.to_dict() for s in report_c.standings],
    }
    write_json(state_path, state)
    write_json(out_dir / "final_report.json", state["final"])
    return state


def status(args: argparse.Namespace) -> dict[str, Any]:
    state = read_json(args.out / "state.json")
    final = read_json(args.out / "final_report.json")
    return {
        "out": str(args.out),
        "has_state": bool(state),
        "record_count": len(state.get("records", [])),
        "stage_a_top": [row.get("name") for row in state.get("stage_a_top", [])[:5]],
        "stage_b_top": [row.get("name") for row in state.get("stage_b_top", [])[:5]],
        "stage_c_top": [row.get("name") for row in state.get("stage_c_top", [])[:5]],
        "final": final,
    }


def export_final_from_state(args: argparse.Namespace) -> dict[str, Any]:
    state = read_json(args.out / "state.json")
    final = state.get("final") or {}
    best = final.get("best")
    if not best:
        raise SystemExit(f"No final best candidate in {args.out / 'state.json'}")
    record = CandidateRecord(**best)
    export_final(record, args.promote, args.submission_out, args.strip_search_wrapper)
    return {
        "promote": str(args.promote),
        "submission": str(args.submission_out),
        "submission_sha256": sha256_file(args.submission_out),
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="a800_gold", choices=["a800_gold", "smoke"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--promote", type=Path, default=DEFAULT_PROMOTE)
    parser.add_argument("--submission-out", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--population", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--run-timeout", type=float)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--finalists", type=int)
    parser.add_argument("--record-sample-rate", type=float, default=0.01)
    parser.add_argument("--pool", nargs="*", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--strip-search-wrapper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--manual-slots", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gold-oriented end-to-end factory for Pokemon TCG AI Battle submissions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Build candidates, evaluate them, and export finalists for manual upload.")
    add_common_args(p_run)

    p_resume = sub.add_parser("resume", help="Alias for run with state reuse.")
    add_common_args(p_resume)
    p_resume.set_defaults(resume=True)

    p_status = sub.add_parser("status", help="Print current factory state.")
    p_status.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p_export = sub.add_parser("export-final", help="Re-export the current best final submission.")
    p_export.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_export.add_argument("--promote", type=Path, default=DEFAULT_PROMOTE)
    p_export.add_argument("--submission-out", type=Path, default=DEFAULT_SUBMISSION)
    p_export.add_argument("--strip-search-wrapper", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args(argv)
    if args.cmd in {"run", "resume"}:
        result = run_factory(args)
    elif args.cmd == "status":
        result = status(args)
    elif args.cmd == "export-final":
        result = export_final_from_state(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
