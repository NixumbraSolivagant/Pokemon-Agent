from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file
from local_eval.evaluator import run_ladder, save_report
from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.auto_iterate import (
    DEFAULT_POOL,
    config_from_dict,
    config_to_dict,
    hall_of_fame_paths,
    initial_champion_config,
    load_state,
    run_generation,
    save_state,
)
from tools.build_models import DEFAULT_BASE, BuildConfig
from tools.build_submission import build_submission
from tools.export_kaggle_submission import export_kaggle_submission


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_metadata(tarball: Path) -> dict[str, Any]:
    try:
        with tarfile.open(tarball, "r:gz") as tar:
            member = tar.extractfile("build_metadata.json")
            if member is None:
                return {}
            return json.loads(member.read().decode("utf-8"))
    except Exception:
        return {}


def state_from_source(state_path: Path, source_state: Path | None, base: Path) -> dict[str, Any]:
    if state_path.exists():
        return load_state(state_path, base)
    if source_state is not None and source_state.exists():
        state = load_json(source_state)
        state["incumbent"] = config_to_dict(config_from_dict(state["incumbent"]))
        return state
    cfg = initial_champion_config(base)
    return {"generation": 0, "incumbent": config_to_dict(cfg), "history": []}


def ensure_promoted_tarball(state: dict[str, Any], promote: Path) -> None:
    state.setdefault("incumbent", {})["out"] = str(promote)
    if promote.exists():
        return
    cfg = config_from_dict(state["incumbent"])
    cfg.name = "incumbent"
    cfg.out = promote
    build_submission(cfg)


def unique_existing(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen_paths: set[Path] = set()
    seen_sha: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        try:
            digest = sha256_file(resolved)
        except Exception:
            continue
        if digest in seen_sha:
            continue
        seen_paths.add(resolved)
        seen_sha.add(digest)
        out.append(resolved)
    return out


def evaluation_tarball_map(paths: list[Path], out_dir: Path, kaggle_safe: bool) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not kaggle_safe:
        return {path.name.removesuffix(".tar.gz"): path for path in paths if path.exists()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            continue
        target = out_dir / path.name
        export_kaggle_submission(
            path,
            target,
            strip_search_wrapper=False,
            exclude_internal_files=False,
        )
        out[path.name.removesuffix(".tar.gz")] = target
    return out


def resolve_report_tarball(report_path: Path, value: str) -> Path | None:
    path = Path(value)
    if path.exists():
        return path.resolve()
    fallbacks = [
        report_path.parent.parent / "variants" / path.name,
        Path("outputs/submissions") / path.name,
        Path("outputs/reference_submissions") / path.name,
        Path("基准") / path.name,
    ]
    for fallback in fallbacks:
        if fallback.exists():
            return fallback.resolve()
    return None


def is_promotable_path(path: Path, allow_cross_archetype: bool) -> bool:
    meta = load_metadata(path)
    if not meta:
        return False
    return allow_cross_archetype or str(meta.get("family", "great_tusk")) == "great_tusk"


def top_tarballs_from_report(path: Path, limit: int, blocked: set[Path], allow_cross_archetype: bool) -> list[Path]:
    if not path.exists():
        return []
    try:
        data = load_json(path)
    except Exception:
        return []
    out: list[Path] = []
    for row in data.get("standings", []):
        resolved = resolve_report_tarball(path, str(row.get("tarball", "")))
        if resolved is None:
            continue
        if resolved in blocked:
            continue
        if not is_promotable_path(resolved, allow_cross_archetype):
            continue
        out.append(resolved)
        if len(out) >= limit:
            break
    return out


def report_paths(out: Path, seed_reports: list[Path]) -> list[Path]:
    paths = [p for p in seed_reports if p.exists()]
    paths.extend(sorted(out.glob("generation_*/stage2/report.json")))
    paths.extend(sorted(out.glob("generation_*/stage1/report.json")))
    return paths


def collect_candidates(
    out: Path,
    promote: Path,
    pool: list[Path],
    state: dict[str, Any],
    seed_reports: list[Path],
    top_per_report: int,
    limit: int,
) -> list[Path]:
    blocked = {p.resolve() for p in pool if p.exists()}
    promote_resolved = promote.resolve() if promote.exists() else promote
    blocked.add(promote_resolved)
    candidates: list[Path] = []

    for path in report_paths(out, seed_reports):
        candidates.extend(top_tarballs_from_report(path, top_per_report, blocked, bool(state.get("allow_cross_archetype_promotion", False))))
    candidates.extend(
        path
        for path in hall_of_fame_paths(state, limit)
        if is_promotable_path(path, bool(state.get("allow_cross_archetype_promotion", False)))
    )
    candidates.extend(
        path
        for path in sorted((out).glob("generation_*/variants/*.tar.gz"))
        if is_promotable_path(path, bool(state.get("allow_cross_archetype_promotion", False)))
    )

    return unique_existing(candidates)[:limit]


def clean_enough(stats: AgentStats, max_no_result_rate: float) -> bool:
    no_result_rate = stats.no_results / max(1, stats.games)
    return (
        stats.crashes == 0
        and stats.timeouts == 0
        and stats.invalids == 0
        and no_result_rate <= max_no_result_rate
    )


def score_delta(report: MatchReport, candidate: str, incumbent: str) -> tuple[float, int, int]:
    stats = {s.name: s for s in report.standings}
    cand = stats[candidate]
    inc = stats[incumbent]
    rec = cand.opponents.get(incumbent, {"wins": 0, "losses": 0, "draws": 0})
    return cand.kaggle_score_estimate - inc.kaggle_score_estimate, rec["wins"], rec["losses"]


def incumbent_config_from_tarball(candidate: Path, promote: Path) -> dict[str, Any]:
    meta = load_metadata(candidate)
    if not meta:
        raise ValueError(f"{candidate} has no build_metadata.json; refusing to promote it as an evolving incumbent")
    cfg = BuildConfig(
        name="incumbent",
        family=str(meta.get("family", "great_tusk")),
        base=Path(str(meta.get("base", DEFAULT_BASE))),
        out=promote,
        enable_search=bool(meta.get("enable_search", True)),
        injection=str(meta.get("injection", "great_tusk")),
        search_candidates=int(meta.get("search_candidates", 8)),
        search_budget_s=float(meta.get("search_budget_s", 0.25)),
        search_margin=float(meta.get("search_margin", 1200.0)),
        search_rollout_steps=int(meta.get("search_rollout_steps", 16)),
        deck_swaps=[tuple(map(int, pair)) for pair in meta.get("deck_swaps", [])],
        deck_files=tuple(meta.get("deck_files", ("deck.csv",))),
        notes=str(meta.get("notes", "")),
    )
    return config_to_dict(cfg)


def promote_candidate(
    candidate: AgentStats,
    state: dict[str, Any],
    promote: Path,
    stage: str,
    delta: float,
    h2h_wins: int,
    h2h_losses: int,
    candidate_path_override: Path | None = None,
) -> None:
    candidate_path = candidate_path_override or Path(candidate.tarball)
    promote.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate_path, promote)
    state["incumbent"] = incumbent_config_from_tarball(candidate_path, promote)
    state.setdefault("history", []).append(
        {
            "generation": state.get("generation", 0),
            "stage": stage,
            "candidate": candidate.name,
            "promoted": True,
            "score_delta": f"{delta:.6f}",
            "h2h_wins": h2h_wins,
            "h2h_losses": h2h_losses,
            "tarball": str(candidate_path.resolve()),
            "reason": "gold_sprint verified promotion",
        }
    )


def export_current_submission(args: argparse.Namespace) -> None:
    if not args.promote.exists():
        return
    export_kaggle_submission(
        args.promote,
        args.submission_out,
        strip_search_wrapper=args.strip_search_wrapper_for_submission,
    )


def write_sprint_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "promoted",
        "candidate",
        "score_delta",
        "h2h_wins",
        "h2h_losses",
        "games",
        "tarball",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def verify_and_promote(
    stage: str,
    candidates: list[Path],
    state: dict[str, Any],
    args: argparse.Namespace,
    games: int,
) -> MatchReport | None:
    pool = [p for p in args.pool if p.exists()]
    original_tarballs = unique_existing([args.promote, *candidates, *pool])
    original_by_name = {path.name.removesuffix(".tar.gz"): path for path in original_tarballs}
    eval_by_name = evaluation_tarball_map(
        original_tarballs,
        args.out / "verification" / stage / "eval_tarballs",
        args.evaluate_kaggle_safe,
    )
    tarballs = unique_existing(list(eval_by_name.values()))
    if len(tarballs) < 2:
        return None
    out_dir = args.out / "verification" / stage
    config = EvalConfig(
        seed=args.seed + int(hashlib.sha256(stage.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000,
        workers=args.workers,
        max_actions=args.max_actions,
        run_timeout_s=args.run_timeout,
        record_mode=args.record_mode,
        record_focus="",
        record_sample_rate=args.record_sample_rate,
        record_gzip=True,
    )
    report = run_ladder(tarballs, games, config, Path.cwd())
    save_report(report, out_dir)

    incumbent_name = args.promote.name.removesuffix(".tar.gz")
    candidate_names = {p.name.removesuffix(".tar.gz") for p in candidates}
    stats = {s.name: s for s in report.standings}
    promotable = [s for s in report.standings if s.name in candidate_names and s.name in stats]
    promotable = [s for s in promotable if clean_enough(s, args.max_no_result_rate)]
    best = promotable[0] if promotable else None

    row: dict[str, Any] = {
        "stage": stage,
        "promoted": False,
        "candidate": best.name if best else "",
        "score_delta": "",
        "h2h_wins": "",
        "h2h_losses": "",
        "games": games,
        "tarball": best.tarball if best else "",
    }
    if best is not None and incumbent_name in stats:
        delta, h2h_wins, h2h_losses = score_delta(report, best.name, incumbent_name)
        should_promote = delta >= args.verify_min_score_delta and (
            h2h_wins >= h2h_losses + args.verify_h2h_margin or delta >= args.verify_large_score_delta
        )
        row.update(
            {
                "promoted": should_promote,
                "score_delta": f"{delta:.6f}",
                "h2h_wins": h2h_wins,
                "h2h_losses": h2h_losses,
            }
        )
        if should_promote:
            promote_candidate(
                best,
                state,
                args.promote,
                stage,
                delta,
                h2h_wins,
                h2h_losses,
                candidate_path_override=original_by_name.get(best.name),
            )
            save_state(args.state, state)
    export_current_submission(args)
    write_sprint_row(args.out / "sprint_summary.csv", row)
    write_json(
        args.out / "latest_verification.json",
        {
            "stage": stage,
            "out": str(out_dir),
            "row": row,
            "standings": [s.to_dict() for s in report.standings],
            "champion": str(args.promote),
            "champion_sha256": sha256_file(args.promote) if args.promote.exists() else "",
        },
    )
    return report


def auto_iterate_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        base=args.base,
        out=args.out,
        promote=args.promote,
        submission_out=args.submission_out,
        strip_search_wrapper_for_submission=args.strip_search_wrapper_for_submission,
        generations=1,
        population=args.population,
        stage1_games=args.stage1_games,
        stage2_games=args.stage2_games,
        finalists=args.finalists,
        workers=args.workers,
        max_actions=args.max_actions,
        run_timeout=args.run_timeout,
        seed=args.seed,
        min_score_delta=args.inner_min_score_delta,
        large_score_delta=args.inner_large_score_delta,
        max_no_result_rate=args.max_no_result_rate,
        evaluate_kaggle_safe=args.evaluate_kaggle_safe,
        promotion_policy=args.inner_promotion_policy,
        include_portfolio_seeds=args.include_portfolio_seeds,
        allow_cross_archetype_promotion=args.allow_cross_archetype_promotion,
        record_mode=args.record_mode,
        record_sample_rate=args.record_sample_rate,
        pool=args.pool,
        hof_size=args.hof_size,
    )


def run_cycles(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    runner_args = auto_iterate_args(args)
    for _ in range(args.cycles):
        for _ in range(args.generations_per_cycle):
            generation = int(state.get("generation", 0)) + 1
            state = run_generation(state, runner_args, generation)
            save_state(args.state, state)
            print(json.dumps({"generation": generation, "incumbent": state["incumbent"]}, indent=2))
        candidates = collect_candidates(
            args.out,
            args.promote,
            args.pool,
            state,
            args.seed_reports,
            args.top_per_report,
            args.verify_candidates,
        )
        verify_and_promote(f"cycle_{state.get('generation', 0):03d}", candidates, state, args, args.verify_games)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unattended high-confidence sprint for Pokemon TCG submissions.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=Path("outputs/gold_sprint"))
    parser.add_argument("--state", type=Path, default=Path("outputs/gold_sprint/state.json"))
    parser.add_argument("--source-state", type=Path, default=Path("outputs/auto_iterate_server_gold/state.json"))
    parser.add_argument("--promote", type=Path, default=Path("outputs/submissions/champion_latest.tar.gz"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submissions/submission.tar.gz"))
    parser.add_argument("--strip-search-wrapper-for-submission", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--evaluate-kaggle-safe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--generations-per-cycle", type=int, default=2)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--stage1-games", type=int, default=3)
    parser.add_argument("--stage2-games", type=int, default=40)
    parser.add_argument("--verify-games", type=int, default=80)
    parser.add_argument("--final-games", type=int, default=160)
    parser.add_argument("--verify-candidates", type=int, default=10)
    parser.add_argument("--top-per-report", type=int, default=4)
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--inner-min-score-delta", type=float, default=12.0)
    parser.add_argument("--inner-large-score-delta", type=float, default=50.0)
    parser.add_argument("--inner-promotion-policy", choices=["balanced", "strict", "score"], default="balanced")
    parser.add_argument("--verify-min-score-delta", type=float, default=10.0)
    parser.add_argument("--verify-large-score-delta", type=float, default=35.0)
    parser.add_argument("--verify-h2h-margin", type=int, default=1)
    parser.add_argument("--max-no-result-rate", type=float, default=0.02)
    parser.add_argument("--include-portfolio-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cross-archetype-promotion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-mode", choices=["none", "losses", "sample", "all"], default="losses")
    parser.add_argument("--record-sample-rate", type=float, default=0.01)
    parser.add_argument("--pool", nargs="*", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--hof-size", type=int, default=8)
    parser.add_argument(
        "--seed-reports",
        nargs="*",
        type=Path,
        default=[Path("outputs/auto_iterate_server_gold/generation_004/stage2/report.json")],
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    args.promote.parent.mkdir(parents=True, exist_ok=True)
    state = state_from_source(args.state, args.source_state, args.base)
    state["allow_cross_archetype_promotion"] = args.allow_cross_archetype_promotion
    ensure_promoted_tarball(state, args.promote)
    save_state(args.state, state)

    if not args.skip_bootstrap:
        candidates = collect_candidates(
            args.out,
            args.promote,
            args.pool,
            state,
            args.seed_reports,
            args.top_per_report,
            args.verify_candidates,
        )
        verify_and_promote("bootstrap", candidates, state, args, args.verify_games)

    state = run_cycles(state, args)
    final_candidates = collect_candidates(
        args.out,
        args.promote,
        args.pool,
        state,
        args.seed_reports,
        args.top_per_report,
        args.verify_candidates,
    )
    final_report = verify_and_promote("final", final_candidates, state, args, args.final_games)
    export_current_submission(args)
    write_json(
        args.out / "final_report.json",
        {
            "champion": str(args.promote),
            "champion_sha256": sha256_file(args.promote) if args.promote.exists() else "",
            "submission": str(args.submission_out),
            "submission_sha256": sha256_file(args.submission_out) if args.submission_out.exists() else "",
            "state": state,
            "final_standings": [s.to_dict() for s in final_report.standings] if final_report else [],
        },
    )
    print(json.dumps(load_json(args.out / "final_report.json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
