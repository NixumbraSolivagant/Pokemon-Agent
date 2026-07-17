from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from local_eval.evaluator import run_ladder, save_report
from local_eval.models import EvalConfig
from tools.build_submission import BuildConfig, DEFAULT_BASE, build_submission
from tools.export_kaggle_submission import export_kaggle_submission


DEFAULT_POOL = [
    "outputs/reference_submissions/i-have-one-rear-card.tar.gz",
    "outputs/reference_submissions/pokemon-steel.tar.gz",
    "outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz",
    "outputs/reference_submissions/improved-probabilistic-agent.tar.gz",
    "outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz",
    "基准/submission_sorce_700.tar.gz",
]


def preset_variants(base: Path, out_dir: Path) -> list[BuildConfig]:
    presets = [
        BuildConfig(
            name="champion_gt_search",
            base=base,
            out=out_dir / "champion_gt_search.tar.gz",
            search_candidates=8,
            search_budget_s=0.25,
            search_margin=1200.0,
            notes="Default shallow search wrapper.",
        ),
        BuildConfig(
            name="champion_gt_wide",
            base=base,
            out=out_dir / "champion_gt_wide.tar.gz",
            search_candidates=14,
            search_budget_s=0.45,
            search_margin=900.0,
            notes="Wider one-step search.",
        ),
        BuildConfig(
            name="champion_gt_conservative",
            base=base,
            out=out_dir / "champion_gt_conservative.tar.gz",
            search_candidates=6,
            search_budget_s=0.15,
            search_margin=2400.0,
            notes="Only overrides heuristic on large search advantage.",
        ),
        BuildConfig(
            name="champion_gt_ultraball",
            base=base,
            out=out_dir / "champion_gt_ultraball.tar.gz",
            search_candidates=8,
            search_budget_s=0.25,
            search_margin=1200.0,
            deck_swaps=[(1121, 1123)],
            notes="+1 Ultra Ball, -1 Switch.",
        ),
        BuildConfig(
            name="champion_gt_hammer",
            base=base,
            out=out_dir / "champion_gt_hammer.tar.gz",
            search_candidates=8,
            search_budget_s=0.25,
            search_margin=1200.0,
            deck_swaps=[(1081, 1123)],
            notes="+1 Enhanced Hammer, -1 Switch.",
        ),
        BuildConfig(
            name="champion_gt_hand_trim",
            base=base,
            out=out_dir / "champion_gt_hand_trim.tar.gz",
            search_candidates=8,
            search_budget_s=0.25,
            search_margin=1200.0,
            deck_swaps=[(1087, 1182)],
            notes="+1 Hand Trimmer, -1 Boss.",
        ),
    ]
    return presets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Great Tusk variants and run a local ladder.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=Path("outputs/iterations/gt_run"))
    parser.add_argument("--games-per-pair", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--pool", nargs="*", default=DEFAULT_POOL)
    parser.add_argument("--promote", type=Path, default=Path("outputs/submissions/champion_latest.tar.gz"))
    parser.add_argument("--record-mode", choices=["none", "losses", "sample", "all"], default="losses")
    parser.add_argument("--record-focus", default="")
    parser.add_argument("--record-sample-rate", type=float, default=0.05)
    parser.add_argument("--record-gzip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submissions/submission.tar.gz"))
    parser.add_argument("--strip-search-wrapper-for-submission", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    variants_dir = args.out / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    built = [build_submission(replace(cfg, out=variants_dir / cfg.out.name)) for cfg in preset_variants(args.base, variants_dir)]
    pool = [Path(p) for p in args.pool if Path(p).exists()]
    tarballs = [*built, *pool]
    cfg = EvalConfig(
        seed=args.seed,
        workers=max(1, args.workers),
        max_actions=args.max_actions,
        run_timeout_s=args.run_timeout,
        record_mode=args.record_mode,
        record_focus=args.record_focus,
        record_sample_rate=args.record_sample_rate,
        record_gzip=args.record_gzip,
    )
    report = run_ladder(tarballs, args.games_per_pair, cfg, Path.cwd())
    save_report(report, args.out / "ladder")
    champion_names = {p.name.removesuffix(".tar.gz") for p in built}
    best = next((s for s in report.standings if s.name in champion_names), None)
    if best is not None:
        args.promote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best.tarball, args.promote)
        export_kaggle_submission(
            args.promote,
            args.submission_out,
            strip_search_wrapper=args.strip_search_wrapper_for_submission,
        )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "promoted": str(args.promote) if best is not None else None,
                "best_variant": best.to_dict() if best is not None else None,
                "standings": [s.to_dict() for s in report.standings],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
