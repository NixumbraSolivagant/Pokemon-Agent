from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_eval.evaluator import run_candidate_pool, save_report
from local_eval.models import EvalConfig
from tools.robust_gold_search import combine_rankings, rank_report, write_ranking
from tools.server_candidate_eval import parse_paths


def evaluate(
    label: str,
    candidates: list[Path],
    opponents: list[Path],
    games: int,
    args: argparse.Namespace,
    seed_offset: int,
    profile: str = "legacy",
) -> list:
    out = args.out / label
    out.mkdir(parents=True, exist_ok=True)
    report = run_candidate_pool(
        candidates,
        opponents,
        games,
        EvalConfig(
            profile=profile,
            seed=args.seed + seed_offset,
            workers=args.workers,
            run_timeout_s=args.run_timeout if profile == "legacy" else 2000.0,
            max_actions=args.max_actions if profile == "legacy" else 10_000_000,
            record_mode="none",
            progress=True,
            progress_label=label,
            progress_mode="line",
            archive_cache_dir=str(out / "cache"),
            max_in_flight=args.workers * 2,
            common_random_seeds=profile == "legacy",
        ),
        Path.cwd(),
        peer_span=0,
    )
    save_report(report, out)
    rows = rank_report(report, candidates, opponents)
    write_ranking(out / "ranking.json", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Low-priority remote train/holdout autoscreener. It never submits to Kaggle.")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-actions", type=int, default=600)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--stage1-games", type=int, default=12)
    parser.add_argument("--stage2-games", type=int, default=40)
    parser.add_argument("--holdout-games", type=int, default=80)
    parser.add_argument("--stage2-count", type=int, default=6)
    parser.add_argument("--holdout-count", type=int, default=4)
    args = parser.parse_args(argv)
    candidates = parse_paths(args.candidates)
    train = parse_paths(args.train)
    holdout = parse_paths(args.holdout)
    if not candidates or not train or not holdout:
        raise ValueError("candidates, train, and holdout must be non-empty")

    stage1 = evaluate("stage1", candidates, train, args.stage1_games, args, 0)
    stage2_candidates = [Path(row.tarball) for row in stage1[: args.stage2_count]]
    stage2 = evaluate("stage2", stage2_candidates, train, args.stage2_games, args, 100_000)
    holdout_candidates = [Path(row.tarball) for row in stage2[: args.holdout_count]]
    holdout_rows = evaluate("holdout", holdout_candidates, holdout, args.holdout_games, args, 200_000, profile="kaggle")
    combined = combine_rankings(stage2, holdout_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "combined_ranking.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps({"winner": combined[0] if combined else None, "stage1": len(stage1), "stage2": len(stage2), "holdout": len(holdout_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
