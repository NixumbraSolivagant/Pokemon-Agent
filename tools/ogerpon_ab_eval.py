from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from local_eval.evaluator import run_candidate_pool, save_report
from local_eval.models import EvalConfig
from tools.robust_gold_search import rank_report, write_ranking


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a strict candidate-vs-opponent Ogerpon A/B evaluation.")
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--profile", choices=("legacy", "kaggle"), default="kaggle")
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(
        profile=args.profile,
        workers=args.workers,
        seed=args.seed,
        record_mode="none",
        progress=True,
        progress_label=args.out.name,
        progress_mode="line",
        archive_cache_dir=str(args.out / "cache"),
        max_in_flight=args.workers * 2,
    )
    report = run_candidate_pool(args.candidates, args.opponents, args.games, config, Path.cwd(), peer_span=0)
    save_report(report, args.out)
    rows = rank_report(report, args.candidates, args.opponents)
    write_ranking(args.out / "ranking.json", rows)
    print(json.dumps([asdict(row) for row in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
