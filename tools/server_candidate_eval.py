from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from local_eval.evaluator import run_candidate_pool, save_report
from local_eval.models import EvalConfig
from tools.robust_gold_search import rank_report, write_ranking


def parse_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches:
            matches = [Path(pattern)]
        for path in matches:
            resolved = path.resolve()
            if resolved in seen:
                continue
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(resolved)
            paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Server-safe candidate-pool evaluator with a spawnable file entry point.")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--record-mode", choices=("none", "losses", "sample", "all"), default="losses")
    parser.add_argument("--record-gzip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--peer-span", type=int, default=0)
    args = parser.parse_args(argv)

    candidates = parse_paths(args.candidates)
    opponents = parse_paths(args.opponents)
    args.out.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(
        profile="kaggle",
        seed=args.seed,
        workers=args.workers,
        record_mode=args.record_mode,
        record_gzip=args.record_gzip,
        progress=True,
        progress_label=args.out.name,
        progress_mode="line",
        archive_cache_dir=str(args.out / "cache"),
        max_in_flight=args.workers * 2,
    )
    report = run_candidate_pool(
        candidates,
        opponents,
        args.games,
        config,
        Path.cwd(),
        peer_span=args.peer_span,
    )
    save_report(report, args.out)
    rows = rank_report(report, candidates, opponents)
    write_ranking(args.out / "ranking.json", rows)
    summary = [
        {
            "name": row.name,
            "games": row.games,
            "wins": row.wins,
            "losses": row.losses,
            "no_results": row.no_results,
            "mean_rate": row.mean_rate,
            "worst_rate": row.worst_rate,
            "worst_lower": row.worst_lower,
            "robust_score": row.robust_score,
        }
        for row in rows
    ]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
