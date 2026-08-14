from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .evaluator import run_ladder, run_match, save_report
from .models import EvalConfig, default_output_dir


def _config_from_args(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        profile=args.profile,
        act_timeout_s=args.act_timeout,
        import_timeout_s=args.import_timeout,
        deck_timeout_s=args.deck_timeout,
        overage_time_s=args.overage_time,
        run_timeout_s=args.run_timeout,
        max_actions=args.max_actions,
        seed=args.seed,
        workers=args.workers,
        record_mode=args.record_mode,
        record_focus=args.record_focus or "",
        record_sample_rate=args.record_sample_rate,
        record_gzip=args.record_gzip,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Kaggle-style evaluator for Pokemon TCG AI Battle submissions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", choices=("legacy", "kaggle"), default="kaggle")
    common.add_argument("--seed", type=int, default=20260717)
    common.add_argument("--import-timeout", type=float, default=6.0)
    common.add_argument("--deck-timeout", type=float, default=6.0)
    common.add_argument("--act-timeout", type=float, default=6.0)
    common.add_argument("--overage-time", type=float, default=12.0)
    common.add_argument("--run-timeout", type=float, default=1200.0)
    common.add_argument("--max-actions", type=int, default=1000)
    common.add_argument("--out", type=Path, default=None)
    common.add_argument("--workers", type=int, default=1)
    common.add_argument("--record-mode", choices=["all", "none", "losses", "sample", "training"], default="losses")
    common.add_argument("--record-focus", default="")
    common.add_argument("--record-sample-rate", type=float, default=0.05)
    common.add_argument("--record-gzip", action="store_true")

    p_match = sub.add_parser("match", parents=[common], help="Evaluate two submission tarballs.")
    p_match.add_argument("tarball_a")
    p_match.add_argument("tarball_b")
    p_match.add_argument("--games", type=int, default=10)

    p_ladder = sub.add_parser("ladder", parents=[common], help="Run round-robin ladder over submission tarballs.")
    p_ladder.add_argument("--submissions", required=True, help="Glob pattern for submission tarballs.")
    p_ladder.add_argument("--baseline", help="Optional baseline tarball to include if not matched by the glob.")
    p_ladder.add_argument("--games-per-pair", type=int, default=10)

    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = _config_from_args(args)
    out_dir = args.out or default_output_dir()

    if args.cmd == "match":
        report = run_match(args.tarball_a, args.tarball_b, args.games, config, project_root)
    elif args.cmd == "ladder":
        tarballs = sorted(glob.glob(args.submissions))
        if args.baseline and str(Path(args.baseline)) not in tarballs:
            tarballs.append(args.baseline)
        if len(tarballs) < 2:
            raise SystemExit("Need at least two submission tarballs for a ladder.")
        report = run_ladder(tarballs, args.games_per_pair, config, project_root)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    save_report(report, out_dir)
    print(json.dumps({"out": str(out_dir), "standings": [s.to_dict() for s in report.standings]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
