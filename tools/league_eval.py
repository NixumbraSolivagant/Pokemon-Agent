from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from local_eval.evaluator import run_candidate_pool, save_report, submission_name
from local_eval.models import EvalConfig
from tools.robust_gold_search import rank_report


def load_manifest(path: Path) -> tuple[list[Path], dict[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("opponents", [])
    return [Path(row["path"]) for row in rows], {submission_name(Path(row["path"])): row for row in rows}


def league_score(row, metadata: dict[str, dict]) -> dict:
    weighted_sum = weighted_weight = 0.0
    top_sum = top_weight = 0.0
    mirror_values = []
    lower_values = []
    for name, matchup in row.matchups.items():
        meta = metadata.get(name, {})
        weight = float(meta.get("weight", 1.0))
        lower = float(matchup["lower"])
        weighted_sum += weight * lower
        weighted_weight += weight
        lower_values.append(lower)
        if meta.get("category") == "top100":
            top_sum += weight * lower
            top_weight += weight
        if meta.get("category") == "mirror":
            mirror_values.append(lower)
    lower_values.sort()
    tail_count = max(1, round(len(lower_values) * 0.20)) if lower_values else 1
    cvar = sum(lower_values[:tail_count]) / tail_count if lower_values else 0.0
    weighted = weighted_sum / max(weighted_weight, 1e-9)
    top100 = top_sum / max(top_weight, 1e-9) if top_weight else weighted
    mirror = sum(mirror_values) / len(mirror_values) if mirror_values else weighted
    runtime_penalty = 0.03 * min(1.0, row.runtime_failures / max(1, row.games))
    objective = 0.40 * weighted + 0.25 * top100 + 0.20 * cvar + 0.10 * mirror + 0.05 - runtime_penalty
    return {
        **asdict(row),
        "weighted_lower": weighted,
        "top100_lower": top100,
        "worst20_cvar": cvar,
        "mirror_lower": mirror,
        "league_objective": objective,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidates against a weighted opponent league manifest.")
    parser.add_argument("--candidates", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--profile", choices=("legacy", "kaggle"), default="kaggle")
    parser.add_argument("--record-mode", choices=("all", "none", "losses", "sample", "training"), default="losses")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    opponents, metadata = load_manifest(args.manifest)
    config = EvalConfig(
        profile=args.profile, seed=args.seed, workers=args.workers, record_mode=args.record_mode, record_gzip=True,
        progress=True, progress_label=args.out.name, progress_mode="line", archive_cache_dir=str(args.out / "cache"),
        max_in_flight=args.workers * 2, common_random_seeds=args.profile == "legacy",
    )
    report = run_candidate_pool(args.candidates, opponents, args.games, config, Path.cwd(), peer_span=0)
    save_report(report, args.out)
    rows = [league_score(row, metadata) for row in rank_report(report, args.candidates, opponents)]
    rows.sort(key=lambda row: row["league_objective"], reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "league_ranking.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
