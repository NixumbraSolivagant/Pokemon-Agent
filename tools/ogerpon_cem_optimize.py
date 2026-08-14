from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_eval.evaluator import run_candidate_pool, save_report, submission_name
from local_eval.models import EvalConfig
from tools.behavior_clone_audit import audit
from tools.build_meta_submission import build
from tools.robust_gold_search import rank_report, write_ranking


PARAMETERS = {
    "model_weight": (200000.0, 50000.0, 100000.0, 350000.0),
    "residual_weight": (90000.0, 50000.0, 0.0, 250000.0),
    "risk_penalty": (0.20, 0.12, 0.0, 0.80),
    "margin": (2500.0, 5000.0, 0.0, 30000.0),
    "budget_s": (0.16, 0.06, 0.05, 0.35),
    "value_model_weight": (24000.0, 14000.0, 0.0, 80000.0),
    "candidates": (4.0, 1.0, 2.0, 6.0),
    "belief_worlds": (3.0, 1.0, 1.0, 8.0),
    "rollout_steps": (8.0, 2.0, 4.0, 16.0),
}
INTEGER_PARAMETERS = {"candidates", "belief_worlds", "rollout_steps"}


def centered_distribution(
    center: dict[str, float | int] | None = None,
    std_scale: float = 1.0,
) -> tuple[dict[str, dict[str, float]], dict[str, float | int] | None]:
    if std_scale <= 0.0:
        raise ValueError("std_scale must be positive")
    distribution = {
        name: {"mean": mean, "std": std * std_scale, "low": low, "high": high}
        for name, (mean, std, low, high) in PARAMETERS.items()
    }
    if center is None:
        return distribution, None
    missing = sorted(set(PARAMETERS) - set(center))
    if missing:
        raise ValueError(f"center vector is missing parameters: {', '.join(missing)}")
    normalized: dict[str, float | int] = {}
    for name, values in distribution.items():
        value = max(values["low"], min(values["high"], float(center[name])))
        normalized[name] = int(round(value)) if name in INTEGER_PARAMETERS else value
        values["mean"] = float(normalized[name])
    return distribution, normalized


def sample_vector(distribution: dict[str, dict[str, float]], rng: random.Random) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, values in distribution.items():
        value = max(values["low"], min(values["high"], rng.gauss(values["mean"], values["std"])))
        result[name] = int(round(value)) if name in INTEGER_PARAMETERS else value
    return result


def overrides(vector: dict[str, float | int]) -> dict[str, Any]:
    return {
        "model_weight": vector["model_weight"],
        "residual_weight": vector["residual_weight"],
        "search": {
            "risk_penalty": vector["risk_penalty"],
            "margin": vector["margin"],
            "budget_s": vector["budget_s"],
            "value_model_weight": vector["value_model_weight"],
            "candidates": vector["candidates"],
            "belief_worlds": vector["belief_worlds"],
            "rollout_steps": vector["rollout_steps"],
        },
    }


def update_distribution(
    distribution: dict[str, dict[str, float]],
    elites: list[dict[str, float | int]],
    smoothing: float,
) -> None:
    for name, values in distribution.items():
        samples = [float(vector[name]) for vector in elites]
        elite_mean = sum(samples) / len(samples)
        elite_std = math.sqrt(sum((sample - elite_mean) ** 2 for sample in samples) / len(samples))
        values["mean"] = smoothing * values["mean"] + (1.0 - smoothing) * elite_mean
        minimum_std = (values["high"] - values["low"]) * 0.015
        values["std"] = max(minimum_std, smoothing * values["std"] + (1.0 - smoothing) * elite_std)


def latency_penalty(report, candidate_name: str, threshold_s: float) -> float:
    durations = [game.duration_s for game in report.games if candidate_name in {game.p0, game.p1} and game.ranking_eligible]
    if not durations:
        return 1.0
    mean_duration = sum(durations) / len(durations)
    return max(0.0, mean_duration - threshold_s) * 0.002


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize Ogerpon residual/search parameters with CEM and successive fidelity.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--target-name", default="keidroid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="clone_residual_search")
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--smoothing", type=float, default=0.60)
    parser.add_argument("--games", nargs="+", type=int, default=[4, 12, 32, 64, 64])
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--min-semantic-rate", type=float, default=0.80)
    parser.add_argument("--min-ability-rate", type=float, default=0.85)
    parser.add_argument("--latency-threshold-s", type=float, default=20.0)
    parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--center-vector", type=Path)
    parser.add_argument("--std-scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    test_episode_ids = {str(value).split(":")[-1] for value in artifact.get("split_stats", {}).get("test_episodes", [])}
    center = json.loads(args.center_vector.read_text(encoding="utf-8")) if args.center_vector else None
    distribution, center = centered_distribution(center, args.std_scale)
    rng = random.Random(args.seed)
    history = []
    global_best: list[dict[str, Any]] = []
    for generation in range(args.generations):
        generation_dir = args.out / f"generation_{generation:02d}"
        candidate_dir = generation_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        vectors = [center] if center is not None else []
        vectors.extend(sample_vector(distribution, rng) for _ in range(args.population - len(vectors)))
        candidates = []
        vector_by_name = {}
        for index, vector in enumerate(vectors):
            package = candidate_dir / f"ogerpon_cem_g{generation:02d}_{index:03d}.tar.gz"
            build(
                args.artifact,
                package,
                args.runtime,
                args.cg_dir,
                args.variant,
                linux_only=True,
                config_overrides=overrides(vector),
            )
            candidates.append(package)
            vector_by_name[submission_name(package)] = vector
        config = EvalConfig(
            profile="legacy",
            workers=args.workers,
            seed=args.seed + generation * 100_000,
            record_mode="none",
            progress=True,
            progress_label=f"cem-g{generation}",
            progress_mode="line",
            archive_cache_dir=str(generation_dir / "cache"),
            max_in_flight=args.workers * 2,
            common_random_seeds=True,
        )
        games = args.games[min(generation, len(args.games) - 1)]
        report = run_candidate_pool(candidates, args.opponents, games, config, Path.cwd(), peer_span=0)
        save_report(report, generation_dir)
        ranking = rank_report(report, candidates, args.opponents)
        write_ranking(generation_dir / "ranking.json", ranking)
        scored = []
        for row in ranking:
            score = row.robust_score - latency_penalty(report, row.name, args.latency_threshold_s)
            scored.append({"name": row.name, "score": score, "row": asdict(row), "vector": vector_by_name[row.name]})
        scored.sort(key=lambda item: item["score"], reverse=True)
        elite_count = max(2, math.ceil(args.population * args.elite_fraction))
        audited = []
        for item in scored[: max(elite_count * 3, elite_count)]:
            package = candidate_dir / f"{item['name']}.tar.gz"
            result = audit(package, args.replays, args.target_name, test_episode_ids)
            ability = float(result.get("by_group", {}).get("ability", {}).get("semantic_rate", 0.0))
            passed = result["semantic_rate"] >= args.min_semantic_rate and ability >= args.min_ability_rate
            item["audit"] = {"semantic_rate": result["semantic_rate"], "ability_rate": ability, "passed": passed}
            if passed:
                audited.append(item)
            if len(audited) >= elite_count:
                break
        if len(audited) < 2:
            raise RuntimeError(f"generation {generation} produced fewer than two audit-safe elites")
        elites = audited[:elite_count]
        update_distribution(distribution, [item["vector"] for item in elites], args.smoothing)
        generation_summary = {
            "generation": generation,
            "games_per_pair": games,
            "distribution": distribution,
            "elites": elites,
        }
        (generation_dir / "cem_summary.json").write_text(json.dumps(generation_summary, indent=2), encoding="utf-8")
        history.append(generation_summary)
        global_best.extend(
            {**item, "generation": generation, "package": str(candidate_dir / f"{item['name']}.tar.gz")}
            for item in elites
        )
    global_best.sort(key=lambda item: item["score"], reverse=True)
    final_dir = args.out / "finalists"
    final_dir.mkdir(exist_ok=True)
    finalists = []
    seen_vectors = set()
    for item in global_best:
        signature = json.dumps(item["vector"], sort_keys=True)
        if signature in seen_vectors:
            continue
        seen_vectors.add(signature)
        source = Path(item["package"])
        target = final_dir / source.name
        shutil.copy2(source, target)
        finalists.append({**item, "package": str(target)})
        if len(finalists) >= 8:
            break
    result = {"history": history, "final_distribution": distribution, "finalists": finalists}
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"finalists": finalists, "final_distribution": distribution}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
