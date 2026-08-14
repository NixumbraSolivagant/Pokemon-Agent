from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

from local_eval.archive import sha256_file
from local_eval.evaluator import run_candidate_pool, save_report, submission_name
from local_eval.models import EvalConfig
from tools.behavior_clone_audit import audit
from tools.build_meta_submission import build
from tools.robust_gold_search import rank_report, write_ranking


def run_module(module: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], check=True)


def package_deck(path: Path) -> list[int]:
    with tarfile.open(path, "r:gz") as archive:
        payload = archive.extractfile("deck.csv")
        if payload is None:
            return []
        return [int(line) for line in payload.read().decode("utf-8").splitlines() if line.strip()]


def deck_sha256(deck: list[int]) -> str:
    return hashlib.sha256("\n".join(map(str, deck)).encode("utf-8")).hexdigest()


def load_opponent_manifest(path: Path, expected_split: str) -> list[Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest_split = str(data.get("split", expected_split))
    if manifest_split != expected_split:
        raise ValueError(f"{path} has split={manifest_split}, expected {expected_split}")
    opponents = [Path(row["path"]) for row in data.get("opponents", [])]
    if not opponents:
        raise ValueError(f"{path} contains no opponents")
    missing = [str(opponent) for opponent in opponents if not opponent.exists()]
    if missing:
        raise FileNotFoundError(f"{path} references missing opponents: {missing[:3]}")
    if len({str(opponent.resolve()) for opponent in opponents}) != len(opponents):
        raise ValueError(f"{path} contains duplicate opponent packages")
    return opponents


def resolve_opponents(direct: list[Path], manifest: Path | None, split: str) -> list[Path]:
    if manifest is not None and direct:
        raise ValueError(f"use either --{split}-opponents or --{split}-manifest, not both")
    if manifest is not None:
        return load_opponent_manifest(manifest, split)
    if not direct:
        raise ValueError(f"no {split} opponents configured")
    missing = [str(opponent) for opponent in direct if not opponent.exists()]
    if missing:
        raise FileNotFoundError(f"missing {split} opponents: {missing[:3]}")
    return direct


def validate_pool_splits(summary_path: Path, splits: dict[str, list[Path]], minimum_size: int) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = int(summary.get("selected", 0))
    if selected < minimum_size:
        raise RuntimeError(f"opponent pool has {selected} entries; minimum is {minimum_size}")
    expected_counts = summary.get("splits", {})
    seen: dict[str, str] = {}
    for split, opponents in splits.items():
        if split in expected_counts and int(expected_counts[split]) != len(opponents):
            raise RuntimeError(
                f"{split} manifest has {len(opponents)} opponents; summary declares {expected_counts[split]}"
            )
        for opponent in opponents:
            key = str(opponent.resolve())
            if key in seen:
                raise RuntimeError(f"opponent leakage between {seen[key]} and {split}: {opponent}")
            seen[key] = split
    if len(seen) != selected:
        raise RuntimeError(f"pool manifests contain {len(seen)} unique opponents; summary declares {selected}")
    return summary


def training_games(candidate: Path, opponents: list[Path], games: int, workers: int, seed: int, out: Path) -> None:
    config = EvalConfig(
        profile="legacy",
        workers=workers,
        seed=seed,
        record_mode="training",
        record_focus=submission_name(candidate),
        record_gzip=True,
        progress=True,
        progress_label=out.name,
        progress_mode="line",
        archive_cache_dir=str(out / "cache"),
        max_in_flight=workers * 2,
        common_random_seeds=True,
    )
    report = run_candidate_pool([candidate], opponents, games, config, Path.cwd(), peer_span=0)
    save_report(report, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full Ogerpon teacher-initialized residual optimization loop.")
    parser.add_argument("--v2-out", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--baseline-artifact", type=Path)
    parser.add_argument("--train-opponents", nargs="*", type=Path, default=[])
    parser.add_argument("--dev-opponents", nargs="*", type=Path, default=[])
    parser.add_argument("--holdout-opponents", nargs="*", type=Path, default=[])
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--pool-summary", type=Path)
    parser.add_argument("--minimum-pool-size", type=int, default=80)
    parser.add_argument("--expected-deck-sha256")
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--target-name", default="keidroid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--training-games", type=int, default=24)
    parser.add_argument("--holdout-games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--model-jobs", type=int, default=12)
    parser.add_argument("--gpu-device", type=int, default=-1)
    parser.add_argument("--max-states", type=int, default=20_000)
    parser.add_argument("--max-states-per-episode", type=int, default=250)
    parser.add_argument("--counterfactual-workers", type=int, default=1)
    parser.add_argument("--belief-worlds", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--state-budget-s", type=float, default=0.50)
    parser.add_argument("--cem-population", type=int, default=48)
    parser.add_argument("--cem-generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(argv)

    args.train_opponents = resolve_opponents(args.train_opponents, args.train_manifest, "train")
    args.dev_opponents = resolve_opponents(args.dev_opponents, args.dev_manifest, "dev")
    args.holdout_opponents = resolve_opponents(args.holdout_opponents, args.holdout_manifest, "holdout")
    pool_summary = None
    if args.pool_summary is not None:
        pool_summary = validate_pool_splits(
            args.pool_summary,
            {
                "train": args.train_opponents,
                "dev": args.dev_opponents,
                "holdout": args.holdout_opponents,
            },
            args.minimum_pool_size,
        )

    if bool(args.baseline_package) != bool(args.baseline_artifact):
        raise ValueError("--baseline-package and --baseline-artifact must be provided together")
    if args.baseline_package is not None:
        baseline_package = args.baseline_package
        baseline_artifact = args.baseline_artifact
    else:
        v2_result_path = args.v2_out / "result.json"
        if not v2_result_path.exists():
            raise FileNotFoundError(f"V2 result is not ready: {v2_result_path}")
        v2_result = json.loads(v2_result_path.read_text(encoding="utf-8"))
        if not v2_result.get("submission_queue"):
            raise RuntimeError("V2 produced no submission candidates")
        baseline_package = Path(v2_result["submission_queue"][0]["package"])
        baseline_name = baseline_package.name.removeprefix("ogerpon_").split("_clone", 1)[0]
        baseline_artifact = args.v2_out / "artifacts" / f"{baseline_name}.json"
    if not baseline_package.exists() or not baseline_artifact.exists():
        raise FileNotFoundError("V2 baseline package or artifact is missing")

    args.out.mkdir(parents=True, exist_ok=True)
    augmented_artifact = args.out / "baseline_artifact.json"
    artifact_data = json.loads(baseline_artifact.read_text(encoding="utf-8"))
    baseline_deck_hash = deck_sha256(package_deck(baseline_package))
    artifact_deck_hash = str(artifact_data.get("deck_sha256", ""))
    if artifact_deck_hash and artifact_deck_hash != baseline_deck_hash:
        raise RuntimeError("baseline artifact and package deck hashes differ")
    if args.expected_deck_sha256 and baseline_deck_hash != args.expected_deck_sha256:
        raise RuntimeError(
            f"baseline deck hash {baseline_deck_hash} does not match expected {args.expected_deck_sha256}"
        )
    belief_decks = artifact_data.get("opponent_decks", [])
    for opponent in [*args.train_opponents, *args.dev_opponents, *args.holdout_opponents]:
        deck = package_deck(opponent)
        if len(deck) == 60 and deck not in belief_decks:
            belief_decks.append(deck)
    artifact_data["opponent_decks"] = belief_decks[:24]
    artifact_data["version"] = max(3, int(artifact_data.get("version", 1)))
    augmented_artifact.write_text(json.dumps(artifact_data, separators=(",", ":")), encoding="utf-8")
    current_package = baseline_package
    current_artifact = augmented_artifact
    data_paths: list[Path] = []
    iterations = []
    for iteration in range(args.iterations):
        iteration_dir = args.out / f"iteration_{iteration:02d}"
        games_dir = iteration_dir / "training_games"
        if (games_dir / "report.json").exists() and (games_dir / "game_records").is_dir():
            print(f"[resume] reusing training games from {games_dir}", flush=True)
        else:
            training_games(
                current_package,
                args.train_opponents,
                args.training_games,
                args.workers,
                args.seed + iteration * 100_000,
                games_dir,
            )
        counterfactual_path = iteration_dir / "counterfactual.jsonl"
        if counterfactual_path.exists() and counterfactual_path.with_suffix(".jsonl.summary.json").exists():
            print(f"[resume] reusing counterfactual data from {counterfactual_path}", flush=True)
        else:
            run_module(
                "tools.ogerpon_counterfactual_data",
                [
                    "--artifact", str(current_artifact),
                    "--records", str(games_dir / "game_records"),
                    "--focus-name", submission_name(current_package),
                    "--out", str(counterfactual_path),
                    "--variant", "clone_residual_search" if iteration else "clone_combat_search",
                    "--max-states", str(args.max_states),
                    "--max-states-per-episode", str(args.max_states_per_episode),
                    "--belief-worlds", str(args.belief_worlds),
                    "--rollout-steps", str(args.rollout_steps),
                    "--state-budget-s", str(args.state_budget_s),
                    "--workers", str(args.counterfactual_workers),
                ],
            )
        data_paths.append(counterfactual_path)
        next_artifact = iteration_dir / "residual_artifact.json"
        if next_artifact.exists():
            print(f"[resume] reusing residual artifact from {next_artifact}", flush=True)
        else:
            run_module(
                "tools.ogerpon_residual_train",
                [
                    "--base-artifact", str(current_artifact),
                    "--data", *map(str, data_paths),
                    "--out", str(next_artifact),
                    "--n-jobs", str(args.model_jobs),
                    "--gpu-device", str(args.gpu_device),
                    "--seed", str(args.seed + iteration),
                ],
            )
        test_episode_ids = {
            str(value).split(":")[-1]
            for value in json.loads(next_artifact.read_text(encoding="utf-8")).get("split_stats", {}).get("test_episodes", [])
        }
        calibrated_weight = 0.0
        next_package = iteration_dir / "ogerpon_residual_search_w0.tar.gz"
        for weight in (90000.0, 60000.0, 30000.0, 15000.0, 0.0):
            candidate = iteration_dir / f"ogerpon_residual_search_w{int(weight)}.tar.gz"
            build(
                next_artifact,
                candidate,
                Path("agents/meta_runtime.py"),
                Path("cg"),
                "clone_residual_search",
                linux_only=True,
                config_overrides={"residual_weight": weight},
            )
            result = audit(candidate, args.replays, args.target_name, test_episode_ids)
            ability = float(result.get("by_group", {}).get("ability", {}).get("semantic_rate", 0.0))
            if result.get("semantic_rate", 0.0) >= 0.80 and ability >= 0.85:
                calibrated_weight = weight
                next_package = candidate
                break
        iterations.append(
            {
                "iteration": iteration,
                "training_games": str(games_dir),
                "counterfactual": str(counterfactual_path),
                "artifact": str(next_artifact),
                "package": str(next_package),
                "calibrated_residual_weight": calibrated_weight,
            }
        )
        current_artifact = next_artifact
        current_package = next_package

    cem_dir = args.out / "cem"
    run_module(
        "tools.ogerpon_cem_optimize",
        [
            "--artifact", str(current_artifact),
            "--opponents", *map(str, args.dev_opponents),
            "--replays", str(args.replays),
            "--target-name", args.target_name,
            "--out", str(cem_dir),
            "--variant", "clone_residual_search",
            "--population", str(args.cem_population),
            "--generations", str(args.cem_generations),
            "--workers", str(args.workers),
            "--seed", str(args.seed + 900_000),
        ],
    )
    cem_result = json.loads((cem_dir / "result.json").read_text(encoding="utf-8"))
    finalists = [Path(row["package"]) for row in cem_result.get("finalists", [])]
    if not finalists:
        raise RuntimeError("CEM produced no finalists")

    holdout_dir = args.out / "holdout"
    holdout_config = EvalConfig(
        profile="kaggle",
        workers=args.workers,
        seed=args.seed + 1_900_000,
        record_mode="none",
        progress=True,
        progress_label="residual-holdout",
        progress_mode="line",
        archive_cache_dir=str(holdout_dir / "cache"),
        max_in_flight=args.workers * 2,
    )
    holdout_report = run_candidate_pool(finalists, args.holdout_opponents, args.holdout_games, holdout_config, Path.cwd(), peer_span=0)
    save_report(holdout_report, holdout_dir)
    holdout_ranking = rank_report(holdout_report, finalists, args.holdout_opponents)
    write_ranking(holdout_dir / "ranking.json", holdout_ranking)
    submission_queue = [
        {
            "priority": index + 1,
            "name": row.name,
            "package": row.tarball,
            "package_sha256": sha256_file(row.tarball),
            "robust_score": row.robust_score,
            "requires_explicit_kaggle_execute": True,
        }
        for index, row in enumerate(holdout_ranking[:5])
    ]
    final_dir = args.out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    best_submission = final_dir / "best_submission.tar.gz"
    shutil.copy2(Path(submission_queue[0]["package"]), best_submission)
    best_sha256 = sha256_file(best_submission)
    result = {
        "baseline": {"artifact": str(baseline_artifact), "package": str(baseline_package)},
        "iterations": iterations,
        "cem_result": str(cem_dir / "result.json"),
        "holdout": [asdict(row) for row in holdout_ranking],
        "submission_queue": submission_queue,
        "best_submission": {"package": str(best_submission), "package_sha256": best_sha256},
        "online_submission_performed": False,
        "deck_is_fixed": True,
        "deck_sha256": baseline_deck_hash,
        "pool_summary": pool_summary,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "submission_queue.json").write_text(json.dumps(submission_queue, indent=2), encoding="utf-8")
    print(json.dumps({"submission_queue": submission_queue}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
