from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tools.replay_imitation import dump_catboost_model


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[dict[str, Any]], validation_fraction: float, test_fraction: float):
    groups = sorted(
        {str(row.get("group_id") or row["episode_id"]) for row in rows},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(groups) < 3:
        raise ValueError("advantage training requires at least three isolated groups")
    test_count = max(1, round(len(groups) * test_fraction))
    validation_count = max(1, round(len(groups) * validation_fraction))
    if test_count + validation_count >= len(groups):
        test_count = validation_count = 1
    test = set(groups[-test_count:])
    validation = set(groups[-test_count - validation_count:-test_count])

    def group(row: dict[str, Any]) -> str:
        return str(row.get("group_id") or row["episode_id"])

    return (
        [row for row in rows if group(row) not in validation | test],
        [row for row in rows if group(row) in validation],
        [row for row in rows if group(row) in test],
    )


def examples(rows: list[dict[str, Any]]):
    features = []
    targets = []
    weights = []
    for row in rows:
        anchor_index = int(row["anchor_index"])
        for action in row["actions"]:
            if int(action["index"]) == anchor_index:
                continue
            advantage_features = action.get("advantage_features")
            if not advantage_features:
                continue
            features.append(advantage_features)
            targets.append(float(action["advantage_return"]))
            weights.append(max(0.1, len(action.get("samples", [])) / (1.0 + float(action.get("std_return", 0.0)))))
    return features, targets, weights


def row_predictions(models, row: dict[str, Any], uncertainty_weight: float):
    predictions = []
    anchor_index = int(row["anchor_index"])
    for action in row["actions"]:
        if int(action["index"]) == anchor_index or not action.get("advantage_features"):
            continue
        values = [float(model.predict([action["advantage_features"]])[0]) for model in models]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        predictions.append((mean - uncertainty_weight * math.sqrt(variance), float(action["advantage_return"])))
    return predictions


def calibrate(
    models,
    rows: list[dict[str, Any]],
    uncertainty_weight: float,
    min_coverage: float,
    max_coverage: float,
    min_positive_rate: float,
):
    states = []
    for row in rows:
        predictions = row_predictions(models, row, uncertainty_weight)
        if predictions:
            states.append(max(predictions, key=lambda item: item[0]))
    if not states:
        raise ValueError("validation rows contain no advantage candidates")
    thresholds = sorted({score for score, _ in states})
    best = None
    for threshold in thresholds:
        selected = [actual for score, actual in states if score > threshold]
        coverage = len(selected) / len(states)
        if not min_coverage <= coverage <= max_coverage or not selected:
            continue
        mean_advantage = sum(selected) / len(selected)
        positive_rate = sum(value > 0 for value in selected) / len(selected)
        if mean_advantage <= 0.0 or positive_rate < min_positive_rate:
            continue
        objective = mean_advantage + 0.1 * positive_rate
        if best is None or objective > best[0]:
            best = (objective, threshold, coverage, mean_advantage, positive_rate, len(selected))
    if best is None:
        threshold = max(score for score, _ in states) + 1e-9
        best = (
            0.0,
            threshold,
            0.0,
            0.0,
            0.0,
            0,
        )
    return {
        "threshold": best[1],
        "coverage": best[2],
        "mean_actual_advantage": best[3],
        "positive_rate": best[4],
        "selected_states": best[5],
        "states": len(states),
    }


def evaluate(models, rows: list[dict[str, Any]], threshold: float, uncertainty_weight: float) -> dict[str, float | int]:
    selected = []
    states = 0
    for row in rows:
        predictions = row_predictions(models, row, uncertainty_weight)
        if not predictions:
            continue
        states += 1
        best_score, actual = max(predictions, key=lambda item: item[0])
        if best_score > threshold:
            selected.append(actual)
    return {
        "states": states,
        "selected_states": len(selected),
        "coverage": len(selected) / max(1, states),
        "mean_actual_advantage": sum(selected) / max(1, len(selected)),
        "positive_rate": sum(value > 0 for value in selected) / max(1, len(selected)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a champion-gated Ogerpon advantage ensemble.")
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor-package-sha256", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026081401)
    parser.add_argument("--iterations", type=int, default=650)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--min-data-in-leaf", type=int, default=24)
    parser.add_argument("--gpu-device", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--uncertainty-weight", type=float, default=1.5)
    parser.add_argument("--min-coverage", type=float, default=0.02)
    parser.add_argument("--max-coverage", type=float, default=0.08)
    parser.add_argument("--min-positive-rate", type=float, default=0.65)
    args = parser.parse_args(argv)

    rows = read_rows(args.data)
    train_rows, validation_rows, test_rows = split_rows(rows, args.validation_fraction, args.test_fraction)
    train_features, train_targets, train_weights = examples(train_rows)
    if not train_features:
        raise ValueError("no advantage examples found")
    from catboost import CatBoostRegressor

    models = []
    for index in range(args.models):
        params: dict[str, Any] = {
            "loss_function": "Huber:delta=1.0",
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "min_data_in_leaf": args.min_data_in_leaf,
            "l2_leaf_reg": 3.0,
            "random_seed": args.seed + index,
            "task_type": "GPU" if args.gpu_device >= 0 else "CPU",
            "verbose": False,
            "allow_writing_files": False,
        }
        if args.gpu_device >= 0:
            params["devices"] = str(args.gpu_device)
        else:
            params["thread_count"] = args.threads
        model = CatBoostRegressor(**params)
        model.fit(train_features, train_targets, sample_weight=train_weights)
        models.append(model)

    calibration = calibrate(
        models,
        validation_rows,
        args.uncertainty_weight,
        args.min_coverage,
        args.max_coverage,
        args.min_positive_rate,
    )
    artifact = json.loads(args.base_artifact.read_text(encoding="utf-8"))
    artifact.update(
        {
            "version": max(4, int(artifact.get("version", 1))),
            "advantage_ensemble": [dump_catboost_model(model) for model in models],
            "advantage_calibration": {
                "anchor_package_sha256": args.anchor_package_sha256,
                "default_threshold": calibration["threshold"],
                "context_thresholds": {},
                "terminal_threshold": calibration["threshold"] * 2.0,
                "uncertainty_weight": args.uncertainty_weight,
                "minimum_models": min(3, args.models),
                "candidate_limit": 4,
                "max_overrides_per_game": 8,
                "allow_terminal": False,
                "validation": calibration,
            },
            "advantage_metrics": {
                "validation": evaluate(models, validation_rows, calibration["threshold"], args.uncertainty_weight),
                "test": evaluate(models, test_rows, calibration["threshold"], args.uncertainty_weight),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
                "data_sha256": hashlib.sha256(
                    "".join(path.read_text(encoding="utf-8") for path in args.data).encode("utf-8")
                ).hexdigest(),
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(artifact["advantage_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
