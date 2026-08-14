from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.replay_imitation import dump_catboost_model


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[dict[str, Any]], validation_fraction: float, test_fraction: float):
    episodes = sorted({str(row["episode_id"]) for row in rows})
    if len(episodes) < 3:
        raise ValueError("counterfactual training requires at least three complete episodes")
    test_count = max(1, round(len(episodes) * test_fraction))
    validation_count = max(1, round(len(episodes) * validation_fraction))
    if test_count + validation_count >= len(episodes):
        test_count = validation_count = 1
    test = set(episodes[-test_count:])
    validation = set(episodes[-test_count - validation_count:-test_count])
    train_rows = [row for row in rows if row["episode_id"] not in validation | test]
    validation_rows = [row for row in rows if row["episode_id"] in validation]
    test_rows = [row for row in rows if row["episode_id"] in test]
    return train_rows, validation_rows, test_rows


def q_examples(rows: list[dict[str, Any]]):
    features = []
    targets = []
    weights = []
    for row in rows:
        for action in row["actions"]:
            features.append(action["features"])
            targets.append(float(action["robust_return"]))
            weights.append(max(0.1, len(action.get("samples", [])) / (1.0 + float(action.get("std_return", 0.0)))))
    return features, targets, weights


def value_examples(rows: list[dict[str, Any]]):
    features = []
    labels = []
    for row in rows:
        actual = row.get("actual_action") or []
        if len(actual) != 1 or int(row.get("game_reward", 0)) == 0:
            continue
        selected = next((action for action in row["actions"] if action["index"] == actual[0]), None)
        if selected is None:
            continue
        features.append(selected["features"])
        labels.append(1 if int(row["game_reward"]) > 0 else 0)
    return features, labels


def q_metrics(model: LGBMRegressor, rows: list[dict[str, Any]]) -> dict[str, float | int]:
    absolute_error = regret = 0.0
    best_correct = comparisons = examples = 0
    for row in rows:
        actions = row["actions"]
        if not actions:
            continue
        predicted = model.predict([action["features"] for action in actions]).tolist()
        actual = [float(action["robust_return"]) for action in actions]
        absolute_error += sum(abs(left - right) for left, right in zip(predicted, actual))
        examples += len(actions)
        predicted_best = max(range(len(actions)), key=lambda index: predicted[index])
        actual_best = max(range(len(actions)), key=lambda index: actual[index])
        best_correct += int(predicted_best == actual_best)
        regret += actual[actual_best] - actual[predicted_best]
        comparisons += 1
    return {
        "states": comparisons,
        "actions": examples,
        "mae": absolute_error / max(1, examples),
        "best_action_accuracy": best_correct / max(1, comparisons),
        "mean_regret": regret / max(1, comparisons),
    }


def value_metrics(model: LGBMClassifier | None, rows: list[dict[str, Any]]) -> dict[str, float | int]:
    features, labels = value_examples(rows)
    if model is None or not features:
        return {"examples": len(features), "brier": 1.0, "log_loss": float("inf"), "accuracy": 0.0}
    probabilities = model.predict_proba(features)[:, 1].tolist()
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)
    log_loss = -sum(
        label * math.log(max(1e-9, probability)) + (1 - label) * math.log(max(1e-9, 1.0 - probability))
        for probability, label in zip(probabilities, labels)
    ) / len(labels)
    accuracy = sum((probability >= 0.5) == bool(label) for probability, label in zip(probabilities, labels)) / len(labels)
    return {"examples": len(labels), "brier": brier, "log_loss": log_loss, "accuracy": accuracy}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Ogerpon residual Q and calibrated win-value models.")
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=360)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--gpu-device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--safety-residual-weight", type=float, default=0.25)
    args = parser.parse_args(argv)

    rows = read_rows(args.data)
    train_rows, validation_rows, test_rows = split_rows(rows, args.validation_fraction, args.test_fraction)
    train_features, train_targets, train_weights = q_examples(train_rows)
    if not train_features:
        raise ValueError("no residual Q examples found")
    from catboost import CatBoostClassifier, CatBoostRegressor

    q_params: dict[str, Any] = {
        "loss_function": "Huber:delta=1.0",
        "iterations": args.n_estimators,
        "learning_rate": args.learning_rate,
        "depth": max(1, (args.num_leaves - 1).bit_length()),
        "min_data_in_leaf": args.min_child_samples,
        "l2_leaf_reg": 1.0,
        "random_seed": args.seed,
        "task_type": "GPU" if args.gpu_device >= 0 else "CPU",
        "verbose": False,
        "allow_writing_files": False,
    }
    if args.gpu_device >= 0:
        q_params["devices"] = str(args.gpu_device)
    else:
        q_params["thread_count"] = args.n_jobs
    q_model = CatBoostRegressor(**q_params)
    q_model.fit(train_features, train_targets, sample_weight=train_weights)

    value_features, value_labels = value_examples(train_rows)
    value_model = None
    if value_features and len(set(value_labels)) == 2:
        value_params: dict[str, Any] = {
            "loss_function": "Logloss",
            "iterations": min(280, args.n_estimators),
            "learning_rate": args.learning_rate,
            "depth": max(1, (min(31, args.num_leaves) - 1).bit_length()),
            "min_data_in_leaf": args.min_child_samples,
            "l2_leaf_reg": 1.0,
            "random_seed": args.seed + 1,
            "task_type": "GPU" if args.gpu_device >= 0 else "CPU",
            "verbose": False,
            "allow_writing_files": False,
        }
        if args.gpu_device >= 0:
            value_params["devices"] = str(args.gpu_device)
        else:
            value_params["thread_count"] = args.n_jobs
        value_model = CatBoostClassifier(**value_params)
        value_model.fit(value_features, value_labels)

    artifact = json.loads(args.base_artifact.read_text(encoding="utf-8"))
    artifact.update(
        {
            "version": max(3, int(artifact.get("version", 1))),
            "residual_q_model": dump_catboost_model(q_model),
            "win_value_model": dump_catboost_model(value_model) if value_model is not None else {},
            "residual_metrics": {
                "train": q_metrics(q_model, train_rows),
                "validation": q_metrics(q_model, validation_rows),
                "test": q_metrics(q_model, test_rows),
                "value_train": value_metrics(value_model, train_rows),
                "value_validation": value_metrics(value_model, validation_rows),
                "value_test": value_metrics(value_model, test_rows),
            },
            "counterfactual_stats": {
                "rows": len(rows),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
                "train_episodes": sorted({row["episode_id"] for row in train_rows}),
                "validation_episodes": sorted({row["episode_id"] for row in validation_rows}),
                "test_episodes": sorted({row["episode_id"] for row in test_rows}),
                "data_sha256": hashlib.sha256(
                    "".join(path.read_text(encoding="utf-8") for path in args.data).encode("utf-8")
                ).hexdigest(),
            },
        }
    )
    router_calibration = dict(artifact.get("router_calibration") or {})
    router_calibration["safety_residual_weight"] = min(1.0, max(0.0, args.safety_residual_weight))
    artifact["router_calibration"] = router_calibration
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(artifact["residual_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
