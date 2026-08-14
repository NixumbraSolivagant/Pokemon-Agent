from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from agents.meta_runtime import ROUTER_FEATURE_NAMES, TYPE_FEATURE_NAMES, aggregate_option_type_features
from cg.api import OptionType
from tools import replay_imitation as imitation


def _actual_type(decision: imitation.Decision) -> int:
    return next(
        (option_type for option_type, label in zip(decision.option_types, decision.labels) if label),
        -1,
    )


def _threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    confusion = Counter()
    for row in rows:
        confusion[(row["actual_terminal"], row["margin"] >= threshold)] += 1
    continue_recall = confusion[(False, False)] / max(1, confusion[(False, False)] + confusion[(False, True)])
    stop_recall = confusion[(True, True)] / max(1, confusion[(True, True)] + confusion[(True, False)])
    return {
        "threshold": threshold,
        "continue_recall": continue_recall,
        "stop_recall": stop_recall,
        "balanced_accuracy": 0.5 * (continue_recall + stop_recall),
        "confusion": {
            "continue_correct": confusion[(False, False)],
            "false_stop": confusion[(False, True)],
            "terminal_correct": confusion[(True, True)],
            "false_continue": confusion[(True, False)],
        },
    }


def _best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted({row["margin"] for row in rows})
    thresholds = [values[0] - 1e-9, *((left + right) / 2 for left, right in zip(values, values[1:])), values[-1] + 1e-9]
    metrics = [_threshold_metrics(rows, threshold) for threshold in thresholds]
    return max(metrics, key=lambda row: (row["balanced_accuracy"], row["continue_recall"], row["stop_recall"]))


def _best_with_minimum(rows: list[dict[str, Any]], key: str, minimum: float) -> dict[str, Any]:
    values = sorted({row["margin"] for row in rows})
    thresholds = [values[0] - 1e-9, *((left + right) / 2 for left, right in zip(values, values[1:])), values[-1] + 1e-9]
    feasible = [metrics for threshold in thresholds if (metrics := _threshold_metrics(rows, threshold))[key] >= minimum]
    return max(feasible, key=lambda row: (row["balanced_accuracy"], row["continue_recall"], row["stop_recall"]), default={})


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        str(quantile): float(np.quantile(values, quantile))
        for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def _group_metrics(rows: list[dict[str, Any]], threshold: float, key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: {"rows": len(group), **_threshold_metrics(group, threshold)}
        for name, group in sorted(grouped.items())
    }


def _turn_bin(turn: int) -> str:
    if turn <= 2:
        return "1-2"
    if turn <= 4:
        return "3-4"
    if turn <= 7:
        return "5-7"
    return "8+"


def _action_bin(count: int) -> str:
    if count <= 1:
        return "0-1"
    if count <= 3:
        return "2-3"
    if count <= 6:
        return "4-6"
    return "7+"


def _split_usage(model: dict[str, Any]) -> list[dict[str, Any]]:
    feature_names = [
        *(f"continue::{name}" for name in TYPE_FEATURE_NAMES),
        *(f"terminal::{name}" for name in TYPE_FEATURE_NAMES),
        *(f"delta::{name}" for name in TYPE_FEATURE_NAMES),
        *imitation.STOP_EXTRA_FEATURE_NAMES,
    ]
    counts = Counter()
    for tree in model.get("oblivious_trees", []):
        for split in tree.get("splits", []):
            counts[int(split.get("float_feature_index", -1))] += 1
    return [
        {"feature": feature_names[index] if 0 <= index < len(feature_names) else str(index), "splits": count}
        for index, count in counts.most_common(40)
    ]


def analyze(manifest_path: Path, artifact_path: Path, output_path: Path) -> dict[str, Any]:
    _, sources, config = imitation.load_manifest(manifest_path)
    decisions = []
    source_stats = {}
    cache_dir = Path(config["cache_dir"]) if config.get("cache_dir") else None
    for source in sources:
        rows, _, stats = imitation.load_source(
            source,
            parse_workers=int(config.get("parse_workers", 1)),
            cache_dir=cache_dir,
        )
        decisions.extend(rows)
        source_stats[source.name] = stats
    train, validation, test = imitation.split_by_episode(
        decisions,
        float(config.get("validation_fraction", 0.15)),
        float(config.get("test_fraction", 0.15)),
        int(config.get("split_seed", config.get("seed", 0))),
    )
    parts_dir = artifact_path.parent / f"{artifact_path.stem}_parts"
    base_model = imitation._load_model_part(parts_dir, "head_main") or imitation._load_model_part(parts_dir, "exported")
    base_seed = int(config.get("seed", 0))
    stop_models = {}
    stop_members = {}
    for regime in imitation.STOP_REGIMES:
        members = []
        for member in range(int(config.get("stop_ensemble_size", 3))):
            seed = base_seed + 1009 * member
            model = imitation._load_model_part(parts_dir, f"head_main_stop_{regime}_seed{seed}")
            if model:
                members.append(model)
        if members:
            stop_members[regime] = members
            stop_models[regime] = imitation.average_catboost_models(members)
    if not stop_models:
        ensemble = []
        continue_weight = float(config.get("stop_continue_weight", 1.0))
        for member in range(int(config.get("stop_ensemble_size", 1))):
            seed = base_seed + 1009 * member
            model = imitation._load_model_part(parts_dir, f"head_main_stop_full_v5_w{continue_weight:g}_seed{seed}")
            if model:
                ensemble.append(model)
        if not ensemble:
            raise FileNotFoundError(f"no stop models found in {parts_dir}")
        stop_models["global"] = imitation.average_catboost_models(ensemble)
        stop_members["global"] = ensemble
    feature_index = {name: index for index, name in enumerate(imitation.FEATURE_NAMES)}

    def rows_for(split_rows: list[imitation.Decision]) -> list[dict[str, Any]]:
        output = []
        for decision in split_rows:
            if decision.head != "main":
                continue
            base_scores = imitation.predict_batch(base_model, decision.features)
            type_rows, present_types = aggregate_option_type_features(
                decision.features, decision.option_types, base_scores, feature_mode="full"
            )
            aggregated = imitation._aggregate_layout_decision(decision, type_rows, present_types)
            stop_features = imitation._stop_decision_features(aggregated)
            if stop_features is None:
                continue
            regime = imitation.stop_regime(aggregated)
            model_key = regime if regime in stop_models else "global"
            member_scores = [imitation.predict_model(model, stop_features) for model in stop_members[model_key]]
            actual_type = _actual_type(decision)
            first = decision.features[0]
            option_types = set(decision.option_types)
            output.append({
                "margin": sum(member_scores) / len(member_scores),
                "margin_std": float(np.std(member_scores)),
                "regime": regime,
                "actual_terminal": actual_type in imitation.TERMINAL_MAIN_TYPES,
                "actual_type": OptionType(actual_type).name.lower() if actual_type >= 0 else "unknown",
                "won": decision.won,
                "seat": decision.seat,
                "turn": int(first[feature_index["turn"]]),
                "turn_bin": _turn_bin(int(first[feature_index["turn"]])),
                "turn_action_count": int(first[feature_index["turn_action_count"]]),
                "action_bin": _action_bin(int(first[feature_index["turn_action_count"]])),
                "attack_available": int(OptionType.ATTACK) in option_types,
                "end_available": int(OptionType.END) in option_types,
                "attack_ko_available": any(
                    option_type == int(OptionType.ATTACK) and features[feature_index["attack_would_ko"]] > 0
                    for features, option_type in zip(decision.features, decision.option_types)
                ),
                "max_attack_damage": max(
                    (features[feature_index["effective_attack_damage"]] for features, option_type in zip(decision.features, decision.option_types)
                     if option_type == int(OptionType.ATTACK)),
                    default=0.0,
                ),
            })
        return output

    validation_rows = rows_for(validation)
    test_rows = rows_for(test)
    regime_thresholds = {
        regime: _best_threshold([row for row in validation_rows if row["regime"] == regime])
        for regime in imitation.STOP_REGIMES
    }
    for row in validation_rows:
        row["predicted_terminal"] = row["margin"] >= float(regime_thresholds[row["regime"]]["threshold"])
    false_stops = [row for row in validation_rows if not row["actual_terminal"] and row["predicted_terminal"]]
    false_continues = [row for row in validation_rows if row["actual_terminal"] and not row["predicted_terminal"]]
    report = {
        "manifest": str(manifest_path),
        "artifact": str(artifact_path),
        "config": {
            "seed": config.get("seed"),
            "win_weight": config.get("win_weight"),
            "loss_weight": config.get("loss_weight"),
            "stop_no_attack_continue_weight": config.get("stop_no_attack_continue_weight", 1.0),
            "stop_attack_ready_continue_weight": config.get("stop_attack_ready_continue_weight", 1.5),
            "stop_ko_ready_continue_weight": config.get("stop_ko_ready_continue_weight", 2.0),
            "stop_weight_cap": config.get("stop_weight_cap", 2.70),
            "stop_ensemble_size": config.get("stop_ensemble_size"),
            "router_feature_mode": config.get("router_feature_mode"),
        },
        "source_stats": source_stats,
        "split": {
            "train_episodes": len({row.episode_id for row in train}),
            "validation_episodes": len({row.episode_id for row in validation}),
            "test_episodes": len({row.episode_id for row in test}),
            "train_decisions": len(train),
            "validation_decisions": len(validation),
            "test_decisions": len(test),
        },
        "validation": {
            "rows": len(validation_rows),
            "class_counts": dict(Counter("terminal" if row["actual_terminal"] else "continue" for row in validation_rows)),
            "regime_thresholds": regime_thresholds,
            "margin_quantiles": {
                "continue": _quantiles([row["margin"] for row in validation_rows if not row["actual_terminal"]]),
                "terminal": _quantiles([row["margin"] for row in validation_rows if row["actual_terminal"]]),
            },
            "by_regime": {
                regime: _threshold_metrics(
                    [row for row in validation_rows if row["regime"] == regime],
                    float(metrics["threshold"]),
                )
                for regime, metrics in regime_thresholds.items()
            },
            "false_stop": {
                "rows": len(false_stops),
                "actual_types": dict(Counter(row["actual_type"] for row in false_stops)),
                "turn_bins": dict(Counter(row["turn_bin"] for row in false_stops)),
                "action_bins": dict(Counter(row["action_bin"] for row in false_stops)),
                "attack_available": sum(row["attack_available"] for row in false_stops),
                "attack_ko_available": sum(row["attack_ko_available"] for row in false_stops),
                "won": sum(row["won"] for row in false_stops),
            },
            "false_continue": {
                "rows": len(false_continues),
                "actual_types": dict(Counter(row["actual_type"] for row in false_continues)),
                "turn_bins": dict(Counter(row["turn_bin"] for row in false_continues)),
                "action_bins": dict(Counter(row["action_bin"] for row in false_continues)),
            },
        },
        "test_at_validation_thresholds": {
            regime: _threshold_metrics(
                [row for row in test_rows if row["regime"] == regime],
                float(metrics["threshold"]),
            )
            for regime, metrics in regime_thresholds.items()
        },
        "model_split_usage": {
            regime: _split_usage(model)
            for regime, model in stop_models.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.artifact, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
