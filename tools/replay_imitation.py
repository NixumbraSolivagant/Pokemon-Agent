from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.meta_runtime import (
    FEATURE_NAMES,
    ROUTER_FEATURE_NAMES,
    STOP_EXTRA_FEATURE_NAMES,
    TYPE_FEATURE_NAMES,
    _LOPUNNY_MEMORY,
    _OGERPON_MEMORY,
    _record_policy_action,
    _reset_policy_memory,
    _sync_policy_memory,
    _update_lopunny_memory,
    _update_ogerpon_memory,
    aggregate_option_type_features,
    feature_vector,
    score_main_options_v7,
    stop_decision_features,
)
from cg.api import OptionType, SelectContext, to_observation_class
from tools.kaggle_replay_miner import full_deck


@dataclass(slots=True)
class SourceSpec:
    name: str
    path: Path
    target_name: str
    split: str = "train"
    weight: float = 1.0


@dataclass(slots=True)
class Decision:
    features: list[list[float]]
    labels: list[int]
    source: str
    won: bool
    episode_id: str = ""
    head: str = "global"
    seat: int = 0
    reward: int = 0
    option_types: list[int] = field(default_factory=list)
    min_count: int = 1
    max_count: int = 1
    weight: float = 1.0


def deck_signature(deck: list[int]) -> tuple[int, ...]:
    return tuple(sorted(deck))


def _option_type(option: Any) -> Any:
    if isinstance(option, dict):
        return option.get("type")
    return getattr(option, "type", None)


_load_source_cache: dict[tuple[str, str], tuple[list[Decision], tuple[int, ...], dict[str, Any]]] = {}
_load_source_cache_lock = threading.Lock()
_replay_content_cache: dict[str, dict[str, Any]] = {}
_replay_content_cache_lock = threading.Lock()
_catboost_gpu_slots = threading.Semaphore(max(1, int(os.environ.get("CATBOOST_GPU_CONCURRENCY", "1"))))

PARSE_CACHE_VERSION = 9
STOP_ROUTER_SCHEMA_VERSION = 2
STOP_REGIMES = ("no_attack", "attack_ready", "ko_ready")
STOP_ABLATIONS = ("full", "no_base", "delta_only", "damped_base")


def replay_fingerprint(path: Path) -> str:
    paths = sorted({*path.glob("*replay.json"), *path.glob("*replay.json.gz")})
    digest = hashlib.sha256()
    for replay_path in paths:
        stat = replay_path.stat()
        digest.update(
            f"{replay_path.name}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_ino}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _replay_content_entry(replay_path: Path) -> dict[str, str]:
    opener = gzip.open if replay_path.suffix == ".gz" else open
    with opener(replay_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"file": replay_path.name, "sha256": hashlib.sha256(normalized).hexdigest()}


def replay_content_fingerprint(path: Path, workers: int = 1) -> dict[str, Any]:
    replay_paths = sorted({*path.glob("*replay.json"), *path.glob("*replay.json.gz")})
    if workers > 1 and len(replay_paths) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            entries = list(executor.map(_replay_content_entry, replay_paths))
    else:
        entries = [_replay_content_entry(replay_path) for replay_path in replay_paths]
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['file']}:{entry['sha256']}\n".encode("utf-8"))
    return {"sha256": digest.hexdigest(), "replays": len(entries), "entries": entries}


def split_fingerprint(train: list[Decision], validation: list[Decision], test: list[Decision]) -> dict[str, Any]:
    result = {}
    for name, decisions in (("train", train), ("validation", validation), ("test", test)):
        episodes = sorted({decision.episode_id for decision in decisions})
        result[name] = {
            "episodes": len(episodes),
            "sha256": hashlib.sha256("\n".join(episodes).encode("utf-8")).hexdigest(),
        }
    return result


def verify_frozen_dataset(
    frozen_path: Path,
    source_fingerprints: dict[str, dict[str, Any]],
    train: list[Decision],
    validation: list[Decision],
    test: list[Decision],
) -> None:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if len(source_fingerprints) != 1:
        raise ValueError("frozen dataset verification requires exactly one source")
    actual_dataset = next(iter(source_fingerprints.values()))
    checks = {
        "dataset_sha256": actual_dataset["sha256"] == frozen.get("dataset_sha256"),
        "replay_count": actual_dataset["replays"] == frozen.get("replay_count"),
        "splits": split_fingerprint(train, validation, test) == frozen.get("splits"),
        "episodes": {
            "train": sorted({row.episode_id for row in train}),
            "validation": sorted({row.episode_id for row in validation}),
            "test": sorted({row.episode_id for row in test}),
        } == frozen.get("episodes"),
    }
    if not all(checks.values()):
        raise ValueError(f"frozen dataset verification failed: {checks}")


def _feature_schema_sha256() -> str:
    payload = "\n".join((*FEATURE_NAMES, "--TYPE--", *TYPE_FEATURE_NAMES, "--ROUTER--", *ROUTER_FEATURE_NAMES, "--STOP--", *STOP_EXTRA_FEATURE_NAMES))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_cache_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _load_cache_file(path: Path) -> Any | None:
    try:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None


def _load_model_part(parts_dir: Path, name: str) -> dict[str, Any] | None:
    part_path = parts_dir / f"{name}.json"
    if not part_path.exists():
        return None
    try:
        return json.loads(part_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_model_part(parts_dir: Path, name: str, value: dict[str, Any]) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = parts_dir / f"{name}.json"
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp_path, part_path)


MAIN_TYPE_OPTIONS = {
    "main_play": int(OptionType.PLAY),
    "main_attach": int(OptionType.ATTACH),
    "main_evolve": int(OptionType.EVOLVE),
    "main_ability": int(OptionType.ABILITY),
    "main_discard": int(OptionType.DISCARD),
    "main_retreat": int(OptionType.RETREAT),
    "main_attack": int(OptionType.ATTACK),
    "main_end": int(OptionType.END),
}
MAIN_TYPE_HEADS = {option_type: head for head, option_type in MAIN_TYPE_OPTIONS.items()}
TERMINAL_MAIN_TYPES = {int(OptionType.ATTACK), int(OptionType.END)}
CONTINUE_MAIN_TYPES = set(MAIN_TYPE_HEADS) - TERMINAL_MAIN_TYPES


def stop_regime(decision: Decision) -> str:
    if int(OptionType.ATTACK) not in decision.option_types:
        return "no_attack"
    feature_names = (
        ROUTER_FEATURE_NAMES
        if decision.features and len(decision.features[0]) == len(ROUTER_FEATURE_NAMES)
        else TYPE_FEATURE_NAMES
        if decision.features and len(decision.features[0]) == len(TYPE_FEATURE_NAMES)
        else FEATURE_NAMES
    )
    ko_name = "attack_ko_any" if "attack_ko_any" in feature_names else "attack_would_ko"
    ko_index = feature_names.index(ko_name)
    return "ko_ready" if any(
        option_type == int(OptionType.ATTACK) and features[ko_index] > 0
        for features, option_type in zip(decision.features, decision.option_types)
    ) else "attack_ready"


def stop_sample_weight(label: int, decision: Decision, manifest_data: dict[str, Any]) -> float:
    regime = stop_regime(decision)
    continue_multiplier = {
        "no_attack": float(manifest_data.get("stop_no_attack_continue_weight", 1.0)),
        "attack_ready": float(manifest_data.get("stop_attack_ready_continue_weight", 1.5)),
        "ko_ready": float(manifest_data.get("stop_ko_ready_continue_weight", 2.0)),
    }[regime] if label == 0 else 1.0
    outcome_multiplier = float(
        manifest_data.get("win_weight", 1.0) if decision.won else manifest_data.get("loss_weight", 0.35)
    )
    cap = float(manifest_data.get("stop_weight_cap", 2.70))
    return min(cap, decision.weight * continue_multiplier * outcome_multiplier)


def stop_weight_audit(
    rows: list[tuple[list[float], int, Decision]],
    manifest_data: dict[str, Any],
) -> dict[str, Any]:
    audit: dict[str, Any] = {regime: {"continue": [], "terminal": []} for regime in STOP_REGIMES}
    for _, label, decision in rows:
        audit[stop_regime(decision)]["terminal" if label else "continue"].append(
            stop_sample_weight(label, decision, manifest_data)
        )
    result: dict[str, Any] = {}
    for regime, classes in audit.items():
        result[regime] = {}
        for label, weights in classes.items():
            result[regime][label] = {
                "count": len(weights),
                "mass": sum(weights),
                "mean_weight": sum(weights) / max(1, len(weights)),
                "max_weight": max(weights, default=0.0),
            }
    result["weight_cap"] = float(manifest_data.get("stop_weight_cap", 2.70))
    return result


def stop_feature_names() -> tuple[str, ...]:
    return (
        *(f"continue::{name}" for name in TYPE_FEATURE_NAMES),
        *(f"terminal::{name}" for name in TYPE_FEATURE_NAMES),
        *(f"delta::{name}" for name in TYPE_FEATURE_NAMES),
        *STOP_EXTRA_FEATURE_NAMES,
    )


def apply_stop_ablation(features: list[float], mode: str) -> list[float]:
    if mode == "full":
        return features
    result = list(features)
    for index, name in enumerate(stop_feature_names()):
        is_base = "::base_score_" in name
        if not is_base:
            continue
        if mode == "no_base" or (mode == "delta_only" and not name.startswith("delta::")):
            result[index] = 0.0
        elif mode == "damped_base":
            result[index] *= 0.25
    return result


def apply_stop_ablation_rows(
    rows: list[tuple[list[float], int, Decision]],
    mode: str,
) -> list[tuple[list[float], int, Decision]]:
    return [(apply_stop_ablation(features, mode), label, decision) for features, label, decision in rows]


def mine_oof_hard_continue(
    rows: list[tuple[list[float], int, Decision]],
    manifest_data: dict[str, Any],
) -> tuple[list[tuple[list[float], int, Decision]], dict[str, Any]]:
    folds = int(manifest_data.get("stop_oof_folds", 0))
    multiplier = float(manifest_data.get("stop_oof_false_stop_multiplier", 1.35))
    if folds < 2:
        return rows, {"enabled": False, "folds": folds, "false_stops": 0}
    episode_ids = sorted({decision.episode_id for _, _, decision in rows})
    fold_by_episode = {
        episode_id: int(hashlib.sha256(f"{manifest_data.get('seed', 0)}:{episode_id}".encode("utf-8")).hexdigest(), 16) % folds
        for episode_id in episode_ids
    }
    scored: list[tuple[float, int, int]] = []
    fold_audit = []
    for fold in range(folds):
        fit_rows = [row for row in rows if fold_by_episode[row[2].episode_id] != fold]
        holdout_rows = [row for row in rows if fold_by_episode[row[2].episode_id] == fold]
        if not fit_rows or not holdout_rows or len({label for _, label, _ in fit_rows}) < 2:
            raise ValueError(f"invalid stop OOF fold={fold}: train={len(fit_rows)} holdout={len(holdout_rows)}")
        model = train_stop_classifier(fit_rows, manifest_data, seed=int(manifest_data.get("seed", 0)) + 7919 * (fold + 1))
        predictions = predict_batch(model, [features for features, _, _ in holdout_rows])
        for row, score in zip(holdout_rows, predictions):
            scored.append((float(score), int(row[1]), id(row[2])))
        fold_audit.append({"fold": fold, "train": len(fit_rows), "holdout": len(holdout_rows)})
    threshold_metrics = _best_stop_threshold([(score, bool(label)) for score, label, _ in scored])
    threshold = float(threshold_metrics["threshold"])
    hard_ids = {decision_id for score, label, decision_id in scored if label == 0 and score >= threshold}
    weighted = [
        (features, label, replace(decision, weight=decision.weight * multiplier) if id(decision) in hard_ids else decision)
        for features, label, decision in rows
    ]
    return weighted, {
        "enabled": True,
        "folds": folds,
        "multiplier": multiplier,
        "false_stops": len(hard_ids),
        "threshold_metrics": threshold_metrics,
        "folds_detail": fold_audit,
    }


def select_stop_ablation(
    rows: list[tuple[list[float], int, Decision]],
    manifest_data: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not bool(manifest_data.get("stop_run_ablations", False)):
        return str(manifest_data.get("stop_ablation_mode", "full")), {"enabled": False}
    episodes = sorted({decision.episode_id for _, _, decision in rows})
    validation_ids = {
        episode_id for episode_id in episodes
        if int(hashlib.sha256(f"ablation:{manifest_data.get('seed', 0)}:{episode_id}".encode("utf-8")).hexdigest(), 16) % 5 == 0
    }
    train_rows = [row for row in rows if row[2].episode_id not in validation_ids]
    validation_rows = [row for row in rows if row[2].episode_id in validation_ids]
    if not train_rows or not validation_rows:
        raise ValueError("stop ablation split is empty")
    report = {}
    best = None
    for number, mode in enumerate(STOP_ABLATIONS):
        transformed_train = apply_stop_ablation_rows(train_rows, mode)
        transformed_validation = apply_stop_ablation_rows(validation_rows, mode)
        model = train_stop_classifier(
            transformed_train,
            manifest_data,
            seed=int(manifest_data.get("seed", 0)) + 104729 * (number + 1),
        )
        scored = [
            (score, bool(label))
            for score, (_, label, _) in zip(
                predict_batch(model, [features for features, _, _ in transformed_validation]),
                transformed_validation,
            )
        ]
        metrics = _best_stop_threshold(scored)
        report[mode] = metrics
        objective = (metrics["balanced_accuracy"], metrics["continue_recall"], metrics["stop_recall"])
        if best is None or objective > best[0]:
            best = (objective, mode)
    return best[1], {"enabled": True, "selected": best[1], "metrics": report}


def _filter_decision_by_type(decision: Decision, option_type: int) -> Decision | None:
    rows = [
        (features, label)
        for features, label, candidate_type in zip(decision.features, decision.labels, decision.option_types)
        if candidate_type == option_type
    ]
    if not rows:
        return None
    features, labels = zip(*rows)
    return Decision(
        features=list(features),
        labels=list(labels),
        source=decision.source,
        won=decision.won,
        episode_id=decision.episode_id,
        head=decision.head,
        seat=decision.seat,
        reward=decision.reward,
        option_types=[option_type] * len(features),
        min_count=decision.min_count,
        max_count=decision.max_count,
        weight=decision.weight,
    )


def _type_selected(decision: Decision, option_type: int) -> bool:
    return any(
        candidate_type == option_type and label == 1
        for candidate_type, label in zip(decision.option_types, decision.labels)
    )


def _aggregate_decision_by_type(
    decision: Decision,
    base_model: dict[str, Any] | None = None,
    feature_mode: str = "full",
) -> Decision | None:
    base_scores = predict_batch(base_model, decision.features) if base_model else None
    features, option_types = aggregate_option_type_features(
        decision.features, decision.option_types, base_scores, feature_mode=feature_mode
    )
    if len(features) < 2:
        return None
    selected_types = {
        option_type
        for option_type, label in zip(decision.option_types, decision.labels)
        if label
    }
    labels = [int(option_type in selected_types) for option_type in option_types]
    if not any(labels) or all(labels):
        return None
    return Decision(
        features=features,
        labels=labels,
        source=decision.source,
        won=decision.won,
        episode_id=decision.episode_id,
        head="main_type",
        seat=decision.seat,
        reward=decision.reward,
        option_types=option_types,
        min_count=decision.min_count,
        max_count=decision.max_count,
        weight=decision.weight,
    )


def _stage_decision(decision: Decision) -> Decision | None:
    selected = {
        option_type
        for option_type, label in zip(decision.option_types, decision.labels)
        if label
    }
    if not selected:
        return None
    terminal = any(option_type in TERMINAL_MAIN_TYPES for option_type in selected)
    labels = [int((option_type in TERMINAL_MAIN_TYPES) == terminal) for option_type in decision.option_types]
    if not any(labels) or all(labels):
        return None
    return replace(decision, labels=labels, head="main_stop")


def _binary_stage_decision(decision: Decision) -> Decision | None:
    feature_names = ROUTER_FEATURE_NAMES if len(decision.features[0]) == len(ROUTER_FEATURE_NAMES) else TYPE_FEATURE_NAMES
    score_index = feature_names.index("base_score_max")
    continuing = [
        index for index, option_type in enumerate(decision.option_types)
        if option_type in CONTINUE_MAIN_TYPES
    ]
    terminal = [
        index for index, option_type in enumerate(decision.option_types)
        if option_type in TERMINAL_MAIN_TYPES
    ]
    if not continuing or not terminal:
        return None
    continue_index = max(continuing, key=lambda index: decision.features[index][score_index])
    terminal_index = max(terminal, key=lambda index: decision.features[index][score_index])
    selected_terminal = any(
        label and option_type in TERMINAL_MAIN_TYPES
        for label, option_type in zip(decision.labels, decision.option_types)
    )
    return replace(
        decision,
        features=[decision.features[continue_index], decision.features[terminal_index]],
        labels=[int(not selected_terminal), int(selected_terminal)],
        option_types=[decision.option_types[continue_index], decision.option_types[terminal_index]],
        head="main_stop",
    )


def _stop_decision_features(decision: Decision) -> list[float] | None:
    return stop_decision_features(dict(zip(decision.option_types, decision.features)))


def _stop_classifier_rows(decisions: list[Decision]) -> list[tuple[list[float], int, Decision]]:
    rows = []
    for decision in decisions:
        features = _stop_decision_features(decision)
        if features is None:
            continue
        terminal = int(any(
            label and option_type in TERMINAL_MAIN_TYPES
            for label, option_type in zip(decision.labels, decision.option_types)
        ))
        rows.append((features, terminal, decision))
    return rows


def _stage_type_decision(decision: Decision, terminal: bool) -> Decision | None:
    wanted = TERMINAL_MAIN_TYPES if terminal else CONTINUE_MAIN_TYPES
    rows = [
        (features, label, option_type)
        for features, label, option_type in zip(decision.features, decision.labels, decision.option_types)
        if option_type in wanted
    ]
    if len(rows) < 2 or not any(label for _, label, _ in rows):
        return None
    features, labels, option_types = zip(*rows)
    if all(labels):
        return None
    return replace(
        decision,
        features=list(features),
        labels=list(labels),
        option_types=list(option_types),
        head="main_terminal_type" if terminal else "main_continue_type",
    )


def balance_type_decisions(decisions: list[Decision], maximum_weight: float = 6.0) -> list[Decision]:
    selected_counts: Counter[int] = Counter()
    selected_by_decision: list[int] = []
    for decision in decisions:
        selected = next((option_type for option_type, label in zip(decision.option_types, decision.labels) if label), -1)
        selected_by_decision.append(selected)
        if selected >= 0:
            selected_counts[selected] += 1
    if not selected_counts:
        return decisions
    largest = max(selected_counts.values())
    return [
        replace(
            decision,
            weight=decision.weight * min(maximum_weight, largest / max(1, selected_counts[selected])),
        )
        for decision, selected in zip(decisions, selected_by_decision)
    ]


def balance_stage_decisions(decisions: list[Decision], maximum_weight: float = 4.0) -> list[Decision]:
    counts = Counter()
    stages = []
    for decision in decisions:
        stage = int(any(
            label and option_type in TERMINAL_MAIN_TYPES
            for label, option_type in zip(decision.labels, decision.option_types)
        ))
        stages.append(stage)
        counts[stage] += 1
    largest = max(counts.values(), default=1)
    return [
        replace(decision, weight=decision.weight * min(maximum_weight, largest / max(1, counts[stage])))
        for decision, stage in zip(decisions, stages)
    ]


def stage_weight_audit(
    decisions: list[Decision],
    win_weight: float,
    loss_weight: float,
) -> dict[str, Any]:
    rows = {"continue": {"count": 0, "mass": 0.0}, "terminal": {"count": 0, "mass": 0.0}}
    for decision in decisions:
        terminal = any(
            label and option_type in TERMINAL_MAIN_TYPES
            for label, option_type in zip(decision.labels, decision.option_types)
        )
        key = "terminal" if terminal else "continue"
        weight = decision.weight * (win_weight if decision.won else loss_weight)
        rows[key]["count"] += 1
        rows[key]["mass"] += weight
    total_count = sum(row["count"] for row in rows.values())
    total_mass = sum(row["mass"] for row in rows.values())
    for row in rows.values():
        row["mean_weight"] = row["mass"] / max(1, row["count"])
    return {
        **rows,
        "terminal_count_fraction": rows["terminal"]["count"] / max(1, total_count),
        "terminal_mass_fraction": rows["terminal"]["mass"] / max(1e-12, total_mass),
    }


def evaluate_stop_router(model: dict[str, Any], decisions: list[Decision]) -> dict[str, Any]:
    confusion = Counter()
    for decision in decisions:
        scores = predict_batch(model, decision.features)
        if not scores:
            continue
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
        actual_terminal = any(
            label and option_type in TERMINAL_MAIN_TYPES
            for label, option_type in zip(decision.labels, decision.option_types)
        )
        predicted_terminal = decision.option_types[predicted_index] in TERMINAL_MAIN_TYPES
        confusion[(actual_terminal, predicted_terminal)] += 1
    continue_recall = confusion[(False, False)] / max(1, confusion[(False, False)] + confusion[(False, True)])
    stop_recall = confusion[(True, True)] / max(1, confusion[(True, True)] + confusion[(True, False)])
    return {
        "decisions": sum(confusion.values()),
        "balanced_accuracy": 0.5 * (continue_recall + stop_recall),
        "continue_recall": continue_recall,
        "stop_recall": stop_recall,
    }


def evaluate_stop_classifier(
    model: dict[str, Any],
    rows: list[tuple[list[float], int, Decision]],
    threshold: float = 0.0,
) -> dict[str, Any]:
    confusion = Counter()
    for features, actual_terminal, _ in rows:
        predicted_terminal = predict_model(model, features) >= threshold
        confusion[(bool(actual_terminal), predicted_terminal)] += 1
    continue_recall = confusion[(False, False)] / max(1, confusion[(False, False)] + confusion[(False, True)])
    stop_recall = confusion[(True, True)] / max(1, confusion[(True, True)] + confusion[(True, False)])
    return {
        "decisions": sum(confusion.values()),
        "balanced_accuracy": 0.5 * (continue_recall + stop_recall),
        "continue_recall": continue_recall,
        "stop_recall": stop_recall,
    }


def evaluate_hard_main(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    decisions: list[Decision],
) -> dict[str, Any]:
    stop_model = head_models.get("main_stop")
    if not stop_model:
        return {"decisions": 0, "top1": 0.0, "top3": 0.0, "by_type": {}, "stop": {}}
    top1 = top3 = 0
    by_type: dict[str, dict[str, int]] = {}
    stop_confusion = Counter()
    names = {value: name.removeprefix("main_") for name, value in MAIN_TYPE_OPTIONS.items()}
    for decision in (row for row in decisions if row.head == "main"):
        base_scores = predict_batch(head_models.get("main", global_model), decision.features)
        type_rows, present_types = aggregate_option_type_features(decision.features, decision.option_types, base_scores)
        type_features = dict(zip(present_types, type_rows))
        stop_scores = {value: predict_model(stop_model, type_features[value]) for value in present_types}
        terminal_types = [value for value in present_types if value in TERMINAL_MAIN_TYPES]
        continue_types = [value for value in present_types if value in CONTINUE_MAIN_TYPES]
        choose_terminal = bool(terminal_types) and (
            not continue_types or max(stop_scores[value] for value in terminal_types) > max(stop_scores[value] for value in continue_types)
        )
        stage_types = terminal_types if choose_terminal else continue_types
        stage_model = head_models.get("main_terminal_type" if choose_terminal else "main_continue_type")
        chosen_type = max(
            stage_types,
            key=lambda value: predict_model(stage_model, type_features[value]) if stage_model else stop_scores[value],
        )
        scores = []
        for option_index, (features, option_type, base_score) in enumerate(
            zip(decision.features, decision.option_types, base_scores)
        ):
            if option_type != chosen_type:
                scores.append(float("-inf"))
                continue
            head = head_models.get(MAIN_TYPE_HEADS.get(option_type, ""))
            scores.append(base_score + (predict_model(head, features) if head else 0.0))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        selected = {index for index, label in enumerate(decision.labels) if label}
        matched = int(order[0] in selected)
        top1 += matched
        top3 += int(bool(selected.intersection(order[:3])))
        actual_type = next((value for value, label in zip(decision.option_types, decision.labels) if label), -1)
        row = by_type.setdefault(names.get(actual_type, str(actual_type)), {"decisions": 0, "top1": 0})
        row["decisions"] += 1
        row["top1"] += matched
        actual_terminal = actual_type in TERMINAL_MAIN_TYPES
        stop_confusion[(actual_terminal, choose_terminal)] += 1
    for row in by_type.values():
        row["recall"] = row["top1"] / max(1, row["decisions"])
    count = sum(row["decisions"] for row in by_type.values())
    continue_recall = stop_confusion[(False, False)] / max(1, stop_confusion[(False, False)] + stop_confusion[(False, True)])
    stop_recall = stop_confusion[(True, True)] / max(1, stop_confusion[(True, True)] + stop_confusion[(True, False)])
    return {
        "decisions": count,
        "top1": top1 / max(1, count),
        "top3": top3 / max(1, count),
        "by_type": by_type,
        "stop": {
            "balanced_accuracy": 0.5 * (continue_recall + stop_recall),
            "continue_recall": continue_recall,
            "stop_recall": stop_recall,
        },
    }


def _standardize_type_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    scale = variance ** 0.5
    if scale <= 1e-9:
        return {key: 0.0 for key in values}
    return {key: (value - mean) / scale for key, value in values.items()}


def _stage_representatives(type_features: dict[int, list[float]], feature_mode: str) -> tuple[int | None, int | None]:
    feature_names = ROUTER_FEATURE_NAMES if feature_mode == "compact_v1" else TYPE_FEATURE_NAMES
    score_index = feature_names.index("base_score_max")
    continuing = [value for value in type_features if value in CONTINUE_MAIN_TYPES]
    terminal = [value for value in type_features if value in TERMINAL_MAIN_TYPES]
    continue_type = max(continuing, key=lambda value: type_features[value][score_index]) if continuing else None
    terminal_type = max(terminal, key=lambda value: type_features[value][score_index]) if terminal else None
    return continue_type, terminal_type


def _aggregate_layout_decision(decision: Decision, type_rows: list[list[float]], present_types: list[int]) -> Decision:
    selected_types = {
        option_type
        for option_type, label in zip(decision.option_types, decision.labels)
        if label
    }
    return replace(
        decision,
        features=type_rows,
        labels=[int(option_type in selected_types) for option_type in present_types],
        option_types=present_types,
    )


def _soft_router_layouts(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    decisions: list[Decision],
    feature_mode: str,
    ablation_modes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    ablation_modes = ablation_modes or {}
    layouts = []
    for decision in (row for row in decisions if row.head == "main"):
        base_scores = predict_batch(head_models.get("main", global_model), decision.features)
        type_rows, present_types = aggregate_option_type_features(
            decision.features, decision.option_types, base_scores, feature_mode=feature_mode
        )
        type_features = dict(zip(present_types, type_rows))
        stop_rows, stop_types = aggregate_option_type_features(
            decision.features, decision.option_types, base_scores, feature_mode="full"
        )

        def type_head_scores(name: str) -> dict[int, float]:
            model = head_models.get(name)
            if not model:
                return {option_type: 0.0 for option_type in present_types}
            return {
                option_type: predict_model(model, type_features[option_type])
                for option_type in present_types
            }

        main_type = _standardize_type_scores(type_head_scores("main_type"))
        stop_decision = _aggregate_layout_decision(decision, stop_rows, stop_types)
        regime = stop_regime(stop_decision)
        stop_model = head_models.get(f"main_stop_{regime}") or head_models.get("main_stop")
        stop_features = _stop_decision_features(stop_decision)
        transformed_stop_features = (
            apply_stop_ablation(stop_features, str(ablation_modes.get(regime, "full")))
            if stop_features is not None else None
        )
        stop_margin = predict_model(stop_model, transformed_stop_features) if stop_model and transformed_stop_features is not None else float("inf")
        member_scores = [
            predict_model(model, transformed_stop_features)
            for name, model in head_models.items()
            if name.startswith(f"main_stop_{regime}_seed") and transformed_stop_features is not None
        ]
        continue_type = type_head_scores("main_continue_type")
        terminal_type = type_head_scores("main_terminal_type")
        stage = _standardize_type_scores({
            option_type: terminal_type[option_type] if option_type in TERMINAL_MAIN_TYPES else continue_type[option_type]
            for option_type in present_types
        })
        option_head_scores = []
        for features, option_type in zip(decision.features, decision.option_types):
            model = head_models.get(MAIN_TYPE_HEADS.get(option_type, ""))
            option_head_scores.append(predict_model(model, features) if model else 0.0)
        layouts.append({
            "decision": decision,
            "base": list(_standardize_type_scores(dict(enumerate(base_scores))).values()),
            "option": list(_standardize_type_scores(dict(enumerate(option_head_scores))).values()),
            "main_type": [main_type.get(value, 0.0) for value in decision.option_types],
            "stage": [stage.get(value, 0.0) for value in decision.option_types],
            "stop_margin": stop_margin,
            "stop_std": float(np.std(member_scores)) if member_scores else 0.0,
            "regime": regime,
        })
    return layouts


def evaluate_soft_main_layouts(layouts: list[dict[str, Any]], calibration: dict[str, Any]) -> dict[str, Any]:
    top1 = top3 = 0
    by_type: dict[str, dict[str, int]] = {}
    stop_confusion = Counter()
    names = {value: name.removeprefix("main_") for name, value in MAIN_TYPE_OPTIONS.items()}
    by_regime: dict[str, Counter] = {regime: Counter() for regime in STOP_REGIMES}
    late_turn_confusion = Counter()
    for layout in layouts:
        decision = layout["decision"]
        regime = str(layout.get("regime", "no_attack"))
        regime_calibration = (calibration.get("regimes") or {}).get(regime, calibration)
        threshold = float(regime_calibration.get("stop_threshold", 0.0))
        terminal_preferred = layout["stop_margin"] >= threshold
        uncertain = (
            abs(layout["stop_margin"] - threshold) < float(regime_calibration.get("stop_confidence_band", 0.0))
            or float(layout.get("stop_std", 0.0)) > float(regime_calibration.get("uncertainty_threshold", float("inf")))
        )
        multiplier = float(calibration.get("low_confidence_multiplier", 0.25)) if uncertain else 1.0
        stage_gates = [
            1.0 if (option_type in TERMINAL_MAIN_TYPES) == terminal_preferred else -1.0
            for option_type in decision.option_types
        ]
        scores = [
            base
            + float(calibration["option_weight"]) * option
            + float(calibration["main_type_weight"]) * main_type
            + float(calibration["stage_type_weight"]) * stage
            + float(calibration["stop_weight"]) * multiplier * stage_gate
            for base, option, main_type, stage, stage_gate in zip(
                layout["base"], layout["option"], layout["main_type"], layout["stage"], stage_gates
            )
        ]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        selected = {index for index, label in enumerate(decision.labels) if label}
        top1 += int(bool(order) and order[0] in selected)
        top3 += int(bool(selected.intersection(order[:3])))
        actual_type = next((value for value, label in zip(decision.option_types, decision.labels) if label), -1)
        predicted_type = decision.option_types[order[0]] if order else -1
        row = by_type.setdefault(names.get(actual_type, str(actual_type)), {"decisions": 0, "top1": 0})
        row["decisions"] += 1
        row["top1"] += int(predicted_type == actual_type and order[0] in selected)
        stop_confusion[(actual_type in TERMINAL_MAIN_TYPES, predicted_type in TERMINAL_MAIN_TYPES)] += 1
        by_regime[regime][(actual_type in TERMINAL_MAIN_TYPES, predicted_type in TERMINAL_MAIN_TYPES)] += 1
        if decision.features and decision.features[0][FEATURE_NAMES.index("turn")] >= 8:
            late_turn_confusion[(actual_type in TERMINAL_MAIN_TYPES, predicted_type in TERMINAL_MAIN_TYPES)] += 1
    for row in by_type.values():
        row["recall"] = row["top1"] / max(1, row["decisions"])
    count = len(layouts)
    continue_recall = stop_confusion[(False, False)] / max(1, stop_confusion[(False, False)] + stop_confusion[(False, True)])
    stop_recall = stop_confusion[(True, True)] / max(1, stop_confusion[(True, True)] + stop_confusion[(True, False)])
    macro = sum(row["recall"] for row in by_type.values()) / max(1, len(by_type))
    regime_metrics = {}
    for regime, confusion in by_regime.items():
        continue_regime = confusion[(False, False)] / max(1, confusion[(False, False)] + confusion[(False, True)])
        stop_regime_recall = confusion[(True, True)] / max(1, confusion[(True, True)] + confusion[(True, False)])
        regime_metrics[regime] = {
            "continue_recall": continue_regime,
            "stop_recall": stop_regime_recall,
            "balanced_accuracy": 0.5 * (continue_regime + stop_regime_recall),
        }
    late_continue = late_turn_confusion[(False, False)] / max(
        1, late_turn_confusion[(False, False)] + late_turn_confusion[(False, True)]
    )
    return {
        "decisions": count,
        "top1": top1 / max(1, count),
        "top3": top3 / max(1, count),
        "macro_recall": macro,
        "by_type": by_type,
        "stop": {
            "balanced_accuracy": 0.5 * (continue_recall + stop_recall),
            "continue_recall": continue_recall,
            "stop_recall": stop_recall,
        },
        "regimes": regime_metrics,
        "subgroups": {"turn_8_plus_continue_recall": late_continue},
    }


def enforce_stop_gates(metrics: dict[str, Any], config: dict[str, Any], label: str) -> None:
    checks = stop_gate_checks(metrics, config)
    if not all(checks.values()):
        raise ValueError(f"{label} stop gates failed: checks={checks} metrics={metrics}")


def stop_gate_checks(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    stop = metrics.get("stop") or {}
    regimes = metrics.get("regimes") or {}
    subgroups = metrics.get("subgroups") or {}
    return {
        "continue": float(stop.get("continue_recall", 0.0)) >= float(config.get("stop_gate_continue", 0.90)),
        "terminal": float(stop.get("stop_recall", 0.0)) >= float(config.get("stop_gate_terminal", 0.70)),
        "balanced": float(stop.get("balanced_accuracy", 0.0)) >= float(config.get("stop_gate_balanced", 0.85)),
        "attack_continue": float((regimes.get("attack_ready") or {}).get("continue_recall", 0.0)) >= float(config.get("stop_gate_attack_continue", 0.0)),
        "ko_continue": float((regimes.get("ko_ready") or {}).get("continue_recall", 0.0)) >= float(config.get("stop_gate_ko_continue", 0.0)),
        "turn_8_plus_continue": float(subgroups.get("turn_8_plus_continue_recall", 0.0)) >= float(config.get("stop_gate_turn_8_continue", 0.0)),
    }


def _stop_gate_deficit(metrics: dict[str, Any], config: dict[str, Any]) -> float:
    stop = metrics.get("stop") or {}
    regimes = metrics.get("regimes") or {}
    subgroups = metrics.get("subgroups") or {}
    pairs = (
        (float(stop.get("continue_recall", 0.0)), float(config.get("stop_gate_continue", 0.90))),
        (float(stop.get("stop_recall", 0.0)), float(config.get("stop_gate_terminal", 0.70))),
        (float(stop.get("balanced_accuracy", 0.0)), float(config.get("stop_gate_balanced", 0.85))),
        (float((regimes.get("attack_ready") or {}).get("continue_recall", 0.0)), float(config.get("stop_gate_attack_continue", 0.0))),
        (float((regimes.get("ko_ready") or {}).get("continue_recall", 0.0)), float(config.get("stop_gate_ko_continue", 0.0))),
        (float(subgroups.get("turn_8_plus_continue_recall", 0.0)), float(config.get("stop_gate_turn_8_continue", 0.0))),
    )
    return sum(max(0.0, required - actual) for actual, required in pairs)


def _joint_threshold_candidates(rows: list[tuple[float, bool]], current: float, limit: int = 12) -> list[float]:
    thresholds = _candidate_stop_thresholds(rows)
    if len(thresholds) <= limit:
        return sorted(set([*thresholds, current]))
    indices = np.linspace(0, len(thresholds) - 1, num=max(2, limit - 1), dtype=int)
    return sorted({current, *(thresholds[int(index)] for index in indices)})


def calibrate_soft_router(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    decisions: list[Decision],
    feature_mode: str = "compact_v1",
    gate_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_config = gate_config or {}
    if int(gate_config.get("router_schema_version", 1)) < STOP_ROUTER_SCHEMA_VERSION:
        layouts = _soft_router_layouts(global_model, head_models, decisions, feature_mode)
        labeled_margins = []
        for layout in layouts:
            if not np.isfinite(layout["stop_margin"]):
                continue
            terminal = any(
                label and option_type in TERMINAL_MAIN_TYPES
                for label, option_type in zip(layout["decision"].labels, layout["decision"].option_types)
            )
            labeled_margins.append((float(layout["stop_margin"]), terminal))
        thresholds = _feasible_stop_thresholds(labeled_margins)
        if not thresholds:
            raise ValueError(f"stop router cannot satisfy strict recall constraints; best={_best_stop_threshold(labeled_margins)}")
        best = None
        for stop_weight in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            for main_type_weight in (0.25, 0.5, 1.0, 2.0):
                for stage_type_weight in (0.0, 0.25, 0.5, 1.0):
                    for option_weight in (0.5, 1.0, 1.5):
                        for threshold in thresholds:
                            margins = [abs(value - threshold) for value, _ in labeled_margins]
                            calibration = {
                                "mode": "soft_confidence",
                                "feature_mode": feature_mode,
                                "stop_feature_mode": "full",
                                "stop_weight": stop_weight,
                                "main_type_weight": main_type_weight,
                                "stage_type_weight": stage_type_weight,
                                "option_weight": option_weight,
                                "stop_threshold": threshold,
                                "stop_confidence_band": float(np.quantile(margins, 0.25)) if margins else 0.0,
                                "low_confidence_multiplier": 0.25,
                            }
                            metrics = evaluate_soft_main_layouts(layouts, calibration)
                            stop = metrics["stop"]
                            if stop["continue_recall"] < 0.90 or stop["stop_recall"] < 0.70 or stop["balanced_accuracy"] < 0.85:
                                continue
                            objective = (metrics["top1"], metrics["macro_recall"], metrics["top3"])
                            if best is None or objective > best[0]:
                                best = (objective, calibration, metrics)
        if best is None:
            raise ValueError("no soft router calibration satisfies stop recall constraints")
        return best[1], best[2]
    ablation_modes = dict(gate_config.get("stop_ablation_modes", {}))
    layouts = _soft_router_layouts(global_model, head_models, decisions, feature_mode, ablation_modes)
    regime_calibrations = {}
    regime_margins = {}
    regime_threshold_candidates = {}
    for regime in STOP_REGIMES:
        regime_layouts = [row for row in layouts if row.get("regime") == regime and np.isfinite(row["stop_margin"])]
        labeled_margins = []
        for layout in regime_layouts:
            terminal = any(
                label and option_type in TERMINAL_MAIN_TYPES
                for label, option_type in zip(layout["decision"].labels, layout["decision"].option_types)
            )
            labeled_margins.append((float(layout["stop_margin"]), terminal))
        if not labeled_margins:
            raise ValueError(f"stop router has no validation rows for regime={regime}")
        best_stop = _best_stop_threshold(labeled_margins)
        regime_margins[regime] = labeled_margins
        regime_threshold_candidates[regime] = _joint_threshold_candidates(
            labeled_margins, float(best_stop["threshold"])
        )
        distances = [abs(value - float(best_stop["threshold"])) for value, _ in labeled_margins]
        uncertainties = [float(row.get("stop_std", 0.0)) for row in regime_layouts]
        regime_calibrations[regime] = {
            "stop_threshold": float(best_stop["threshold"]),
            "stop_confidence_band": float(np.quantile(distances, 0.25)) if distances else 0.0,
            "uncertainty_threshold": float(np.quantile(uncertainties, 0.90)) if uncertainties else float("inf"),
            "threshold_metrics": best_stop,
            "ablation_mode": str(gate_config.get("stop_ablation_modes", {}).get(regime, "full")),
        }
    best: tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]] | None = None
    nearest: tuple[tuple[float, float, float, float], dict[str, Any], dict[str, Any]] | None = None
    for stop_weight in (0.5, 1.0, 1.5, 2.0):
        for main_type_weight in (0.25, 0.5, 1.0, 2.0):
            for stage_type_weight in (0.0, 0.25, 0.5, 1.0):
                for option_weight in (0.5, 1.0, 1.5):
                    calibration = {
                        "schema_version": STOP_ROUTER_SCHEMA_VERSION,
                        "mode": "regime_soft_v2",
                        "feature_mode": feature_mode,
                        "stop_feature_mode": "full",
                        "stop_weight": stop_weight,
                        "main_type_weight": main_type_weight,
                        "stage_type_weight": stage_type_weight,
                        "option_weight": option_weight,
                        "score_clip": 4.0,
                        "safety_residual_weight": float(gate_config.get("safety_residual_weight", 0.0)),
                        "low_confidence_multiplier": 0.25,
                        "regimes": regime_calibrations,
                    }
                    metrics = evaluate_soft_main_layouts(layouts, calibration)
                    objective = (metrics["top1"], metrics["macro_recall"], metrics["top3"])
                    rank = (_stop_gate_deficit(metrics, gate_config), -objective[0], -objective[1], -objective[2])
                    if nearest is None or rank < nearest[0]:
                        nearest = (rank, copy.deepcopy(calibration), metrics)
                    if not all(stop_gate_checks(metrics, gate_config).values()):
                        continue
                    if best is None or objective > best[0]:
                        best = (objective, calibration, metrics)
    if best is None:
        assert nearest is not None
        calibration = nearest[1]
        for _ in range(2):
            for regime in STOP_REGIMES:
                regime_best = None
                for threshold in regime_threshold_candidates[regime]:
                    candidate = copy.deepcopy(calibration)
                    candidate_regime = candidate["regimes"][regime]
                    candidate_regime["stop_threshold"] = float(threshold)
                    distances = [abs(value - float(threshold)) for value, _ in regime_margins[regime]]
                    candidate_regime["stop_confidence_band"] = float(np.quantile(distances, 0.25)) if distances else 0.0
                    metrics = evaluate_soft_main_layouts(layouts, candidate)
                    objective = (metrics["top1"], metrics["macro_recall"], metrics["top3"])
                    rank = (_stop_gate_deficit(metrics, gate_config), -objective[0], -objective[1], -objective[2])
                    if regime_best is None or rank < regime_best[0]:
                        regime_best = (rank, candidate, metrics)
                assert regime_best is not None
                calibration = regime_best[1]
                if all(stop_gate_checks(regime_best[2], gate_config).values()):
                    objective = (
                        regime_best[2]["top1"],
                        regime_best[2]["macro_recall"],
                        regime_best[2]["top3"],
                    )
                    if best is None or objective > best[0]:
                        best = (objective, calibration, regime_best[2])
        if best is None:
            metrics = evaluate_soft_main_layouts(layouts, calibration)
            raise ValueError(
                "no soft router calibration satisfies stop recall constraints; "
                f"checks={stop_gate_checks(metrics, gate_config)} metrics={metrics}"
            )
    return best[1], best[2]


def _stop_threshold_metrics(rows: list[tuple[float, bool]], threshold: float) -> dict[str, float]:
    true_terminal = sum(actual and margin >= threshold for margin, actual in rows)
    false_continue = sum(actual and margin < threshold for margin, actual in rows)
    true_continue = sum(not actual and margin < threshold for margin, actual in rows)
    false_terminal = sum(not actual and margin >= threshold for margin, actual in rows)
    continue_recall = true_continue / max(1, true_continue + false_terminal)
    terminal_recall = true_terminal / max(1, true_terminal + false_continue)
    return {
        "threshold": threshold,
        "continue_recall": continue_recall,
        "stop_recall": terminal_recall,
        "balanced_accuracy": 0.5 * (continue_recall + terminal_recall),
    }


def _candidate_stop_thresholds(rows: list[tuple[float, bool]]) -> list[float]:
    values = sorted({margin for margin, _ in rows})
    if not values:
        return []
    return [values[0] - 1e-9, *((left + right) / 2 for left, right in zip(values, values[1:])), values[-1] + 1e-9]


def _best_stop_threshold(rows: list[tuple[float, bool]]) -> dict[str, float]:
    metrics = [_stop_threshold_metrics(rows, threshold) for threshold in _candidate_stop_thresholds(rows)]
    return max(metrics, key=lambda row: (row["balanced_accuracy"], row["continue_recall"], row["stop_recall"]), default={})


def _feasible_stop_thresholds(rows: list[tuple[float, bool]], limit: int = 24) -> list[float]:
    feasible = [
        metrics
        for threshold in _candidate_stop_thresholds(rows)
        if (metrics := _stop_threshold_metrics(rows, threshold))["continue_recall"] >= 0.90
        and metrics["stop_recall"] >= 0.70
        and metrics["balanced_accuracy"] >= 0.85
    ]
    feasible.sort(key=lambda row: (row["balanced_accuracy"], row["continue_recall"], row["stop_recall"]), reverse=True)
    return [row["threshold"] for row in feasible[:limit]]


def temporal_hard_main_metrics(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    decisions: list[Decision],
    folds: int = 3,
) -> list[dict[str, Any]]:
    episode_ids = sorted({decision.episode_id for decision in decisions if decision.head == "main"})
    if not episode_ids:
        return []
    result = []
    for fold in range(min(folds, len(episode_ids))):
        start = len(episode_ids) * fold // min(folds, len(episode_ids))
        end = len(episode_ids) * (fold + 1) // min(folds, len(episode_ids))
        selected = set(episode_ids[start:end])
        metrics = evaluate_hard_main(
            global_model,
            head_models,
            [decision for decision in decisions if decision.episode_id in selected],
        )
        result.append({"fold": fold, "episodes": len(selected), **metrics})
    return result


def _read_replay_file(replay_path: Path) -> dict[str, Any] | None:
    try:
        if replay_path.suffix == ".gz":
            replay = json.loads(gzip.open(replay_path, "rt", encoding="utf-8").read())
        else:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay["_local_replay_path"] = str(replay_path)
        name = replay_path.name.removesuffix(".gz").removesuffix(".json")
        replay["_local_episode_id"] = name.replace("episode-", "").replace("-replay", "")
        return replay
    except (OSError, json.JSONDecodeError):
        return None


def read_replays(path: Path, workers: int = 1) -> list[dict[str, Any]]:
    result = []
    replay_paths = sorted({*path.glob("*replay.json"), *path.glob("*replay.json.gz")})
    if workers > 1 and len(replay_paths) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_read_replay_file, replay_path) for replay_path in replay_paths]
            for number, future in enumerate(as_completed(futures), 1):
                replay = future.result()
                if replay is not None:
                    result.append(replay)
                if number % 100 == 0 or number == len(futures):
                    print(f"[read] {number}/{len(futures)}", flush=True)
        return result
    for replay_path in replay_paths:
        replay = _read_replay_file(replay_path)
        if replay is not None:
            result.append(replay)
    return result


def infer_target_deck(replays: list[dict[str, Any]], target_name: str) -> tuple[int, ...]:
    signatures: Counter[tuple[int, ...]] = Counter()
    for replay in replays:
        names = replay.get("info", {}).get("TeamNames") or []
        if target_name not in names:
            continue
        index = names.index(target_name)
        deck = full_deck(replay, index)
        if len(deck) == 60:
            signatures[deck_signature(deck)] += 1
    if not signatures:
        raise ValueError(f"could not infer target deck for {target_name}")
    return signatures.most_common(1)[0][0]


def target_index(replay: dict[str, Any], target_name: str, target_deck: tuple[int, ...]) -> int | None:
    names = replay.get("info", {}).get("TeamNames") or []
    signatures = [deck_signature(full_deck(replay, index)) for index in range(2)]
    if target_name in names:
        index = names.index(target_name)
        return index if signatures[index] == target_deck else None
    matches = [index for index, signature in enumerate(signatures) if signature == target_deck]
    if len(matches) == 1:
        return matches[0]
    return None


def decision_head(obs: Any, selected: set[int]) -> str:
    context = SelectContext(getattr(obs.select, "context", SelectContext.MAIN))
    if context == SelectContext.MAIN:
        return "main"
    if context in {SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO}:
        return "attach"
    if context in {
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.SWITCH_ENERGY_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.TO_HAND_ENERGY,
        SelectContext.TO_DECK_ENERGY,
        SelectContext.SWITCH_ENERGY,
    }:
        return "energy"
    if context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
        return "retreat"
    selected_options = [obs.select.option[value] for value in selected if value < len(obs.select.option)]
    if selected_options and all(option.type == OptionType.ATTACK for option in selected_options):
        return "attack"
    return "card"


def extract_decisions(replay: dict[str, Any], index: int, source: str) -> list[Decision]:
    reward = int((replay.get("rewards") or [0, 0])[index] or 0)
    episode_id = f"{source}:{replay.get('_local_episode_id') or replay.get('id') or id(replay)}"
    decisions = []
    _LOPUNNY_MEMORY.update(turn=None, active_serial=None, bench_serials=set(), moved=False)
    _OGERPON_MEMORY.update(turn=None, primary_serial=None)
    _reset_policy_memory()
    previous_observation: dict[str, Any] | None = None
    for step in replay.get("steps", []):
        if index >= len(step):
            continue
        agent_step = step[index] or {}
        action = agent_step.get("action")
        observation = previous_observation or {}
        select = observation.get("select") or {}
        options = select.get("option") or []
        if isinstance(action, list) and len(action) != 60 and options:
            selected = {value for value in action if isinstance(value, int) and 0 <= value < len(options)}
            try:
                obs = to_observation_class(observation)
                if hasattr(obs, "current"):
                    _update_lopunny_memory(obs)
                    _update_ogerpon_memory(obs)
                    _sync_policy_memory(obs)
                if selected and len(selected) != len(options) and len(options) > 1:
                    rows = [feature_vector(obs, option) for option in obs.select.option]
                else:
                    rows = []
            except Exception:
                rows = []
                obs = None
            if rows:
                option_types = [
                    int(_option_type(option).value if hasattr(_option_type(option), "value") else _option_type(option) or 0)
                    for option in obs.select.option
                ]
                decisions.append(
                    Decision(
                        features=rows,
                        labels=[1 if option_index in selected else 0 for option_index in range(len(rows))],
                        option_types=option_types,
                        source=source,
                        won=reward > 0,
                        episode_id=episode_id,
                        head=decision_head(obs, selected),
                        seat=index,
                        reward=reward,
                        min_count=int(getattr(obs.select, "minCount", 1) or 0),
                        max_count=int(getattr(obs.select, "maxCount", len(rows)) or len(rows)),
                    )
                )
            if obs is not None and selected:
                _record_policy_action(obs, selected)
        previous_observation = agent_step.get("observation") or None
    return decisions


def _process_replay_chunk(
    chunk: list[dict[str, Any]],
    target_name: str,
    source: str,
    target_deck: tuple[int, ...],
) -> tuple[list[Decision], int, int, int, list[list[int]]]:
    decisions: list[Decision] = []
    resolved = 0
    ambiguous = 0
    deck_mismatch = 0
    opponent_decks: list[list[int]] = []
    for replay in chunk:
        index = target_index(replay, target_name, target_deck)
        if index is None:
            names = replay.get("info", {}).get("TeamNames") or []
            if target_name in names:
                deck_mismatch += 1
            else:
                ambiguous += 1
            continue
        resolved += 1
        decisions.extend(extract_decisions(replay, index, source))
        opponent_deck = full_deck(replay, 1 - index)
        if len(opponent_deck) == 60:
            opponent_decks.append(list(opponent_deck))
    return decisions, resolved, ambiguous, deck_mismatch, opponent_decks


def load_source(
    spec: SourceSpec,
    parse_workers: int = 1,
    cache_dir: Path | None = None,
) -> tuple[list[Decision], tuple[int, ...], dict[str, Any]]:
    fingerprint = replay_fingerprint(spec.path) if cache_dir is not None else ""
    parse_cache_path = None
    if cache_dir is not None:
        parse_cache_path = cache_dir / f"parse-{fingerprint[:16]}.pickle.gz"
        payload = _load_cache_file(parse_cache_path)
        if (
            isinstance(payload, dict)
            and payload.get("version") == PARSE_CACHE_VERSION
            and payload.get("fingerprint") == fingerprint
            and payload.get("feature_schema_sha256") == _feature_schema_sha256()
            and isinstance(payload.get("decisions"), list)
        ):
            print(f"[cache] parse hit {parse_cache_path.name} decisions={len(payload['decisions'])}", flush=True)
            return payload["decisions"], tuple(payload["deck"]), payload["stats"]
    replays = read_replays(spec.path, parse_workers)
    target_deck = infer_target_deck(replays, spec.target_name)
    decisions: list[Decision] = []
    resolved = ambiguous = deck_mismatch = 0
    opponent_decks: Counter[tuple[int, ...]] = Counter()
    if parse_workers > 1 and len(replays) > 1:
        chunk_size = max(1, len(replays) // (parse_workers * 4))
        chunks = [replays[i:i + chunk_size] for i in range(0, len(replays), chunk_size)]
        with ProcessPoolExecutor(max_workers=parse_workers) as executor:
            futures = [
                executor.submit(_process_replay_chunk, chunk, spec.target_name, spec.name, target_deck)
                for chunk in chunks
            ]
            report_every = max(1, len(futures) // 10)
            for number, future in enumerate(as_completed(futures), 1):
                chunk_decisions, chunk_resolved, chunk_ambiguous, chunk_deck_mismatch, chunk_opponents = future.result()
                decisions.extend(chunk_decisions)
                resolved += chunk_resolved
                ambiguous += chunk_ambiguous
                deck_mismatch += chunk_deck_mismatch
                for opponent_deck in chunk_opponents:
                    opponent_decks[deck_signature(opponent_deck)] += 1
                if number % report_every == 0 or number == len(futures):
                    print(f"[parse] {number}/{len(futures)} chunks decisions={len(decisions)}", flush=True)
    else:
        chunk_decisions, chunk_resolved, chunk_ambiguous, chunk_deck_mismatch, chunk_opponents = _process_replay_chunk(
            replays, spec.target_name, spec.name, target_deck
        )
        decisions.extend(chunk_decisions)
        resolved += chunk_resolved
        ambiguous += chunk_ambiguous
        deck_mismatch += chunk_deck_mismatch
        for opponent_deck in chunk_opponents:
            opponent_decks[deck_signature(opponent_deck)] += 1
    decisions.sort(key=lambda decision: decision.episode_id)
    stats = {
        "replays": len(replays),
        "resolved": resolved,
        "ambiguous": ambiguous,
        "deck_mismatch": deck_mismatch,
        "decisions": len(decisions),
        "opponent_decks": [list(deck) for deck, _ in opponent_decks.most_common(16)],
    }
    if parse_cache_path is not None:
        _write_cache_file(parse_cache_path, {
            "version": PARSE_CACHE_VERSION,
            "fingerprint": fingerprint,
            "feature_schema_sha256": _feature_schema_sha256(),
            "decisions": decisions,
            "deck": list(target_deck),
            "stats": stats,
        })
        _write_cache_file(cache_dir / f"replays-{fingerprint[:16]}.pickle.gz", replays)
        print(f"[cache] parse written {parse_cache_path.name} decisions={len(decisions)}", flush=True)
    return decisions, target_deck, stats


def flatten(decisions: list[Decision], win_weight: float, loss_weight: float) -> tuple[list[list[float]], list[int], list[int], list[float]]:
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    weights: list[float] = []
    for decision in decisions:
        features.extend(decision.features)
        labels.extend(decision.labels)
        groups.append(len(decision.labels))
        base_weight = win_weight if decision.won else loss_weight
        weights.extend([base_weight * decision.weight] * len(decision.labels))
    return features, labels, groups, weights


def rankable_decisions(decisions: list[Decision]) -> list[Decision]:
    return [
        decision
        for decision in decisions
        if len(decision.labels) > 1 and any(decision.labels) and not all(decision.labels)
    ]


def _catboost_depth(num_leaves: int) -> int:
    return max(1, (num_leaves - 1).bit_length())


def _catboost_scale_bias(model: dict[str, Any]) -> tuple[float, float]:
    scale_and_bias = model.get("scale_and_bias")
    if isinstance(scale_and_bias, dict):
        return float(scale_and_bias.get("scale", 1.0)), float(scale_and_bias.get("bias", 0.0))
    if isinstance(scale_and_bias, (list, tuple)) and scale_and_bias:
        scale = float(scale_and_bias[0])
        bias_raw = scale_and_bias[1] if len(scale_and_bias) > 1 else 0.0
        bias = float(bias_raw[0]) if isinstance(bias_raw, (list, tuple)) else float(bias_raw)
        return scale, bias
    return 1.0, 0.0


def _predict_catboost(model: dict[str, Any], features: list[float]) -> float:
    scale, bias = _catboost_scale_bias(model)
    total = 0.0
    for tree in model.get("oblivious_trees", []):
        index = 0
        for level, split in enumerate(tree.get("splits", [])):
            feature = int(split.get("float_feature_index", 0))
            value = features[feature] if feature < len(features) else 0.0
            if value > float(split.get("border", 0.0)):
                index |= 1 << level
        leaf_values = tree.get("leaf_values", [])
        total += float(leaf_values[index]) if index < len(leaf_values) else 0.0
    return total * scale + bias


def predict_batch(model: dict[str, Any], features: list[list[float]]) -> list[float]:
    if not features:
        return []
    scale, bias = _catboost_scale_bias(model)
    max_feature = -1
    for tree in model.get("oblivious_trees", []):
        for split in tree.get("splits", []):
            max_feature = max(max_feature, int(split.get("float_feature_index", 0)))
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if max_feature >= matrix.shape[1]:
        pad = np.zeros((matrix.shape[0], max_feature + 1 - matrix.shape[1]), dtype=np.float64)
        matrix = np.concatenate([matrix, pad], axis=1)
    total = np.zeros(matrix.shape[0], dtype=np.float64)
    for tree in model.get("oblivious_trees", []):
        splits = tree.get("splits", [])
        leaf_values = tree.get("leaf_values", [])
        if not splits:
            total += float(leaf_values[0]) if leaf_values else 0.0
            continue
        index = np.zeros(matrix.shape[0], dtype=np.intp)
        for level, split in enumerate(splits):
            column = matrix[:, int(split.get("float_feature_index", 0))]
            index |= (column > float(split.get("border", 0.0))).astype(np.intp) << level
        np.clip(index, 0, max(0, len(leaf_values) - 1), out=index)
        total += np.take(leaf_values, index)
    return (total * scale + bias).tolist()


def _train_catboost(
    features: list[list[float]],
    labels: list[int],
    groups: list[int],
    weights: list[float],
    manifest_data: dict[str, Any],
    baseline_scores: list[float] | None = None,
) -> dict[str, Any]:
    from catboost import CatBoostRanker

    group_ids = []
    for group_id, size in enumerate(groups):
        group_ids.extend([group_id] * size)
    loss_function = str(manifest_data.get("loss_function", "YetiRank"))
    params: dict[str, Any] = {
        "iterations": int(manifest_data.get("n_estimators", 180)),
        "learning_rate": float(manifest_data.get("learning_rate", 0.045)),
        "depth": _catboost_depth(int(manifest_data.get("num_leaves", 31))),
        "min_data_in_leaf": int(manifest_data.get("min_child_samples", 30)),
        "l2_leaf_reg": float(manifest_data.get("reg_lambda", 1.0)),
        "rsm": float(manifest_data.get("colsample_bytree", 0.9)),
        "random_seed": int(manifest_data.get("seed", 20260731)),
        "loss_function": loss_function,
        "eval_metric": str(manifest_data.get("eval_metric", "NDCG:top=3")),
        "metric_period": 50,
        "task_type": str(manifest_data.get("task_type", "CPU")),
        "thread_count": int(manifest_data.get("n_jobs", 12)),
        "verbose": False,
        "allow_writing_files": False,
    }
    if params["task_type"].upper() == "GPU":
        params.pop("rsm", None)
    devices = str(manifest_data.get("devices", "") or "")
    if devices:
        params["devices"] = devices
    model = CatBoostRanker(**params)
    fit_kwargs = {
        "group_id": np.asarray(group_ids, dtype=np.int32),
        "group_weight": np.asarray(weights, dtype=np.float32),
    }
    if baseline_scores is not None:
        fit_kwargs["baseline"] = np.asarray(baseline_scores, dtype=np.float32)
    if params["task_type"].upper() == "GPU":
        with _catboost_gpu_slots:
            model.fit(
                np.asarray(features, dtype=np.float32),
                np.asarray(labels, dtype=np.float32),
                **fit_kwargs,
            )
            return dump_catboost_model(model)
    model.fit(
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        **fit_kwargs,
    )
    return dump_catboost_model(model)


def train_stop_classifier(
    rows: list[tuple[list[float], int, Decision]],
    manifest_data: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    params: dict[str, Any] = {
        "iterations": int(manifest_data.get("router_n_estimators", 450)),
        "learning_rate": float(manifest_data.get("learning_rate", 0.045)),
        "depth": _catboost_depth(int(manifest_data.get("router_num_leaves", 31))),
        "min_data_in_leaf": int(manifest_data.get("router_min_child_samples", 32)),
        "l2_leaf_reg": float(manifest_data.get("router_l2_leaf_reg", 8.0)),
        "random_seed": int(manifest_data.get("seed", 20260731) if seed is None else seed),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "metric_period": 50,
        "task_type": str(manifest_data.get("task_type", "CPU")),
        "thread_count": int(manifest_data.get("n_jobs", 12)),
        "verbose": False,
        "allow_writing_files": False,
    }
    devices = str(manifest_data.get("devices", "") or "")
    if devices:
        params["devices"] = devices
    features = np.asarray([features for features, _, _ in rows], dtype=np.float32)
    labels = np.asarray([label for _, label, _ in rows], dtype=np.int32)
    if int(manifest_data.get("router_schema_version", 1)) >= STOP_ROUTER_SCHEMA_VERSION:
        weights = np.asarray([
            stop_sample_weight(label, decision, manifest_data)
            for _, label, decision in rows
        ], dtype=np.float32)
    else:
        continue_weight = float(manifest_data.get("stop_continue_weight", 1.0))
        weights = np.asarray([
            (continue_weight if label == 0 else 1.0) * decision.weight * (
                float(manifest_data.get("win_weight", 1.5))
                if decision.won else float(manifest_data.get("loss_weight", 0.15))
            )
            for _, label, decision in rows
        ], dtype=np.float32)
    model = CatBoostClassifier(**params)
    if params["task_type"].upper() == "GPU":
        with _catboost_gpu_slots:
            model.fit(features, labels, sample_weight=weights)
    else:
        model.fit(features, labels, sample_weight=weights)
    return dump_catboost_model(model)


def average_catboost_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    if not models:
        raise ValueError("at least one CatBoost model is required")
    scale = 1.0 / len(models)
    merged = copy.deepcopy(models[0])
    merged["oblivious_trees"] = []
    bias = 0.0
    for model in models:
        model_scale, model_bias = _catboost_scale_bias(model)
        trees = copy.deepcopy(model.get("oblivious_trees", []))
        for tree in trees:
            tree["leaf_values"] = [float(value) * model_scale * scale for value in tree.get("leaf_values", [])]
        merged["oblivious_trees"].extend(trees)
        bias += model_bias * scale
    merged["scale_and_bias"] = [1.0, [bias]]
    return merged


def merge_catboost_models(base_model: dict[str, Any], residual_model: dict[str, Any]) -> dict[str, Any]:
    base_scale, base_bias = _catboost_scale_bias(base_model)
    residual_scale, residual_bias = _catboost_scale_bias(residual_model)

    def scaled_trees(model: dict[str, Any], scale: float) -> list[dict[str, Any]]:
        trees = copy.deepcopy(model.get("oblivious_trees", []))
        if scale != 1.0:
            for tree in trees:
                tree["leaf_values"] = [float(value) * scale for value in tree.get("leaf_values", [])]
        return trees

    merged = copy.deepcopy(residual_model)
    merged["oblivious_trees"] = scaled_trees(base_model, base_scale) + scaled_trees(residual_model, residual_scale)
    merged["scale_and_bias"] = [1.0, [base_bias + residual_bias]]
    return merged


def dump_catboost_model(model: Any) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        dump_path = handle.name
    try:
        model.save_model(dump_path, format="json")
        with open(dump_path, encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        os.unlink(dump_path)


def split_by_episode(
    decisions: list[Decision],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[Decision], list[Decision], list[Decision]]:
    episode_rows: dict[str, tuple[int, int]] = {}
    for decision in decisions:
        episode_rows.setdefault(decision.episode_id, (decision.reward, decision.seat))
    episode_ids = sorted(episode_rows)
    if len(episode_ids) < 3:
        split = max(1, len(decisions) // 5)
        return decisions[:-split], decisions[-split:], []
    test_count = max(1, round(len(episode_ids) * test_fraction)) if test_fraction > 0 else 0
    validation_count = max(1, round(len(episode_ids) * validation_fraction)) if validation_fraction > 0 else 0
    if test_count + validation_count >= len(episode_ids):
        test_count = 1
        validation_count = 1
    groups: dict[tuple[int, int], list[str]] = {}
    for episode_id in episode_ids:
        groups.setdefault(episode_rows[episode_id], []).append(episode_id)

    def allocate(candidates: set[str], count: int, label: str) -> set[str]:
        if count <= 0:
            return set()
        active_groups = {
            key: [episode_id for episode_id in values if episode_id in candidates]
            for key, values in groups.items()
        }
        total = sum(len(values) for values in active_groups.values())
        quotas = {key: len(values) * count / max(1, total) for key, values in active_groups.items()}
        allocated = {key: int(value) for key, value in quotas.items()}
        remaining = count - sum(allocated.values())
        order = sorted(
            active_groups,
            key=lambda key: (quotas[key] - allocated[key], len(active_groups[key]), repr(key)),
            reverse=True,
        )
        for key in order[:remaining]:
            allocated[key] += 1
        selected: set[str] = set()
        for key, values in active_groups.items():
            ranked = sorted(
                values,
                key=lambda episode_id: hashlib.sha256(
                    f"{seed}:{label}:{episode_id}".encode("utf-8")
                ).digest(),
            )
            selected.update(ranked[:allocated[key]])
        return selected

    all_ids = set(episode_ids)
    test_ids = allocate(all_ids, test_count, "test")
    validation_ids = allocate(all_ids - test_ids, validation_count, "validation")
    train = [decision for decision in decisions if decision.episode_id not in validation_ids | test_ids]
    validation = [decision for decision in decisions if decision.episode_id in validation_ids]
    test = [decision for decision in decisions if decision.episode_id in test_ids]
    return train, validation, test


def predict_model(model: dict[str, Any], features: list[float]) -> float:
    return _predict_catboost(model, features)


def evaluate_model(model: dict[str, Any], decisions: list[Decision]) -> dict[str, Any]:
    all_features: list[list[float]] = []
    offsets: list[int] = []
    for decision in decisions:
        offsets.append(len(decision.features))
        all_features.extend(decision.features)
    all_scores = predict_batch(model, all_features) if all_features else []
    top1 = top3 = selected_total = 0
    by_source: dict[str, dict[str, int]] = {}
    cursor = 0
    for decision, count in zip(decisions, offsets):
        scores = all_scores[cursor:cursor + count]
        cursor += count
        if not scores:
            continue
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        selected = {index for index, label in enumerate(decision.labels) if label}
        top1 += int(order[0] in selected)
        top3 += int(bool(selected.intersection(order[:3])))
        selected_total += 1
        row = by_source.setdefault(decision.source, {"decisions": 0, "top1": 0, "top3": 0})
        row["decisions"] += 1
        row["top1"] += int(order[0] in selected)
        row["top3"] += int(bool(selected.intersection(order[:3])))
    for row in by_source.values():
        count = max(1, row["decisions"])
        row["top1_rate"] = row["top1"] / count
        row["top3_rate"] = row["top3"] / count
    return {
        "decisions": selected_total,
        "top1": top1 / max(1, selected_total),
        "top3": top3 / max(1, selected_total),
        "by_source": by_source,
    }


def evaluate_hierarchical_main(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    decisions: list[Decision],
    type_weight: float,
    option_weight: float,
) -> dict[str, Any]:
    main_decisions = [decision for decision in decisions if decision.head == "main"]
    if not main_decisions:
        return {"decisions": 0, "top1": 0.0, "top3": 0.0}
    flattened = [features for decision in main_decisions for features in decision.features]
    global_flat = predict_batch(global_model, flattened)
    type_model = head_models.get("main_type")
    aggregated_rows: list[list[float]] = []
    aggregated_layout: list[list[int]] = []
    if type_model:
        base_cursor = 0
        for decision in main_decisions:
            base_scores = global_flat[base_cursor:base_cursor + len(decision.features)]
            base_cursor += len(decision.features)
            rows, option_types = aggregate_option_type_features(
                decision.features,
                decision.option_types,
                base_scores,
            )
            aggregated_rows.extend(rows)
            aggregated_layout.append(option_types)
        aggregated_scores = predict_batch(type_model, aggregated_rows)
    else:
        aggregated_layout = [[] for _ in main_decisions]
        aggregated_scores = []
    option_scores: dict[str, list[float]] = {}
    option_positions: dict[str, list[tuple[int, int]]] = {}
    for decision_index, decision in enumerate(main_decisions):
        for option_index, (features, option_type) in enumerate(zip(decision.features, decision.option_types)):
            head = MAIN_TYPE_HEADS.get(option_type)
            if head in head_models:
                option_scores.setdefault(head, []).append(features)
                option_positions.setdefault(head, []).append((decision_index, option_index))
    within = [[0.0] * len(decision.features) for decision in main_decisions]
    for head, rows in option_scores.items():
        for (decision_index, option_index), score in zip(option_positions[head], predict_batch(head_models[head], rows)):
            within[decision_index][option_index] = score
    top1 = top3 = cursor = type_cursor = 0
    by_type: dict[str, dict[str, int]] = {}
    for decision_index, decision in enumerate(main_decisions):
        count = len(decision.features)
        scores = global_flat[cursor:cursor + count]
        cursor += count
        type_scores = {}
        present_types = aggregated_layout[decision_index]
        if type_model:
            values = aggregated_scores[type_cursor:type_cursor + len(present_types)]
            type_cursor += len(present_types)
            type_scores = dict(zip(present_types, values))
        combined = [
            score
            + type_weight * type_scores.get(option_type, 0.0)
            + option_weight * within[decision_index][option_index]
            for option_index, (score, option_type) in enumerate(zip(scores, decision.option_types))
        ]
        order = sorted(range(count), key=lambda index: combined[index], reverse=True)
        selected = {index for index, label in enumerate(decision.labels) if label}
        top1_match = int(order[0] in selected)
        top1 += top1_match
        top3 += int(bool(selected.intersection(order[:3])))
        selected_type = next(
            (option_type for option_type, label in zip(decision.option_types, decision.labels) if label),
            -1,
        )
        type_name = OptionType(selected_type).name.lower() if selected_type >= 0 else "unknown"
        row = by_type.setdefault(type_name, {"decisions": 0, "top1": 0})
        row["decisions"] += 1
        row["top1"] += top1_match
    total = len(main_decisions)
    return {
        "decisions": total,
        "top1": top1 / total,
        "top3": top3 / total,
        "by_type": {
            name: {**row, "recall": row["top1"] / max(1, row["decisions"])}
            for name, row in by_type.items()
        },
    }


def _calibration_score(metrics: dict[str, Any]) -> tuple[float, ...]:
    by_type = metrics.get("by_type", {})
    recall = lambda name, default=1.0: float((by_type.get(name) or {}).get("recall", default))
    critical_pass = float(
        recall("attack") >= 0.85
        and recall("ability") >= 0.85
        and min(recall("retreat"), recall("end")) >= 0.70
    )
    present = [float(row.get("recall", 0.0)) for row in by_type.values() if int(row.get("decisions", 0)) > 0]
    macro = sum(present) / max(1, len(present))
    return critical_pass, macro, float(metrics["top1"]), float(metrics["top3"])


def calibrate_main_weights(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    validation_decisions: list[Decision],
) -> tuple[dict[str, float], dict[str, Any]]:
    best_weights = {"main_type_weight": 0.0, "main_option_weight": 0.0}
    best_metrics = evaluate_hierarchical_main(global_model, head_models, validation_decisions, 0.0, 0.0)
    for type_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        for option_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
            metrics = evaluate_hierarchical_main(
                global_model, head_models, validation_decisions, type_weight, option_weight
            )
            if (*_calibration_score(metrics), -type_weight - option_weight) > (
                *_calibration_score(best_metrics),
                -best_weights["main_type_weight"] - best_weights["main_option_weight"],
            ):
                best_weights = {"main_type_weight": type_weight, "main_option_weight": option_weight}
                best_metrics = metrics
    return best_weights, best_metrics


def _select_with_margin(scores: list[float], min_count: int, max_count: int, margin: float) -> set[int]:
    if not scores or max_count <= 0:
        return set()
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    top = scores[order[0]]
    selected = [index for index in order if top - scores[index] <= margin][:max_count]
    if len(selected) < min_count:
        selected = order[:min(min_count, max_count)]
    return set(selected)


def calibrate_selection_margins(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    validation_decisions: list[Decision],
) -> dict[str, float]:
    grouped: dict[str, list[Decision]] = {}
    for decision in validation_decisions:
        if decision.max_count <= decision.min_count or not decision.features:
            continue
        context = int(decision.features[0][0])
        grouped.setdefault(f"{decision.head}:{context}", []).append(decision)
    thresholds: dict[str, float] = {}
    for key, decisions in grouped.items():
        if len(decisions) < 10:
            continue
        head = decisions[0].head
        model = head_models.get(head, global_model)
        flattened = [features for decision in decisions for features in decision.features]
        flat_scores = predict_batch(model, flattened)
        layouts = []
        gaps = [0.0]
        cursor = 0
        for decision in decisions:
            scores = flat_scores[cursor:cursor + len(decision.features)]
            cursor += len(decision.features)
            layouts.append(scores)
            if scores:
                top = max(scores)
                gaps.extend(max(0.0, top - score) for score in scores)
        candidates = sorted(set(float(value) for value in np.quantile(gaps, np.linspace(0.0, 1.0, 41))))
        best = (float("-inf"), float("-inf"), 0.0)
        for margin in candidates:
            exact = overlap = 0.0
            for decision, scores in zip(decisions, layouts):
                predicted = _select_with_margin(scores, decision.min_count, decision.max_count, margin)
                actual = {index for index, label in enumerate(decision.labels) if label}
                exact += float(predicted == actual)
                overlap += len(predicted & actual) / max(1, len(predicted | actual))
            score = (exact / len(decisions), overlap / len(decisions), -margin)
            if score > best:
                best = score
                thresholds[key] = margin
    return thresholds


def evaluate_runtime_policy(
    global_model: dict[str, Any],
    head_models: dict[str, dict[str, Any]],
    decisions: list[Decision],
    calibration: dict[str, float],
) -> dict[str, Any]:
    main_metrics = evaluate_hierarchical_main(
        head_models.get("main", global_model),
        head_models,
        decisions,
        calibration["main_type_weight"],
        calibration["main_option_weight"],
    )
    totals = {
        "decisions": int(main_metrics.get("decisions", 0)),
        "top1_count": float(main_metrics.get("top1", 0.0)) * int(main_metrics.get("decisions", 0)),
        "top3_count": float(main_metrics.get("top3", 0.0)) * int(main_metrics.get("decisions", 0)),
    }
    by_head = {"main": main_metrics}
    for head in sorted({decision.head for decision in decisions if decision.head != "main"}):
        rows = [decision for decision in decisions if decision.head == head]
        metrics = evaluate_model(head_models.get(head, global_model), rows)
        count = int(metrics.get("decisions", 0))
        totals["decisions"] += count
        totals["top1_count"] += float(metrics.get("top1", 0.0)) * count
        totals["top3_count"] += float(metrics.get("top3", 0.0)) * count
        by_head[head] = metrics
    count = max(1, totals["decisions"])
    return {
        "decisions": totals["decisions"],
        "top1": totals["top1_count"] / count,
        "top3": totals["top3_count"] / count,
        "by_head": by_head,
    }


def derive_generic_policy(decisions: list[Decision]) -> dict[str, dict[str, float]]:
    card_counts: Counter[int] = Counter()
    attack_counts: Counter[int] = Counter()
    for decision in decisions:
        for features, label in zip(decision.features, decision.labels):
            if not label:
                continue
            card_id = int(features[2])
            attack_id = int(features[3])
            if card_id > 0:
                card_counts[card_id] += 1
            if attack_id > 0:
                attack_counts[attack_id] += 1

    def scores(counts: Counter[int], high: float, step: float, limit: int = 32) -> dict[str, float]:
        return {str(value): max(100.0, high - index * step) for index, (value, _) in enumerate(counts.most_common(limit))}

    cards = scores(card_counts, 1400.0, 35.0)
    return {
        "setup_active_priority": dict(list(cards.items())[:12]),
        "setup_bench_priority": dict(list(cards.items())[:16]),
        "play_priority": cards,
        "search_priority": cards,
        "keep_priority": cards,
        "attach_priority": dict(list(cards.items())[:16]),
        "active_priority": dict(list(cards.items())[:12]),
        "board_priority": dict(list(cards.items())[:16]),
        "attack_priority": scores(attack_counts, 1400.0, 60.0, 16),
    }


def confidence_threshold(
    model: dict[str, Any],
    decisions: list[Decision],
    target_precision: float = 0.80,
    minimum_samples: int = 10,
) -> float:
    all_features: list[list[float]] = []
    offsets: list[int] = []
    for decision in decisions:
        offsets.append(len(decision.features))
        all_features.extend(decision.features)
    all_scores = predict_batch(model, all_features) if all_features else []
    rows: list[tuple[float, int]] = []
    cursor = 0
    for decision, count in zip(decisions, offsets):
        scores = all_scores[cursor:cursor + count]
        cursor += count
        order = sorted(range(len(scores)), key=lambda option_index: scores[option_index], reverse=True)
        if not order:
            continue
        selected = {index for index, label in enumerate(decision.labels) if label}
        margin = scores[order[0]] - scores[order[1]] if len(order) > 1 else float("inf")
        rows.append((margin, int(order[0] in selected)))
    rows.sort(reverse=True)
    correct = 0
    threshold = float("inf")
    for index, (margin, is_correct) in enumerate(rows, 1):
        correct += is_correct
        if index >= minimum_samples and correct / index >= target_precision:
            threshold = margin
    return threshold if threshold != float("inf") else 0.0


def train_ranker(
    decisions: list[Decision],
    manifest_data: dict[str, Any],
    initial_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decisions = rankable_decisions(decisions)
    if not decisions:
        raise ValueError("no rankable decisions")
    features, labels, groups, weights = flatten(
        decisions,
        float(manifest_data.get("win_weight", 1.5)),
        float(manifest_data.get("loss_weight", 0.15)),
    )
    baseline_scores = predict_batch(initial_model, features) if initial_model else None
    residual = _train_catboost(features, labels, groups, weights, manifest_data, baseline_scores)
    return merge_catboost_models(initial_model, residual) if initial_model else residual


def load_manifest(path: Path) -> tuple[str, list[SourceSpec], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = [
        SourceSpec(
            name=str(row["name"]),
            path=Path(row["path"]),
            target_name=str(row["target_name"]),
            split=str(row.get("split", "train")),
            weight=float(row.get("weight", 1.0)),
        )
        for row in data.get("sources", [])
    ]
    if not sources:
        raise ValueError("manifest requires at least one source")
    return str(data["family"]), sources, data


def train(manifest: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        print(f"[train] artifact exists, skipping: {out}", flush=True)
        return json.loads(out.read_text(encoding="utf-8"))
    family, sources, manifest_data = load_manifest(manifest)
    shared_artifact = None
    shared_path = manifest_data.get("shared_init_artifact")
    if shared_path:
        shared_artifact = json.loads(Path(shared_path).read_text(encoding="utf-8"))
        if shared_artifact.get("feature_schema_sha256") != _feature_schema_sha256():
            raise ValueError("shared_init_feature_schema_mismatch")

    def initial_model(name: str) -> dict[str, Any] | None:
        if not shared_artifact:
            return None
        if bool(manifest_data.get("low_level_shared_only", False)) and name in {
            "exported", "main", "main_type", "main_stop", "main_continue_type", "main_terminal_type", "value",
        }:
            return None
        shared_loss = str((shared_artifact.get("effective_params") or {}).get("loss_function", "YetiRank"))
        target_loss = str(manifest_data.get("loss_function", "YetiRank"))
        if shared_loss != target_loss:
            return None
        if name == "exported":
            return shared_artifact.get("model")
        if name == "value":
            return shared_artifact.get("value_model")
        return (shared_artifact.get("head_models") or {}).get(name)
    parse_workers = int(manifest_data.get("parse_workers", 1))
    cache_dir = Path(manifest_data["cache_dir"]) if manifest_data.get("cache_dir") else None
    parts_dir = out.parent / f"{out.stem}_parts"
    train_decisions: list[Decision] = []
    validation_decisions: list[Decision] = []
    test_decisions: list[Decision] = []
    source_stats = {}
    source_fingerprints = {}
    decks: Counter[tuple[int, ...]] = Counter()
    opponent_decks: Counter[tuple[int, ...]] = Counter()
    required_deck: tuple[int, ...] | None = None
    for source in sources:
        content_cache_key = str(source.path.resolve())
        with _replay_content_cache_lock:
            source_fingerprint = _replay_content_cache.get(content_cache_key)
        if source_fingerprint is None:
            computed_fingerprint = replay_content_fingerprint(source.path, workers=parse_workers)
            with _replay_content_cache_lock:
                source_fingerprint = _replay_content_cache.setdefault(content_cache_key, computed_fingerprint)
        source_fingerprints[source.name] = source_fingerprint
        cache_key = (str(source.path), source.target_name)
        with _load_source_cache_lock:
            cached = _load_source_cache.get(cache_key)
        if cached is None:
            loaded = load_source(source, parse_workers=parse_workers, cache_dir=cache_dir)
            with _load_source_cache_lock:
                cached = _load_source_cache.setdefault(cache_key, loaded)
        decisions, deck, stats = cached
        if required_deck is None:
            required_deck = deck
        elif bool(manifest_data.get("require_same_deck_sources", False)) and deck != required_deck:
            source_stats[source.name] = {
                **dict(stats),
                "split": source.split,
                "weight": source.weight,
                "status": "skipped_deck_mismatch",
            }
            print(f"[train] skip source deck mismatch: {source.name}", flush=True)
            continue
        stats_copy = dict(stats)
        for opponent_deck in stats_copy.pop("opponent_decks", []):
            opponent_decks[tuple(map(int, opponent_deck))] += 1
        source_stats[source.name] = {**stats_copy, "split": source.split, "weight": source.weight}
        decks[deck] += stats_copy["resolved"]
        weighted_decisions = [replace(decision, weight=decision.weight * source.weight) for decision in decisions]
        if source.split == "validation":
            validation_decisions.extend(weighted_decisions)
        elif source.split == "test":
            test_decisions.extend(weighted_decisions)
        else:
            train_decisions.extend(weighted_decisions)
    if not train_decisions:
        raise ValueError("no train decisions extracted")
    if not validation_decisions:
        train_decisions, validation_decisions, automatic_test = split_by_episode(
            train_decisions,
            float(manifest_data.get("validation_fraction", 0.15)),
            float(manifest_data.get("test_fraction", 0.15)),
            int(manifest_data.get("split_seed", manifest_data.get("seed", 0))),
        )
        if not test_decisions:
            test_decisions = automatic_test
    if manifest_data.get("frozen_dataset"):
        verify_frozen_dataset(
            Path(manifest_data["frozen_dataset"]),
            source_fingerprints,
            train_decisions,
            validation_decisions,
            test_decisions,
        )
    exported = _load_model_part(parts_dir, "exported")
    if exported is None:
        exported = train_ranker(train_decisions, manifest_data, initial_model("exported"))
        _save_model_part(parts_dir, "exported", exported)
        print("[train] exported model trained", flush=True)
    else:
        print("[train] exported model loaded from parts", flush=True)
    value_manifest = {
        **manifest_data,
        "win_weight": float(manifest_data.get("value_win_weight", 2.0)),
        "loss_weight": float(manifest_data.get("value_loss_weight", 0.05)),
        "n_estimators": int(manifest_data.get("value_n_estimators", min(240, int(manifest_data.get("n_estimators", 180))))),
        "num_leaves": int(manifest_data.get("value_num_leaves", min(31, int(manifest_data.get("num_leaves", 31))))),
    }
    value_model = _load_model_part(parts_dir, "value")
    if value_model is None:
        value_model = train_ranker(train_decisions, value_manifest, initial_model("value"))
        _save_model_part(parts_dir, "value", value_model)
        print("[train] value model trained", flush=True)
    else:
        print("[train] value model loaded from parts", flush=True)
    minimum_head_decisions = int(manifest_data.get("minimum_head_decisions", 30))
    fast_finalize = bool(manifest_data.get("fast_finalize", False))

    def metric_pair(model: dict[str, Any], train_rows: list[Decision], validation_rows: list[Decision]) -> dict[str, Any]:
        if fast_finalize:
            return {"train": {}, "validation": {}}
        return {
            "train": evaluate_model(model, train_rows),
            "validation": evaluate_model(model, validation_rows),
        }

    head_models: dict[str, dict[str, Any]] = {}
    head_metrics: dict[str, dict[str, Any]] = {}
    weight_audit: dict[str, Any] = {}
    confidence: dict[str, float] = {
        "global": 0.0
        if fast_finalize
        else confidence_threshold(
            exported,
            validation_decisions,
            float(manifest_data.get("confidence_precision", 0.80)),
        )
    }
    for head in sorted({decision.head for decision in train_decisions}):
        head_train = [decision for decision in train_decisions if decision.head == head]
        head_validation = [decision for decision in validation_decisions if decision.head == head]
        head_label_values = {label for decision in head_train for label in decision.labels}
        if len(head_label_values) < 2:
            print(f"[train] skip head {head}: constant labels", flush=True)
            continue
        if len(head_train) < minimum_head_decisions or not head_validation:
            continue
        head_manifest = {
            **manifest_data,
            "n_estimators": int(manifest_data.get("head_n_estimators", min(180, int(manifest_data.get("n_estimators", 180))))),
            "num_leaves": int(manifest_data.get("head_num_leaves", min(31, int(manifest_data.get("num_leaves", 31))))),
        }
        head_model = _load_model_part(parts_dir, f"head_{head}")
        if head_model is None:
            head_model = train_ranker(head_train, head_manifest, initial_model(head))
            _save_model_part(parts_dir, f"head_{head}", head_model)
            print(f"[train] head model trained: {head}", flush=True)
        else:
            print(f"[train] head model loaded from parts: {head}", flush=True)
        head_models[head] = head_model
        head_metrics[head] = metric_pair(head_model, head_train, head_validation)
        confidence[head] = (
            0.0
            if fast_finalize
            else confidence_threshold(
                head_model,
                head_validation,
                float(manifest_data.get("confidence_precision", 0.80)),
            )
        )
    if fast_finalize and bool(manifest_data.get("enable_main_type_heads", False)):
        for head in ("main_type", *MAIN_TYPE_OPTIONS):
            part = _load_model_part(parts_dir, f"head_{head}")
            if part is not None:
                head_models[head] = part
                head_metrics[head] = {"train": {}, "validation": {}}
                confidence[head] = 0.0
                print(f"[train] fast-finalize type head loaded: {head}", flush=True)
    elif bool(manifest_data.get("enable_main_type_heads", False)):
        main_train = [decision for decision in train_decisions if decision.head == "main"]
        main_validation = [decision for decision in validation_decisions if decision.head == "main"]
        main_base_model = head_models.get("main", exported)
        router_feature_mode = str(manifest_data.get("router_feature_mode", "compact_v1"))
        raw_type_train = [
            aggregated
            for decision in main_train
            if (aggregated := _aggregate_decision_by_type(decision, main_base_model, router_feature_mode))
        ]
        type_validation = [
            aggregated
            for decision in main_validation
            if (aggregated := _aggregate_decision_by_type(decision, main_base_model, router_feature_mode))
        ]
        stop_type_train = [
            aggregated
            for decision in main_train
            if (aggregated := _aggregate_decision_by_type(decision, main_base_model, "full"))
        ]
        stop_type_validation = [
            aggregated
            for decision in main_validation
            if (aggregated := _aggregate_decision_by_type(decision, main_base_model, "full"))
        ]
        type_train = balance_type_decisions(
            raw_type_train,
            float(manifest_data.get("type_balance_max_weight", 6.0)),
        )
        if len(type_train) >= minimum_head_decisions and type_validation:
            type_manifest = {
                **manifest_data,
                "n_estimators": int(manifest_data.get("head_n_estimators", min(180, int(manifest_data.get("n_estimators", 180))))),
                "num_leaves": int(manifest_data.get("head_num_leaves", min(31, int(manifest_data.get("num_leaves", 31))))),
            }
            type_model = _load_model_part(parts_dir, "head_main_type")
            if type_model is None:
                type_model = train_ranker(type_train, type_manifest, initial_model("main_type"))
                _save_model_part(parts_dir, "head_main_type", type_model)
                print("[train] type head trained: main_type", flush=True)
            else:
                print("[train] type head loaded from parts: main_type", flush=True)
            head_models["main_type"] = type_model
            head_metrics["main_type"] = metric_pair(type_model, type_train, type_validation)
            confidence["main_type"] = (
                0.0
                if fast_finalize
                else confidence_threshold(
                    type_model,
                    type_validation,
                    float(manifest_data.get("confidence_precision", 0.80)),
                )
            )
        if bool(manifest_data.get("enable_main_router", manifest_data.get("enable_hard_main_router", False))):
            router_manifest = {
                **manifest_data,
                "n_estimators": int(manifest_data.get("router_n_estimators", 450)),
                "num_leaves": int(manifest_data.get("router_num_leaves", 31)),
                "min_child_samples": int(manifest_data.get("router_min_child_samples", 32)),
                "reg_lambda": float(manifest_data.get("router_l2_leaf_reg", 8.0)),
            }
            stop_train_rows = _stop_classifier_rows(stop_type_train)
            stop_validation_rows = _stop_classifier_rows(stop_type_validation)
            if len(stop_train_rows) >= minimum_head_decisions and stop_validation_rows:
                ensemble_size = int(manifest_data.get("stop_ensemble_size", 3))
                base_seed = int(manifest_data.get("seed", 20260731))
                if int(manifest_data.get("router_schema_version", 1)) >= STOP_ROUTER_SCHEMA_VERSION:
                    oof_audit = {}
                    ablation_audit = {}
                    ablation_modes = {}
                    final_stop_train_rows = []
                    for regime in STOP_REGIMES:
                        regime_train = [row for row in stop_train_rows if stop_regime(row[2]) == regime]
                        regime_validation = [row for row in stop_validation_rows if stop_regime(row[2]) == regime]
                        if len(regime_train) < minimum_head_decisions or not regime_validation:
                            raise ValueError(f"insufficient stop rows for regime={regime}: train={len(regime_train)} validation={len(regime_validation)}")
                        if len({label for _, label, _ in regime_train}) < 2:
                            raise ValueError(f"stop regime has constant labels: regime={regime}")
                        ablation_mode, ablation_audit[regime] = select_stop_ablation(regime_train, router_manifest)
                        ablation_modes[regime] = ablation_mode
                        regime_train = apply_stop_ablation_rows(regime_train, ablation_mode)
                        regime_validation = apply_stop_ablation_rows(regime_validation, ablation_mode)
                        regime_train, oof_audit[regime] = mine_oof_hard_continue(regime_train, router_manifest)
                        final_stop_train_rows.extend(regime_train)
                        parts = []
                        for member in range(ensemble_size):
                            member_seed = base_seed + 1009 * member
                            head_name = f"main_stop_{regime}_seed{member_seed}"
                            part_name = f"head_{head_name}"
                            member_part = _load_model_part(parts_dir, part_name)
                            if member_part is None:
                                member_part = train_stop_classifier(regime_train, router_manifest, seed=member_seed)
                                _save_model_part(parts_dir, part_name, member_part)
                                print(f"[train] router classifier trained: {head_name}", flush=True)
                            head_models[head_name] = member_part
                            parts.append(member_part)
                        ensemble = average_catboost_models(parts)
                        head_name = f"main_stop_{regime}"
                        head_models[head_name] = ensemble
                        head_metrics[head_name] = {
                            "train_stop": evaluate_stop_classifier(ensemble, regime_train),
                            "validation_stop": evaluate_stop_classifier(ensemble, regime_validation),
                            "ensemble_size": ensemble_size,
                        }
                    weight_audit = stop_weight_audit(final_stop_train_rows, manifest_data)
                    weight_audit["oof_hard_continue"] = oof_audit
                    weight_audit["ablation"] = ablation_audit
                    manifest_data["stop_ablation_modes"] = ablation_modes
                else:
                    parts = []
                    for member in range(ensemble_size):
                        member_seed = base_seed + 1009 * member
                        part_name = f"head_main_stop_seed{member_seed}"
                        member_part = _load_model_part(parts_dir, part_name)
                        if member_part is None:
                            member_part = train_stop_classifier(stop_train_rows, router_manifest, seed=member_seed)
                            _save_model_part(parts_dir, part_name, member_part)
                        parts.append(member_part)
                    head_models["main_stop"] = average_catboost_models(parts)
                    head_metrics["main_stop"] = {
                        "train_stop": evaluate_stop_classifier(head_models["main_stop"], stop_train_rows),
                        "validation_stop": evaluate_stop_classifier(head_models["main_stop"], stop_validation_rows),
                        "ensemble_size": ensemble_size,
                    }
                    weight_audit = stage_weight_audit(
                        [decision for _, _, decision in stop_train_rows],
                        float(manifest_data.get("win_weight", 1.5)),
                        float(manifest_data.get("loss_weight", 0.15)),
                    )
            for head, rows in (
                ("main_continue_type", [row for decision in raw_type_train if (row := _stage_type_decision(decision, False))]),
                ("main_terminal_type", [row for decision in raw_type_train if (row := _stage_type_decision(decision, True))]),
            ):
                validation_rows = []
                for decision in type_validation:
                    row = _stage_type_decision(decision, head == "main_terminal_type")
                    if row is not None:
                        validation_rows.append(row)
                if len(rows) < minimum_head_decisions or not validation_rows:
                    print(f"[train] skip router head {head}: samples={len(rows)}", flush=True)
                    continue
                part_name = f"head_{head}"
                part = _load_model_part(parts_dir, part_name)
                if part is None:
                    part = train_ranker(rows, router_manifest, None)
                    _save_model_part(parts_dir, part_name, part)
                    print(f"[train] router head trained: {head}", flush=True)
                head_models[head] = part
                head_metrics[head] = metric_pair(part, rows, validation_rows)
        for head, wanted_type in MAIN_TYPE_OPTIONS.items():
            head_train = [
                filtered
                for decision in main_train
                if _type_selected(decision, wanted_type)
                for filtered in [_filter_decision_by_type(decision, wanted_type)]
                if filtered is not None
            ]
            head_validation = [
                filtered
                for decision in main_validation
                if _type_selected(decision, wanted_type)
                for filtered in [_filter_decision_by_type(decision, wanted_type)]
                if filtered is not None
            ]
            head_label_values = {label for decision in head_train for label in decision.labels}
            if len(head_label_values) < 2 or len(head_train) < minimum_head_decisions or not head_validation:
                print(f"[train] skip type head {head}: samples={len(head_train)}", flush=True)
                continue
            head_manifest = {
                **manifest_data,
                "n_estimators": int(manifest_data.get("head_n_estimators", min(180, int(manifest_data.get("n_estimators", 180))))),
                "num_leaves": int(manifest_data.get("head_num_leaves", min(31, int(manifest_data.get("num_leaves", 31))))),
            }
            head_model = _load_model_part(parts_dir, f"head_{head}")
            if head_model is None:
                head_model = train_ranker(head_train, head_manifest, initial_model(head))
                _save_model_part(parts_dir, f"head_{head}", head_model)
                print(f"[train] type head trained: {head}", flush=True)
            else:
                print(f"[train] type head loaded from parts: {head}", flush=True)
            head_models[head] = head_model
            head_metrics[head] = metric_pair(head_model, head_train, head_validation)
            confidence[head] = (
                0.0
                if fast_finalize
                else confidence_threshold(
                    head_model,
                    head_validation,
                    float(manifest_data.get("confidence_precision", 0.80)),
                )
            )
    main_base_model = head_models.get("main", exported)
    router_enabled = bool(manifest_data.get("enable_main_router", manifest_data.get("enable_hard_main_router", False))) and (
        "main_stop" in head_models or all(f"main_stop_{regime}" in head_models for regime in STOP_REGIMES)
    )
    hard_router = bool(manifest_data.get("enable_hard_main_router", False)) and "main_stop" in head_models
    if router_enabled:
        calibration = {"main_type_weight": 0.0, "main_option_weight": 1.0}
        hierarchical_validation = {}
        hierarchical_test = {}
    else:
        calibration, hierarchical_validation = calibrate_main_weights(main_base_model, head_models, validation_decisions)
        hierarchical_test = evaluate_hierarchical_main(
            main_base_model,
            head_models,
            test_decisions,
            calibration["main_type_weight"],
            calibration["main_option_weight"],
        ) if test_decisions else {}
    hard_validation = evaluate_hard_main(main_base_model, head_models, validation_decisions) if hard_router else {}
    hard_test = evaluate_hard_main(main_base_model, head_models, test_decisions) if hard_router and test_decisions else {}
    hard_temporal = temporal_hard_main_metrics(
        main_base_model,
        head_models,
        [*validation_decisions, *test_decisions],
    ) if hard_router else []
    router_feature_mode = str(manifest_data.get("router_feature_mode", "compact_v1"))
    if router_enabled:
        router_calibration, runtime_validation = calibrate_soft_router(
            main_base_model, head_models, validation_decisions, router_feature_mode, manifest_data
        )
        enforce_stop_gates(runtime_validation, manifest_data, "validation")
        runtime_test = evaluate_soft_main_layouts(
            _soft_router_layouts(
                main_base_model,
                head_models,
                test_decisions,
                router_feature_mode,
                dict(manifest_data.get("stop_ablation_modes", {})),
            ),
            router_calibration,
        ) if test_decisions else {}
        if test_decisions and bool(manifest_data.get("stop_require_test_gate", False)):
            enforce_stop_gates(runtime_test, manifest_data, "test")
    else:
        router_calibration = {}
        runtime_validation = evaluate_runtime_policy(exported, head_models, validation_decisions, calibration)
        runtime_test = evaluate_runtime_policy(exported, head_models, test_decisions, calibration) if test_decisions else {}
    selection_margins = calibrate_selection_margins(exported, head_models, validation_decisions)
    artifact = {
        "version": 7 if router_enabled else 6,
        "clone_algorithm_version": int(manifest_data.get("clone_algorithm_version", 0)),
        "family": family,
        "features": list(FEATURE_NAMES),
        "type_features": list(TYPE_FEATURE_NAMES),
        "router_features": list(ROUTER_FEATURE_NAMES),
        "feature_schema_sha256": _feature_schema_sha256(),
        "router_feature_schema_sha256": hashlib.sha256("\n".join(ROUTER_FEATURE_NAMES).encode("utf-8")).hexdigest(),
        "model": exported,
        "value_model": value_model,
        "head_models": head_models,
        "calibration": calibration,
        "router_calibration": router_calibration,
        "weight_audit": weight_audit,
        "selection_margin_thresholds": selection_margins,
        "hierarchical_main_metrics": {
            "validation": hierarchical_validation,
            "test": hierarchical_test,
        },
        "hard_main_metrics": {"validation": hard_validation, "test": hard_test, "temporal_folds": hard_temporal},
        "runtime_policy_metrics": {
            "validation": runtime_validation,
            "test": runtime_test,
        },
        "effective_params": {
            "model_family": "catboost",
            "task_type": str(manifest_data.get("task_type", "CPU")),
            "devices": str(manifest_data.get("devices", "") or ""),
            "thread_count": int(manifest_data.get("n_jobs", 12)),
            "loss_function": str(manifest_data.get("loss_function", "YetiRank")),
            "eval_metric": str(manifest_data.get("eval_metric", "NDCG:top=3")),
            "metric_period": 50,
            "num_leaves": int(manifest_data.get("num_leaves", 31)),
            "depth": _catboost_depth(int(manifest_data.get("num_leaves", 31))),
            "rsm": None if str(manifest_data.get("task_type", "CPU")).upper() == "GPU" else manifest_data.get("colsample_bytree"),
            "subsample": None,
            "parse_workers": int(manifest_data.get("parse_workers", 1)),
            "seed": int(manifest_data.get("seed", 20260731)),
            "head_num_leaves": int(manifest_data.get("head_num_leaves", min(31, int(manifest_data.get("num_leaves", 31))))),
            "type_balance_max_weight": float(manifest_data.get("type_balance_max_weight", 6.0)),
            "router_feature_mode": router_feature_mode,
            "win_weight": float(manifest_data.get("win_weight", 1.5)),
            "loss_weight": float(manifest_data.get("loss_weight", 0.15)),
        },
        "confidence_thresholds": confidence,
        "deck": list(decks.most_common(1)[0][0]),
        "deck_sha256": hashlib.sha256("\n".join(map(str, decks.most_common(1)[0][0])).encode("utf-8")).hexdigest(),
        "opponent_decks": [list(deck) for deck, _ in opponent_decks.most_common(16)],
        "source_stats": source_stats,
        "dataset_fingerprint": {
            "sources": source_fingerprints,
            "splits": split_fingerprint(train_decisions, validation_decisions, test_decisions),
        },
        "shared_init_artifact": str(shared_path) if shared_path else None,
        "train_metrics": {} if fast_finalize else evaluate_model(exported, train_decisions),
        "validation_metrics": {} if fast_finalize else evaluate_model(exported, validation_decisions),
        "test_metrics": {} if fast_finalize else evaluate_model(exported, test_decisions) if test_decisions else {},
        "head_metrics": head_metrics,
        "split_stats": {
            "train_episodes": sorted({decision.episode_id for decision in train_decisions}),
            "validation_episodes": sorted({decision.episode_id for decision in validation_decisions}),
            "test_episodes": sorted({decision.episode_id for decision in test_decisions}),
            "train_decisions": len(train_decisions),
            "validation_decisions": len(validation_decisions),
            "test_decisions": len(test_decisions),
        },
        "manifest": manifest_data,
        "generic_policy": manifest_data.get("generic_policy", derive_generic_policy(train_decisions)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a compact option-ranking model from public Kaggle replays.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = train(args.manifest, args.out)
    print(json.dumps({
        "family": artifact["family"],
        "deck_cards": len(artifact["deck"]),
        "train": artifact["train_metrics"],
        "validation": artifact["validation_metrics"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
