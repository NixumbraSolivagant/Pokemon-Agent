from __future__ import annotations

import io
import base64
import gzip
import json
import subprocess
import sys
import pytest
import tarfile
from pathlib import Path
from types import SimpleNamespace

from agents.meta_runtime import (
    BUDDY_BUDDY_POFFIN,
    DAWN,
    FEATURE_NAMES,
    ROUTER_FEATURE_NAMES,
    TYPE_FEATURE_NAMES,
    FROSLASS,
    IMPIDIMP,
    POKE_PAD,
    TEAM_ROCKETS_PETREL,
    _POLICY_MEMORY,
    _POLICY_ROUTER_MEMORY,
    _record_policy_action,
    _reset_policy_memory,
    _reset_policy_router,
    _select_policy_config,
    _LOPUNNY_MEMORY,
    _update_lopunny_memory,
    _grim_search_score,
    _ogerpon_attack_damage,
    _sample_hidden_worlds,
    ensure_mandatory_minimums,
    fallback_score,
    forced_action,
    choose_from_scores,
    model_score,
    model_head,
    resolve_model,
    score_options,
    score_main_options_v7,
    aggregate_option_type_features,
    advantage_override,
    search_action,
    state_value,
)
from cg.api import AreaType, OptionType, SelectContext
from tools.build_meta_submission import build
from tools.build_meta_submission import family_config
from tools.great_tusk_candidates import inspect_package
from tools.ogerpon_cem_optimize import PARAMETERS, centered_distribution, sample_vector, update_distribution
from tools.ogerpon_residual_train import split_rows as split_counterfactual_rows
from tools.ogerpon_advantage_train import split_rows as split_advantage_rows
from tools import replay_imitation
from tools.replay_imitation import (
    Decision,
    _binary_stage_decision,
    _feasible_stop_thresholds,
    _joint_threshold_candidates,
    _stop_gate_deficit,
    average_catboost_models,
    deck_signature,
    extract_decisions,
    split_by_episode,
    stop_regime,
    stop_sample_weight,
    stop_weight_audit,
    apply_stop_ablation,
    enforce_stop_gates,
    stage_weight_audit,
    target_index,
)


def replay_with_decks(first: list[int], second: list[int], names=("target", "other")) -> dict:
    return {
        "info": {"TeamNames": list(names)},
        "steps": [
            [{"action": None}, {"action": None}],
            [{"action": first}, {"action": second}],
        ],
    }


def pokemon(card_id: int, serial: int, energies: int = 0, hp: int = 100):
    return SimpleNamespace(id=card_id, serial=serial, energyCards=[SimpleNamespace(id=7)] * energies, hp=hp)


def test_policy_router_requires_evidence_and_locks_selected_expert():
    _reset_policy_router()
    config = {
        "model_weight": 10.0,
        "search": {"candidates": 3, "margin": 10.0},
        "policy_router": {
            "enabled": True,
            "min_evidence": 2,
            "experts": {
                "g32": {
                    "card_scores": {"700": 0.04, "701": 0.03},
                    "threshold": 0.06,
                    "overrides": {"model_weight": 20.0, "search": {"margin": 2.0}},
                }
            },
        },
    }

    def observation(cards):
        me = SimpleNamespace(active=[], bench=[], discard=[])
        opponent = SimpleNamespace(active=[SimpleNamespace(id=card_id) for card_id in cards], bench=[], discard=[])
        return SimpleNamespace(current=SimpleNamespace(players=[me, opponent], yourIndex=0, turn=0))

    assert _select_policy_config(observation([700]), config) is config
    routed = _select_policy_config(observation([701]), config)
    assert routed["model_weight"] == 20.0
    assert routed["search"] == {"candidates": 3, "margin": 2.0}
    assert _POLICY_ROUTER_MEMORY["expert"] == "g32"
    assert _select_policy_config(observation([]), config)["model_weight"] == 20.0


def test_policy_router_reset_restores_anchor_default():
    _POLICY_ROUTER_MEMORY["seen_opponent_cards"].update({700, 701})
    _POLICY_ROUTER_MEMORY["expert"] = "g32"
    _reset_policy_router()
    assert _POLICY_ROUTER_MEMORY == {"seen_opponent_cards": set(), "expert": None}


def test_advantage_override_requires_positive_lower_bound(monkeypatch):
    options = [SimpleNamespace(type=OptionType.PLAY), SimpleNamespace(type=OptionType.PLAY)]
    obs = SimpleNamespace(select=SimpleNamespace(context=SelectContext.MAIN, option=options, maxCount=1))
    models = [{"tree_info": []} for _ in range(3)]
    model = {
        "advantage_ensemble": models,
        "advantage_calibration": {
            "default_threshold": 0.2,
            "uncertainty_weight": 1.5,
            "minimum_models": 3,
        },
    }
    monkeypatch.setattr("agents.meta_runtime.advantage_feature_vector", lambda *_: [1.0])
    monkeypatch.setattr("agents.meta_runtime.model_score", lambda *_: 0.5)
    assert advantage_override(obs, [0], [1.0, 0.9], None, model) == [1]
    monkeypatch.setattr("agents.meta_runtime.model_score", lambda *_: 0.1)
    assert advantage_override(obs, [0], [1.0, 0.9], None, model) is None


def test_advantage_override_blocks_unproven_terminal(monkeypatch):
    options = [SimpleNamespace(type=OptionType.PLAY), SimpleNamespace(type=OptionType.ATTACK)]
    obs = SimpleNamespace(select=SimpleNamespace(context=SelectContext.MAIN, option=options, maxCount=1))
    model = {
        "advantage_ensemble": [{"tree_info": []} for _ in range(3)],
        "advantage_calibration": {
            "default_threshold": 0.0,
            "terminal_threshold": 0.0,
            "allow_terminal": True,
            "minimum_models": 3,
        },
    }
    monkeypatch.setattr("agents.meta_runtime.advantage_feature_vector", lambda *_: [1.0])
    monkeypatch.setattr("agents.meta_runtime.model_score", lambda *_: 1.0)
    assert advantage_override(obs, [0], [1.0, 0.9], None, model) is None
    assert advantage_override(obs, [0], [1.0, 0.9], [1], model) == [1]


def test_advantage_split_keeps_opponents_isolated():
    rows = [
        {"episode_id": f"e{index}", "group_id": f"opponent-{index % 4}"}
        for index in range(12)
    ]
    train, validation, test = split_advantage_rows(rows, 0.25, 0.25)
    groups = [
        {row["group_id"] for row in split}
        for split in (train, validation, test)
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def grim_observation(context, options, *, active=None, bench=(), deck=None, effect=None, context_card=None, minimum=1, maximum=1, turn=6):
    def player(active_cards=(), bench_cards=()):
        return SimpleNamespace(
            active=list(active_cards), bench=list(bench_cards), hand=[], discard=[], prize=[],
            deckCount=40, handCount=0,
        )

    me = player([] if active is None else [active], bench)
    opponent = player()
    select = SimpleNamespace(
        context=context,
        option=options,
        deck=deck,
        effect=effect,
        contextCard=context_card,
        minCount=minimum,
        maxCount=maximum,
    )
    current = SimpleNamespace(yourIndex=0, players=[me, opponent], turn=turn, stadium=[], looking=[], result=-1)
    return SimpleNamespace(select=select, current=current)


def option(option_type, **kwargs):
    values = {
        "type": option_type,
        "number": None,
        "area": None,
        "index": None,
        "playerIndex": None,
        "inPlayArea": None,
        "inPlayIndex": None,
        "attackId": None,
        "cardId": None,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_tree_runtime_uses_exported_lightgbm_shape():
    model = {
        "tree_info": [
            {
                "shrinkage": 1.0,
                "tree_structure": {
                    "split_feature": 0,
                    "threshold": 2.0,
                    "decision_type": "<=",
                    "left_child": {"leaf_value": -1.5},
                    "right_child": {"leaf_value": 2.5},
                },
            }
        ]
    }
    assert model_score(model, [1.0]) == -1.5
    assert model_score(model, [3.0]) == 2.5
    assert len(FEATURE_NAMES) > 100


def test_policy_memory_records_selected_action(monkeypatch):
    obs = SimpleNamespace(
        current=SimpleNamespace(turn=4),
        select=SimpleNamespace(option=[SimpleNamespace(type=OptionType.PLAY)]),
    )
    monkeypatch.setattr("agents.meta_runtime.option_card", lambda _obs, _option: SimpleNamespace(id=1120))
    _reset_policy_memory()
    _record_policy_action(obs, [0])
    assert _POLICY_MEMORY["turn"] == 4
    assert _POLICY_MEMORY["counts"][int(OptionType.PLAY)] == 1
    assert _POLICY_MEMORY["last_type"] == int(OptionType.PLAY)
    assert _POLICY_MEMORY["last_card_id"] == 1120
    assert _POLICY_MEMORY["recent_types"] == [int(OptionType.PLAY)]
    assert _POLICY_MEMORY["recent_card_ids"] == [1120]
    assert _POLICY_MEMORY["card_counts"][1120 % 64] == 1


def test_type_features_include_candidate_card_set():
    rows = [[0.0] * len(FEATURE_NAMES) for _ in range(2)]
    rows[0][FEATURE_NAMES.index("card_id")] = 1141
    rows[1][FEATURE_NAMES.index("card_id")] = 1152
    features, types = aggregate_option_type_features(rows, [int(OptionType.PLAY)] * 2, [3.0, 2.0])
    assert types == [int(OptionType.PLAY)]
    assert features[0][TYPE_FEATURE_NAMES.index(f"candidate_card_hash_{1141 % 64}")] == 1.0
    assert features[0][TYPE_FEATURE_NAMES.index(f"candidate_card_hash_{1152 % 64}")] == 1.0
    assert features[0][TYPE_FEATURE_NAMES.index("base_score_second")] == 2.0


def test_v6_hard_router_blocks_cross_stage_scores(monkeypatch):
    global_model = {"oblivious_trees": [{}]}
    stop_model = {"oblivious_trees": [{}]}
    stage_model = {"oblivious_trees": [{}]}
    play_model = {"oblivious_trees": [{}]}
    options = [SimpleNamespace(type=OptionType.PLAY), SimpleNamespace(type=OptionType.ATTACK)]
    obs = SimpleNamespace(select=SimpleNamespace(context=SelectContext.MAIN, option=options))
    model = {
        "version": 6,
        "global": global_model,
        "heads": {"main_stop": stop_model, "main_continue_type": stage_model, "main_play": play_model},
    }
    monkeypatch.setattr("agents.meta_runtime.resolve_model", lambda _model, _obs: (global_model, 0.0))
    monkeypatch.setattr("agents.meta_runtime.feature_vector", lambda _obs, selected: [float(selected.type)])
    monkeypatch.setattr("agents.meta_runtime.aggregate_option_type_features", lambda *_: ([[1.0], [2.0]], [7, 13]))
    monkeypatch.setattr("agents.meta_runtime.fallback_score", lambda *_: 0.0)
    monkeypatch.setattr(
        "agents.meta_runtime.model_score",
        lambda selected, features: ({id(stop_model): -features[0], id(stage_model): features[0], id(play_model): 1.0}.get(id(selected), 0.0)),
    )
    scores = score_options(obs, {"model_weight": 1.0}, model)
    assert scores[0] > float("-inf")
    assert scores[1] == float("-inf")


def test_v7_soft_router_keeps_all_legal_main_options_finite(monkeypatch):
    global_model = {"oblivious_trees": [], "scale_and_bias": [1, [0]]}
    heads = {
        name: {"oblivious_trees": [], "scale_and_bias": [1, [0]]}
        for name in ("main_stop", "main_type", "main_continue_type", "main_terminal_type", "main_play", "main_attack")
    }
    model = {
        "version": 7,
        "global": global_model,
        "heads": heads,
        "router_calibration": {
            "feature_mode": "compact_v1",
            "stop_weight": 0.5,
            "main_type_weight": 0.5,
            "stage_type_weight": 0.25,
            "option_weight": 1.0,
            "stop_threshold": 0.0,
            "low_confidence_multiplier": 0.25,
        },
    }
    options = [SimpleNamespace(type=OptionType.PLAY), SimpleNamespace(type=OptionType.ATTACK)]
    obs = SimpleNamespace(select=SimpleNamespace(context=SelectContext.MAIN, option=options))
    monkeypatch.setattr("agents.meta_runtime.resolve_model", lambda _model, _obs: (global_model, 0.0))
    monkeypatch.setattr("agents.meta_runtime.feature_vector", lambda _obs, _option: [0.0] * len(FEATURE_NAMES))
    monkeypatch.setattr("agents.meta_runtime.fallback_score", lambda *_: 0.0)
    scores = score_options(obs, {"model_weight": 1.0}, model)
    assert len(ROUTER_FEATURE_NAMES) < len(TYPE_FEATURE_NAMES)
    assert all(score != float("-inf") and score == score for score in scores)


def test_stage_weight_audit_preserves_real_prior_without_double_balance():
    rows = [
        Decision([[0.0, 0.0]], [1], "source", True, option_types=[int(OptionType.PLAY)]),
        Decision([[0.0, 0.0]], [1], "source", False, option_types=[int(OptionType.ATTACK)]),
        Decision([[0.0, 0.0]], [1], "source", False, option_types=[int(OptionType.END)]),
        Decision([[0.0, 0.0]], [1], "source", False, option_types=[int(OptionType.ATTACH)]),
    ]
    audit = stage_weight_audit(rows, win_weight=1.0, loss_weight=0.35)
    assert audit["terminal_count_fraction"] == 0.5
    assert audit["terminal_mass_fraction"] == 0.7 / 2.05


def _type_stop_decision(*, ko: bool, won: bool = True) -> Decision:
    play = [0.0] * len(TYPE_FEATURE_NAMES)
    attack = [0.0] * len(TYPE_FEATURE_NAMES)
    play[TYPE_FEATURE_NAMES.index("base_score_max")] = 0.5
    attack[TYPE_FEATURE_NAMES.index("base_score_max")] = 1.0
    attack[TYPE_FEATURE_NAMES.index("attack_ko_any")] = float(ko)
    return Decision(
        [play, attack], [1, 0], "source", won,
        option_types=[int(OptionType.PLAY), int(OptionType.ATTACK)],
    )


def test_stop_regime_distinguishes_attack_and_ko_states():
    no_attack = Decision(
        [[0.0] * len(TYPE_FEATURE_NAMES), [0.0] * len(TYPE_FEATURE_NAMES)],
        [1, 0], "source", True,
        option_types=[int(OptionType.PLAY), int(OptionType.END)],
    )
    assert stop_regime(no_attack) == "no_attack"
    assert stop_regime(_type_stop_decision(ko=False)) == "attack_ready"
    assert stop_regime(_type_stop_decision(ko=True)) == "ko_ready"


def test_stop_sample_weights_are_targeted_and_capped():
    manifest = {
        "win_weight": 1.0,
        "loss_weight": 0.35,
        "stop_attack_ready_continue_weight": 1.5,
        "stop_ko_ready_continue_weight": 2.0,
        "stop_weight_cap": 2.70,
    }
    attack = _type_stop_decision(ko=False)
    ko = _type_stop_decision(ko=True)
    ko.weight = 2.0
    assert stop_sample_weight(0, attack, manifest) == 1.5
    assert stop_sample_weight(1, attack, manifest) == 1.0
    assert stop_sample_weight(0, ko, manifest) == 2.70
    audit = stop_weight_audit([
        ([0.0], 0, attack),
        ([0.0], 0, ko),
        ([0.0], 1, attack),
    ], manifest)
    assert audit["attack_ready"]["continue"]["mass"] == 1.5
    assert audit["ko_ready"]["continue"]["max_weight"] == 2.70


def test_v7_regime_router_uses_ko_expert_and_keeps_scores_finite(monkeypatch):
    tree = {"oblivious_trees": [{}]}
    ko_head = {"oblivious_trees": [{}]}
    heads = {
        "main_stop_ko_ready": ko_head,
        "main_type": tree,
        "main_continue_type": tree,
        "main_terminal_type": tree,
        "main_play": tree,
        "main_attack": tree,
    }
    play = [0.0] * len(TYPE_FEATURE_NAMES)
    attack = [0.0] * len(TYPE_FEATURE_NAMES)
    attack[TYPE_FEATURE_NAMES.index("attack_ko_any")] = 1.0
    play[TYPE_FEATURE_NAMES.index("base_score_max")] = 0.5
    attack[TYPE_FEATURE_NAMES.index("base_score_max")] = 1.0
    monkeypatch.setattr(
        "agents.meta_runtime.aggregate_option_type_features",
        lambda *_args, **_kwargs: ([play, attack], [int(OptionType.PLAY), int(OptionType.ATTACK)]),
    )
    monkeypatch.setattr(
        "agents.meta_runtime.model_score",
        lambda selected, _features: 2.0 if selected is ko_head else 0.0,
    )
    model = {
        "heads": heads,
        "router_calibration": {
            "schema_version": 2,
            "feature_mode": "full",
            "stop_feature_mode": "full",
            "stop_weight": 2.0,
            "regimes": {"ko_ready": {"stop_threshold": 0.0}},
        },
    }
    scores = score_main_options_v7(
        model,
        [[0.0] * len(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES)],
        [int(OptionType.PLAY), int(OptionType.ATTACK)],
        [0.0, 0.0],
    )
    assert all(score != float("-inf") for score in scores)
    assert scores[1] > scores[0]


def test_stop_ablation_removes_only_requested_base_features():
    names = replay_imitation.stop_feature_names()
    features = [1.0] * len(names)
    no_base = apply_stop_ablation(features, "no_base")
    delta_only = apply_stop_ablation(features, "delta_only")
    assert all(
        value == 0.0 if "::base_score_" in name else value == 1.0
        for name, value in zip(names, no_base)
    )
    assert all(
        value == 0.0 if "::base_score_" in name and not name.startswith("delta::") else value == 1.0
        for name, value in zip(names, delta_only)
    )


def test_stop_gates_require_global_and_subgroup_safety():
    metrics = {
        "stop": {"continue_recall": 0.92, "stop_recall": 0.75, "balanced_accuracy": 0.87},
        "regimes": {
            "attack_ready": {"continue_recall": 0.89},
            "ko_ready": {"continue_recall": 0.85},
        },
        "subgroups": {"turn_8_plus_continue_recall": 0.85},
    }
    config = {
        "stop_gate_continue": 0.92,
        "stop_gate_terminal": 0.75,
        "stop_gate_balanced": 0.87,
        "stop_gate_attack_continue": 0.89,
        "stop_gate_ko_continue": 0.85,
        "stop_gate_turn_8_continue": 0.85,
    }
    enforce_stop_gates(metrics, config, "test")
    metrics["regimes"]["ko_ready"]["continue_recall"] = 0.84
    with pytest.raises(ValueError, match="test stop gates failed"):
        enforce_stop_gates(metrics, config, "test")


def test_stop_gate_deficit_includes_late_turn_safety():
    metrics = {
        "stop": {"continue_recall": 0.92, "stop_recall": 0.75, "balanced_accuracy": 0.87},
        "regimes": {
            "attack_ready": {"continue_recall": 0.89},
            "ko_ready": {"continue_recall": 0.85},
        },
        "subgroups": {"turn_8_plus_continue_recall": 0.80},
    }
    config = {
        "stop_gate_continue": 0.92,
        "stop_gate_terminal": 0.75,
        "stop_gate_balanced": 0.87,
        "stop_gate_attack_continue": 0.89,
        "stop_gate_ko_continue": 0.85,
        "stop_gate_turn_8_continue": 0.85,
    }
    assert _stop_gate_deficit(metrics, config) == pytest.approx(0.05)


def test_joint_threshold_candidates_preserve_current_threshold():
    rows = [(float(value), value >= 5) for value in range(20)]
    candidates = _joint_threshold_candidates(rows, current=4.25, limit=6)
    assert 4.25 in candidates
    assert len(candidates) <= 6


def test_binary_stage_decision_uses_one_representative_per_stage():
    score_index = ROUTER_FEATURE_NAMES.index("base_score_max")
    rows = [[0.0] * len(ROUTER_FEATURE_NAMES) for _ in range(4)]
    for row, score in zip(rows, (1.0, 4.0, 2.0, 3.0)):
        row[score_index] = score
    decision = Decision(
        rows,
        [0, 0, 1, 0],
        "source",
        True,
        option_types=[int(OptionType.PLAY), int(OptionType.ATTACH), int(OptionType.ATTACK), int(OptionType.END)],
    )
    binary = _binary_stage_decision(decision)
    assert binary is not None
    assert binary.labels == [0, 1]
    assert binary.features[0][score_index] == 4.0
    assert binary.features[1][score_index] == 3.0


def test_feasible_stop_thresholds_enforces_all_strict_gates():
    rows = [(-3.0, False), (-2.0, False), (-1.0, False), (1.0, True), (2.0, True)]
    thresholds = _feasible_stop_thresholds(rows)
    assert thresholds
    assert all(-1.0 < threshold < 1.0 for threshold in thresholds)


def test_average_catboost_models_averages_scores():
    first = {"oblivious_trees": [{"splits": [], "leaf_values": [2.0]}], "scale_and_bias": [1.0, [1.0]]}
    second = {"oblivious_trees": [{"splits": [], "leaf_values": [6.0]}], "scale_and_bias": [1.0, [-1.0]]}
    averaged = average_catboost_models([first, second])
    assert replay_imitation.predict_model(averaged, []) == 4.0


def test_hierarchical_main_type_score_changes_selected_type(monkeypatch):
    global_model = {"oblivious_trees": [{}]}
    type_model = {"oblivious_trees": [{}]}
    options = [SimpleNamespace(type=OptionType.PLAY), SimpleNamespace(type=OptionType.END)]
    obs = SimpleNamespace(select=SimpleNamespace(context=SelectContext.MAIN, option=options))
    model = {
        "global": global_model,
        "heads": {"main_type": type_model},
        "calibration": {"main_type_weight": 1.0, "main_option_weight": 0.0},
    }
    monkeypatch.setattr("agents.meta_runtime.resolve_model", lambda _model, _obs: (global_model, 0.0))
    monkeypatch.setattr("agents.meta_runtime.feature_vector", lambda _obs, selected: [float(selected.type)])
    monkeypatch.setattr("agents.meta_runtime.fallback_score", lambda *_: 0.0)
    monkeypatch.setattr(
        "agents.meta_runtime.model_score",
        lambda selected, features: 0.0 if selected is global_model else features[0],
    )
    assert score_options(obs, {"model_weight": 1.0}, model) == [7.0, 14.0]


def test_margin_selection_supports_optional_multi_select_without_empty_result():
    select = SimpleNamespace(minCount=0, maxCount=3)
    assert choose_from_scores(select, [-4.0, -4.1, -9.0], margin=0.2) == [0, 1]
    assert choose_from_scores(select, [-4.0, -4.1, -9.0], margin=0.0) == [0]


def test_target_index_prefers_unique_deck_hash():
    target_deck = [1] * 56 + [1121] * 4
    other_deck = [2] * 56 + [1121] * 4
    replay = replay_with_decks(other_deck, target_deck, names=("renamed", "also renamed"))
    assert target_index(replay, "target", deck_signature(target_deck)) == 1


def test_replay_actions_label_the_previous_observation(monkeypatch):
    monkeypatch.setattr(
        replay_imitation,
        "to_observation_class",
        lambda observation: SimpleNamespace(select=SimpleNamespace(option=observation["select"]["option"])),
    )
    monkeypatch.setattr(replay_imitation, "feature_vector", lambda obs, option: [float(option["value"])])
    replay = {
        "rewards": [1, -1],
        "steps": [
            [{"action": [], "observation": {"select": None}}, {}],
            [{"action": [1, 2, 3], "observation": {"select": {"option": [{"value": 10}, {"value": 20}]}}}, {}],
            [{"action": [1], "observation": {"select": {"option": [{"value": 30}, {"value": 40}, {"value": 50}]}}}, {}],
        ],
    }
    decisions = extract_decisions(replay, 0, "leader")
    assert len(decisions) == 1
    assert decisions[0].features == [[10.0], [20.0]]
    assert decisions[0].labels == [0, 1]


def test_episode_split_never_leaks_decisions_between_sets():
    decisions = [
        Decision([[float(index)]], [1], "source", True, episode_id=f"source:{episode}")
        for episode in range(10)
        for index in range(2)
    ]
    train_rows, validation_rows, test_rows = split_by_episode(decisions, 0.2, 0.2)
    train_ids = {row.episode_id for row in train_rows}
    validation_ids = {row.episode_id for row in validation_rows}
    test_ids = {row.episode_id for row in test_rows}
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids
    assert train_ids | validation_ids | test_ids == {f"source:{episode}" for episode in range(10)}


def test_episode_split_is_reproducibly_stratified_by_outcome_and_seat():
    decisions = [
        Decision(
            [[float(index)]], [1], "source", episode % 2 == 0,
            episode_id=f"source:{episode}", seat=episode % 2,
        )
        for episode in range(20)
        for index in range(2)
    ]
    first = split_by_episode(decisions, 0.2, 0.2, seed=20260813)
    second = split_by_episode(decisions, 0.2, 0.2, seed=20260813)
    assert [[row.episode_id for row in part] for part in first] == [[row.episode_id for row in part] for part in second]
    sets = [{row.episode_id for row in part} for part in first]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sets[0] | sets[1] | sets[2] == {f"source:{episode}" for episode in range(20)}


def test_meta_builder_creates_valid_reference_free_package(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "family": "grimmsnarl",
                "deck": [1] * 56 + [1121] * 4,
                "model": {"tree_info": []},
                "validation_metrics": {"top1": 0.5},
            }
        )
    )
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("pass\n")
    (cg_dir / "game.py").write_text("pass\n")
    out = tmp_path / "candidate.tar.gz"

    build(artifact, out, runtime, cg_dir, "rules")

    info = inspect_package(out)
    assert len(info["deck"]) == 60
    with tarfile.open(out, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
    assert {"main.py", "meta_runtime.py", "deck.csv", "cg/api.py", "cg/game.py"} <= names


def test_meta_builder_embeds_v2_models_and_strips_non_linux_libraries(tmp_path: Path):
    tree = {"tree_info": []}
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "version": 2,
        "family": "teal_ogerpon",
        "deck": [1] * 60,
        "model": tree,
        "head_models": {"main": tree},
        "value_model": tree,
        "confidence_thresholds": {"main": 0.5},
    }))
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    for name in ("api.py", "game.py", "libcg.so", "cg.dll", "libcg.dylib", "libcg-arm64.so"):
        (cg_dir / name).write_bytes(b"x")
    package = tmp_path / "candidate.tar.gz"
    build(artifact, package, runtime, cg_dir, "clone_search")
    with tarfile.open(package, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        model_payload = json.loads(gzip.decompress(archive.extractfile("model.json.gz").read()))
        metadata = json.loads(archive.extractfile("build_metadata.json").read())
    assert "cg/libcg.so" in names
    assert not {"cg/cg.dll", "cg/libcg.dylib", "cg/libcg-arm64.so"} & names
    assert model_payload["version"] == 2
    assert "main" in model_payload["heads"]
    assert metadata["linux_only"] is True
    assert metadata["deck_sha256"] and metadata["model_sha256"]


def test_meta_builder_preserves_packaged_anchor_model_exactly(tmp_path: Path):
    packaged_model = {
        "version": 7,
        "global": {"oblivious_trees": [{"splits": [], "leaf_values": [0.125]}]},
        "heads": {"main_play": {"oblivious_trees": [], "scale_and_bias": [1, [0]]}},
        "unknown_future_field": {"order": [3, 1, 2]},
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "version": 4,
        "family": "teal_ogerpon",
        "deck": [1] * 60,
        "model": {"oblivious_trees": []},
        "head_models": {},
        "__packaged_model__": packaged_model,
    }))
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("pass\n")
    package = tmp_path / "candidate.tar.gz"
    build(artifact, package, runtime, cg_dir, "clone_residual_search")
    with tarfile.open(package, "r:gz") as archive:
        payload = json.loads(gzip.decompress(archive.extractfile("model.json.gz").read()))
    assert payload == packaged_model


def test_meta_builder_preserves_packaged_model_bytes_when_requested(tmp_path: Path):
    raw_model = gzip.compress(b'{"version":9,"global":{}}', compresslevel=9)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "version": 4,
        "family": "teal_ogerpon",
        "deck": [1] * 60,
        "__packaged_model__": {"version": 9, "global": {}},
        "__packaged_model_gzip_b64__": base64.b64encode(raw_model).decode("ascii"),
    }))
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("pass\n")
    package = tmp_path / "candidate.tar.gz"
    build(artifact, package, runtime, cg_dir, "clone_residual_search")
    with tarfile.open(package, "r:gz") as archive:
        payload = archive.extractfile("model.json.gz").read()
    assert payload == raw_model


def test_meta_builder_embeds_v4_hierarchical_calibration(tmp_path: Path):
    tree = {"oblivious_trees": [], "scale_and_bias": [1, [0]]}
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "version": 4,
        "family": "teal_ogerpon",
        "deck": [1] * 60,
        "model": tree,
        "head_models": {"main_type": tree, "main_play": tree},
        "value_model": {},
        "confidence_thresholds": {},
        "calibration": {"main_type_weight": 2.0, "main_option_weight": 0.5},
    }))
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("pass\n")
    package = tmp_path / "candidate.tar.gz"
    build(artifact, package, runtime, cg_dir, "clone_fidelity")
    with tarfile.open(package, "r:gz") as archive:
        payload = json.loads(gzip.decompress(archive.extractfile("model.json.gz").read()))
    assert payload["version"] == 4
    assert payload["calibration"] == {"main_type_weight": 2.0, "main_option_weight": 0.5}
    assert {"main_type", "main_play"} <= payload["heads"].keys()


def test_meta_builder_main_loads_when_kaggle_exec_omits_file(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "version": 2,
        "family": "teal_ogerpon",
        "deck": [1] * 60,
        "model": {"tree_info": []},
        "head_models": {},
        "value_model": {},
        "confidence_thresholds": {},
    }))
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("pass\n")
    package = tmp_path / "candidate.tar.gz"
    build(artifact, package, runtime, cg_dir, "clone_search")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(package, "r:gz") as archive:
        archive.extractall(extracted)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(extracted)!r}); "
            f"source=open({str(extracted / 'main.py')!r}, encoding='utf-8').read(); "
            "namespace={}; exec(compile(source, 'main.py', 'exec'), namespace); "
            "assert callable(namespace['agent'])",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_meta_builder_embeds_v3_residual_models(tmp_path: Path):
    tree = {"oblivious_trees": [{"leaf_values": [0.25], "leaf_weights": [1], "splits": []}], "scale_and_bias": [1, [0]]}
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "version": 3,
        "family": "teal_ogerpon",
        "deck": [1] * 60,
        "model": {"oblivious_trees": [], "scale_and_bias": [1, [0]]},
        "head_models": {},
        "value_model": {},
        "residual_q_model": tree,
        "win_value_model": tree,
        "confidence_thresholds": {},
        "opponent_decks": [[96] * 4 + [1] * 56],
    }))
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("pass\n")
    package = tmp_path / "candidate.tar.gz"
    build(artifact, package, runtime, cg_dir, "clone_residual_search")
    with tarfile.open(package, "r:gz") as archive:
        payload = json.loads(gzip.decompress(archive.extractfile("model.json.gz").read()))
        main_source = archive.extractfile("main.py").read().decode("utf-8")
    assert payload["version"] == 3
    assert payload["residual_q"]["oblivious_trees"]
    assert payload["win_value"]["oblivious_trees"]
    assert "opponent_decks" in main_source


def test_meta_builder_supports_generic_replay_family(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "version": 3,
        "family": "generic_replay",
        "deck": [1] * 60,
        "model": {"tree_info": []},
        "head_models": {},
        "value_model": {},
        "confidence_thresholds": {},
        "generic_policy": {"play_priority": {"1": 900.0}},
    }))
    runtime = tmp_path / "meta_runtime.py"
    runtime.write_text("def run_agent(obs, config, model):\n    return []\n")
    cg_dir = tmp_path / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("pass\n")
    package = tmp_path / "generic.tar.gz"
    build(artifact, package, runtime, cg_dir, "clone_fidelity")
    with tarfile.open(package, "r:gz") as archive:
        main_source = archive.extractfile("main.py").read().decode("utf-8")
    assert "generic_replay" in main_source
    assert "'1': 900.0" in main_source


def test_counterfactual_split_never_leaks_episodes():
    rows = [{"episode_id": f"g{episode}", "actions": []} for episode in range(10)]
    train, validation, test = split_counterfactual_rows(rows, 0.2, 0.2)
    train_ids = {row["episode_id"] for row in train}
    validation_ids = {row["episode_id"] for row in validation}
    test_ids = {row["episode_id"] for row in test}
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_cem_sampling_stays_in_bounds_and_elites_move_distribution():
    distribution = {
        name: {"mean": mean, "std": std, "low": low, "high": high}
        for name, (mean, std, low, high) in PARAMETERS.items()
    }
    sample = sample_vector(distribution, __import__("random").Random(7))
    for name, value in sample.items():
        assert distribution[name]["low"] <= value <= distribution[name]["high"]
    old_mean = distribution["residual_weight"]["mean"]
    elites = [{name: values["high"] for name, values in distribution.items()} for _ in range(3)]
    update_distribution(distribution, elites, 0.6)
    assert distribution["residual_weight"]["mean"] > old_mean


def test_cem_center_vector_sets_mean_and_scales_search_radius():
    center = {name: mean for name, (mean, _, _, _) in PARAMETERS.items()}
    center["candidates"] = 5.4
    distribution, normalized = centered_distribution(center, 0.25)
    assert normalized is not None
    assert normalized["candidates"] == 5
    assert distribution["candidates"]["mean"] == 5.0
    assert distribution["model_weight"]["std"] == PARAMETERS["model_weight"][1] * 0.25


def test_model_head_routes_observable_contexts():
    obs = SimpleNamespace(select=SimpleNamespace(context=SelectContext.ATTACH_TO))
    assert model_head(obs) == "attach"
    bundle = {"global": {"tree_info": []}, "heads": {"attach": {"tree_info": [{"tree_structure": {"leaf_value": 1}}]}}, "confidence_thresholds": {"attach": 0.4}}
    selected, threshold = resolve_model(bundle, obs)
    assert selected is bundle["heads"]["attach"]
    assert threshold == 0.4


def test_grim_evolution_trigger_cannot_choose_no():
    obs = grim_observation(
        SelectContext.ACTIVATE,
        [option(OptionType.NO), option(OptionType.YES)],
        effect=SimpleNamespace(id=648, serial=10),
    )
    assert forced_action(obs, {"family": "grimmsnarl"}) == [1]


def test_grim_punk_up_selects_all_available_darkness_energy():
    deck = [SimpleNamespace(id=7), SimpleNamespace(id=7), SimpleNamespace(id=7), SimpleNamespace(id=112)]
    options = [
        option(OptionType.CARD, area=AreaType.DECK, index=0, playerIndex=0),
        option(OptionType.CARD, area=AreaType.DECK, index=1, playerIndex=0),
        option(OptionType.CARD, area=AreaType.DECK, index=2, playerIndex=0),
    ]
    obs = grim_observation(
        SelectContext.ATTACH_TO,
        options,
        deck=deck,
        effect=SimpleNamespace(id=648, serial=10),
        minimum=0,
        maximum=5,
    )
    assert forced_action(obs, {"family": "grimmsnarl"}) is None
    assert ensure_mandatory_minimums(obs, [], [3.0, 2.0, 1.0]) == [0, 1]


def test_grim_punk_up_charges_new_grim_before_other_targets():
    grim = pokemon(648, serial=10, energies=1, hp=320)
    munkidori = pokemon(112, serial=11, energies=0, hp=110)
    options = [
        option(OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0),
        option(OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=0),
    ]
    obs = grim_observation(
        SelectContext.ATTACH_FROM,
        options,
        bench=[munkidori, grim],
        effect=SimpleNamespace(id=648, serial=10),
    )
    assert forced_action(obs, {"family": "grimmsnarl"}) == [1]


def test_grim_punk_up_distributes_after_source_is_attack_ready():
    grim = pokemon(648, serial=10, energies=2, hp=320)
    munkidori = pokemon(112, serial=11, energies=0, hp=110)
    options = [
        option(OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0),
        option(OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=0),
    ]
    obs = grim_observation(
        SelectContext.ATTACH_FROM,
        options,
        bench=[munkidori, grim],
        effect=SimpleNamespace(id=648, serial=10),
    )
    assert forced_action(obs, {"family": "grimmsnarl"}) == [0]


def test_ready_bench_grim_is_promoted_over_impidimp():
    impidimp = pokemon(646, serial=1, energies=1, hp=70)
    grim = pokemon(648, serial=10, energies=2, hp=320)
    options = [
        option(OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0),
        option(OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=0),
    ]
    obs = grim_observation(SelectContext.TO_ACTIVE, options, active=impidimp, bench=[pokemon(112, 2), grim])
    assert forced_action(obs, {"family": "grimmsnarl"}) == [1]


def test_ready_bench_grim_leaves_main_order_to_corrected_model():
    impidimp = pokemon(646, serial=1, energies=1, hp=70)
    grim = pokemon(648, serial=10, energies=2, hp=320)
    options = [
        option(OptionType.ATTACK, attackId=934),
        option(OptionType.RETREAT),
        option(OptionType.END),
    ]
    obs = grim_observation(SelectContext.MAIN, options, active=impidimp, bench=[grim])
    assert forced_action(obs, {"family": "grimmsnarl"}) is None


def test_legal_grim_evolution_leaves_main_order_to_corrected_model():
    impidimp = pokemon(646, serial=1, energies=1, hp=70)
    grim_card = SimpleNamespace(id=648, serial=20)
    me_hand = [SimpleNamespace(id=1227), grim_card]
    options = [
        option(OptionType.ATTACK, attackId=934),
        option(OptionType.EVOLVE, area=AreaType.HAND, index=1, playerIndex=0, inPlayArea=AreaType.ACTIVE, inPlayIndex=0),
    ]
    obs = grim_observation(SelectContext.MAIN, options, active=impidimp)
    obs.current.players[0].hand = me_hand
    assert forced_action(obs, {"family": "grimmsnarl"}) is None


def test_skill_order_selects_required_froslass_skills_first():
    options = [
        option(OptionType.SKILL, cardId=112),
        option(OptionType.SKILL, cardId=104),
    ]
    obs = grim_observation(SelectContext.SKILL_ORDER, options, minimum=2, maximum=2)
    assert forced_action(obs, {"family": "grimmsnarl"}) == [1, 0]


def test_kang_teal_dance_cannot_select_zero_energy():
    deck = [SimpleNamespace(id=1), SimpleNamespace(id=4)]
    options = [
        option(OptionType.CARD, area=AreaType.DECK, index=0, playerIndex=0),
        option(OptionType.CARD, area=AreaType.DECK, index=1, playerIndex=0),
    ]
    obs = grim_observation(
        SelectContext.ATTACH_TO,
        options,
        deck=deck,
        effect=SimpleNamespace(id=96, serial=10),
        minimum=0,
        maximum=1,
    )
    assert forced_action(obs, {"family": "mega_kangaskhan"}) == [0]


def test_kang_optional_abilities_are_activated():
    obs = grim_observation(
        SelectContext.ACTIVATE,
        [option(OptionType.NO), option(OptionType.YES)],
        effect=SimpleNamespace(id=756, serial=10),
    )
    assert forced_action(obs, {"family": "mega_kangaskhan"}) == [1]


def test_poke_pad_rescues_missing_impidimp_before_froslass():
    obs = grim_observation(SelectContext.TO_HAND, [], active=pokemon(860, 1))
    obs.current.players[0].hand = []
    assert _grim_search_score(obs, IMPIDIMP, POKE_PAD) > _grim_search_score(obs, FROSLASS, POKE_PAD)


def test_petrel_rescues_missing_impidimp_with_same_turn_poffin():
    obs = grim_observation(SelectContext.TO_HAND, [], active=pokemon(112, 1))
    obs.current.players[0].hand = []
    assert _grim_search_score(obs, BUDDY_BUDDY_POFFIN, TEAM_ROCKETS_PETREL) > _grim_search_score(obs, DAWN, TEAM_ROCKETS_PETREL)


def test_grim_target_priority_includes_lucario_evolution_basics():
    config = family_config("grimmsnarl", "clone")
    assert config["target_priority"]


def test_lopunny_family_prefers_dunsparce_opening_and_gale_thrust():
    config = family_config("mega_lopunny", "clone")
    assert config["setup_active_priority"]["305"] > config["setup_active_priority"]["848"]
    assert config["attack_priority"]["1225"] > config["attack_priority"]["1226"]


def test_teal_ogerpon_family_supports_core_loop():
    config = family_config("teal_ogerpon", "clone")
    assert config["setup_active_priority"]["96"] > 0
    assert config["attack_priority"]["120"] > 0


def test_teal_ogerpon_combat_variant_enables_tactical_search():
    config = family_config("teal_ogerpon", "clone_combat_search")
    assert config["ogerpon_tactical"] is True
    assert config["ogerpon_anti_mill"] is False
    assert config["ogerpon_force_terminal_attack"] is True
    assert config["search"]["low_confidence_only"] is True
    assert config["search"]["contexts"] == ["MAIN", "SWITCH", "TO_ACTIVE"]


def test_teal_ogerpon_myriad_leaf_damage_counts_both_active_energy_and_weakness():
    obs = grim_observation(
        SelectContext.MAIN,
        [option(OptionType.ATTACK, attackId=120)],
        active=pokemon(96, serial=10, energies=3, hp=210),
    )
    obs.current.players[1].active = [pokemon(607, serial=20, energies=2, hp=140)]
    assert _ogerpon_attack_damage(obs, 120) == 360.0


def test_teal_ogerpon_tactical_target_avoids_crustle_and_hits_dwebble():
    options = [
        option(OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=1),
        option(OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=1),
    ]
    obs = grim_observation(
        SelectContext.SWITCH,
        options,
        active=pokemon(96, serial=10, energies=3, hp=210),
    )
    obs.current.players[1].bench = [pokemon(344, serial=20, hp=70), pokemon(345, serial=21, hp=150)]
    config = family_config("teal_ogerpon", "clone_combat")
    assert fallback_score(obs, options[0], config) > fallback_score(obs, options[1], config)


def test_teal_ogerpon_anti_mill_stops_optional_draw_after_attacker_is_ready():
    options = [
        option(OptionType.PLAY, area=AreaType.HAND, index=0, playerIndex=0),
        option(OptionType.PLAY, area=AreaType.HAND, index=1, playerIndex=0),
    ]
    obs = grim_observation(
        SelectContext.MAIN,
        options,
        active=pokemon(96, serial=10, energies=3, hp=210),
    )
    obs.current.players[0].hand = [SimpleNamespace(id=1094), SimpleNamespace(id=1120)]
    obs.current.players[0].deckCount = 24
    obs.current.players[1].deckCount = 20
    obs.current.players[1].active = [pokemon(58, serial=20, energies=2, hp=140)]
    config = family_config("teal_ogerpon", "clone_anti_mill")
    assert fallback_score(obs, options[1], config) > fallback_score(obs, options[0], config)


def test_search_state_value_recognizes_terminal_result():
    obs = grim_observation(SelectContext.MAIN, [option(OptionType.END)])
    obs.current.result = 0
    assert state_value(obs, {"family": "teal_ogerpon"}) == 1_000_000_000.0
    obs.current.result = 1
    assert state_value(obs, {"family": "teal_ogerpon"}) == -1_000_000_000.0


def test_belief_worlds_preserve_hidden_zone_counts():
    obs = grim_observation(SelectContext.MAIN, [option(OptionType.END), option(OptionType.ATTACK, attackId=120)])
    obs.current.players[0].deckCount = 20
    obs.current.players[0].prize = [SimpleNamespace(id=1)] * 4
    obs.current.players[1].deckCount = 25
    obs.current.players[1].prize = [SimpleNamespace(id=1)] * 5
    obs.current.players[1].handCount = 6
    deck = [96] * 4 + [1] * 56
    worlds = _sample_hidden_worlds(obs, {"own_deck": deck, "opponent_decks": [deck]}, 3)
    assert len(worlds) == 3
    for own_deck, own_prize, opponent_deck, opponent_prize, opponent_hand, hidden_active in worlds:
        assert len(own_deck) == 20
        assert len(own_prize) == 4
        assert len(opponent_deck) == 25
        assert len(opponent_prize) == 5
        assert len(opponent_hand) == 6
        assert hidden_active == []


def test_search_uses_official_search_ids_and_can_change_action(monkeypatch):
    options = [option(OptionType.END), option(OptionType.ATTACK, attackId=120)]
    obs = grim_observation(SelectContext.MAIN, options, active=pokemon(96, 10, energies=3, hp=210))
    obs.search_begin_input = "encoded-state"
    losing = grim_observation(SelectContext.MAIN, [], active=pokemon(96, 10), turn=6)
    losing.current.result = 1
    losing.select = None
    winning = grim_observation(SelectContext.MAIN, [], active=pokemon(96, 10), turn=6)
    winning.current.result = 0
    winning.select = None
    calls = []

    def fake_begin(*args):
        calls.append(("begin", len(args)))
        return SimpleNamespace(searchId=77, observation=obs)

    def fake_step(search_id, action):
        calls.append(("step", search_id, action))
        return SimpleNamespace(searchId=100 + action[0], observation=winning if action == [1] else losing)

    monkeypatch.setattr("cg.api.search_begin", fake_begin)
    monkeypatch.setattr("cg.api.search_step", fake_step)
    monkeypatch.setattr("cg.api.search_end", lambda: calls.append(("end",)))
    monkeypatch.setattr("cg.api.search_release", lambda search_id: calls.append(("release", search_id)))
    monkeypatch.setattr(
        "agents.meta_runtime._sample_hidden_worlds",
        lambda *_: [([1], [1], [96], [1], [1], [])],
    )
    config = {
        "family": "teal_ogerpon",
        "model_weight": 0.0,
        "search": {
            "enabled": True,
            "low_confidence_only": False,
            "candidates": 2,
            "belief_worlds": 1,
            "rollout_steps": 1,
            "budget_s": 1.0,
            "minimum_overage_s": 0.0,
        },
    }
    assert search_action({"remainingOverageTime": 600.0}, obs, [0], config, None) == [1]
    assert ("begin", 8) in calls
    assert ("step", 77, [0]) in calls
    assert ("step", 77, [1]) in calls


def test_teal_ogerpon_activates_teal_dance():
    ogerpon = pokemon(96, serial=10, energies=1, hp=210)
    options = [option(OptionType.ABILITY, area=AreaType.ACTIVE, index=0, playerIndex=0), option(OptionType.END)]
    obs = grim_observation(SelectContext.MAIN, options, active=ogerpon)
    assert forced_action(obs, {"family": "teal_ogerpon", "ogerpon_force_ability": True}) == [0]


def test_teal_ogerpon_defaults_to_learned_ability_target():
    ogerpon = pokemon(96, serial=10, energies=1, hp=210)
    options = [option(OptionType.ABILITY, area=AreaType.ACTIVE, index=0, playerIndex=0), option(OptionType.END)]
    obs = grim_observation(SelectContext.MAIN, options, active=ogerpon)
    assert forced_action(obs, {"family": "teal_ogerpon"}) is None


def test_teal_ogerpon_attaches_basic_grass_with_ability():
    deck = [SimpleNamespace(id=1), SimpleNamespace(id=18)]
    options = [
        option(OptionType.CARD, area=AreaType.DECK, index=0, playerIndex=0),
        option(OptionType.CARD, area=AreaType.DECK, index=1, playerIndex=0),
    ]
    obs = grim_observation(
        SelectContext.ATTACH_TO,
        options,
        deck=deck,
        effect=SimpleNamespace(id=96, serial=10),
    )
    assert forced_action(obs, {"family": "teal_ogerpon"}) == [0]


def test_lopunny_always_chooses_to_go_first():
    obs = grim_observation(
        SelectContext.IS_FIRST,
        [option(OptionType.NO), option(OptionType.YES)],
    )
    assert forced_action(obs, {"family": "mega_lopunny"}) == [1]


def test_lopunny_promotion_prefers_mega_lopunny():
    lopunny = pokemon(849, serial=10, energies=1, hp=330)
    froslass = pokemon(861, serial=11, energies=3, hp=310)
    options = [
        option(OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0),
        option(OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=0),
    ]
    obs = grim_observation(SelectContext.TO_ACTIVE, options, bench=[froslass, lopunny])
    assert forced_action(obs, {"family": "mega_lopunny"}) == [1]


def test_lopunny_uses_spiky_hopper_without_movement_bonus():
    _LOPUNNY_MEMORY.update(turn=6, active_serial=10, moved=False)
    options = [
        option(OptionType.ATTACK, attackId=1225),
        option(OptionType.ATTACK, attackId=1226),
        option(OptionType.END),
    ]
    obs = grim_observation(SelectContext.MAIN, options, active=pokemon(849, 10, energies=2, hp=330))
    assert forced_action(obs, {"family": "mega_lopunny", "lopunny_attack_fix": True}) == [1]


def test_lopunny_memory_does_not_treat_active_evolution_as_movement():
    _LOPUNNY_MEMORY.update(turn=None, active_serial=None, bench_serials=set(), moved=False)
    before = grim_observation(SelectContext.MAIN, [], active=pokemon(848, 10), bench=[pokemon(305, 30)], turn=6)
    evolved = grim_observation(SelectContext.MAIN, [], active=pokemon(849, 20), bench=[pokemon(305, 30)], turn=6)
    _update_lopunny_memory(before)
    _update_lopunny_memory(evolved)
    assert _LOPUNNY_MEMORY["moved"] is False


def test_lopunny_memory_requires_previous_bench_membership():
    _LOPUNNY_MEMORY.update(turn=None, active_serial=None, bench_serials=set(), moved=False)
    before = grim_observation(SelectContext.MAIN, [], active=pokemon(66, 10), bench=[pokemon(849, 20)], turn=6)
    promoted = grim_observation(SelectContext.MAIN, [], active=pokemon(849, 20), bench=[pokemon(66, 10)], turn=6)
    _update_lopunny_memory(before)
    _update_lopunny_memory(promoted)
    assert _LOPUNNY_MEMORY["moved"] is True


def test_lopunny_uses_gale_after_movement_when_spiky_does_not_ko():
    _LOPUNNY_MEMORY.update(turn=6, active_serial=10, moved=True)
    options = [
        option(OptionType.ATTACK, attackId=1225),
        option(OptionType.ATTACK, attackId=1226),
        option(OptionType.END),
    ]
    obs = grim_observation(SelectContext.MAIN, options, active=pokemon(849, 10, energies=2, hp=330))
    obs.current.players[1].active = [pokemon(648, 20, hp=330)]
    assert forced_action(obs, {"family": "mega_lopunny", "lopunny_attack_fix": True}) == [0]


def test_lopunny_retreats_into_free_pivot_before_attacking():
    _LOPUNNY_MEMORY.update(turn=6, active_serial=10, moved=False)
    pivot = pokemon(66, 11, hp=140)
    options = [
        option(OptionType.RETREAT),
        option(OptionType.ATTACK, attackId=1225),
        option(OptionType.END),
    ]
    obs = grim_observation(
        SelectContext.MAIN,
        options,
        active=pokemon(849, 10, energies=1, hp=330),
        bench=[pivot],
    )
    assert forced_action(obs, {"family": "mega_lopunny", "lopunny_cycle": True}) == [0]


def test_froslass_attack_choice_is_left_to_imitation_model():
    _LOPUNNY_MEMORY.update(turn=6, active_serial=10, moved=False)
    options = [
        option(OptionType.RETREAT),
        option(OptionType.ATTACK, attackId=1240),
        option(OptionType.END),
    ]
    obs = grim_observation(
        SelectContext.MAIN,
        options,
        active=pokemon(861, 10, energies=1, hp=310),
        bench=[pokemon(849, 11, energies=1, hp=330)],
    )
    obs.current.players[1].active = [pokemon(648, 20, hp=200)]
    obs.current.players[1].handCount = 4
    assert forced_action(obs, {"family": "mega_lopunny"}) is None
