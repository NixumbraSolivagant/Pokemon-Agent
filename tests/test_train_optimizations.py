from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import agents.meta_runtime as meta_runtime
from agents.meta_runtime import FEATURE_NAMES, TYPE_FEATURE_NAMES, _main_type_head
from cg.api import OptionType, SelectContext
from tools.ogerpon_front100_search import make_manifest
from tools.replay_imitation import (
    Decision,
    _aggregate_decision_by_type,
    balance_type_decisions,
    _load_cache_file,
    _load_model_part,
    _predict_catboost,
    _save_model_part,
    _train_catboost,
    _write_cache_file,
    _filter_decision_by_type,
    decision_head,
    merge_catboost_models,
    predict_batch,
    rankable_decisions,
    replay_fingerprint,
    replay_content_fingerprint,
    target_index,
)


def _synthetic_model() -> dict:
    rng = np.random.default_rng(5)
    trees = []
    for _ in range(5):
        depth = 3
        splits = [
            {
                "border": float(rng.normal()),
                "float_feature_index": int(rng.integers(0, 12)),
                "split_index": level,
                "split_type": "FloatFeature",
            }
            for level in range(depth)
        ]
        leaf_values = [float(rng.normal()) for _ in range(1 << depth)]
        trees.append({"leaf_values": leaf_values, "leaf_weights": [1] * len(leaf_values), "splits": splits})
    return {"oblivious_trees": trees, "scale_and_bias": [1.25, [0.5]]}


def test_predict_batch_matches_pure_python() -> None:
    model = _synthetic_model()
    rng = np.random.default_rng(9)
    features = rng.normal(size=(50, 10)).tolist()
    features[0][0] = float("nan")
    features[1][1] = float("nan")
    batch = predict_batch(model, features)
    for row, expected in zip(features, batch):
        assert abs(_predict_catboost(model, row) - expected) < 1e-12


def test_predict_batch_empty() -> None:
    assert predict_batch(_synthetic_model(), []) == []


def test_merged_catboost_model_matches_additive_scores() -> None:
    base = _synthetic_model()
    residual = _synthetic_model()
    features = np.random.default_rng(11).normal(size=(12, 10)).tolist()
    merged = merge_catboost_models(base, residual)
    expected = [left + right for left, right in zip(predict_batch(base, features), predict_batch(residual, features))]
    assert np.allclose(predict_batch(merged, features), expected)


def test_parse_cache_roundtrip(tmp_path: Path) -> None:
    decisions = [
        Decision(
            features=[[1.0, 2.0], [3.0, 4.0]],
            labels=[1, 0],
            source="Majkel1337",
            won=True,
            episode_id="e1",
            head="main",
            seat=0,
            reward=1,
        )
    ]
    payload = {
        "version": 2,
        "fingerprint": "abc",
        "feature_schema_sha256": "x",
        "decisions": decisions,
        "deck": [1, 2],
        "stats": {"resolved": 1, "ambiguous": 0},
    }
    path = tmp_path / "parse-abc.pickle.gz"
    _write_cache_file(path, payload)
    loaded = _load_cache_file(path)
    assert loaded is not None
    assert loaded["decisions"][0].features == decisions[0].features
    assert loaded["decisions"][0].labels == [1, 0]
    assert loaded["deck"] == [1, 2]


def test_replay_fingerprint_changes_with_content(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "episode-1-replay.json").write_text('{"a": 1}', encoding="utf-8")
    (second / "episode-1-replay.json").write_text('{"a": 2}', encoding="utf-8")
    assert replay_fingerprint(first) != replay_fingerprint(second)
    assert replay_fingerprint(first) == replay_fingerprint(first)


def test_replay_content_fingerprint_ignores_json_formatting(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "episode-1-replay.json").write_text('{"b":2,"a":1}', encoding="utf-8")
    (second / "episode-1-replay.json").write_text('{\n  "a": 1,\n  "b": 2\n}', encoding="utf-8")
    assert replay_content_fingerprint(first)["sha256"] == replay_content_fingerprint(second)["sha256"]
    (second / "episode-1-replay.json").write_text('{"a":1,"b":3}', encoding="utf-8")
    assert replay_content_fingerprint(first)["sha256"] != replay_content_fingerprint(second)["sha256"]


def test_model_parts_roundtrip(tmp_path: Path) -> None:
    parts = tmp_path / "balanced_parts"
    _save_model_part(parts, "exported", {"oblivious_trees": [], "scale_and_bias": [1, [0]]})
    assert _load_model_part(parts, "exported") == {"oblivious_trees": [], "scale_and_bias": [1, [0]]}
    assert _load_model_part(parts, "missing") is None


def test_make_manifest_source_name_and_cache(tmp_path: Path) -> None:
    args = SimpleNamespace(
        seed=1,
        model_jobs=12,
        parse_workers=4,
        gpu_device=1,
        minimum_head_decisions=30,
        confidence_precision=0.80,
        replays=Path("outputs/replays"),
        target_name="Majkel1337",
        out=tmp_path,
    )
    manifest_path = make_manifest(args, "balanced", tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["sources"][0]["name"] == "Majkel1337"
    assert data["sources"][0]["target_name"] == "Majkel1337"
    assert data["cache_dir"] == str(tmp_path / "cache")
    assert data["task_type"] == "GPU"
    assert data["devices"] == "1"
    assert data["enable_main_type_heads"] is True


def test_make_manifest_uses_repeat_specific_seed_and_name(tmp_path: Path) -> None:
    args = SimpleNamespace(
        seed=10,
        seed_repeats=3,
        model_jobs=2,
        parse_workers=1,
        gpu_device=-1,
        gpu_devices="",
        minimum_head_decisions=30,
        confidence_precision=0.8,
        replays=Path("replays"),
        target_name="Majkel1337",
        out=tmp_path,
    )
    path = make_manifest(args, "deep", tmp_path, repeat=2)
    assert path.name == "deep_s2.json"
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 212


def _fake_obs(option_types: list[OptionType]) -> SimpleNamespace:
    return SimpleNamespace(
        select=SimpleNamespace(
            context=SelectContext.MAIN,
            option=[SimpleNamespace(type=option_type) for option_type in option_types],
        )
    )


def test_decision_head_keeps_main_for_all_action_types() -> None:
    assert decision_head(_fake_obs([OptionType.PLAY]), {0}) == "main"
    assert decision_head(_fake_obs([OptionType.ATTACH]), {0}) == "main"
    assert decision_head(_fake_obs([OptionType.ATTACK]), {0}) == "main"
    assert decision_head(_fake_obs([OptionType.END]), {0}) == "main"
    assert decision_head(_fake_obs([OptionType.ABILITY]), {0}) == "main"


def test_filter_decision_by_type_keeps_only_wanted_rows() -> None:
    decision = Decision(
        features=[[1.0], [2.0], [3.0]],
        labels=[0, 1, 0],
        option_types=[int(OptionType.PLAY), int(OptionType.ATTACH), int(OptionType.PLAY)],
        source="s",
        won=True,
    )
    filtered = _filter_decision_by_type(decision, int(OptionType.PLAY))
    assert filtered is not None
    assert filtered.features == [[1.0], [3.0]]
    assert filtered.labels == [0, 0]
    assert filtered.option_types == [int(OptionType.PLAY), int(OptionType.PLAY)]
    assert _filter_decision_by_type(decision, int(OptionType.END)) is None


def test_aggregate_decision_by_type_creates_type_ranking_query() -> None:
    def row(card_id: int, attack_damage: float = 0.0) -> list[float]:
        values = [0.0] * len(FEATURE_NAMES)
        values[FEATURE_NAMES.index("context")] = float(SelectContext.MAIN)
        values[FEATURE_NAMES.index("turn")] = 7.0
        values[FEATURE_NAMES.index("card_id")] = float(card_id)
        values[FEATURE_NAMES.index("effective_attack_damage")] = attack_damage
        return values

    decision = Decision(
        features=[row(100), row(200), row(0)],
        labels=[0, 1, 0],
        option_types=[int(OptionType.PLAY), int(OptionType.PLAY), int(OptionType.END)],
        source="s",
        won=True,
        head="main",
    )
    aggregated = _aggregate_decision_by_type(decision)
    assert aggregated is not None
    assert all(len(features) == len(TYPE_FEATURE_NAMES) for features in aggregated.features)
    play = aggregated.features[0]
    assert play[TYPE_FEATURE_NAMES.index("candidate_count")] == 2.0
    assert play[TYPE_FEATURE_NAMES.index("total_candidate_count")] == 3.0
    assert "card_id" not in TYPE_FEATURE_NAMES
    assert play[TYPE_FEATURE_NAMES.index("turn")] == 7.0
    assert aggregated.labels == [1, 0]
    assert aggregated.option_types == [int(OptionType.PLAY), int(OptionType.END)]


def test_type_balance_upweights_rare_selected_types() -> None:
    play = Decision([[0.0], [1.0]], [1, 0], "s", True, option_types=[7, 13])
    attack = Decision([[0.0], [1.0]], [0, 1], "s", True, option_types=[7, 13])
    balanced = balance_type_decisions([play, play, play, attack], maximum_weight=6.0)
    assert balanced[-1].weight == 3.0
    assert balanced[0].weight == 1.0


def test_main_type_head_lookup() -> None:
    play_head = {"oblivious_trees": [], "scale_and_bias": [1, [0]]}
    model = {"heads": {"main_play": play_head, "main": {"oblivious_trees": [], "scale_and_bias": [1, [0]]}}}
    name, head = _main_type_head(model, SimpleNamespace(type=OptionType.PLAY))
    assert (name, head) == ("main_play", play_head)
    assert _main_type_head(model, SimpleNamespace(type=OptionType.ATTACK)) == ("main_attack", None)
    assert _main_type_head(None, SimpleNamespace(type=OptionType.PLAY)) == ("main_play", None)


def test_target_index_rejects_named_player_on_different_deck(monkeypatch) -> None:
    replay = {"info": {"TeamNames": ["Majkel1337", "opponent"]}}
    monkeypatch.setattr(
        "tools.replay_imitation.full_deck",
        lambda _replay, index: [1] * 60 if index == 0 else [2] * 60,
    )
    assert target_index(replay, "Majkel1337", tuple([2] * 60)) is None
    assert target_index(replay, "unknown", tuple([2] * 60)) == 1


def test_rankable_decisions_remove_constant_and_singleton_queries() -> None:
    decisions = [
        Decision(features=[[1.0], [2.0]], labels=[0, 1], source="s", won=True),
        Decision(features=[[1.0], [2.0]], labels=[0, 0], source="s", won=True),
        Decision(features=[[1.0]], labels=[1], source="s", won=True),
    ]
    assert rankable_decisions(decisions) == decisions[:1]


def test_catboost_receives_group_weights(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeRanker:
        def __init__(self, **params):
            captured["params"] = params

        def fit(self, features, labels, **kwargs):
            captured["features"] = features
            captured["labels"] = labels
            captured.update(kwargs)

        def save_model(self, path, format):
            Path(path).write_text('{"oblivious_trees": [], "scale_and_bias": [1, [0]]}', encoding="utf-8")

    monkeypatch.setitem(sys.modules, "catboost", SimpleNamespace(CatBoostRanker=FakeRanker))
    _train_catboost(
        [[1.0], [2.0]],
        [0, 1],
        [2],
        [0.25, 0.75],
        {"task_type": "CPU", "n_jobs": 1},
    )
    assert captured["group_id"].tolist() == [0, 0]
    assert captured["group_weight"].tolist() == [0.25, 0.75]
    assert captured["params"]["eval_metric"] == "NDCG:top=3"


def test_query_softmax_uses_compatible_ranking_metric(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeRanker:
        def __init__(self, **params):
            captured["params"] = params

        def fit(self, *_args, **_kwargs):
            pass

        def save_model(self, path, format):
            Path(path).write_text('{"oblivious_trees": [], "scale_and_bias": [1, [0]]}', encoding="utf-8")

    monkeypatch.setitem(sys.modules, "catboost", SimpleNamespace(CatBoostRanker=FakeRanker))
    _train_catboost(
        [[1.0], [2.0]], [0, 1], [2], [1.0, 1.0],
        {"task_type": "CPU", "loss_function": "QuerySoftMax"},
    )
    assert captured["params"]["loss_function"] == "QuerySoftMax"
    assert captured["params"]["eval_metric"] == "NDCG:top=3"


def test_type_heads_are_disabled_without_explicit_weight(monkeypatch) -> None:
    global_model = {"oblivious_trees": [{}]}
    type_model = {"oblivious_trees": [{}]}
    model = {"global": global_model, "heads": {"main_play": type_model}}
    obs = SimpleNamespace(
        select=SimpleNamespace(context=SelectContext.MAIN, option=[SimpleNamespace(type=OptionType.PLAY)]),
    )
    monkeypatch.setattr(meta_runtime, "resolve_model", lambda _model, _obs: (global_model, 0.0))
    monkeypatch.setattr(meta_runtime, "feature_vector", lambda _obs, _option: [0.0])
    monkeypatch.setattr(meta_runtime, "fallback_score", lambda _obs, _option, _config: 0.0)
    monkeypatch.setattr(meta_runtime, "model_score", lambda selected, _features: 1.0 if selected is global_model else 100.0)
    assert meta_runtime.score_options(obs, {"model_weight": 1.0}, model) == [1.0]
    assert meta_runtime.score_options(obs, {"model_weight": 1.0, "type_head_weight": 1.0}, model) == [101.0]
