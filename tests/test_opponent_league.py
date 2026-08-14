from __future__ import annotations

import gzip
import json
import threading
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import tools.opponent_league_pipeline as opponent_league
from tools.opponent_league_pipeline import AUTHENTICATION_ERROR, CLONE_ALGORITHM_VERSION, CLONE_PROFILES, behavior_signature, build_sources, deterministic_sample, download_episode, ensure_shared_pretrain, is_rate_limit_error, parser, quality_gate, quality_pass, retry_after_seconds, select_episodes, select_teams
from tools.ogerpon_residual_pipeline import load_opponent_manifest, validate_pool_splits
from tools.replay_imitation import Decision, derive_generic_policy, read_replays


def test_select_teams_uses_fixed_rank_bands():
    rows = [{"rank": rank, "team_id": rank} for rank in range(1, 6219)]
    selected = select_teams(rows)
    assert len(selected) == 500
    assert sum(int(row["rank"]) <= 100 for row in selected) == 100
    assert sum(101 <= int(row["rank"]) <= 500 for row in selected) == 150
    assert sum(501 <= int(row["rank"]) <= 1500 for row in selected) == 120
    assert sum(1501 <= int(row["rank"]) <= 3000 for row in selected) == 80
    assert sum(int(row["rank"]) > 3000 for row in selected) == 50


def test_deterministic_sample_keeps_range_edges():
    rows = [{"rank": rank} for rank in range(101, 501)]
    sample = deterministic_sample(rows, 50)
    assert len(sample) == 50
    assert sample[0]["rank"] == 101
    assert sample[-1]["rank"] == 500


def test_select_episodes_filters_and_combines_recent_with_history():
    rows = [
        {"id": index, "state": "EpisodeState.COMPLETED", "type": "EpisodeType.EPISODE_TYPE_PUBLIC", "createTime": f"2026-08-04T00:{index:02d}:00"}
        for index in range(20)
    ]
    rows.append({"id": 99, "state": "EpisodeState.ERRORED", "type": "EpisodeType.EPISODE_TYPE_PUBLIC", "createTime": "2026-08-04T01:00:00"})
    selected = select_episodes(rows, 10)
    assert len(selected) == 10
    assert 99 not in selected
    assert 19 in selected
    assert min(selected) < 10


def test_quality_gate_requires_semantic_and_critical_groups():
    result = {
        "semantic_rate": 0.90,
        "errors": 0,
        "by_group": {
            "ability": {"semantic_rate": 0.86},
            "attack:1": {"semantic_rate": 0.92},
            "SETUP": {"semantic_rate": 0.91},
        },
    }
    assert quality_pass(result)
    result["by_group"]["ability"]["semantic_rate"] = 0.80
    assert not quality_pass(result)


def test_quality_gate_aggregates_attack_groups_and_ignores_rare_group_veto():
    result = {
        "semantic_rate": 0.94,
        "errors": 0,
        "by_group": {
            "ability": {"decisions": 40, "semantic": 38, "semantic_rate": 0.95},
            "attack:common": {"decisions": 100, "semantic": 95, "semantic_rate": 0.95},
            "attack:rare": {"decisions": 1, "semantic": 0, "semantic_rate": 0.0},
            "SETUP": {"decisions": 30, "semantic": 29, "semantic_rate": 29 / 30},
        },
    }
    gate = quality_gate(result)
    assert gate["passed"]
    assert gate["metrics"]["attack_decisions"] == 101
    assert gate["metrics"]["attack"] > 0.94


def test_quality_gate_rejects_insufficient_attack_coverage():
    result = {
        "semantic_rate": 0.95,
        "errors": 0,
        "by_group": {
            "ability": {"decisions": 5, "semantic": 5, "semantic_rate": 1.0},
            "attack:only": {"decisions": 10, "semantic": 10, "semantic_rate": 1.0},
            "SETUP": {"decisions": 5, "semantic": 5, "semantic_rate": 1.0},
        },
    }
    gate = quality_gate(result)
    assert not gate["passed"]
    assert "attack_coverage" in gate["failed"]


def test_quality_gate_uses_turn_fidelity_and_keeps_legacy_semantic_diagnostic():
    result = {
        "semantic_rate": 0.60,
        "errors": 0,
        "turn_fidelity": {
            "main_macro_recall": 0.84,
            "attack_recall": 0.90,
            "ability_recall": 0.88,
            "terminal_recall": 0.75,
            "action_jaccard_median": 0.80,
        },
        "by_group": {
            "ability": {"decisions": 40, "semantic": 36, "semantic_rate": 0.90},
            "attack:1": {"decisions": 40, "semantic": 38, "semantic_rate": 0.95},
            "end": {"decisions": 20, "semantic": 15, "semantic_rate": 0.75},
        },
    }
    gate = quality_gate(result)
    assert gate["passed"]
    assert gate["metrics"]["legacy_semantic_passed"] is False


def test_quality_gate_requires_v5_stopping_accuracy():
    result = {
        "semantic_rate": 0.95,
        "errors": 0,
        "turn_fidelity": {
            "main_macro_recall": 0.90,
            "attack_recall": 0.90,
            "ability_recall": 0.90,
            "terminal_recall": 0.80,
            "action_jaccard_median": 0.80,
            "stop_balanced_accuracy": 0.84,
        },
        "by_group": {
            "ability": {"decisions": 40, "semantic": 38, "semantic_rate": 0.95},
            "attack:1": {"decisions": 40, "semantic": 38, "semantic_rate": 0.95},
            "end": {"decisions": 20, "semantic": 16, "semantic_rate": 0.80},
        },
    }
    gate = quality_gate(result)
    assert not gate["passed"]
    assert "stop_balanced" in gate["failed"]


def test_clone_profiles_make_deep_heads_deeper_and_softmax_compatible():
    profiles = dict(CLONE_PROFILES)
    assert profiles["deep"]["head_num_leaves"] > profiles["balanced"]["head_num_leaves"]
    assert profiles["softmax"]["loss_function"] == "QuerySoftMax"
    assert profiles["softmax"]["eval_metric"].startswith("NDCG")


def test_shared_pretrain_excludes_pilot_targets(tmp_path: Path, monkeypatch):
    rows = []
    for index in range(3):
        replay_path = Path("replays") / f"source_{index}"
        (tmp_path / replay_path).mkdir(parents=True)
        rows.append((index + 1, {
            "source_id": f"source_{index}",
            "replay_path": str(replay_path),
            "team_name": f"team_{index}",
            "rank": index + 1,
        }, None))

    def fake_train(manifest: Path, artifact: Path):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert {row["name"] for row in data["sources"]} == {"source_1", "source_2"}
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({
            "version": 5,
            "clone_algorithm_version": CLONE_ALGORITHM_VERSION,
        }), encoding="utf-8")
        return {}

    monkeypatch.setattr(opponent_league, "train", fake_train)
    artifact = ensure_shared_pretrain(
        tmp_path,
        rows,
        [rows[0]],
        Namespace(
            disable_shared_pretrain=False,
            force_shared_pretrain=False,
            shared_pretrain_sources=2,
            seed=1,
            model_jobs=2,
            task_type="CPU",
            gpu_devices="",
        ),
    )
    assert artifact == tmp_path / "artifacts" / f"shared_pretrain_v{CLONE_ALGORITHM_VERSION}.json"


def test_behavior_signature_is_stable_for_equivalent_audits():
    first = {"semantic_rate": 0.901, "by_group": {"ability": {"decisions": 10, "semantic_rate": 0.861}}}
    second = {"by_group": {"ability": {"semantic_rate": 0.864, "decisions": 10}}, "semantic_rate": 0.904}
    assert behavior_signature(first) == behavior_signature(second)


def test_read_replays_accepts_gzip(tmp_path: Path):
    replay = {"info": {"TeamNames": ["a", "b"]}, "steps": []}
    path = tmp_path / "episode-123-replay.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(replay, handle)
    rows = read_replays(tmp_path)
    assert len(rows) == 1
    assert rows[0]["_local_episode_id"] == "123"


def test_generic_policy_uses_selected_teacher_cards_and_attacks():
    decision = Decision(
        features=[[0.0, 0.0, 96.0, 120.0], [0.0, 0.0, 1.0, 0.0]],
        labels=[1, 0], source="teacher", won=True,
    )
    policy = derive_generic_policy([decision])
    assert policy["play_priority"]["96"] == 1400.0
    assert policy["attack_priority"]["120"] == 1400.0


def test_build_sources_runs_four_models_in_parallel(tmp_path: Path, monkeypatch):
    sources = []
    for index in range(4):
        source_id = f"source_{index}"
        replay_path = Path("replays") / source_id
        replay_dir = tmp_path / replay_path
        replay_dir.mkdir(parents=True)
        (replay_dir / "episode-1-replay.json").write_text("{}", encoding="utf-8")
        sources.append({
            "source_id": source_id,
            "replay_path": str(replay_path),
            "team_name": source_id,
            "rank": index + 1,
        })
    (tmp_path / "snapshot.json").write_text(json.dumps({"sources": sources}), encoding="utf-8")

    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_train(manifest: Path, artifact: Path):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
        with lock:
            active -= 1
        return {
            "deck_sha256": "deck",
            "split_stats": {
                "train_decisions": 1000,
                "validation_decisions": 100,
                "test_decisions": 100,
                "test_episodes": [],
            },
        }

    def fake_build_submission(_artifact: Path, package: Path, *_args):
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_bytes(b"package")

    def fake_audit(*_args):
        return {
            "semantic_rate": 0.95,
            "errors": 0,
            "by_group": {
                "ability": {"semantic_rate": 0.95},
                "attack:1": {"semantic_rate": 0.95},
                "SETUP": {"semantic_rate": 0.95},
            },
        }

    monkeypatch.setattr(opponent_league, "train", fake_train)
    monkeypatch.setattr(opponent_league, "build_submission", fake_build_submission)
    monkeypatch.setattr(opponent_league, "audit", fake_audit)
    build_sources(Namespace(
        root=tmp_path,
        runtime=Path("runtime.py"),
        cg_dir=Path("cg"),
        min_episodes=1,
        min_decisions=1000,
        build_workers=4,
        model_jobs=12,
        seed=20260804,
    ))

    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert maximum_active == 4
    assert len(registry["entries"]) == 4
    assert {entry["status"] for entry in registry["entries"]} == {"qualified"}


def test_build_sources_rebuilds_stale_algorithm_version(tmp_path: Path, monkeypatch):
    source = {"source_id": "stale", "replay_path": "replays/stale", "team_name": "stale", "rank": 5}
    replay_dir = tmp_path / source["replay_path"]
    replay_dir.mkdir(parents=True)
    (replay_dir / "episode-1-replay.json").write_text("{}", encoding="utf-8")
    (tmp_path / "snapshot.json").write_text(json.dumps({"sources": [source]}), encoding="utf-8")
    (tmp_path / "registry.json").write_text(json.dumps({"entries": [{
        **source,
        "replay_count": 1,
        "status": "qualified",
        "clone_algorithm_version": CLONE_ALGORITHM_VERSION - 1,
    }]}), encoding="utf-8")
    calls = []

    def fake_build_source(_root, selected, *_args, **_kwargs):
        calls.append(selected["source_id"])
        return {**selected, "status": "qualified", "clone_algorithm_version": CLONE_ALGORITHM_VERSION}

    monkeypatch.setattr(opponent_league, "build_source", fake_build_source)
    build_sources(Namespace(
        root=tmp_path, min_episodes=1, build_workers=1, model_jobs=1, seed=1,
        force_rebuild=False, pilot_size=0,
    ))
    assert calls == ["stale"]


def test_residual_pool_manifests_are_disjoint_and_match_summary(tmp_path: Path):
    splits = {}
    for split, count in (("train", 48), ("dev", 16), ("holdout", 16)):
        rows = []
        for index in range(count):
            package = tmp_path / "packages" / f"{split}_{index}.tar.gz"
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_bytes(b"package")
            rows.append({"path": str(package)})
        manifest = tmp_path / f"{split}.json"
        manifest.write_text(json.dumps({"split": split, "opponents": rows}), encoding="utf-8")
        splits[split] = load_opponent_manifest(manifest, split)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "selected": 80,
        "splits": {"train": 48, "dev": 16, "holdout": 16},
    }), encoding="utf-8")

    result = validate_pool_splits(summary, splits, minimum_size=80)
    assert result["selected"] == 80


def test_residual_pool_rejects_cross_split_leakage(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    package = tmp_path / "same.tar.gz"
    package.write_bytes(b"package")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "selected": 3,
        "splits": {"train": 1, "dev": 1, "holdout": 1},
    }), encoding="utf-8")

    try:
        validate_pool_splits(
            summary,
            {"train": [package], "dev": [package], "holdout": [tmp_path / "other.tar.gz"]},
            minimum_size=3,
        )
    except RuntimeError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("cross-split leakage was accepted")


def test_replay_download_stops_retrying_on_authentication_failure(tmp_path: Path, monkeypatch):
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="Authentication required to call the Kaggle API.")

    monkeypatch.setattr(opponent_league.subprocess, "run", fake_run)
    key, error = download_episode(
        "kaggle",
        "source:123",
        123,
        tmp_path / "staging",
        tmp_path / "replays",
        max_retries=0,
    )

    assert key == "source:123"
    assert error is not None and error.startswith(f"{AUTHENTICATION_ERROR}:")
    assert calls == 1


def test_collection_defaults_are_rate_limited():
    args = parser().parse_args(["collect", "--root", "pool"])
    assert args.workers == 2
    assert args.request_interval == 1.55
    assert args.request_jitter == 0.0
    assert args.max_retries == 5
    assert args.max_backoff == 300.0
    assert retry_after_seconds("Retry-After: 45", 10.0) == 45.0
    assert is_rate_limit_error("HTTP 429 Too Many Requests")


def test_replay_download_retries_rate_limits_until_success(tmp_path: Path, monkeypatch):
    calls = 0

    class FakePacer:
        def __init__(self):
            self.cooldowns = []

        def wait(self):
            return None

        def cooldown(self, seconds):
            self.cooldowns.append(seconds)

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            return SimpleNamespace(returncode=1, stdout="", stderr="HTTP 429 Too Many Requests; Retry-After: 2")
        work = Path(command[command.index("-p") + 1])
        (work / "download-replay.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    pacer = FakePacer()
    monkeypatch.setattr(opponent_league.subprocess, "run", fake_run)
    key, error = download_episode(
        "kaggle",
        "source:123",
        123,
        tmp_path / "staging",
        tmp_path / "replays",
        pacer=pacer,
        max_retries=0,
    )
    assert key == "source:123"
    assert error is None
    assert calls == 3
    assert pacer.cooldowns == [2.0, 2.0]


def test_build_defaults_to_gpu_catboost():
    args = parser().parse_args(["build", "--root", "pool"])
    assert args.task_type == "GPU"
    assert args.gpu_devices == "0"
