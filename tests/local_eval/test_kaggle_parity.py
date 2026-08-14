from __future__ import annotations

from pathlib import Path

import pytest

from local_eval.kaggle_spec import CABT_CONFIGURATION, OFFICIAL_CG_SHA256, cg_hash_report
from local_eval.models import EvalConfig
from local_eval.referee import _agent_failure, _finished, validate_kaggle_deck_action


def test_kaggle_profile_locks_official_runtime_values():
    config = EvalConfig(
        profile="kaggle",
        act_timeout_s=99,
        overage_time_s=1,
        run_timeout_s=2,
        max_actions=3,
    )

    assert config.act_timeout_s == CABT_CONFIGURATION["actTimeout"]
    assert config.overage_time_s == CABT_CONFIGURATION["remainingOverageTime"]
    assert config.run_timeout_s == CABT_CONFIGURATION["runTimeout"]
    assert config.max_actions == CABT_CONFIGURATION["episodeSteps"]


def test_kaggle_profile_rejects_fake_common_random_seeds():
    with pytest.raises(ValueError, match="random_device"):
        EvalConfig(profile="kaggle", common_random_seeds=True)


def test_legacy_profile_keeps_existing_defaults():
    config = EvalConfig()

    assert config.profile == "legacy"
    assert config.act_timeout_s == 6.0
    assert config.overage_time_s == 12.0
    assert config.run_timeout_s == 1200.0
    assert config.max_actions == 1000


def test_kaggle_deck_validation_leaves_card_legality_to_engine():
    assert validate_kaggle_deck_action([999] * 60) == [999] * 60
    assert validate_kaggle_deck_action(["999"] * 60) == ["999"] * 60
    with pytest.raises(ValueError):
        validate_kaggle_deck_action([1] * 59)


def test_normal_results_have_official_statuses_and_rewards(monkeypatch):
    monkeypatch.setattr("local_eval.referee.time.perf_counter", lambda: 10.0)

    win = _finished("g", "a", "b", 1, 0, 3, 9.0)
    draw = _finished("d", "a", "b", 1, 2, 4, 9.0)

    assert (win.p0_status, win.p1_status, win.p0_reward, win.p1_reward) == ("DONE", "DONE", 1, -1)
    assert (draw.p0_reward, draw.p1_reward) == (0, 0)


def test_kaggle_action_failure_preserves_core_reward_nulling(monkeypatch):
    monkeypatch.setattr("local_eval.referee.time.perf_counter", lambda: 10.0)

    result = _agent_failure(
        EvalConfig(profile="kaggle"), "g", "a", "b", 1, 0,
        "AGENT_ERROR", "ERROR", 2, 9.0, "boom",
    )

    assert result.winner == "b"
    assert result.p0_status == "ERROR"
    assert result.p1_status == "DONE"
    assert result.p0_reward is None
    assert result.p1_reward == 1
    assert result.ranking_eligible is True


def test_kaggle_initial_deck_failure_has_no_reward(monkeypatch):
    monkeypatch.setattr("local_eval.referee.time.perf_counter", lambda: 10.0)

    result = _agent_failure(
        EvalConfig(profile="kaggle"), "g", "a", "b", 1, 0,
        "DECK_ERROR", "INVALID", 0, 9.0, "bad deck", initial=True,
    )

    assert result.outcome == "NO_RESULT"
    assert result.winner is None
    assert result.p0_status == "INVALID"
    assert result.p1_status == "DONE"
    assert result.p0_reward is None and result.p1_reward is None


def test_repository_cg_matches_frozen_official_hashes():
    report = cg_hash_report(Path.cwd())

    assert report["expected"] == OFFICIAL_CG_SHA256
    assert report["matches_official"] is True
