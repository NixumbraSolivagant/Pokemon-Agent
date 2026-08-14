from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.kaggle_eval_calibration import (
    calibrate,
    derive_episode_weights,
    isotonic_predict,
    pairwise_order_accuracy,
    pava_fit,
    spearman_correlation,
    validate_config,
)


def test_validation_rejects_duplicate_submission_ids():
    data = {
        "submissions": [
            {"submission_id": 7, "package": "missing-a", "package_sha256": "a" * 64},
            {"submission_id": 7, "package": "missing-b", "package_sha256": "b" * 64},
        ],
        "opponents": [],
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_config(data)


def test_validation_allows_explicitly_excluded_duplicate():
    data = {
        "excluded_submission_ids": [55188036],
        "submissions": [
            {"submission_id": 55188036, "package": "missing-a", "package_sha256": "a" * 64},
            {"submission_id": 55188036, "package": "missing-b", "package_sha256": "b" * 64},
        ],
        "opponents": [],
    }
    validate_config(data)


def test_validation_rejects_package_sha_mismatch(tmp_path: Path):
    package = tmp_path / "submission.tar.gz"
    package.write_bytes(b"payload")
    data = {
        "submissions": [{"submission_id": 8, "package": str(package), "package_sha256": "0" * 64}],
        "opponents": [],
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_config(data)


def test_isotonic_mapping_is_monotonic():
    model = pava_fit([0.1, 0.2, 0.3, 0.4], [700, 650, 800, 790])
    predictions = [isotonic_predict(model, value) for value in (0.1, 0.2, 0.3, 0.4)]

    assert predictions == sorted(predictions)


def test_order_metrics_reward_correct_candidate_ordering():
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert pairwise_order_accuracy([1, 2, 3], [10, 20, 30]) == 1.0


def test_episode_weights_report_unknown_opponents_and_seats():
    episodes = {
        10: [
            {"agents": [{"submissionId": 10}, {"submissionId": 20}]},
            {"agents": [{"submissionId": 99}, {"submissionId": 10}]},
        ]
    }

    result = derive_episode_weights(episodes, [{"name": "known", "submission_ids": [20]}])

    assert result["derived_opponent_weights"] == {"known": 1.0}
    assert result["derived_seat_weights"] == {"0": 0.5, "1": 0.5}
    assert result["episode_weight_coverage"]["unknown_opponent_coverage"] == 0.5


def test_calibration_reports_insufficient_data(tmp_path: Path):
    package_sha = "a" * 64
    report = {
        "submissions": [{"name": "candidate", "sha256": package_sha}, {"name": "known", "sha256": "b" * 64}],
        "games": [{"p0": "candidate", "p1": "known", "winner": "candidate", "loser": "known", "outcome": "P0_WIN", "ranking_eligible": True}],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    data = {
        "submissions": [{
            "submission_id": 1,
            "package": "missing",
            "package_sha256": package_sha,
            "confirmed": True,
            "observed_public_score": 700,
            "local_report": str(report_path),
        }],
        "opponents": [{"name": "known", "package": "missing", "weight": 1}],
    }

    result = calibrate(data, tmp_path)

    assert result["prediction_status"] == "insufficient_data"
    assert result["auto_submit_allowed"] is False
    assert result["rows"][0]["unknown_opponent_coverage"] == 0
