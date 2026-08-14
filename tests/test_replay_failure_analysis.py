from __future__ import annotations

from tools.kaggle_replay_miner import attack_metrics, classify_failure


def test_attack_metrics_detects_first_land_collapse():
    events = [
        {"type": 15, "playerIndex": 1, "attackId": 100, "turn": 3, "step": 8},
        {"type": 15, "playerIndex": 0, "attackId": 62, "turn": 14, "step": 42},
        {"type": 15, "playerIndex": 0, "attackId": 62, "turn": 16, "step": 48},
    ]

    metrics = attack_metrics(events, 0)

    assert metrics["attack_count"] == 2
    assert metrics["land_collapse_count"] == 2
    assert metrics["first_land_collapse_turn"] == 14


def test_failure_classifies_self_deckout_late_setup_and_abomasnow():
    row = {
        "reward": -1,
        "target_deck_remaining": 0,
        "opponent_deck_remaining": 17,
        "land_collapse_count": 1,
        "first_land_collapse_turn": 41,
        "target_prizes_remaining": 6,
        "opponent_archetype": "mega_abomasnow",
    }

    labels = classify_failure(row)

    assert labels[0] == "self_deckout"
    assert "late_land_collapse" in labels
    assert "no_prizes_taken" in labels
    assert "abomasnow_route_failure" in labels


def test_failure_classifies_crustle_without_mill_attack():
    labels = classify_failure(
        {
            "reward": -1,
            "target_deck_remaining": 12,
            "opponent_deck_remaining": 20,
            "land_collapse_count": 0,
            "first_land_collapse_turn": None,
            "target_prizes_remaining": 5,
            "opponent_archetype": "crustle",
        }
    )

    assert "no_land_collapse" in labels
    assert "wall_or_mirror" in labels
