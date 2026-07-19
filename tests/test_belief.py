from __future__ import annotations

from collections import Counter
from pathlib import Path

from tools.belief import PublicBeliefState, infer_archetype, sample_hidden_worlds
from tools.leaderboard import LeaderboardObservation, append_observation, load_observations


def test_infer_archetype_uses_visible_card_overlap():
    decks = {"mill": [58, 58, 344], "metal": [999, 998, 997]}

    assert infer_archetype([58, 344], decks) == "mill"


def test_hidden_world_sampling_is_reproducible_and_partitions_cards():
    deck = [1] * 20 + [2] * 20 + [3] * 20
    state = PublicBeliefState(
        visible_ids=(1, 2),
        opponent_hand_count=7,
        opponent_deck_count=40,
        opponent_prize_count=6,
        seed=42,
    )

    first = sample_hidden_worlds(state, {"known": deck}, deck, count=4)
    second = sample_hidden_worlds(state, {"known": deck}, deck, count=4)

    assert first == second
    assert len(first) == 4
    for world in first:
        hidden = [*world.opponent_deck, *world.opponent_prize, *world.opponent_hand]
        assert len(hidden) == 53
        combined = Counter(hidden) + Counter(state.visible_ids)
        assert combined[1] <= 20
        assert combined[2] <= 20
        assert combined[3] <= 20


def test_leaderboard_observations_append_without_overwriting(tmp_path: Path):
    ledger = tmp_path / "leaderboard.jsonl"
    first = LeaderboardObservation("a.tar.gz", "a", 900.0, 20, "public", "2026-07-19T00:00:00Z")
    second = LeaderboardObservation("b.tar.gz", "b", 1000.0, 10, "public", "2026-07-19T01:00:00Z")

    append_observation(ledger, first)
    append_observation(ledger, second)

    assert load_observations(ledger) == [first, second]
