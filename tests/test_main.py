from __future__ import annotations

import main


def test_read_deck_csv_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    deck = main.read_deck_csv()

    assert len(deck) == 60


def test_fallback_action_respects_selection_bounds():
    observation = {
        "select": {
            "option": [{}, {}, {}],
            "minCount": 2,
            "maxCount": 1,
        }
    }

    assert main._fallback_action(observation) == [0]
