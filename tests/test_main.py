from __future__ import annotations

from pathlib import Path

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


def test_kaggle_exec_without_file_reads_deck(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.csv").write_text(Path(main.__file__).with_name("deck.csv").read_text(), encoding="utf-8")
    namespace = {"__name__": "kaggle_submission"}

    exec(compile(Path(main.__file__).read_text(encoding="utf-8"), "main.py", "exec"), namespace)

    assert "__file__" not in namespace
    assert len(namespace["agent"]({"select": None, "current": None, "logs": []})) == 60
