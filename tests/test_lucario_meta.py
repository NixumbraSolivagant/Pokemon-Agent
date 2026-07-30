from __future__ import annotations

import io
import tarfile
from pathlib import Path

from tools.lucario_meta_search import SEARCH_PROFILES, deck_variants, read_deck, render_runtime


def write_submission(path: Path) -> Path:
    deck = Path("configs/lucario_replayfix_deck.csv").read_bytes()
    info = tarfile.TarInfo("deck.csv")
    info.size = len(deck)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(deck))
    return path


def test_lucario_runtime_renders_search_profile():
    source = Path("agents/lucario_meta.py").read_text(encoding="utf-8")
    rendered = render_runtime(source, SEARCH_PROFILES[2], {"test": [6] * 60})

    assert "SEARCH_TIME_BUDGET = 0.48" in rendered
    assert "SEARCH_BELIEF_WORLDS = 3" in rendered
    assert "OPPONENT_DECK_MODELS = {'test':" in rendered


def test_lucario_deck_variants_are_legal(tmp_path):
    deck = read_deck(write_submission(tmp_path / "base.tar.gz"))
    variants = deck_variants(deck)

    assert len(variants) >= 15
    assert all(len(candidate) == 60 for _, candidate in variants)
    assert any(LEGACY_ENERGY in candidate for _, candidate in variants)


def test_lucario_runtime_executes_without_file(tmp_path, monkeypatch):
    deck = read_deck(write_submission(tmp_path / "base.tar.gz"))
    (tmp_path / "deck.csv").write_text("\n".join(map(str, deck)) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "kaggle_submission"}

    exec(compile(Path(__file__).parents[1].joinpath("agents/lucario_meta.py").read_text(encoding="utf-8"), "main.py", "exec"), namespace)

    assert "__file__" not in namespace
    assert len(namespace["agent"]({"select": None})) == 60


LEGACY_ENERGY = 12
