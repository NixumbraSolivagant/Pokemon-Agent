from __future__ import annotations

import tarfile
from pathlib import Path

from tools.build_models import BuildConfig as ModelBuildConfig
from tools.build_submission import BuildConfig, build_submission
from tools.deck_rules import apply_deck_swaps, validate_deck_ids
from tools.export_kaggle_submission import SEARCH_WRAPPER_MARKER, export_kaggle_submission
from tools.gold_factory import generate_candidate_configs, profile_from_name


def _read_member(path: Path, member: str) -> str:
    with tarfile.open(path, "r:gz") as tar:
        f = tar.extractfile(member)
        assert f is not None
        return f.read().decode("utf-8")


def test_smoke_profile_is_small():
    profile = profile_from_name("smoke")
    assert profile.population <= 12
    assert profile.stage_a.games_per_pair == 1
    assert profile.manual_slots == 3


def test_build_config_compatibility_export_uses_canonical_model():
    assert BuildConfig is ModelBuildConfig


def test_search_wrapper_embeds_belief_configuration(tmp_path: Path):
    cfg = BuildConfig(
        name="belief_wrapper",
        base=Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
        out=tmp_path / "belief_wrapper.tar.gz",
        belief_worlds=6,
        risk_penalty=0.35,
        opponent_decks={"test_archetype": [1] * 60},
    )

    build_submission(cfg)
    main_py = _read_member(cfg.out, "main.py")

    assert "GT_BELIEF_WORLDS = 6" in main_py
    assert "GT_RISK_PENALTY = 0.35" in main_py
    assert "'test_archetype':" in main_py
    assert "def _gt_sample_hidden_worlds" in main_py


def test_deck_rules_apply_swaps_without_build_dependencies():
    original = "\n".join(["1"] * 56 + ["1121"] * 4) + "\n"

    swapped = apply_deck_swaps(original, [(1122, 1121)])
    deck = [int(line) for line in swapped.splitlines()]

    validate_deck_ids(deck)
    assert deck.count(1121) == 3
    assert deck.count(1122) == 1


def test_lucario_candidate_mutates_both_deck_files(tmp_path: Path):
    cfg = BuildConfig(
        name="lucario_sync",
        family="lucario",
        base=Path("outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz"),
        out=tmp_path / "lucario_sync.tar.gz",
        injection="none",
        enable_search=False,
        deck_swaps=[(1182, 1213)],
        deck_files=("deck.csv", "lucario_deck.csv"),
    )
    build_submission(cfg)
    deck = _read_member(cfg.out, "deck.csv")
    lucario_deck = _read_member(cfg.out, "lucario_deck.csv")
    assert deck == lucario_deck
    assert deck.splitlines().count("1182") == 4
    assert deck.splitlines().count("1213") == 0


def test_candidate_generation_contains_cross_archetype_seeds(tmp_path: Path):
    configs = generate_candidate_configs(tmp_path, 20)
    families = {cfg.family for cfg in configs}
    assert "great_tusk" in families
    assert "metal_tempo" in families
    assert "lucario" in families


def test_kaggle_export_keeps_search_wrapper_by_default(tmp_path: Path):
    cfg = BuildConfig(
        name="search_export_default",
        base=Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
        out=tmp_path / "internal.tar.gz",
        injection="great_tusk",
        enable_search=True,
    )
    build_submission(cfg)

    exported = export_kaggle_submission(cfg.out, tmp_path / "submission.tar.gz")
    main_py = _read_member(exported, "main.py")

    assert SEARCH_WRAPPER_MARKER in main_py
    assert "def _gt_search_action" in main_py
    assert main_py.rfind("kaggle_agent = agent") > main_py.rfind("def _gt_search_action")
    assert main_py.rfind("kaggle_agent = agent") > main_py.rfind("def agent")


def test_kaggle_export_can_strip_search_wrapper_explicitly(tmp_path: Path):
    cfg = BuildConfig(
        name="search_export_strip",
        base=Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
        out=tmp_path / "internal.tar.gz",
        injection="great_tusk",
        enable_search=True,
    )
    build_submission(cfg)

    exported = export_kaggle_submission(
        cfg.out,
        tmp_path / "submission.tar.gz",
        strip_search_wrapper=True,
    )
    main_py = _read_member(exported, "main.py")

    assert SEARCH_WRAPPER_MARKER not in main_py
    assert "def _gt_search_action" not in main_py


def test_deck_override_writes_exact_legal_deck(tmp_path: Path):
    deck_override = [1] * 56 + [1121] * 4
    cfg = BuildConfig(
        name="override_exact",
        base=Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
        out=tmp_path / "override_exact.tar.gz",
        injection="great_tusk",
        deck_override=deck_override,
        strategy_weights={"opp_mill": 1234.0},
        policy_variant="test_variant",
        origin="test",
    )
    build_submission(cfg)

    deck = [int(line) for line in _read_member(cfg.out, "deck.csv").splitlines()]
    metadata = _read_member(cfg.out, "build_metadata.json")

    assert deck == deck_override
    assert '"opp_mill": 1234.0' in metadata
    assert '"policy_variant": "test_variant"' in metadata


def test_search_wrapper_injection_is_idempotent(tmp_path: Path):
    first = BuildConfig(
        name="first_wrapper",
        base=Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
        out=tmp_path / "first.tar.gz",
        injection="great_tusk",
    )
    build_submission(first)
    second = BuildConfig(
        name="second_wrapper",
        base=first.out,
        out=tmp_path / "second.tar.gz",
        injection="great_tusk",
        strategy_weights={"opp_mill": 999.0},
    )
    build_submission(second)

    main_py = _read_member(second.out, "main.py")

    assert main_py.count(SEARCH_WRAPPER_MARKER) == 1
    assert "'opp_mill': 999.0" in main_py
