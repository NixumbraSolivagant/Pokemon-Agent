from __future__ import annotations

import json
import tarfile
from pathlib import Path

from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.build_models import BuildConfig
from tools.build_submission import build_submission
from tools.opponent_models import OpponentArchive, base_opponent_genomes, generate_counter_opponents
from tools.racing import classify_roles, rank_for_racing, robust_stats


def test_builder_does_not_require_reference_archive(tmp_path: Path):
    output = tmp_path / "canonical.tar.gz"
    cfg = BuildConfig(
        name="canonical",
        base=tmp_path / "missing-reference.tar.gz",
        out=output,
        deck_override=[1] * 56 + [1121] * 4,
    )

    build_submission(cfg)

    with tarfile.open(output, "r:gz") as tar:
        names = set(tar.getnames())
        metadata = json.loads(tar.extractfile("build_metadata.json").read().decode("utf-8"))
    assert {"main.py", "deck.csv", "cg/game.py", "cg/api.py"} <= names
    assert metadata["reference_free_runtime"] is True


def test_counter_opponents_are_reproducible_and_archived(tmp_path: Path):
    first = generate_counter_opponents(8, 123, parents=base_opponent_genomes())
    second = generate_counter_opponents(8, 123, parents=base_opponent_genomes())
    archive = OpponentArchive(base_opponent_genomes())

    assert first == second
    assert any(archive.add(genome) for genome in first)
    path = tmp_path / "opponents.json"
    archive.save(path)
    assert len(OpponentArchive.load(path).genomes()) >= len(base_opponent_genomes())


def test_racing_prefers_stable_candidate_and_assigns_three_roles():
    stable = AgentStats(
        name="stable",
        tarball="stable.tar.gz",
        sha256="a",
        games=40,
        wins=28,
        losses=12,
        opponents={
            "fast_ko": {"wins": 7, "losses": 3, "draws": 0},
            "mill_control": {"wins": 7, "losses": 3, "draws": 0},
            "wall": {"wins": 7, "losses": 3, "draws": 0},
            "denial": {"wins": 7, "losses": 3, "draws": 0},
        },
    )
    volatile = AgentStats(
        name="volatile",
        tarball="volatile.tar.gz",
        sha256="b",
        games=40,
        wins=30,
        losses=10,
        opponents={
            "fast_ko": {"wins": 10, "losses": 0, "draws": 0},
            "mill_control": {"wins": 0, "losses": 10, "draws": 0},
            "wall": {"wins": 10, "losses": 0, "draws": 0},
            "denial": {"wins": 10, "losses": 0, "draws": 0},
        },
    )
    specialist = AgentStats(
        name="specialist",
        tarball="specialist.tar.gz",
        sha256="c",
        games=40,
        wins=24,
        losses=16,
        opponents={
            "fast_ko": {"wins": 8, "losses": 2, "draws": 0},
            "mill_control": {"wins": 8, "losses": 2, "draws": 0},
            "wall": {"wins": 4, "losses": 6, "draws": 0},
            "denial": {"wins": 4, "losses": 6, "draws": 0},
        },
    )
    report = MatchReport(EvalConfig(), [], [], [stable, volatile, specialist])

    assert robust_stats(stable)["robust_score"] > robust_stats(volatile)["robust_score"]
    assert rank_for_racing(report, {"stable", "volatile", "specialist"}, 1) == ["stable"]
    assert set(classify_roles(report, {"stable", "volatile", "specialist"}).values()) >= {
        "generalist",
        "anti_fast_ko",
        "anti_control",
    }
