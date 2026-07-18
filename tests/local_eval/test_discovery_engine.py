from __future__ import annotations

from pathlib import Path

from local_eval.models import AgentStats, EvalConfig, MatchReport, SubmissionInfo
from argparse import Namespace

from tools.discovery_engine import profile_from_name, resolve_workers
from tools.discovery_space import generate_discovery_configs
from tools.psro import solve_psro


def test_smoke_discovery_profile_is_bounded():
    profile = profile_from_name("smoke_discovery")
    assert profile.population <= 12
    assert profile.stage_a.games_per_pair == 1
    assert profile.stage_d.games_per_pair == 1


def test_turbo_discovery_profile_uses_auto_workers_and_larger_population():
    normal = profile_from_name("a800_discovery")
    turbo = profile_from_name("a800_turbo_discovery")

    assert normal.workers == 0
    assert turbo.workers == 0
    assert turbo.population > normal.population
    assert turbo.stage_b.candidate_limit > normal.stage_b.candidate_limit


def test_resolve_workers_respects_explicit_and_auto_limits():
    profile = profile_from_name("a800_discovery")

    assert resolve_workers(Namespace(workers=17, cpu_headroom=2, max_workers=None), profile) == 17

    auto = resolve_workers(Namespace(workers=0, cpu_headroom=0, max_workers=3), profile)

    assert auto == 3


def test_discovery_generation_includes_motif_or_tag_deck_overrides(tmp_path: Path):
    configs = generate_discovery_configs(
        Path("outputs/submissions/champion_latest.tar.gz"),
        tmp_path,
        population=14,
        generation=1,
        seed=20260718,
    )
    assert len(configs) == 14
    assert any(cfg.deck_override for cfg in configs)
    assert any(cfg.origin.startswith("motif") or cfg.origin == "tag_sweep" for cfg in configs)


def test_psro_gives_weight_to_cycle_strategy():
    names = ["rock", "paper", "scissors"]
    stats = {
        name: AgentStats(name=name, tarball=f"{name}.tar.gz", sha256=name, games=2)
        for name in names
    }
    stats["rock"].opponents = {"scissors": {"wins": 2, "losses": 0, "draws": 0}, "paper": {"wins": 0, "losses": 2, "draws": 0}}
    stats["paper"].opponents = {"rock": {"wins": 2, "losses": 0, "draws": 0}, "scissors": {"wins": 0, "losses": 2, "draws": 0}}
    stats["scissors"].opponents = {"paper": {"wins": 2, "losses": 0, "draws": 0}, "rock": {"wins": 0, "losses": 2, "draws": 0}}
    report = MatchReport(
        config=EvalConfig(),
        submissions=[SubmissionInfo(name=name, tarball=f"{name}.tar.gz", sha256=name) for name in names],
        games=[],
        standings=[stats[name] for name in names],
    )

    result = solve_psro(report, names)

    assert set(result.weights) == set(names)
    assert all(weight > 0.2 for weight in result.weights.values())
