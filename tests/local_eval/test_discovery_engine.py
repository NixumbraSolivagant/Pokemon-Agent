from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from local_eval.models import AgentStats, EvalConfig, MatchReport, SubmissionInfo
from argparse import Namespace

from tools.discovery_engine import (
    build_generation_feedback,
    candidate_generation_summary,
    classify_final,
    pool_paths_by_role,
    profile_from_name,
    resolve_feedback_target_paths,
    resolve_build_workers,
    resolve_workers,
    write_pool_manifest,
)
from tools.discovery_feedback import aggregate_feedback
from tools.build_submission import BuildConfig
from tools.discovery_space import deck_distance, generate_discovery_configs, same_strategy_shape, seed_decks_from_paths
from tools.loss_mining import digest_scenarios
from tools.psro import solve_psro
from tools.reference_pool import DEFAULT_REFERENCE_PATHS, load_anchor_table
from tools.gold_gate import classify_candidate as classify_gold_candidate


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


def test_reference_pool_includes_submission_820_and_baselines():
    names = {path.name for path in DEFAULT_REFERENCE_PATHS}
    assert "submission_820.tar.gz" in names
    assert "i-have-one-rear-card.tar.gz" in names
    assert "pokemon-ai-battle-best-ptcg-advanced.tar.gz" in names


def test_lb_anchor_table_targets_1200():
    table = load_anchor_table()
    assert table["target_lb_score"] == 1200
    anchors = {row["name"]: row for row in table["anchors"]}
    assert anchors["i-have-one-rear-card"]["lb_score"] == 700
    assert anchors["submission_820"]["lb_score"] == 820


def test_resolve_workers_respects_explicit_and_auto_limits():
    profile = profile_from_name("a800_discovery")

    assert resolve_workers(Namespace(workers=17, cpu_headroom=2, max_workers=None), profile) == 17

    auto = resolve_workers(Namespace(workers=0, cpu_headroom=0, max_workers=3), profile)

    assert auto == 3


def test_resolve_build_workers_defaults_to_bounded_effective_workers():
    profile = profile_from_name("a800_turbo_discovery")

    assert resolve_build_workers(Namespace(build_workers=7, workers=0, cpu_headroom=0, max_workers=99), profile) == 7
    assert resolve_build_workers(Namespace(build_workers=None, workers=0, cpu_headroom=0, max_workers=99), profile) <= 32


def test_discovery_generation_includes_reference_free_policy_genomes(tmp_path: Path):
    configs = generate_discovery_configs(
        Path("outputs/submissions/champion_latest.tar.gz"),
        tmp_path,
        population=14,
        generation=1,
        seed=20260718,
    )
    assert len(configs) == 14
    assert any(cfg.deck_override for cfg in configs)
    assert any(cfg.origin == "synthetic_policy_genome" for cfg in configs)
    assert any(cfg.policy_variant.startswith("synthetic_") for cfg in configs)


def test_loss_digest_can_drive_generation_motifs(tmp_path: Path):
    digest = digest_scenarios(
        [
            {
                "scenario_id": "s1",
                "focus": "candidate",
                "opponent": "opponent",
                "drop": 300.0,
                "tags": ["focus_hand_drop", "opponent_took_prize"],
            }
        ],
        tmp_path / "loss_digest.json",
    )

    configs = generate_discovery_configs(
        Path("outputs/submissions/champion_latest.tar.gz"),
        tmp_path / "variants",
        population=18,
        generation=2,
        seed=20260718,
        loss_digest_path=tmp_path / "loss_digest.json",
    )

    assert "draw_recovery" in digest["recommended_motifs"]
    assert any(cfg.origin.startswith("motif:loss_digest") for cfg in configs)


def test_feedback_analyzer_creates_matchup_and_scenario_pressure(tmp_path: Path):
    report = {
        "standings": [
            {
                "name": "d001_incumbent",
                "games": 4,
                "kaggle_score_estimate": 600.0,
                "opponents": {},
            },
            {
                "name": "candidate",
                "games": 4,
                "kaggle_score_estimate": 610.0,
                "opponents": {"lucario_seed": {"wins": 0, "losses": 4, "draws": 0}},
            },
        ]
    }
    report_path = tmp_path / "stage_b" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(__import__("json").dumps(report), encoding="utf-8")
    scenarios = [
        {
            "scenario_id": "s1",
            "focus": "candidate",
            "opponent": "lucario_seed",
            "drop": 500.0,
            "tags": ["opponent_took_prize"],
        }
    ]
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(__import__("json").dumps(scenarios), encoding="utf-8")

    feedback = aggregate_feedback(
        out=tmp_path / "generation_feedback.json",
        generation=1,
        stage_reports={"stage_b": report_path},
        scenario_paths=[scenario_path],
        incumbent_name="d001_incumbent",
        population=24,
    )

    kinds = {item["kind"] for item in feedback["pressure_items"]}
    assert "repair_matchup" in kinds
    assert "anti_prize_race" in kinds
    assert feedback["population_plan"]["slots"]["repair"] > 0
    repair = next(item for item in feedback["pressure_items"] if item["kind"] == "repair_matchup")
    assert repair["target_family"] == "great_tusk"
    assert repair["evidence"]["opponent_family"] == "lucario"


def test_feedback_pressure_drives_generation_configs(tmp_path: Path):
    feedback = {
        "version": 2,
        "generation": 1,
        "recommended_motifs": ["defensive_tools"],
        "pressure_items": [
            {
                "pressure_id": "stage_b:repair:candidate:vs:lucario",
                "kind": "repair_matchup",
                "severity": 0.8,
                "confidence": 0.8,
                "target_opponent": "lucario",
                "target_family": "great_tusk",
                "target_candidate": "candidate",
                "evidence": {"score_rate": 0.25},
                "recommended_operators": ["motif_insert", "strategy_shift"],
                "recommended_motifs": ["defensive_tools"],
                "avoid_mutations": [],
                "protected_cards": [],
                "strategy_shifts": {"prize_delta": 5200.0},
                "budget_weight": 1.0,
            }
        ],
    }
    feedback_path = tmp_path / "generation_feedback.json"
    feedback_path.write_text(__import__("json").dumps(feedback), encoding="utf-8")

    configs = generate_discovery_configs(
        Path("outputs/submissions/champion_latest.tar.gz"),
        tmp_path / "variants",
        population=18,
        generation=3,
        seed=20260718,
        feedback_path=feedback_path,
    )

    pressure_configs = [cfg for cfg in configs if cfg.origin.startswith("pressure:") or cfg.origin.startswith("motif:pressure:")]
    assert pressure_configs
    assert any(cfg.strategy_weights.get("prize_delta") == 5200.0 for cfg in pressure_configs)


def test_feedback_target_paths_resolve_target_candidate_and_family(tmp_path: Path):
    explicit = tmp_path / "explicit.tar.gz"
    same_family = tmp_path / "family.tar.gz"
    explicit.write_text("x", encoding="utf-8")
    same_family.write_text("x", encoding="utf-8")
    feedback = {
        "pressure_items": [
            {"target_candidate": "candidate_a", "target_family": "great_tusk"},
            {"target_candidate": "missing", "target_family": "lucario"},
        ]
    }
    records = [
        {"name": "candidate_a", "family": "great_tusk", "internal_tarball": str(explicit)},
        {"name": "candidate_b", "family": "lucario", "internal_tarball": str(same_family)},
    ]

    paths = resolve_feedback_target_paths(feedback, records)

    assert explicit.resolve() in paths
    assert same_family.resolve() in paths


def test_seed_decks_from_paths_preserves_tarball_family_metadata(tmp_path: Path):
    tarball = tmp_path / "candidate.tar.gz"
    deck = "\n".join(["1"] * 60) + "\n"
    metadata = json.dumps({"family": "lucario"}).encode("utf-8")
    with tarfile.open(tarball, "w:gz") as tar:
        deck_bytes = deck.encode("utf-8")
        deck_info = tarfile.TarInfo("deck.csv")
        deck_info.size = len(deck_bytes)
        tar.addfile(deck_info, io.BytesIO(deck_bytes))
        meta_info = tarfile.TarInfo("build_metadata.json")
        meta_info.size = len(metadata)
        tar.addfile(meta_info, io.BytesIO(metadata))

    rows = [row for row in seed_decks_from_paths([tarball]) if row[2] == tarball]

    assert rows
    assert rows[0][0] == "lucario"


def test_deck_distance_and_strategy_shape_support_diversity_filter():
    base = BuildConfig(name="a", deck_override=[1, 1, 2, 3], strategy_weights={"x": 1.0})
    near = BuildConfig(name="b", deck_override=[1, 1, 2, 4], strategy_weights={"x": 1.0})
    different_strategy = BuildConfig(name="c", deck_override=[1, 1, 2, 4], strategy_weights={"x": 2.0})

    assert 0.0 < deck_distance(base.deck_override or [], near.deck_override or []) < 0.5
    assert same_strategy_shape(base, near)
    assert not same_strategy_shape(base, different_strategy)


def test_turbo_population_reaches_requested_size_with_diversity_filter(tmp_path: Path):
    configs = generate_discovery_configs(
        Path("outputs/submissions/champion_latest.tar.gz"),
        tmp_path / "variants",
        population=96,
        generation=7,
        seed=20260718,
        min_diversity_distance=0.08,
    )
    summary = candidate_generation_summary(configs, 96)

    assert len(configs) == 96
    assert summary["underfilled"] is False
    assert summary["deck_overrides"] >= 32
    assert "synthetic_policy_genome" in summary["origin_counts"]


def test_pressure_only_microburst_can_fill_requested_slots(tmp_path: Path):
    feedback_path = tmp_path / "generation_feedback.json"
    feedback_path.write_text(
        __import__("json").dumps(
            {
                "recommended_motifs": ["defensive_tools", "draw_recovery"],
                "pressure_items": [
                    {
                        "pressure_id": "stage_b:repair:candidate:vs:lucario",
                        "kind": "repair_matchup",
                        "target_family": "great_tusk",
                        "recommended_motifs": ["defensive_tools", "switch_pivot"],
                        "strategy_shifts": {"prize_delta": 5200.0},
                    },
                    {
                        "pressure_id": "stage_b:scenario:candidate:focus_hand_drop",
                        "kind": "loss_recovery",
                        "target_family": "great_tusk",
                        "recommended_motifs": ["draw_recovery", "resource_safety"],
                        "strategy_shifts": {"hand_delta": 90.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    configs = generate_discovery_configs(
        Path("outputs/submissions/champion_latest.tar.gz"),
        tmp_path / "microburst",
        population=36,
        generation=1501,
        seed=20260718,
        feedback_path=feedback_path,
        pressure_only=True,
        min_diversity_distance=0.08,
    )

    assert len(configs) == 36
    assert any(cfg.origin.startswith("pressure:") or cfg.origin.startswith("motif:pressure:") for cfg in configs)


def test_pool_manifest_splits_holdout_from_discovery_pool(tmp_path: Path):
    paths = []
    for index in range(6):
        path = tmp_path / f"pool{index}.tar.gz"
        path.write_text("x", encoding="utf-8")
        paths.append(path)
    incumbent = tmp_path / "inc.tar.gz"
    incumbent.write_text("x", encoding="utf-8")

    manifest = write_pool_manifest(tmp_path / "pool_manifest.json", incumbent, paths, [], holdout_count=4)
    roles = pool_paths_by_role(manifest)

    assert len(roles["holdout"]) == 2
    assert len(roles["core"]) == 4
    assert not set(roles["holdout"]) & set(roles["core"])


def test_build_generation_feedback_excludes_holdout_when_audit_only(tmp_path: Path):
    gen_dir = tmp_path / "generation_001"
    holdout = gen_dir / "stage_d_holdout"
    holdout.mkdir(parents=True)
    report = {
        "standings": [
            {"name": "d001_incumbent", "games": 4, "kaggle_score_estimate": 600.0, "opponents": {}},
            {
                "name": "candidate",
                "games": 4,
                "kaggle_score_estimate": 610.0,
                "opponents": {"holdout_enemy": {"wins": 0, "losses": 4, "draws": 0}},
            },
        ]
    }
    (holdout / "report.json").write_text(__import__("json").dumps(report), encoding="utf-8")
    (holdout / "psro.json").write_text(
        __import__("json").dumps({"niches": [{"candidate": "candidate", "weight": 0.9, "beats": ["holdout_enemy"], "loses_to": []}]}),
        encoding="utf-8",
    )
    scenarios = [
        {"scenario_id": "s1", "focus": "candidate", "opponent": "holdout_enemy", "drop": 900.0, "tags": ["focus_deck_drop"]}
    ]
    scenario_path = gen_dir / "scenarios.json"
    scenario_path.write_text(__import__("json").dumps(scenarios), encoding="utf-8")

    feedback = build_generation_feedback(
        Namespace(feedback_mode="closed_loop", holdout_feedback_mode="audit_only"),
        gen_dir,
        1,
        "d001_incumbent",
        24,
        ["stage_d_holdout"],
        [scenario_path],
        tmp_path / "feedback.json",
    )

    assert feedback["pressure_items"] == []


def test_classify_final_blocks_h2h_losing_promoted_candidate():
    final = {
        "status": "promoted",
        "name": "candidate",
        "score_delta_vs_incumbent": 42.0,
        "h2h_wins": 80,
        "h2h_losses": 100,
        "stage_d_top": [{"name": "candidate", "games": 1000, "no_results": 0}],
    }

    decision = classify_final(final, {"required_ok": True, "last_alias_ok": True, "error": ""})

    assert decision["submit_ready"] is False
    assert decision["champion_class"] == "portfolio_candidate"


def test_classify_final_marks_near_clean_candidate_unstable():
    final = {
        "status": "promoted",
        "name": "candidate",
        "score_delta_vs_incumbent": 42.0,
        "h2h_wins": 101,
        "h2h_losses": 99,
        "stage_d_top": [{"name": "candidate", "games": 1000, "no_results": 15}],
    }

    decision = classify_final(final, {"required_ok": True, "last_alias_ok": True, "error": ""})

    assert decision["submit_ready"] is False
    assert decision["champion_class"] == "unstable_candidate"


def test_gold_gate_requires_strong_anchor_progress(tmp_path: Path):
    report = {
        "standings": [
            {
                "name": "candidate",
                "games": 100,
                "wins": 60,
                "losses": 40,
                "draws": 0,
                "no_results": 0,
                "crashes": 0,
                "timeouts": 0,
                "invalids": 0,
                "kaggle_score_estimate": 600.0,
                "opponents": {
                    "incumbent": {"wins": 60, "losses": 40, "draws": 0},
                    "i-have-one-rear-card": {"wins": 72, "losses": 28, "draws": 0},
                    "submission_820": {"wins": 62, "losses": 38, "draws": 0},
                    "pokemon-ai-battle-best-ptcg-advanced": {"wins": 55, "losses": 45, "draws": 0},
                    "pokemon-steel": {"wins": 55, "losses": 45, "draws": 0},
                    "pokemon-tcg-rahul-jiwane": {"wins": 55, "losses": 45, "draws": 0},
                    "ptcg-mega-lucario-ex-v63": {"wins": 55, "losses": 45, "draws": 0},
                    "multiply-agent-best-940-lb": {"wins": 57, "losses": 43, "draws": 0},
                    "submission_sorce_700": {"wins": 72, "losses": 28, "draws": 0},
                },
            }
        ]
    }
    decision = classify_gold_candidate(report, "candidate", incumbent_name="incumbent", submission_sha256="")
    assert decision["decision"] == "kaggle_probe_ready"
    assert decision["submit_ready"] is False
    assert decision["gold_gate_passed"] is True


def test_gold_gate_fails_without_required_anchor_matchups():
    report = {
        "standings": [
            {
                "name": "candidate",
                "games": 10,
                "wins": 10,
                "losses": 0,
                "draws": 0,
                "no_results": 0,
                "crashes": 0,
                "timeouts": 0,
                "invalids": 0,
                "kaggle_score_estimate": 600.0,
                "opponents": {"incumbent": {"wins": 10, "losses": 0, "draws": 0}},
            }
        ]
    }

    decision = classify_gold_candidate(report, "candidate", incumbent_name="incumbent")

    assert decision["decision"] == "gold_gate_failed"
    assert "missing required anchor matchup: i-have-one-rear-card" in decision["reasons"]


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
