from __future__ import annotations

from pathlib import Path

from tools.build_models import BuildConfig
from tools.policy_genome import generate_genomes, generate_synthetic_configs


def test_policy_genomes_are_reproducible_and_reference_free():
    first = generate_genomes(8, 123)
    second = generate_genomes(8, 123)

    assert first == second
    assert len(first) == 8
    assert all(genome.name.startswith("synthetic_") for genome in first)


def test_policy_genome_compiler_produces_legal_decks(tmp_path: Path):
    configs = generate_synthetic_configs(BuildConfig(), tmp_path, 12, 456)

    assert len(configs) == 12
    assert all(len(config.deck_override or []) == 60 for config in configs)
    assert all(config.origin == "synthetic_policy_genome" for config in configs)
    assert all(config.policy_variant.startswith("synthetic_") for config in configs)
