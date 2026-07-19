from __future__ import annotations

from pathlib import Path

from tools.build_models import BuildConfig
from tools.policy_genome import (
    StrategyArchive,
    behavior_descriptor,
    crossover_genomes,
    generate_genomes,
    generate_synthetic_configs,
    mutate_genome,
)
import random


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


def test_policy_population_uses_multiple_complete_families():
    genomes = generate_genomes(64, 90210)

    assert len({genome.deck_family for genome in genomes}) >= 4
    assert all(sum(count for _, count in genome.card_counts) == 60 for genome in genomes)


def test_crossover_mutation_and_archive_preserve_legal_genomes(tmp_path: Path):
    first, second = generate_genomes(2, 777)
    child = crossover_genomes(first, second, 2, random.Random(1))
    mutant = mutate_genome(child, 3, random.Random(2), {}, targeted=True)
    archive = StrategyArchive()

    assert sum(count for _, count in child.card_counts) == 60
    assert sum(count for _, count in mutant.card_counts) == 60
    assert archive.add(child, 0.4, behavior_descriptor(child))
    archive.add(mutant, 0.8, behavior_descriptor(mutant))
    archive_path = tmp_path / "archive.json"
    archive.save(archive_path)

    assert StrategyArchive.load(archive_path).elites()
