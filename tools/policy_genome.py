from __future__ import annotations

from dataclasses import dataclass, replace
import random
from pathlib import Path
from typing import Any

from tools.build_models import BuildConfig
from tools.counterexample_search import failure_bias, load_counterexamples
from tools.deck_rules import validate_deck_ids
from tools.opponent_models import SYNTHETIC_OPPONENTS


@dataclass(frozen=True, slots=True)
class StrategyGenome:
    name: str
    route: str
    search_worlds: int
    risk_penalty: float
    wall_threshold: int
    ko_threshold: int
    bench_floor: int
    switch_priority: float
    recovery_priority: float
    disruption_priority: float
    resource_safety: float
    deck_counts: tuple[tuple[int, int], ...]


BASE_CORE = (
    (58, 4), (344, 4), (345, 4), (607, 1),
    (1142, 4), (1152, 4), (1086, 4), (1122, 4),
    (1197, 4), (1185, 4), (20, 4), (11, 4),
)


def _deck_from_genome(genome: StrategyGenome) -> list[int]:
    deck: list[int] = []
    for card_id, count in (*BASE_CORE, *genome.deck_counts):
        deck.extend([card_id] * count)
    if len(deck) < 60:
        deck.extend([1123] * (60 - len(deck)))
    deck = deck[:60]
    validate_deck_ids(deck)
    return deck


def _random_genome(index: int, rng: random.Random, bias: dict[str, float]) -> StrategyGenome:
    routes = ("adaptive", *(model.name.removeprefix("synthetic_") for model in SYNTHETIC_OPPONENTS))
    route = rng.choice(routes)
    switch = min(1.0, max(0.0, rng.random() + 0.12 * bias.get("switch_priority", 0.0)))
    recovery = min(1.0, max(0.0, rng.random() + 0.10 * bias.get("resource_safety", 0.0)))
    disruption = min(1.0, max(0.0, rng.random() + 0.10 * bias.get("prize_delta", 0.0)))
    resource = min(1.0, max(0.0, rng.random() + 0.12 * bias.get("resource_safety", 0.0)))
    flex = [
        (1121, 1 + rng.randrange(3)),
        (1123, 1 + rng.randrange(4)),
        (1182, 1 + rng.randrange(3)),
        (1097, 1 + rng.randrange(3)),
        (1139, 1 if rng.random() < recovery else 0),
        (1194, 1 + rng.randrange(3)),
        (1186, 1 + rng.randrange(2)),
        (1247 if rng.random() < 0.5 else 1147, 1),
    ]
    total = sum(count for _, count in BASE_CORE) + sum(count for _, count in flex)
    while total > 60:
        card_id, count = flex[rng.randrange(len(flex))]
        if count > 0 and card_id not in {1121, 1123}:
            flex[flex.index((card_id, count))] = (card_id, count - 1)
            total -= 1
    if total < 60:
        flex.append((6, 60 - total))
    return StrategyGenome(
        name=f"synthetic_{index:04d}_{route}",
        route=route,
        search_worlds=rng.choice((2, 3, 4, 6)),
        risk_penalty=round(rng.uniform(0.10, 0.45), 3),
        wall_threshold=rng.randint(16, 28),
        ko_threshold=rng.randint(5, 14),
        bench_floor=rng.randint(2, 5),
        switch_priority=round(switch, 3),
        recovery_priority=round(recovery, 3),
        disruption_priority=round(disruption, 3),
        resource_safety=round(resource, 3),
        deck_counts=tuple(flex),
    )


def generate_genomes(population: int, seed: int, feedback_path: Path | None = None) -> list[StrategyGenome]:
    rng = random.Random(seed)
    bias = failure_bias(load_counterexamples(feedback_path))
    return [_random_genome(index, rng, bias) for index in range(max(0, population))]


def compile_genome(genome: StrategyGenome, base: BuildConfig, out_dir: Path) -> BuildConfig:
    weights = dict(base.strategy_weights)
    weights.update(
        {
            "opp_mill": 950.0 * (1.0 + genome.resource_safety),
            "self_mill_penalty": 280.0 * (1.0 + genome.resource_safety),
            "prize_delta": 3500.0 * (1.0 + genome.disruption_priority),
            "hand_delta": 45.0 * (1.0 + genome.disruption_priority),
            "great_tusk_ready": 6500.0 * (1.0 + genome.recovery_priority),
            "wall_threshold": float(genome.wall_threshold),
            "ko_threshold": float(genome.ko_threshold),
            "bench_floor": float(genome.bench_floor),
            "switch_priority": genome.switch_priority,
            "recovery_priority": genome.recovery_priority,
            "disruption_priority": genome.disruption_priority,
        }
    )
    return replace(
        base,
        name=genome.name,
        out=out_dir / f"{genome.name}.tar.gz",
        deck_override=_deck_from_genome(genome),
        strategy_weights=weights,
        policy_variant=f"synthetic_{genome.route}",
        belief_worlds=genome.search_worlds,
        risk_penalty=genome.risk_penalty,
        origin="synthetic_policy_genome",
        notes=f"Reference-free route genome: {genome.route}; thresholds={genome.wall_threshold}/{genome.ko_threshold}/{genome.bench_floor}",
    )


def generate_synthetic_configs(
    base: BuildConfig,
    out_dir: Path,
    population: int,
    seed: int,
    feedback_path: Path | None = None,
) -> list[BuildConfig]:
    return [compile_genome(genome, base, out_dir) for genome in generate_genomes(population, seed, feedback_path)]
