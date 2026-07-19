from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import random
from pathlib import Path
from typing import Any, Iterable

from tools.build_models import BuildConfig
from tools.counterexample_search import failure_bias, load_counterexamples
from tools.deck_rules import validate_deck_ids
from tools.opponent_models import OpponentGenome, SYNTHETIC_OPPONENTS


DECK_FAMILIES: dict[str, tuple[tuple[int, int], ...]] = {
    "adaptive": (
        (58, 4), (344, 4), (345, 4), (607, 1), (1142, 4), (1152, 4),
        (1086, 4), (1122, 4), (1197, 4), (1185, 4), (20, 4), (11, 4),
        (1121, 3), (1123, 3), (1182, 2), (1097, 2), (1139, 1), (1194, 2),
        (1186, 1), (1247, 1),
    ),
    "mill": (
        (58, 4), (344, 4), (345, 4), (607, 1), (1142, 4), (1152, 4),
        (1086, 4), (1122, 4), (1197, 4), (1185, 4), (20, 4), (11, 4),
        (1121, 4), (1097, 3), (1194, 3), (1186, 2), (1123, 1), (1139, 1),
        (1247, 1),
    ),
    "wall": (
        (58, 3), (344, 4), (345, 4), (607, 2), (1142, 4), (1152, 4),
        (1086, 4), (1122, 4), (1197, 3), (1185, 4), (20, 4), (11, 4),
        (1121, 3), (1123, 4), (1097, 3), (1194, 2), (1182, 2), (1139, 1),
        (1247, 1),
    ),
    "denial": (
        (58, 4), (344, 3), (345, 3), (607, 2), (1142, 4), (1152, 4),
        (1086, 4), (1122, 4), (1197, 4), (1185, 4), (20, 4), (11, 4),
        (1121, 3), (1123, 2), (1182, 3), (1097, 2), (1194, 3), (1186, 2),
        (1247, 1),
    ),
    "prize_race": (
        (58, 4), (344, 3), (345, 3), (607, 3), (1142, 4), (1152, 3),
        (1086, 4), (1122, 4), (1197, 3), (1185, 4), (20, 4), (11, 4),
        (1121, 4), (1123, 3), (1182, 3), (1097, 2), (1194, 2), (1186, 1),
        (1247, 1),
    ),
}

MUTABLE_CARD_POOL = (58, 344, 345, 607, 1142, 1152, 1086, 1122, 1197, 1185, 20, 11, 1121, 1123, 1182, 1097, 1139, 1194, 1186, 6)


@dataclass(frozen=True, slots=True)
class StrategyGenome:
    name: str
    deck_family: str
    route: str
    search_worlds: int
    risk_penalty: float
    opening_policy: float
    prize_policy: float
    switch_policy: float
    recovery_policy: float
    disruption_policy: float
    resource_policy: float
    wall_threshold: int
    ko_threshold: int
    bench_floor: int
    card_counts: tuple[tuple[int, int], ...]
    lineage: str = "random"

    @property
    def deck_counts(self) -> tuple[tuple[int, int], ...]:
        return self.card_counts

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["card_counts"] = [list(row) for row in self.card_counts]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyGenome:
        values = dict(data)
        values["card_counts"] = tuple((int(card_id), int(count)) for card_id, count in values["card_counts"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class BehaviorDescriptor:
    tempo: int
    control: int
    resilience: int
    complexity: int

    @property
    def niche(self) -> str:
        return f"t{self.tempo}_c{self.control}_r{self.resilience}_x{self.complexity}"


@dataclass(slots=True)
class ArchiveEntry:
    genome: StrategyGenome
    score: float
    descriptor: BehaviorDescriptor


class StrategyArchive:
    def __init__(self, entries: Iterable[ArchiveEntry] = ()) -> None:
        self._entries: dict[str, ArchiveEntry] = {}
        for entry in entries:
            self.add(entry.genome, entry.score, entry.descriptor)

    def add(self, genome: StrategyGenome, score: float, descriptor: BehaviorDescriptor | None = None) -> bool:
        descriptor = descriptor or behavior_descriptor(genome)
        current = self._entries.get(descriptor.niche)
        if current is not None and current.score >= score:
            return False
        self._entries[descriptor.niche] = ArchiveEntry(genome, float(score), descriptor)
        return True

    def elites(self) -> list[StrategyGenome]:
        return [entry.genome for entry in sorted(self._entries.values(), key=lambda item: item.score, reverse=True)]

    def to_dict(self) -> dict[str, Any]:
        return {
            niche: {"score": entry.score, "descriptor": asdict(entry.descriptor), "genome": entry.genome.to_dict()}
            for niche, entry in sorted(self._entries.items())
        }

    @classmethod
    def load(cls, path: Path | None) -> StrategyArchive:
        if path is None or not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ArchiveEntry(
                StrategyGenome.from_dict(row["genome"]),
                float(row["score"]),
                BehaviorDescriptor(**row["descriptor"]),
            )
            for row in data.values()
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _normalize_counts(counts: Counter[int], rng: random.Random) -> tuple[tuple[int, int], ...]:
    for card_id in list(counts):
        limit = 60 if card_id in {1, 2, 3, 4, 5, 6, 7, 8} else 4
        counts[card_id] = min(limit, max(0, counts[card_id]))
        if counts[card_id] == 0:
            del counts[card_id]
    while sum(counts.values()) > 60:
        choices = [card_id for card_id, count in counts.items() if count > 1 and card_id != 58]
        if not choices:
            choices = list(counts)
        counts[rng.choice(choices)] -= 1
    while sum(counts.values()) < 60:
        choices = [card_id for card_id in MUTABLE_CARD_POOL if card_id in {1, 2, 3, 4, 5, 6, 7, 8} or counts[card_id] < 4]
        counts[rng.choice(choices)] += 1
    deck = [card_id for card_id, count in counts.items() for _ in range(count)]
    validate_deck_ids(deck)
    return tuple(sorted((card_id, count) for card_id, count in counts.items() if count))


def _family_counts(family: str, rng: random.Random) -> tuple[tuple[int, int], ...]:
    return _normalize_counts(Counter(dict(DECK_FAMILIES[family])), rng)


def _deck_from_genome(genome: StrategyGenome) -> list[int]:
    deck = [card_id for card_id, count in genome.card_counts for _ in range(count)]
    validate_deck_ids(deck)
    return deck


def behavior_descriptor(genome: StrategyGenome) -> BehaviorDescriptor:
    return BehaviorDescriptor(
        tempo=min(3, int((genome.opening_policy + genome.prize_policy) * 2.0)),
        control=min(3, int((genome.disruption_policy + (1.0 if genome.deck_family in {"mill", "denial"} else 0.0)) * 1.5)),
        resilience=min(3, int((genome.recovery_policy + genome.resource_policy) * 2.0)),
        complexity=min(3, max(0, genome.search_worlds // 2 - 1)),
    )


def _random_genome(index: int, rng: random.Random, bias: dict[str, float], lineage: str = "random") -> StrategyGenome:
    family = rng.choice(tuple(DECK_FAMILIES))
    routes = ("adaptive", *(model.name.removeprefix("synthetic_") for model in SYNTHETIC_OPPONENTS))
    return StrategyGenome(
        name=f"synthetic_{index:04d}_{family}",
        deck_family=family,
        route=rng.choice(routes),
        search_worlds=rng.choice((2, 3, 4, 6)),
        risk_penalty=round(rng.uniform(0.10, 0.45), 3),
        opening_policy=round(rng.random(), 3),
        prize_policy=round(min(1.0, max(0.0, rng.random() + 0.10 * bias.get("prize_delta", 0.0))), 3),
        switch_policy=round(min(1.0, max(0.0, rng.random() + 0.12 * bias.get("switch_priority", 0.0))), 3),
        recovery_policy=round(min(1.0, max(0.0, rng.random() + 0.10 * bias.get("resource_safety", 0.0))), 3),
        disruption_policy=round(min(1.0, max(0.0, rng.random() + 0.10 * bias.get("prize_delta", 0.0))), 3),
        resource_policy=round(min(1.0, max(0.0, rng.random() + 0.12 * bias.get("resource_safety", 0.0))), 3),
        wall_threshold=rng.randint(16, 28),
        ko_threshold=rng.randint(5, 14),
        bench_floor=rng.randint(2, 5),
        card_counts=_family_counts(family, rng),
        lineage=lineage,
    )


def mutate_genome(parent: StrategyGenome, index: int, rng: random.Random, bias: dict[str, float], targeted: bool = False) -> StrategyGenome:
    counts = Counter(dict(parent.card_counts))
    edits = 1 + rng.randrange(3)
    for _ in range(edits):
        removable = [card_id for card_id, count in counts.items() if count > 0 and card_id != 58]
        addable = [card_id for card_id in MUTABLE_CARD_POOL if card_id in {1, 2, 3, 4, 5, 6, 7, 8} or counts[card_id] < 4]
        counts[rng.choice(removable)] -= 1
        counts[rng.choice(addable)] += 1
    shift = 0.18 if targeted else 0.10
    return replace(
        parent,
        name=f"synthetic_{index:04d}_{parent.deck_family}",
        risk_penalty=round(min(0.65, max(0.02, parent.risk_penalty + rng.uniform(-shift, shift))), 3),
        prize_policy=round(min(1.0, max(0.0, parent.prize_policy + rng.uniform(-shift, shift) + 0.04 * bias.get("prize_delta", 0.0))), 3),
        switch_policy=round(min(1.0, max(0.0, parent.switch_policy + rng.uniform(-shift, shift) + 0.04 * bias.get("switch_priority", 0.0))), 3),
        recovery_policy=round(min(1.0, max(0.0, parent.recovery_policy + rng.uniform(-shift, shift) + 0.04 * bias.get("resource_safety", 0.0))), 3),
        disruption_policy=round(min(1.0, max(0.0, parent.disruption_policy + rng.uniform(-shift, shift))), 3),
        resource_policy=round(min(1.0, max(0.0, parent.resource_policy + rng.uniform(-shift, shift) + 0.04 * bias.get("resource_safety", 0.0))), 3),
        card_counts=_normalize_counts(counts, rng),
        lineage="counterexample_mutation" if targeted else "elite_mutation",
    )


def crossover_genomes(first: StrategyGenome, second: StrategyGenome, index: int, rng: random.Random) -> StrategyGenome:
    family_parent = first if rng.random() < 0.5 else second
    counts = Counter()
    first_counts = dict(first.card_counts)
    second_counts = dict(second.card_counts)
    for card_id in set(first_counts) | set(second_counts):
        counts[card_id] = first_counts.get(card_id, 0) if rng.random() < 0.5 else second_counts.get(card_id, 0)
    return StrategyGenome(
        name=f"synthetic_{index:04d}_{family_parent.deck_family}",
        deck_family=family_parent.deck_family,
        route=first.route if rng.random() < 0.5 else second.route,
        search_worlds=first.search_worlds if rng.random() < 0.5 else second.search_worlds,
        risk_penalty=round((first.risk_penalty + second.risk_penalty) / 2.0, 3),
        opening_policy=round((first.opening_policy + second.opening_policy) / 2.0, 3),
        prize_policy=round((first.prize_policy + second.prize_policy) / 2.0, 3),
        switch_policy=round((first.switch_policy + second.switch_policy) / 2.0, 3),
        recovery_policy=round((first.recovery_policy + second.recovery_policy) / 2.0, 3),
        disruption_policy=round((first.disruption_policy + second.disruption_policy) / 2.0, 3),
        resource_policy=round((first.resource_policy + second.resource_policy) / 2.0, 3),
        wall_threshold=round((first.wall_threshold + second.wall_threshold) / 2),
        ko_threshold=round((first.ko_threshold + second.ko_threshold) / 2),
        bench_floor=round((first.bench_floor + second.bench_floor) / 2),
        card_counts=_normalize_counts(counts, rng),
        lineage="crossover",
    )


def generate_genomes(
    population: int,
    seed: int,
    feedback_path: Path | None = None,
    parents: Iterable[StrategyGenome] = (),
    pressure_only: bool = False,
) -> list[StrategyGenome]:
    rng = random.Random(seed)
    bias = failure_bias(load_counterexamples(feedback_path))
    feedback = {}
    if feedback_path is not None and feedback_path.exists():
        try:
            feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            feedback = {}
    weighted: Counter[str] = Counter()
    for item in feedback.get("pressure_items", []):
        weight = max(0.0, float(item.get("severity", 0.0))) * max(0.0, float(item.get("confidence", 0.0))) * max(0.0, float(item.get("budget_weight", 1.0)))
        for key, value in dict(item.get("strategy_shifts", {})).items():
            weighted[str(key)] += weight * float(value)
    scale = max((abs(value) for value in weighted.values()), default=0.0)
    if scale > 0:
        bias.update({key: value / scale for key, value in weighted.items()})
    elite = list(parents)
    if not elite:
        elite = [_random_genome(index, rng, bias, lineage="bootstrap_parent") for index in range(min(16, max(4, population // 9)))]
    genomes: list[StrategyGenome] = []
    if pressure_only:
        return [mutate_genome(rng.choice(elite), index, rng, bias, targeted=True) for index in range(population)]
    elite_count = round(population * 0.20)
    crossover_count = round(population * 0.20)
    targeted_count = round(population * 0.30)
    family_count = round(population * 0.20)
    random_count = population - elite_count - crossover_count - targeted_count - family_count
    for _ in range(elite_count):
        genomes.append(mutate_genome(rng.choice(elite), len(genomes), rng, bias))
    for _ in range(crossover_count):
        genomes.append(crossover_genomes(rng.choice(elite), rng.choice(elite), len(genomes), rng))
    for _ in range(targeted_count):
        genomes.append(mutate_genome(rng.choice(elite), len(genomes), rng, bias, targeted=True))
    for _ in range(family_count):
        genomes.append(_random_genome(len(genomes), rng, bias, lineage="family_exploration"))
    for _ in range(random_count):
        genomes.append(_random_genome(len(genomes), rng, bias, lineage="random"))
    return genomes[:population]


def compile_genome(genome: StrategyGenome, base: BuildConfig, out_dir: Path) -> BuildConfig:
    weights = dict(base.strategy_weights)
    weights.update(
        {
            "opp_mill": 950.0 * (1.0 + genome.resource_policy),
            "self_mill_penalty": 280.0 * (1.0 + genome.resource_policy),
            "prize_delta": 3500.0 * (1.0 + genome.prize_policy),
            "hand_delta": 45.0 * (1.0 + genome.disruption_policy),
            "great_tusk_ready": 6500.0 * (1.0 + genome.opening_policy),
            "switch_priority": genome.switch_policy,
            "disruption_priority": genome.disruption_policy,
        }
    )
    return replace(
        base,
        name=genome.name,
        family=genome.deck_family,
        out=out_dir / f"{genome.name}.tar.gz",
        deck_override=_deck_from_genome(genome),
        deck_swaps=[],
        strategy_weights=weights,
        strategy_genome=genome.to_dict(),
        policy_variant=f"synthetic_{genome.route}",
        belief_worlds=genome.search_worlds,
        risk_penalty=genome.risk_penalty,
        opponent_decks={},
        origin="synthetic_policy_genome",
        notes=f"Reference-free {genome.deck_family}/{genome.route}; niche={behavior_descriptor(genome).niche}",
    )


def generate_synthetic_configs(
    base: BuildConfig,
    out_dir: Path,
    population: int,
    seed: int,
    feedback_path: Path | None = None,
    parents: Iterable[StrategyGenome] = (),
    pressure_only: bool = False,
) -> list[BuildConfig]:
    return [compile_genome(genome, base, out_dir) for genome in generate_genomes(population, seed, feedback_path, parents, pressure_only)]


def compile_opponent_genome(genome: OpponentGenome, base: BuildConfig, out_dir: Path) -> BuildConfig:
    family = genome.deck_family if genome.deck_family in DECK_FAMILIES else "adaptive"
    rng = random.Random(sum((index + 1) * ord(char) for index, char in enumerate(genome.name)))
    strategy = StrategyGenome(
        name=genome.name,
        deck_family=family,
        route="fast_ko" if genome.tempo >= 0.7 else "denial" if genome.resource_denial >= 0.7 else "mill" if genome.self_mill >= 0.7 else "adaptive",
        search_worlds=2 if genome.tempo >= 0.7 else 4,
        risk_penalty=round(0.10 + 0.35 * (1.0 - genome.aggression), 3),
        opening_policy=genome.tempo,
        prize_policy=genome.prize_race_bias,
        switch_policy=genome.switch_frequency,
        recovery_policy=max(0.0, 1.0 - genome.resource_denial),
        disruption_policy=max(genome.bench_pressure, genome.resource_denial),
        resource_policy=max(genome.self_mill, genome.resource_denial),
        wall_threshold=round(16 + 12 * (1.0 - genome.aggression)),
        ko_threshold=round(5 + 9 * (1.0 - genome.tempo)),
        bench_floor=round(2 + 3 * genome.bench_pressure),
        card_counts=_family_counts(family, rng),
        lineage=genome.lineage,
    )
    cfg = compile_genome(strategy, base, out_dir)
    return replace(cfg, origin="coevolved_opponent", notes=f"Counter opponent: {genome.lineage}/{family}")
