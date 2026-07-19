from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import random
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class OpponentModel:
    name: str
    aggression: float
    bench_pressure: float
    resource_denial: float
    self_mill: float
    switch_frequency: float


@dataclass(frozen=True, slots=True)
class OpponentGenome:
    name: str
    deck_family: str
    aggression: float
    tempo: float
    bench_pressure: float
    resource_denial: float
    self_mill: float
    switch_frequency: float
    prize_race_bias: float
    lineage: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentGenome:
        return cls(**data)

    def model(self) -> OpponentModel:
        return OpponentModel(
            self.name,
            self.aggression,
            self.bench_pressure,
            self.resource_denial,
            self.self_mill,
            self.switch_frequency,
        )


SYNTHETIC_OPPONENTS = (
    OpponentModel("synthetic_fast_ko", 1.0, 0.8, 0.1, 0.0, 0.6),
    OpponentModel("synthetic_slow_ko", 0.7, 0.6, 0.2, 0.0, 0.4),
    OpponentModel("synthetic_wall", 0.3, 0.4, 0.5, 0.1, 0.2),
    OpponentModel("synthetic_mill", 0.2, 0.2, 0.3, 1.0, 0.2),
    OpponentModel("synthetic_denial", 0.4, 0.3, 1.0, 0.0, 0.3),
    OpponentModel("synthetic_bench_pressure", 0.6, 1.0, 0.2, 0.0, 0.5),
    OpponentModel("synthetic_unknown", 0.55, 0.55, 0.55, 0.25, 0.4),
)


def base_opponent_genomes() -> list[OpponentGenome]:
    families = ("prize_race", "adaptive", "wall", "mill", "denial", "adaptive", "unknown")
    tempos = (1.0, 0.65, 0.25, 0.20, 0.45, 0.70, 0.55)
    prize_bias = (1.0, 0.7, 0.2, 0.1, 0.4, 0.8, 0.55)
    return [
        OpponentGenome(
            name=model.name,
            deck_family=families[index],
            aggression=model.aggression,
            tempo=tempos[index],
            bench_pressure=model.bench_pressure,
            resource_denial=model.resource_denial,
            self_mill=model.self_mill,
            switch_frequency=model.switch_frequency,
            prize_race_bias=prize_bias[index],
        )
        for index, model in enumerate(SYNTHETIC_OPPONENTS)
    ]


def mutate_opponent(parent: OpponentGenome, index: int, seed: int, pressure: dict[str, float] | None = None) -> OpponentGenome:
    rng = random.Random(seed + index * 1009)
    pressure = pressure or {}

    def shifted(value: float, key: str) -> float:
        return round(min(1.0, max(0.0, value + rng.uniform(-0.18, 0.18) + 0.08 * pressure.get(key, 0.0))), 3)

    return replace(
        parent,
        name=f"counter_{index:04d}_{parent.deck_family}",
        aggression=shifted(parent.aggression, "aggression"),
        tempo=shifted(parent.tempo, "tempo"),
        bench_pressure=shifted(parent.bench_pressure, "bench_pressure"),
        resource_denial=shifted(parent.resource_denial, "resource_denial"),
        self_mill=shifted(parent.self_mill, "self_mill"),
        switch_frequency=shifted(parent.switch_frequency, "switch_frequency"),
        prize_race_bias=shifted(parent.prize_race_bias, "prize_race_bias"),
        lineage="counterexample",
    )


def generate_counter_opponents(
    count: int,
    seed: int,
    pressure: dict[str, float] | None = None,
    parents: Iterable[OpponentGenome] = (),
) -> list[OpponentGenome]:
    rng = random.Random(seed)
    pool = list(parents) or base_opponent_genomes()
    return [mutate_opponent(rng.choice(pool), index, seed, pressure) for index in range(max(0, count))]


class OpponentArchive:
    def __init__(self, genomes: Iterable[OpponentGenome] = ()) -> None:
        self._genomes: dict[str, OpponentGenome] = {genome.name: genome for genome in genomes}

    def add(self, genome: OpponentGenome) -> bool:
        signature = opponent_signature(genome)
        if any(opponent_signature(existing) == signature for existing in self._genomes.values()):
            return False
        self._genomes[genome.name] = genome
        return True

    def genomes(self) -> list[OpponentGenome]:
        return list(self._genomes.values())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([genome.to_dict() for genome in self.genomes()], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None) -> OpponentArchive:
        if path is None or not path.exists():
            return cls(base_opponent_genomes())
        return cls(OpponentGenome.from_dict(row) for row in json.loads(path.read_text(encoding="utf-8")))


def opponent_signature(genome: OpponentGenome) -> tuple[object, ...]:
    return (
        genome.deck_family,
        *(round(value, 1) for value in (
            genome.aggression,
            genome.tempo,
            genome.bench_pressure,
            genome.resource_denial,
            genome.self_mill,
            genome.switch_frequency,
            genome.prize_race_bias,
        )),
    )


def opponent_model_names() -> list[str]:
    return [model.name for model in SYNTHETIC_OPPONENTS]
