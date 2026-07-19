from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpponentModel:
    name: str
    aggression: float
    bench_pressure: float
    resource_denial: float
    self_mill: float
    switch_frequency: float


SYNTHETIC_OPPONENTS = (
    OpponentModel("synthetic_fast_ko", 1.0, 0.8, 0.1, 0.0, 0.6),
    OpponentModel("synthetic_slow_ko", 0.7, 0.6, 0.2, 0.0, 0.4),
    OpponentModel("synthetic_wall", 0.3, 0.4, 0.5, 0.1, 0.2),
    OpponentModel("synthetic_mill", 0.2, 0.2, 0.3, 1.0, 0.2),
    OpponentModel("synthetic_denial", 0.4, 0.3, 1.0, 0.0, 0.3),
    OpponentModel("synthetic_bench_pressure", 0.6, 1.0, 0.2, 0.0, 0.5),
    OpponentModel("synthetic_unknown", 0.55, 0.55, 0.55, 0.25, 0.4),
)


def opponent_model_names() -> list[str]:
    return [model.name for model in SYNTHETIC_OPPONENTS]
