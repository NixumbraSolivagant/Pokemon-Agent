from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_BASE = Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz")
DEFAULT_OUT = Path("outputs/submissions/champion_gt_search.tar.gz")


@dataclass(slots=True)
class BuildConfig:
    name: str = "champion_gt_search"
    family: str = "great_tusk"
    base: Path = DEFAULT_BASE
    out: Path = DEFAULT_OUT
    enable_search: bool = True
    injection: str = "great_tusk"
    search_candidates: int = 8
    search_budget_s: float = 0.25
    search_margin: float = 1200.0
    search_rollout_steps: int = 16
    belief_worlds: int = 4
    risk_penalty: float = 0.20
    opponent_decks: dict[str, list[int]] = field(default_factory=dict)
    deck_swaps: list[tuple[int, int]] = field(default_factory=list)
    deck_override: list[int] | None = None
    deck_files: tuple[str, ...] = ("deck.csv",)
    strategy_weights: dict[str, float] = field(default_factory=dict)
    policy_variant: str = "default"
    opponent_model: str = "perfect"
    origin: str = ""
    notes: str = ""
    include_build_metadata: bool = True
