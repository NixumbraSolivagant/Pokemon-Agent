from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Outcome = Literal["P0_WIN", "P1_WIN", "DRAW", "P0_LOSS", "P1_LOSS", "NO_RESULT"]
EvalProfile = Literal["legacy", "kaggle"]
AgentStatus = Literal["ACTIVE", "INACTIVE", "DONE", "ERROR", "INVALID", "TIMEOUT"]


@dataclass(slots=True)
class EvalConfig:
    profile: EvalProfile = "legacy"
    act_timeout_s: float = 6.0
    import_timeout_s: float = 6.0
    deck_timeout_s: float = 6.0
    overage_time_s: float = 12.0
    run_timeout_s: float = 1200.0
    max_actions: int = 1000
    seed: int = 20260717
    trueskill_mu: float = 600.0
    trueskill_sigma: float = 200.0
    trueskill_beta: float = 100.0
    trueskill_tau: float = 2.0
    trueskill_draw_probability: float = 0.10
    workers: int = 1
    record_mode: str = "all"
    record_focus: str = ""
    record_sample_rate: float = 0.05
    record_gzip: bool = False
    progress: bool = False
    progress_label: str = ""
    progress_interval_s: float = 5.0
    progress_mode: str = "auto"
    progress_file: str = ""
    archive_cache_dir: str = ""
    max_in_flight: int = 0
    common_random_seeds: bool = False

    def __post_init__(self) -> None:
        if self.profile not in ("legacy", "kaggle"):
            raise ValueError(f"Unknown evaluator profile: {self.profile!r}")
        if self.profile == "kaggle":
            if self.common_random_seeds:
                raise ValueError(
                    "Kaggle profile cannot use common_random_seeds: the official cg engine "
                    "seeds battles from std::random_device and exposes no seeded battle start."
                )
            self.act_timeout_s = 0.0
            self.overage_time_s = 600.0
            self.run_timeout_s = 2000.0
            self.max_actions = 10_000_000


@dataclass(slots=True)
class SubmissionInfo:
    name: str
    tarball: str
    sha256: str


@dataclass(slots=True)
class GameResult:
    game_id: str
    p0: str
    p1: str
    seed: int
    result: int | None
    outcome: Outcome
    winner: str | None
    loser: str | None
    reason: str
    actions: int
    duration_s: float
    p0_seat: int = 0
    p1_seat: int = 1
    error: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list, repr=False)
    p0_status: AgentStatus = "DONE"
    p1_status: AgentStatus = "DONE"
    p0_reward: float | None = None
    p1_reward: float | None = None
    failure_class: str = ""
    ranking_eligible: bool = True

    def to_dict(self, include_trace: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_trace:
            data.pop("trace", None)
        return data


@dataclass(slots=True)
class AgentStats:
    name: str
    tarball: str
    sha256: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    crashes: int = 0
    timeouts: int = 0
    invalids: int = 0
    no_results: int = 0
    mu: float = 600.0
    sigma: float = 200.0
    kaggle_score_estimate: float = 0.0
    local_trueskill_score: float = 0.0
    kaggle_rank_score: float | None = None
    predicted_public_score: float | None = None
    prediction_interval: list[float] | None = None
    calibration_version: str = ""
    opponents: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatchReport:
    config: EvalConfig
    submissions: list[SubmissionInfo]
    games: list[GameResult]
    standings: list[AgentStats]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "metadata": self.metadata,
            "submissions": [asdict(s) for s in self.submissions],
            "games": [g.to_dict() for g in self.games],
            "standings": [s.to_dict() for s in self.standings],
        }


def default_output_dir(base: str | Path = "reports/local_eval") -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"run_{stamp}"
