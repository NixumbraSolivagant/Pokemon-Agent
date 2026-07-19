from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from local_eval.archive import sha256_file


DEFAULT_LEDGER = Path("outputs/leaderboard_observations.jsonl")


@dataclass(frozen=True, slots=True)
class LeaderboardObservation:
    submission: str
    sha256: str
    score: float
    rank: int | None
    leaderboard: str
    observed_at: str
    notes: str = ""


def append_observation(path: Path, observation: LeaderboardObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(observation), ensure_ascii=False) + "\n")


def load_observations(path: Path = DEFAULT_LEDGER) -> list[LeaderboardObservation]:
    if not path.exists():
        return []
    observations: list[LeaderboardObservation] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                observations.append(LeaderboardObservation(**json.loads(line)))
    return observations


def record(args: argparse.Namespace) -> LeaderboardObservation:
    submission = args.submission.resolve()
    observation = LeaderboardObservation(
        submission=str(submission),
        sha256=sha256_file(submission),
        score=float(args.score),
        rank=args.rank,
        leaderboard=args.leaderboard,
        observed_at=args.observed_at or datetime.now(timezone.utc).isoformat(),
        notes=args.notes,
    )
    append_observation(args.ledger, observation)
    return observation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record user-reported Kaggle leaderboard observations.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--leaderboard", choices=["public", "private"], default="public")
    parser.add_argument("--observed-at")
    parser.add_argument("--notes", default="")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)
    print(json.dumps(asdict(record(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
