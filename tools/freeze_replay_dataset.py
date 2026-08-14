from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.replay_imitation import (
    SourceSpec,
    deck_signature,
    load_source,
    replay_content_fingerprint,
    split_by_episode,
    split_fingerprint,
)


def build(
    replays: Path,
    target_name: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    parse_workers: int,
) -> dict:
    source = SourceSpec(name=target_name, path=replays, target_name=target_name)
    decisions, deck, stats = load_source(source, parse_workers=parse_workers, cache_dir=None)
    train, validation, test = split_by_episode(decisions, validation_fraction, test_fraction, seed)
    content = replay_content_fingerprint(replays, workers=parse_workers)
    return {
        "version": 1,
        "target_name": target_name,
        "replays": str(replays),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "dataset_sha256": content["sha256"],
        "replay_count": content["replays"],
        "deck": list(deck_signature(list(deck))),
        "deck_sha256": hashlib.sha256("\n".join(map(str, deck)).encode("utf-8")).hexdigest(),
        "source_stats": {key: value for key, value in stats.items() if key != "opponent_decks"},
        "splits": split_fingerprint(train, validation, test),
        "episodes": {
            "train": sorted({row.episode_id for row in train}),
            "validation": sorted({row.episode_id for row in validation}),
            "test": sorted({row.episode_id for row in test}),
        },
        "entries": content["entries"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify a replay-derived training benchmark.")
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--parse-workers", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = build(
        args.replays,
        args.target_name,
        args.seed,
        args.validation_fraction,
        args.test_fraction,
        max(1, args.parse_workers),
    )
    if args.check:
        expected = json.loads(args.out.read_text(encoding="utf-8"))
        keys = ("dataset_sha256", "replay_count", "deck_sha256", "splits", "episodes")
        mismatches = {
            key: {"expected": expected.get(key), "actual": current.get(key)}
            for key in keys
            if expected.get(key) != current.get(key)
        }
        if mismatches:
            raise SystemExit(json.dumps({"passed": False, "mismatches": mismatches}, ensure_ascii=False, indent=2))
        print(json.dumps({"passed": True, "dataset_sha256": current["dataset_sha256"]}, indent=2))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "dataset_sha256": current["dataset_sha256"], "replays": current["replay_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
