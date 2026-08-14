from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tools.kaggle_replay_miner import full_deck
from tools.opponent_league_pipeline import RequestPacer, download_episode, kaggle_command


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_json(command: list[str]) -> Any:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout or "null")


def replay_deck(path: Path, team_name: str) -> tuple[int, ...]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        replay = json.load(handle)
    names = (replay.get("info") or {}).get("TeamNames") or []
    target = team_name.casefold()
    seat = next((index for index, name in enumerate(names) if str(name).casefold() == target), None)
    if seat is None:
        raise ValueError(f"team {team_name!r} not found")
    deck = tuple(sorted(full_deck(replay, seat)))
    if len(deck) != 60:
        raise ValueError(f"expected 60 cards, found {len(deck)}")
    return deck


def deck_sha256(deck: tuple[int, ...]) -> str:
    return hashlib.sha256("\n".join(map(str, deck)).encode("utf-8")).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--submission", type=int, required=True)
    result.add_argument("--team-name", required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--state", type=Path, required=True)
    result.add_argument("--staging", type=Path, required=True)
    result.add_argument("--kaggle", required=True)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--request-interval", type=float, default=1.55)
    result.add_argument("--request-jitter", type=float, default=0.0)
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--max-backoff", type=float, default=300.0)
    result.add_argument("--expected-deck-sha256")
    return result


def main() -> int:
    args = parser().parse_args()
    rows = run_json([
        *kaggle_command(args.kaggle),
        "competitions",
        "episodes",
        str(args.submission),
        "--format",
        "json",
        "-q",
    ])
    episode_ids = sorted({
        int(row["id"])
        for row in rows
        if row.get("id")
        and str(row.get("state", "")).endswith("COMPLETED")
        and str(row.get("type", "")).endswith("PUBLIC")
    })
    args.out.mkdir(parents=True, exist_ok=True)
    args.staging.mkdir(parents=True, exist_ok=True)
    quarantine = args.out.parent / f"{args.out.name}_deck_mismatch"
    pacer = RequestPacer(args.request_interval, args.request_jitter)
    lock = threading.Lock()
    failures: dict[str, str] = {}
    mismatches: dict[str, str] = {}

    def snapshot() -> None:
        completed = sum((args.out / f"episode-{episode_id}-replay.json.gz").exists() for episode_id in episode_ids)
        write_json(args.state, {
            "submission": args.submission,
            "team_name": args.team_name,
            "expected": len(episode_ids),
            "completed": completed,
            "failures": failures,
            "deck_mismatches": mismatches,
            "request_interval": args.request_interval,
        })

    def download(episode_id: int) -> tuple[int, str | None]:
        key, error = download_episode(
            args.kaggle,
            f"submission_{args.submission}:{episode_id}",
            episode_id,
            args.staging,
            args.out,
            pacer,
            args.max_retries,
            args.max_backoff,
        )
        if error:
            return episode_id, error
        target = args.out / f"episode-{episode_id}-replay.json.gz"
        try:
            digest = deck_sha256(replay_deck(target, args.team_name))
        except Exception as exc:
            return episode_id, f"deck validation failed: {type(exc).__name__}: {exc}"
        if args.expected_deck_sha256 and digest != args.expected_deck_sha256:
            quarantine.mkdir(parents=True, exist_ok=True)
            shutil.move(target, quarantine / target.name)
            with lock:
                mismatches[str(episode_id)] = digest
            return episode_id, f"deck mismatch: {digest}"
        return episode_id, None

    snapshot()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(download, episode_id): episode_id for episode_id in episode_ids}
        for number, future in enumerate(as_completed(futures), 1):
            episode_id, error = future.result()
            if error:
                with lock:
                    failures[str(episode_id)] = error
            if number % 10 == 0 or number == len(futures):
                snapshot()
            completed = sum((args.out / f"episode-{value}-replay.json.gz").exists() for value in episode_ids)
            print(
                f"[download] {number}/{len(episode_ids)} completed={completed} "
                f"failures={len(failures)} deck_mismatches={len(mismatches)}",
                flush=True,
            )
    snapshot()
    if failures:
        raise RuntimeError(f"download incomplete: failures={len(failures)}")
    write_json(args.out.parent / f"{args.out.name}_complete.json", json.loads(args.state.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
