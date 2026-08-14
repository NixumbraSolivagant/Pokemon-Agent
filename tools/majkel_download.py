from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace")
from tools.opponent_league_pipeline import RequestPacer, download_episode, kaggle_command

WORKSPACE = Path("/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace")
KAGGLE = str(WORKSPACE / ".tools/uv/uvx") + " --python 3.12 --from kaggle==2.2.4 kaggle"
SUBMISSIONS = [55186239, 55147326]
OUT = WORKSPACE / "outputs/kaggle_logs/top_leaders_20260804" / "majkel_all"
STAGING = Path("/tmp/majkel-staging")
WORKERS = 3
INTERVAL = 2.0
JITTER = 0.5


def run_json(command):
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout or "null")


def fetch_episodes(submission):
    rows = run_json([*kaggle_command(KAGGLE), "competitions", "episodes", str(submission), "--format", "json", "-q"])
    return [
        int(row["id"])
        for row in rows
        if str(row.get("state", "")).endswith("COMPLETED") and str(row.get("type", "")).endswith("PUBLIC")
    ]


def main() -> None:
    ids: list[int] = []
    for submission in SUBMISSIONS:
        episodes = fetch_episodes(submission)
        ids.extend(episodes)
        print(f"[fetch] submission={submission} episodes={len(episodes)}", flush=True)
    ids = sorted(set(ids))
    print(f"[fetch] total_unique={len(ids)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    pacer = RequestPacer(INTERVAL, JITTER)
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(download_episode, KAGGLE, f"majkel:{episode_id}", episode_id, STAGING, OUT, pacer, 3, 120.0): episode_id
            for episode_id in ids
        }
        for number, future in enumerate(as_completed(futures), 1):
            key, error = future.result()
            if error:
                failures[key] = error
            if number % 25 == 0 or number == len(futures):
                print(f"[download] {number}/{len(ids)} failures={len(failures)}", flush=True)
    with open("/tmp/majkel_failures.json", "w") as handle:
        json.dump(failures, handle, indent=2)
    existing = len(list(OUT.glob("episode-*-replay.json.gz")))
    print(f"[done] files={existing} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
