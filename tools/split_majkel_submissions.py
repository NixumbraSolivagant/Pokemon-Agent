from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace")
KAGGLE = [str(WORKSPACE / ".tools/uv/uvx"), "--python", "3.12", "--from", "kaggle==2.2.4", "kaggle"]
SUBMISSIONS = {"sub1": 55186239, "sub2": 55147326}
SRC = WORKSPACE / "outputs/kaggle_logs/top_leaders_20260804/majkel_all"


def fetch_episodes(submission: int) -> set[int]:
    completed = subprocess.run(
        [*KAGGLE, "competitions", "episodes", str(submission), "--format", "json", "-q"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(completed.stdout or "null")
    return {
        int(row["id"])
        for row in rows
        if str(row.get("state", "")).endswith("COMPLETED") and str(row.get("type", "")).endswith("PUBLIC")
    }


def main() -> None:
    ids: dict[str, set[int]] = {}
    for name, submission in SUBMISSIONS.items():
        ids[name] = fetch_episodes(submission)
        print(f"[split] {name} submission={submission} episodes={len(ids[name])}", flush=True)
    print(
        f"[split] overlap={len(ids['sub1'] & ids['sub2'])} union={len(ids['sub1'] | ids['sub2'])}",
        flush=True,
    )
    for name, episodes in ids.items():
        dst = WORKSPACE / "outputs/kaggle_logs/top_leaders_20260804" / f"majkel_{name}"
        dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for episode in sorted(episodes):
            source = SRC / f"episode-{episode}-replay.json.gz"
            if source.exists():
                shutil.copy2(source, dst / source.name)
                copied += 1
        print(f"[split] {name} copied={copied} dir={dst.name}", flush=True)


if __name__ == "__main__":
    main()
