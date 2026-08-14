from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace")
KAGGLE = [str(WORKSPACE / ".tools/uv/uvx"), "--python", "3.12", "--from", "kaggle==2.2.4", "kaggle"]
DATASET = sys.argv[1] if len(sys.argv) > 1 else "kaggle/pokemon-tcg-ai-battle-episodes-2026-08-08"
PENDING = Path("/tmp/pending_ids.txt")


def fetch_page(token: str | None) -> dict:
    command = [*KAGGLE, "datasets", "files", DATASET, "--format", "json", "--page-size", "200"]
    if token:
        command += ["--page-token", token]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    text = completed.stdout or ""
    next_token: str | None = None
    for line in text.splitlines():
        if line.startswith("Next Page Token = "):
            next_token = line.split("=", 1)[1].strip()
            break
    start = text.find("[")
    files = json.loads(text[start:]) if start >= 0 else []
    return {"files": files, "nextPageToken": next_token}


def main() -> None:
    pending = {int(line) for line in PENDING.read_text(encoding="utf-8").splitlines() if line.strip()}
    export_ids: set[int] = set()
    token: str | None = None
    pages = 0
    while True:
        payload = fetch_page(token)
        pages += 1
        for item in payload.get("files", []):
            name = str(item.get("name", ""))
            if name.endswith(".json"):
                try:
                    export_ids.add(int(name.removesuffix(".json")))
                except ValueError:
                    continue
        token = payload.get("nextPageToken")
        if not token:
            break
    overlap = pending & export_ids
    print(
        f"[overlap] export={len(export_ids)} pending={len(pending)} "
        f"overlap={len(overlap)} coverage={len(overlap) / max(1, len(pending)):.1%} pages={pages}",
        flush=True,
    )
    Path("/tmp/export_ids.txt").write_text("\n".join(map(str, sorted(export_ids))), encoding="utf-8")
    Path("/tmp/export_overlap_ids.txt").write_text("\n".join(map(str, sorted(overlap))), encoding="utf-8")


if __name__ == "__main__":
    main()
