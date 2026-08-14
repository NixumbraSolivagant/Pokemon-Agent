from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


KAGGLE_ENVIRONMENT_VERSION = "1.16.0"
CABT_SPEC_VERSION = "cabt-1.16.0"
CABT_CONFIGURATION = {
    "actTimeout": 0,
    "runTimeout": 2000,
    "episodeSteps": 10_000_000,
    "remainingOverageTime": 600,
}
OFFICIAL_CG_SHA256 = {
    "cg/api.py": "593f1298e52a635f90f8f505a52113e9af114f444c293404e37906f18ee06ced",
    "cg/game.py": "3bd3d4f4a369a11e6d2f5da9094cf15ebc410a2221835e6417b7cff4883f1fc2",
    "cg/sim.py": "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655",
    "cg/utils.py": "60f29665cee0a88525d6f0383bc45959a6262d16fe35ef380aece1e0ea13c49b",
    "cg/libcg.so": "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c",
}


def cg_hash_report(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    actual: dict[str, str | None] = {}
    for relative_path in OFFICIAL_CG_SHA256:
        path = root / relative_path
        actual[relative_path] = _sha256(path) if path.is_file() else None
    return {
        "expected": OFFICIAL_CG_SHA256,
        "actual": actual,
        "matches_official": actual == OFFICIAL_CG_SHA256,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
