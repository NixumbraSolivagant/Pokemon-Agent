from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file


ANCHOR_TABLE = Path("configs/lb_anchor_table.json")

DEFAULT_REFERENCE_PATHS = [
    Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
    Path("outputs/reference_submissions/submission_820.tar.gz"),
    Path("outputs/reference_submissions/pokemon-ai-battle-best-ptcg-advanced.tar.gz"),
    Path("outputs/reference_submissions/pokemon-steel.tar.gz"),
    Path("outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz"),
    Path("outputs/reference_submissions/improved-probabilistic-agent.tar.gz"),
    Path("outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz"),
    Path("outputs/reference_submissions/multiply-agent-best-940-lb.tar.gz"),
    Path("基准/submission_sorce_700.tar.gz"),
]

REFERENCE_BASES = {
    "great_tusk": Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
    "submission_820": Path("outputs/reference_submissions/submission_820.tar.gz"),
    "advanced": Path("outputs/reference_submissions/pokemon-ai-battle-best-ptcg-advanced.tar.gz"),
    "metal_tempo": Path("outputs/reference_submissions/pokemon-steel.tar.gz"),
    "rahul_metal": Path("outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz"),
    "lucario": Path("outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz"),
    "probabilistic": Path("outputs/reference_submissions/improved-probabilistic-agent.tar.gz"),
    "multiply_940": Path("outputs/reference_submissions/multiply-agent-best-940-lb.tar.gz"),
}


@dataclass(slots=True)
class Anchor:
    name: str
    path: Path
    role: str = ""
    lb_score: float | None = None
    score_source: str = ""
    score_date: str = ""
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "role": self.role,
            "lb_score": self.lb_score,
            "score_source": self.score_source,
            "score_date": self.score_date,
            "sha256": self.sha256,
            "exists": self.path.exists(),
        }


def submission_name(path: Path) -> str:
    return path.name.removesuffix(".tar.gz")


def load_anchor_table(path: Path = ANCHOR_TABLE) -> dict[str, Any]:
    if not path.exists():
        return {"target_lb_score": 1200, "anchors": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_anchors(path: Path = ANCHOR_TABLE, fill_sha: bool = False) -> list[Anchor]:
    table = load_anchor_table(path)
    anchors: list[Anchor] = []
    for row in table.get("anchors", []):
        if not isinstance(row, dict):
            continue
        anchor_path = Path(str(row.get("path") or ""))
        sha = str(row.get("sha256") or "")
        if fill_sha and anchor_path.exists():
            try:
                sha = sha256_file(anchor_path)
            except Exception:
                sha = ""
        lb_value = row.get("lb_score")
        anchors.append(
            Anchor(
                name=str(row.get("name") or submission_name(anchor_path)),
                path=anchor_path,
                role=str(row.get("role") or ""),
                lb_score=float(lb_value) if lb_value is not None else None,
                score_source=str(row.get("score_source") or ""),
                score_date=str(row.get("score_date") or ""),
                sha256=sha,
            )
        )
    return anchors


def default_pool(include_missing: bool = False) -> list[Path]:
    if include_missing:
        return list(DEFAULT_REFERENCE_PATHS)
    return unique_existing(DEFAULT_REFERENCE_PATHS)


def unique_existing(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen_sha: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            digest = sha256_file(path)
        except Exception:
            continue
        if digest in seen_sha:
            continue
        seen_sha.add(digest)
        out.append(path)
    return out


def pool_manifest(paths: list[Path] | None = None) -> dict[str, Any]:
    selected = paths or DEFAULT_REFERENCE_PATHS
    anchors_by_path = {anchor.path: anchor for anchor in load_anchors(fill_sha=True)}
    rows = []
    for path in selected:
        anchor = anchors_by_path.get(path)
        if anchor is None:
            rows.append({"name": submission_name(path), "path": str(path), "exists": path.exists()})
        else:
            rows.append(anchor.to_dict())
    return {"target_lb_score": load_anchor_table().get("target_lb_score", 1200), "pool": rows}
