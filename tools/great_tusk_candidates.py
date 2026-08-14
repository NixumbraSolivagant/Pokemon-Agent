from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.build_models import BuildConfig
from tools.build_submission import build_submission
from tools.deck_rules import validate_deck_ids
from tools.export_kaggle_submission import export_kaggle_submission


@dataclass(slots=True)
class CandidateSpec:
    name: str
    hypothesis: str
    parent: str
    base: Path
    deck_swaps: list[tuple[int, int]] = field(default_factory=list)
    deck_override: list[int] | None = None
    runtime_source: Path = Path("main.py")
    runtime_cg_dir: Path = Path("cg")
    policy_variant: str = "adaptive"
    strategy_weights: dict[str, float] = field(default_factory=dict)
    search_candidates: int = 8
    search_budget_s: float = 0.17
    search_margin: float = 2300.0
    search_rollout_steps: int = 11
    belief_worlds: int = 8
    risk_penalty: float = 0.40


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, Any], list[CandidateSpec]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = dict(data.get("defaults", {}))
    specs: list[CandidateSpec] = []
    names: set[str] = set()
    for raw in data.get("candidates", []):
        merged = {**defaults, **raw}
        name = str(merged["name"])
        if name in names:
            raise ValueError(f"duplicate candidate name: {name}")
        names.add(name)
        swaps = [tuple(map(int, pair)) for pair in merged.get("deck_swaps", [])]
        deck_override = merged.get("deck_override")
        specs.append(
            CandidateSpec(
                name=name,
                hypothesis=str(merged["hypothesis"]),
                parent=str(merged.get("parent", "champion")),
                base=Path(merged["base"]),
                deck_swaps=swaps,
                deck_override=[int(card_id) for card_id in deck_override] if deck_override is not None else None,
                runtime_source=Path(merged.get("runtime_source", "main.py")),
                runtime_cg_dir=Path(merged.get("runtime_cg_dir", "cg")),
                policy_variant=str(merged.get("policy_variant", "adaptive")),
                strategy_weights={str(key): float(value) for key, value in merged.get("strategy_weights", {}).items()},
                search_candidates=int(merged.get("search_candidates", 8)),
                search_budget_s=float(merged.get("search_budget_s", 0.17)),
                search_margin=float(merged.get("search_margin", 2300.0)),
                search_rollout_steps=int(merged.get("search_rollout_steps", 11)),
                belief_worlds=int(merged.get("belief_worlds", 8)),
                risk_penalty=float(merged.get("risk_penalty", 0.40)),
            )
        )
    if not specs:
        raise ValueError(f"manifest has no candidates: {path}")
    return data, specs


def inspect_package(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        required = {"main.py", "deck.csv", "cg/game.py", "cg/api.py"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{path} missing required files: {missing}")
        main_payload = archive.extractfile("main.py")
        deck_payload = archive.extractfile("deck.csv")
        if main_payload is None or deck_payload is None:
            raise ValueError(f"{path} has unreadable required files")
        main_data = main_payload.read()
        deck = [int(line) for line in deck_payload.read().decode("utf-8").splitlines() if line.strip()]
    compile(main_data, f"{path}:main.py", "exec")
    validate_deck_ids(deck)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "main_sha256": hashlib.sha256(main_data).hexdigest(),
        "deck_sha256": hashlib.sha256("\n".join(map(str, deck)).encode()).hexdigest(),
        "deck": deck,
    }


def build_candidates(manifest: Path, out: Path) -> list[dict[str, Any]]:
    _, specs = load_manifest(manifest)
    internal_dir = out / "internal"
    kaggle_dir = out / "kaggle"
    internal_dir.mkdir(parents=True, exist_ok=True)
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for spec in specs:
        internal = internal_dir / f"{spec.name}.tar.gz"
        submission = kaggle_dir / f"{spec.name}.tar.gz"
        config = BuildConfig(
            name=spec.name,
            family="great_tusk",
            base=spec.base,
            out=internal,
            runtime_source=spec.runtime_source,
            runtime_cg_dir=spec.runtime_cg_dir,
            enable_search=True,
            injection="great_tusk",
            search_candidates=spec.search_candidates,
            search_budget_s=spec.search_budget_s,
            search_margin=spec.search_margin,
            search_rollout_steps=spec.search_rollout_steps,
            belief_worlds=spec.belief_worlds,
            risk_penalty=spec.risk_penalty,
            deck_swaps=spec.deck_swaps,
            deck_override=spec.deck_override,
            strategy_weights=spec.strategy_weights,
            policy_variant=spec.policy_variant,
            origin="great_tusk_gold_private_manifest",
            notes=spec.hypothesis,
        )
        build_submission(config)
        export_kaggle_submission(internal, submission)
        package = inspect_package(submission)
        records.append(
            {
                **asdict(spec),
                "base": str(spec.base),
                "runtime_source": str(spec.runtime_source),
                "runtime_cg_dir": str(spec.runtime_cg_dir),
                "internal": str(internal),
                "submission": str(submission),
                "package": package,
            }
        )
    index_path = out / "candidate_index.json"
    index_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build isolated Great Tusk hypotheses from a private manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/great_tusk_gold_private.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/great_tusk_gold"))
    args = parser.parse_args(argv)
    records = build_candidates(args.manifest, args.out)
    print(json.dumps({"out": str(args.out), "candidates": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
