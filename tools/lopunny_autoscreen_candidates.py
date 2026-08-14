from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from tools.build_meta_submission import build
from tools.replay_imitation import train


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    estimators: int
    learning_rate: float
    leaves: int
    min_child: int
    win_weight: float
    loss_weight: float
    seed_offset: int


MODEL_SPECS = (
    ModelSpec("compact", 180, 0.040, 15, 24, 1.8, 0.08, 11),
    ModelSpec("reference", 260, 0.035, 31, 18, 1.8, 0.08, 23),
    ModelSpec("deep", 420, 0.025, 31, 14, 2.0, 0.05, 37),
    ModelSpec("wide", 300, 0.030, 47, 18, 1.8, 0.08, 41),
    ModelSpec("regularized", 240, 0.030, 23, 34, 1.6, 0.12, 53),
    ModelSpec("wins_only", 320, 0.030, 31, 16, 2.8, 0.01, 67),
    ModelSpec("balanced", 300, 0.030, 31, 20, 1.0, 1.0, 79),
    ModelSpec("loss_aware", 340, 0.028, 31, 18, 1.5, 0.35, 97),
)


def generate(
    base_manifest: Path,
    out: Path,
    runtime: Path,
    cg_dir: Path,
    baseline: Path | None,
    n_jobs: int,
) -> list[Path]:
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    manifest_dir = out / "manifests"
    artifact_dir = out / "artifacts"
    candidate_dir = out / "candidates"
    for directory in (manifest_dir, artifact_dir, candidate_dir):
        directory.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    if baseline is not None:
        baseline_out = candidate_dir / "lopunny_previous_best.tar.gz"
        shutil.copy2(baseline, baseline_out)
        candidates.append(baseline_out)

    for spec in MODEL_SPECS:
        manifest = dict(base)
        manifest.update(
            n_estimators=spec.estimators,
            learning_rate=spec.learning_rate,
            num_leaves=spec.leaves,
            min_child_samples=spec.min_child,
            win_weight=spec.win_weight,
            loss_weight=spec.loss_weight,
            seed=int(base.get("seed", 20260802)) + spec.seed_offset,
            n_jobs=n_jobs,
        )
        manifest_path = manifest_dir / f"{spec.name}.json"
        artifact_path = artifact_dir / f"{spec.name}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact = train(manifest_path, artifact_path)
        metrics = artifact.get("validation_metrics", {})
        for profile in ("baseline", "cycle"):
            package = candidate_dir / f"lopunny_{spec.name}_{profile}.tar.gz"
            build(artifact_path, package, runtime, cg_dir, "clone", None, profile)
            candidates.append(package)
        print(json.dumps({
            "model": spec.name,
            "validation_top1": metrics.get("top1"),
            "validation_top3": metrics.get("top3"),
        }, ensure_ascii=False), flush=True)

    index = {
        "count": len(candidates),
        "candidates": [str(path) for path in candidates],
    }
    (out / "candidate_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate diverse Mega Lopunny imitation candidates for server autoscreening.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    candidates = generate(args.manifest, args.out, args.runtime, args.cg_dir, args.baseline, args.n_jobs)
    print(json.dumps({"generated": len(candidates), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
