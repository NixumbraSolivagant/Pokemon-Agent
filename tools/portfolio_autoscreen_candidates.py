from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from tools.build_meta_submission import build
from tools.replay_imitation import train


MODEL_VARIANTS = (
    ("reference", {}),
    (
        "deep",
        {
            "n_estimators": 440,
            "learning_rate": 0.024,
            "num_leaves": 47,
            "min_child_samples": 14,
            "win_weight": 2.2,
            "loss_weight": 0.05,
        },
    ),
)

MODEL_WEIGHTS = (80000.0, 140000.0, 220000.0, 320000.0)


def expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches:
            matches = [Path(pattern)]
        for path in matches:
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def train_variants(manifests: list[Path], out: Path, n_jobs: int) -> list[Path]:
    artifacts: list[Path] = []
    manifest_dir = out / "manifests"
    artifact_dir = out / "artifacts"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for source in manifests:
        base = json.loads(source.read_text(encoding="utf-8"))
        for variant_name, overrides in MODEL_VARIANTS:
            data = {**base, **overrides, "n_jobs": n_jobs}
            stem = f"{source.stem.replace('_private', '')}_{variant_name}"
            manifest_path = manifest_dir / f"{stem}.json"
            artifact_path = artifact_dir / f"{stem}.json"
            manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            artifact = train(manifest_path, artifact_path)
            artifacts.append(artifact_path)
            print(json.dumps({
                "trained": stem,
                "family": artifact.get("family"),
                "validation": artifact.get("validation_metrics"),
            }, ensure_ascii=False), flush=True)
    return artifacts


def package_artifacts(artifacts: list[Path], out: Path, runtime: Path, cg_dir: Path) -> list[Path]:
    candidate_dir = out / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    for artifact_path in artifacts:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        family = str(artifact["family"])
        profiles = ("baseline", "cycle") if family == "mega_lopunny" else ("baseline",)
        for model_weight in MODEL_WEIGHTS:
            for variant in ("clone", "clone_search"):
                for profile in profiles:
                    weight_tag = int(model_weight // 1000)
                    name = f"{artifact_path.stem}_{variant}_{profile}_w{weight_tag}k"
                    package = candidate_dir / f"{name}.tar.gz"
                    build(artifact_path, package, runtime, cg_dir, variant, model_weight, profile)
                    result.append(package)
    return result


def copy_existing(paths: list[Path], out: Path, seen_hashes: set[str]) -> list[Path]:
    candidate_dir = out / "candidates"
    result = []
    for source in paths:
        sha = digest(source)
        if sha in seen_hashes:
            continue
        seen_hashes.add(sha)
        destination = candidate_dir / f"existing_{source.stem}_{sha[:8]}.tar.gz"
        shutil.copy2(source, destination)
        result.append(destination)
    return result


def write_index(paths: list[Path], out: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append({"name": path.stem, "path": str(path), "sha256": digest(path)})
    (out / "candidate_index.json").write_text(
        json.dumps({"count": len(rows), "candidates": rows}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a broad multi-archetype candidate portfolio for server autoscreening.")
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--train-manifests", nargs="*", default=[])
    parser.add_argument("--existing", nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = expand(args.artifacts)
    artifacts.extend(train_variants(expand(args.train_manifests), args.out, args.n_jobs))
    generated = package_artifacts(artifacts, args.out, args.runtime, args.cg_dir)
    seen_hashes = {digest(path) for path in generated}
    existing = copy_existing(expand(args.existing), args.out, seen_hashes)
    candidates = generated + existing
    write_index(candidates, args.out)
    print(json.dumps({
        "artifacts": len(artifacts),
        "generated": len(generated),
        "existing": len(existing),
        "total": len(candidates),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
