from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def run(module: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], check=True)


def anchor_config(package: Path) -> dict:
    with tarfile.open(package, "r:gz") as archive:
        source = archive.extractfile("main.py")
        if source is None:
            raise FileNotFoundError(f"{package} has no main.py")
        module = ast.parse(source.read().decode("utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "CONFIG" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{package} main.py has no literal CONFIG")


def anchor_model(package: Path) -> dict:
    with tarfile.open(package, "r:gz") as archive:
        payload = archive.extractfile("model.json.gz")
        if payload is None:
            raise FileNotFoundError(f"{package} has no model.json.gz")
        import gzip

        return json.loads(gzip.decompress(payload.read()))


def anchor_model_gzip_b64(package: Path) -> str:
    with tarfile.open(package, "r:gz") as archive:
        payload = archive.extractfile("model.json.gz")
        if payload is None:
            raise FileNotFoundError(f"{package} has no model.json.gz")
        return base64.b64encode(payload.read()).decode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run champion-gated Ogerpon advantage training and evaluation.")
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--anchor-package", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--focus-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--data-workers", type=int, default=32)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--max-states", type=int, default=20_000)
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=650)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / "counterfactual_advantage.jsonl"
    artifact = args.out / "advantage_artifact.json"
    package = args.out / "ogerpon_advantage.tar.gz"
    evaluation = args.out / "evaluation"
    config_overrides = args.out / "anchor_config.json"
    anchor_sha256 = hashlib.sha256(args.anchor_package.read_bytes()).hexdigest()
    overrides = anchor_config(args.anchor_package)
    overrides["__replace_config__"] = True
    config_overrides.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    base_artifact = json.loads(args.base_artifact.read_text(encoding="utf-8"))
    base_artifact["__packaged_model__"] = anchor_model(args.anchor_package)
    base_artifact["__packaged_model_gzip_b64__"] = anchor_model_gzip_b64(args.anchor_package)
    anchored_artifact = args.out / "anchor_artifact.json"
    anchored_artifact.write_text(json.dumps(base_artifact, separators=(",", ":")), encoding="utf-8")

    run("tools.ogerpon_counterfactual_data", [
        "--artifact", str(args.base_artifact),
        "--records", str(args.records),
        "--focus-name", args.focus_name,
        "--config-overrides", str(config_overrides),
        "--out", str(data),
        "--max-states", str(args.max_states),
        "--max-candidates", "4",
        "--belief-worlds", "5",
        "--rollout-steps", "12",
        "--risk-penalty", "0.20",
        "--state-budget-s", "0.75",
        "--workers", str(args.data_workers),
    ])
    run("tools.ogerpon_advantage_train", [
        "--base-artifact", str(anchored_artifact),
        "--data", str(data),
        "--out", str(artifact),
        "--anchor-package-sha256", anchor_sha256,
        "--gpu-device", str(args.gpu_device),
        "--models", str(args.models),
        "--iterations", str(args.iterations),
    ])
    run("tools.build_meta_submission", [
        "--artifact", str(artifact),
        "--out", str(package),
        "--variant", "clone_residual_search",
        "--config-overrides", str(config_overrides),
    ])
    run("tools.league_eval", [
        "--candidates", str(args.anchor_package), str(package),
        "--manifest", str(args.manifest),
        "--games", str(args.games),
        "--workers", str(args.workers),
        "--record-mode", "none",
        "--out", str(evaluation),
    ])
    result = {
        "anchor_package": str(args.anchor_package),
        "anchor_sha256": anchor_sha256,
        "artifact": str(artifact),
        "candidate_package": str(package),
        "evaluation": str(evaluation / "league_ranking.json"),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
