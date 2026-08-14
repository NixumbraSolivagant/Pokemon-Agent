from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.build_meta_submission import build
from tools.ogerpon_hybrid_router import package_config


def blend_number(left: float | int, right: float | int, alpha: float):
    value = float(left) + alpha * (float(right) - float(left))
    return int(round(value)) if isinstance(left, int) and isinstance(right, int) else value


def blended_overrides(anchor: dict[str, Any], alternate: dict[str, Any], alpha: float) -> dict[str, Any]:
    ignored = {"own_deck", "opponent_decks", "policy_router"}
    result = {key: value for key, value in anchor.items() if key not in ignored}
    for key in sorted(set(anchor) & set(alternate)):
        if key in ignored or anchor[key] == alternate[key]:
            continue
        if key == "search" and isinstance(anchor[key], dict) and isinstance(alternate[key], dict):
            search = dict(anchor[key])
            for nested in sorted(set(anchor[key]) & set(alternate[key])):
                left, right = anchor[key][nested], alternate[key][nested]
                if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
                    search[nested] = blend_number(left, right, alpha)
            result[key] = search
        else:
            left, right = anchor[key], alternate[key]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
                result[key] = blend_number(left, right, alpha)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build static Ogerpon parameter blends between anchor and g32.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    anchor = package_config(args.anchor)
    alternate = package_config(args.alternate)
    alphas = (-0.25, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.125, 1.25)
    candidates = []
    for alpha in alphas:
        name = f"ogerpon_blend_{alpha:+.3f}".replace("+", "p").replace("-", "m").replace(".", "d")
        package = args.out / f"{name}.tar.gz"
        overrides = blended_overrides(anchor, alternate, alpha)
        build(args.artifact, package, args.runtime, args.cg_dir, "clone_residual_search", linux_only=True, config_overrides=overrides)
        candidates.append({"name": name, "alpha": alpha, "package": str(package)})
    (args.out / "blend_search.json").write_text(json.dumps({"candidates": candidates}, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": len(candidates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
