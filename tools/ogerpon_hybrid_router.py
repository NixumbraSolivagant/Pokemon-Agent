from __future__ import annotations

import argparse
import ast
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from agents.meta_runtime import CARD_TABLE
from tools.build_meta_submission import build


def package_config(package: Path) -> dict[str, Any]:
    with tarfile.open(package, "r:gz") as archive:
        source = archive.extractfile("main.py").read().decode("utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"CONFIG not found in {package}")


def package_deck(package: Path) -> list[int]:
    with tarfile.open(package, "r:gz") as archive:
        text = archive.extractfile("deck.csv").read().decode("utf-8")
    return [int(value.strip()) for value in text.replace(",", "\n").splitlines() if value.strip()][:60]


def package_name(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".tar.gz") else path.stem


def alternate_overrides(anchor: dict[str, Any], alternate: dict[str, Any]) -> dict[str, Any]:
    ignored = {"own_deck", "opponent_decks", "policy_router"}
    overrides: dict[str, Any] = {}
    for key in sorted(set(anchor) | set(alternate)):
        if key in ignored or anchor.get(key) == alternate.get(key):
            continue
        if key == "search" and isinstance(anchor.get(key), dict) and isinstance(alternate.get(key), dict):
            changes = {
                nested: alternate[key][nested]
                for nested in sorted(set(anchor[key]) | set(alternate[key]))
                if anchor[key].get(nested) != alternate[key].get(nested)
            }
            if changes:
                overrides[key] = changes
        else:
            overrides[key] = alternate.get(key)
    return overrides


def matchup_rows(ranking_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in ranking}
    anchor = by_name["anchor_8689"]["matchups"]
    alternate = by_name["challenger_g00_032"]["matchups"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["opponents"]
    manifest_by_name = {package_name(Path(row["path"])): row for row in manifest}
    rows = []
    for name in sorted(set(anchor) & set(alternate) & set(manifest_by_name)):
        left = anchor[name]
        right = alternate[name]
        if left["wins"] + left["losses"] < 8 or right["wins"] + right["losses"] < 8:
            continue
        meta = manifest_by_name[name]
        rows.append({
            "name": name,
            "deck_sha256": meta["deck_sha256"],
            "package": Path(meta["path"]),
            "delta": float(right["rate"]) - float(left["rate"]),
            "weight": min(left["wins"] + left["losses"], right["wins"] + right["losses"]),
        })
    return rows


def grouped_decks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["deck_sha256"]].append(row)
    result = []
    for deck_hash, members in groups.items():
        total_weight = sum(row["weight"] for row in members)
        result.append({
            "deck_sha256": deck_hash,
            "cards": sorted(set(package_deck(members[0]["package"]))),
            "delta": sum(row["delta"] * row["weight"] for row in members) / max(1, total_weight),
            "teams": len(members),
        })
    return result


def learn_card_scores(groups: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[int, float]:
    if len(groups) < 8:
        raise ValueError("router training requires at least eight deck groups")
    rng = random.Random(seed)
    global_mean = fmean(group["delta"] for group in groups)
    card_groups: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        for card_id in group["cards"]:
            card_groups[card_id].append(index)
    scores: dict[int, float] = {}
    for card_id, indices in card_groups.items():
        if len(indices) < 2 or len(indices) > len(groups) - 2:
            continue
        effect = fmean(groups[index]["delta"] for index in indices) - global_mean
        if abs(effect) < 0.02:
            continue
        signs = 0
        for _ in range(bootstrap):
            sample = [groups[rng.randrange(len(groups))] for _ in groups]
            containing = [group["delta"] for group in sample if card_id in group["cards"]]
            if len(containing) < 2:
                continue
            sample_effect = fmean(containing) - fmean(group["delta"] for group in sample)
            signs += int(sample_effect * effect > 0.0)
        stability = signs / max(1, bootstrap)
        if stability < 0.80:
            continue
        shrink = len(indices) / (len(indices) + 4.0)
        scores[card_id] = max(-0.25, min(0.25, effect * shrink))
    return dict(sorted(scores.items(), key=lambda item: abs(item[1]), reverse=True)[:64])


def deck_profiles(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frequency: dict[int, int] = defaultdict(int)
    pokemon_by_group = []
    for group in groups:
        pokemon = {
            card_id for card_id in group["cards"]
            if CARD_TABLE.get(card_id) is not None and int(getattr(CARD_TABLE[card_id], "cardType", -1)) == 0
        }
        pokemon_by_group.append(pokemon)
        for card_id in pokemon:
            frequency[card_id] += 1
    profiles = []
    for group, pokemon in zip(groups, pokemon_by_group):
        distinctive = sorted(card_id for card_id in pokemon if frequency[card_id] <= max(2, len(groups) // 2))
        if distinctive:
            profiles.append({"cards": distinctive, "score": group["delta"], "deck_sha256": group["deck_sha256"]})
    return profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build conservative anchor/g32 Ogerpon policy-router candidates.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    anchor_config = package_config(args.anchor)
    alternate_config = package_config(args.alternate)
    overrides = alternate_overrides(anchor_config, alternate_config)
    groups = grouped_decks(matchup_rows(args.ranking, args.manifest))
    card_scores = learn_card_scores(groups, args.bootstrap, args.seed)
    profiles = deck_profiles(groups)
    if not profiles:
        raise RuntimeError("no router deck profiles found")

    candidates = []
    for max_turn in (0, 1):
        for threshold in (0.03, 0.06, 0.09, 0.12):
            min_evidence = 1
            name = f"hybrid_v3_x{max_turn}_t{int(threshold * 1000):03d}"
            package = args.out / f"{name}.tar.gz"
            router = {
                "enabled": True,
                "min_turn": 0,
                "max_turn": max_turn,
                "min_evidence": min_evidence,
                "experts": {
                    "g32": {
                        "deck_profiles": profiles,
                        "threshold": threshold,
                        "overrides": overrides,
                    }
                },
            }
            config = {key: value for key, value in anchor_config.items() if key not in {"own_deck", "opponent_decks"}}
            config["policy_router"] = router
            build(
                args.artifact,
                package,
                args.runtime,
                args.cg_dir,
                "clone_residual_search",
                linux_only=True,
                config_overrides=config,
            )
            candidates.append({
                "name": name,
                "package": str(package),
                "max_turn": max_turn,
                "threshold": threshold,
                "min_evidence": min_evidence,
            })
    result = {
        "anchor": str(args.anchor),
        "alternate": str(args.alternate),
        "alternate_overrides": overrides,
        "deck_groups": len(groups),
        "card_scores": {str(card_id): score for card_id, score in card_scores.items()},
        "deck_profiles": profiles,
        "candidates": candidates,
    }
    (args.out / "router_search.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"deck_groups": len(groups), "card_features": len(card_scores), "candidates": len(candidates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
