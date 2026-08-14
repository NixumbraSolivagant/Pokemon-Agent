from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cg.api import all_attack, all_card_data

from tools.deck_rules import deck_to_text, validate_deck_ids


CARD_IDS = {card.name: card.cardId for card in all_card_data()}
ATTACK_IDS = {attack.name: attack.attackId for attack in all_attack()}


def ids(names: list[str]) -> list[int]:
    return [CARD_IDS[name] for name in names if name in CARD_IDS]


def priority(names: list[str], start: float, step: float = 40.0) -> dict[str, float]:
    return {str(card_id): start - index * step for index, card_id in enumerate(ids(names))}


def compact_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "oblivious_trees": model.get("oblivious_trees", []),
        "scale_and_bias": model.get("scale_and_bias"),
    }


def family_config(
    family: str,
    variant: str,
    model_weight_override: float | None = None,
    lopunny_profile: str = "baseline",
) -> dict[str, Any]:
    search_enabled = variant.endswith("search")
    model_weight = 0.0 if variant.startswith("rules") else float(model_weight_override or 200000.0)
    if family == "generic_replay":
        return {
            "family": family,
            "model_weight": model_weight,
            "value_weight": 12000.0 if search_enabled else 0.0,
            "deck_guard": 3,
            "optional_thin_cards": [],
            "setup_active_priority": {},
            "setup_bench_priority": {},
            "play_priority": {},
            "search_priority": {},
            "keep_priority": {},
            "evolve_priority": {},
            "attach_priority": {},
            "active_priority": {},
            "board_priority": {},
            "target_priority": {},
            "attack_priority": {},
            "search": {"enabled": False, "candidates": 3, "budget_s": 0.08, "rollout_steps": 4},
        }
    if family == "grimmsnarl":
        return {
            "family": family,
            "model_weight": model_weight,
            "deck_guard": 5,
            "optional_thin_cards": ids(["Buddy-Buddy Poffin", "Poké Pad", "Pokégear 3.0"]),
            "setup_active_priority": priority(["Marnie's Impidimp", "Munkidori", "Snorunt"], 1200.0, 180.0),
            "setup_bench_priority": priority(["Marnie's Impidimp", "Snorunt", "Munkidori"], 1200.0, 100.0),
            "play_priority": priority([
                "Rare Candy", "Buddy-Buddy Poffin", "Poké Pad", "Lillie's Determination",
                "Team Rocket's Petrel", "Night Stretcher", "Unfair Stamp", "Boss’s Orders",
                "Spikemuth Gym", "Pokégear 3.0", "Tool Scrapper", "Dawn",
            ], 900.0, 45.0),
            "search_priority": priority([
                "Marnie's Grimmsnarl ex", "Marnie's Morgrem", "Marnie's Impidimp", "Froslass",
                "Snorunt", "Munkidori", "Rare Candy", "Lillie's Determination", "Team Rocket's Petrel",
                "Night Stretcher", "Boss’s Orders", "Basic {D} Energy",
            ], 1300.0, 55.0),
            "keep_priority": priority([
                "Marnie's Grimmsnarl ex", "Rare Candy", "Marnie's Impidimp", "Froslass", "Munkidori",
                "Night Stretcher", "Boss’s Orders", "Unfair Stamp", "Basic {D} Energy",
            ], 1200.0, 70.0),
            "evolve_priority": priority(["Marnie's Impidimp", "Marnie's Morgrem", "Snorunt"], 1000.0, 80.0),
            "attach_priority": priority(["Marnie's Grimmsnarl ex", "Marnie's Morgrem", "Marnie's Impidimp", "Munkidori"], 850.0, 80.0),
            "active_priority": priority(["Marnie's Grimmsnarl ex", "Marnie's Impidimp", "Marnie's Morgrem", "Munkidori", "Froslass"], 1000.0, 120.0),
            "board_priority": priority(["Marnie's Grimmsnarl ex", "Froslass", "Munkidori", "Marnie's Morgrem", "Marnie's Impidimp"], 700.0, 80.0),
            "target_priority": {
                "169": 50000.0,
                **priority([
                    "Riolu", "Makuhita", "Froslass", "Snorunt", "Munkidori", "Lunatone",
                    "Solrock", "Marnie's Impidimp", "Mega Lucario ex",
                ], 1500.0, 110.0),
            },
            "attack_priority": {str(ATTACK_IDS.get("Shadow Bullet", 937)): 850.0},
            "search": {"enabled": search_enabled, "candidates": 3, "budget_s": 0.10, "rollout_steps": 5},
        }
    if family == "mega_kangaskhan":
        return {
            "family": family,
            "model_weight": model_weight,
            "deck_guard": 4,
            "optional_thin_cards": ids(["Ultra Ball", "Cyrano"]),
            "setup_active_priority": priority(["Mega Kangaskhan ex", "Teal Mask Ogerpon ex", "Latias ex", "Raging Bolt ex", "Meowth ex", "Passimian"], 1500.0, 120.0),
            "setup_bench_priority": priority(["Mega Kangaskhan ex", "Meowth ex", "Teal Mask Ogerpon ex", "Latias ex", "Raging Bolt ex"], 1400.0, 80.0),
            "play_priority": priority([
                "Crispin", "Energy Switch", "Ultra Ball", "Area Zero Underdepths", "Cyrano",
                "Glass Trumpet", "Prime Catcher", "Boss’s Orders", "Night Stretcher", "Xerosic’s Machinations",
            ], 950.0, 50.0),
            "search_priority": priority([
                "Mega Kangaskhan ex", "Meowth ex", "Raging Bolt ex", "Teal Mask Ogerpon ex",
                "Wellspring Mask Ogerpon ex", "Latias ex", "Fezandipiti ex", "Passimian",
            ], 1350.0, 60.0),
            "keep_priority": priority([
                "Mega Kangaskhan ex", "Raging Bolt ex", "Crispin", "Energy Switch", "Prime Catcher",
                "Boss’s Orders", "Glass Trumpet", "Night Stretcher",
            ], 1250.0, 65.0),
            "evolve_priority": priority(["Meowth ex"], 1000.0),
            "attach_priority": priority(["Raging Bolt ex", "Mega Kangaskhan ex", "Teal Mask Ogerpon ex", "Wellspring Mask Ogerpon ex"], 900.0, 70.0),
            "active_priority": priority(["Mega Kangaskhan ex", "Raging Bolt ex", "Wellspring Mask Ogerpon ex", "Teal Mask Ogerpon ex", "Passimian"], 1100.0, 100.0),
            "board_priority": priority(["Mega Kangaskhan ex", "Raging Bolt ex", "Teal Mask Ogerpon ex", "Latias ex"], 750.0, 80.0),
            "target_priority": priority(["Froslass", "Snorunt", "Munkidori", "Marnie's Impidimp"], 900.0, 90.0),
            "attack_priority": {
                str(ATTACK_IDS.get("Rapid-Fire Combo", 1092)): 760.0,
                str(ATTACK_IDS.get("Bellowing Thunder", 0)): 820.0,
                str(ATTACK_IDS.get("Torrential Pump", 0)): 700.0,
                str(ATTACK_IDS.get("Coordinated Throwing", 0)): 650.0,
            },
            "search": {"enabled": search_enabled, "candidates": 4, "budget_s": 0.12, "rollout_steps": 5},
        }
    if family == "teal_ogerpon":
        tactical = variant in {
            "clone_combat", "clone_combat_search", "clone_anti_mill", "clone_anti_mill_search",
            "clone_residual", "clone_residual_search", "clone_residual_anti_mill", "clone_residual_anti_mill_search",
        }
        anti_mill = variant in {
            "clone_anti_mill", "clone_anti_mill_search", "clone_residual_anti_mill", "clone_residual_anti_mill_search",
        }
        residual = variant in {
            "clone_residual", "clone_residual_search", "clone_residual_anti_mill", "clone_residual_anti_mill_search",
        }
        value_variant = variant in {
            "clone_value", "clone_search", "clone_combat_search", "clone_anti_mill_search",
            "clone_residual_search", "clone_residual_anti_mill_search",
        }
        targetfix = variant in {"clone_targetfix", "clone_value", "clone_search"} or tactical
        endgame = variant in {"clone_endgame", "clone_value", "clone_search"} or tactical
        return {
            "family": family,
            "model_weight": model_weight,
            "type_head_weight": 0.5,
            "type_head_weights": {"main_play": 0.0, "main_attack": 0.0, "main_attach": 0.5, "main_end": 1.0},
            "residual_weight": 60000.0 if residual else 0.0,
            "residual_low_confidence_only": True,
            "value_weight": 18000.0 if value_variant else 0.0,
            "ogerpon_targetfix": targetfix,
            "ogerpon_endgame": endgame,
            "ogerpon_tactical": tactical,
            "ogerpon_anti_mill": anti_mill,
            "ogerpon_force_ability": False,
            "ogerpon_force_promotion": tactical,
            "ogerpon_force_terminal_attack": tactical,
            "deck_guard": 5,
            "optional_thin_cards": ids(["Bug Catching Set", "Energy Search", "Pokégear 3.0", "Tera Orb"]),
            "setup_active_priority": priority(["Teal Mask Ogerpon ex"], 1800.0),
            "setup_bench_priority": priority(["Teal Mask Ogerpon ex"], 1700.0),
            "play_priority": priority([
                "Teal Mask Ogerpon ex", "Bug Catching Set", "Energy Search", "Energy Retrieval", "Tera Orb",
                "Lively Stadium", "Lillie's Determination", "Judge", "Jumbo Ice Cream", "Crushing Hammer",
                "Boss’s Orders", "Harlequin", "Briar", "N's Plan", "Tool Scrapper",
            ], 1250.0, 45.0),
            "search_priority": priority([
                "Teal Mask Ogerpon ex", "Basic {G} Energy", "Grow Grass Energy", "Lillie's Determination",
                "Judge", "Boss’s Orders", "Briar", "Harlequin", "Jumbo Ice Cream",
            ], 1500.0, 70.0),
            "keep_priority": priority([
                "Teal Mask Ogerpon ex", "Basic {G} Energy", "Grow Grass Energy", "Hero’s Cape",
                "Lillie's Determination", "Boss’s Orders", "Briar", "N's Plan",
            ], 1400.0, 65.0),
            "attach_priority": priority(["Teal Mask Ogerpon ex"], 1500.0),
            "active_priority": priority(["Teal Mask Ogerpon ex"], 1600.0),
            "board_priority": priority(["Teal Mask Ogerpon ex"], 1300.0),
            "target_priority": priority([
                "Marnie's Impidimp", "Snorunt", "Dunsparce", "Buneary", "Riolu", "Teal Mask Ogerpon ex",
            ], 1200.0, 80.0),
            "attack_priority": {str(ATTACK_IDS.get("Myriad Leaf Shower", 120)): 1400.0},
            "search": {
                "enabled": search_enabled,
                "low_confidence_only": True,
                "contexts": ["MAIN", "SWITCH", "TO_ACTIVE"] if tactical else [],
                "candidates": 4,
                "budget_s": 0.16 if tactical else 0.12,
                "rollout_steps": 8 if tactical else 6,
                "belief_worlds": 3,
                "risk_penalty": 0.20,
                "margin": 2500.0,
                "minimum_overage_s": 30.0,
                "value_model_weight": 24000.0,
                "terminal_main_only": True,
            },
        }
    if family == "mega_lopunny":
        attack_fix = lopunny_profile in ("attack", "cycle", "full")
        cycle = lopunny_profile in ("cycle", "full")
        full = lopunny_profile == "full"
        return {
            "family": family,
            "model_weight": model_weight,
            "lopunny_attack_fix": attack_fix,
            "lopunny_cycle": cycle,
            "lopunny_deck_guard": full,
            "lopunny_main_adjustment": full,
            "lopunny_search_adjustment": full,
            "deck_guard": 4,
            "optional_thin_cards": ids(["Buddy-Buddy Poffin", "Poké Pad", "Pokégear 3.0", "Ultra Ball"]),
            "setup_active_priority": priority(["Dunsparce", "Buneary", "Fan Rotom", "Snorunt"], 1600.0, 180.0),
            "setup_bench_priority": priority(["Buneary", "Dunsparce", "Snorunt", "Fan Rotom"], 1500.0, 120.0),
            "play_priority": priority([
                "Buddy-Buddy Poffin", "Ultra Ball", "Poké Pad", "Hilda", "Battle Cage",
                "Lillie's Determination", "Wally's Compassion", "Hand Trimmer", "Boss’s Orders", "Pokégear 3.0",
            ], 1100.0, 55.0),
            "search_priority": priority([
                "Mega Lopunny ex", "Buneary", "Dudunsparce", "Dunsparce", "Mega Froslass ex",
                "Snorunt", "Fan Rotom", "Enriching Energy", "Mist Energy", "Basic {W} Energy",
                "Hilda", "Wally's Compassion", "Lillie's Determination", "Boss’s Orders",
            ], 1600.0, 65.0),
            "keep_priority": priority([
                "Mega Lopunny ex", "Buneary", "Dudunsparce", "Dunsparce", "Mega Froslass ex",
                "Enriching Energy", "Mist Energy", "Wally's Compassion", "Hilda", "Boss’s Orders",
            ], 1400.0, 70.0),
            "evolve_priority": priority(["Buneary", "Dunsparce", "Snorunt"], 1200.0, 100.0),
            "attach_priority": priority(["Mega Lopunny ex", "Mega Froslass ex", "Dunsparce", "Dudunsparce"], 1100.0, 100.0),
            "active_priority": priority(["Mega Lopunny ex", "Mega Froslass ex", "Dunsparce", "Fan Rotom", "Buneary"], 1500.0, 130.0),
            "board_priority": priority(["Mega Lopunny ex", "Dudunsparce", "Mega Froslass ex", "Buneary", "Dunsparce"], 900.0, 90.0),
            "target_priority": priority(["Dunsparce", "Buneary", "Snorunt", "Fan Rotom", "Mega Lopunny ex"], 1200.0, 100.0),
            "attack_priority": {
                str(ATTACK_IDS.get("Gale Thrust", 1225)): 1200.0,
                str(ATTACK_IDS.get("Spiky Hopper", 1226)): 900.0,
                str(ATTACK_IDS.get("Resentful Refrain", 1240)): 850.0,
                str(ATTACK_IDS.get("Absolute Snow", 1241)): 700.0,
            },
            "search": {"enabled": search_enabled, "candidates": 4, "budget_s": 0.12, "rollout_steps": 6},
        }
    raise ValueError(f"unsupported family: {family}")


def build(
    artifact_path: Path,
    out: Path,
    runtime: Path,
    cg_dir: Path,
    variant: str,
    model_weight: float | None = None,
    lopunny_profile: str = "baseline",
    linux_only: bool = True,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    family = str(artifact["family"])
    deck = [int(card_id) for card_id in artifact["deck"]]
    validate_deck_ids(deck)
    config = family_config(family, variant, model_weight, lopunny_profile)
    if config_overrides and config_overrides.get("__replace_config__"):
        config = dict(config_overrides)
        config.pop("__replace_config__", None)
    if family == "generic_replay":
        for key, value in artifact.get("generic_policy", {}).items():
            if key.endswith("_priority") and isinstance(value, dict):
                config[key] = {str(card_id): float(score) for card_id, score in value.items()}
    config["own_deck"] = deck
    if family == "teal_ogerpon" and config.get("search", {}).get("enabled"):
        config["opponent_decks"] = [
            list(map(int, deck)) for deck in artifact.get("opponent_decks", []) if len(deck) == 60
        ][:16]
    if config_overrides:
        for key, value in config_overrides.items():
            if key == "__replace_config__":
                continue
            if key == "search" and isinstance(value, dict):
                config["search"] = {**config.get("search", {}), **value}
            else:
                config[key] = value
    packaged_model = artifact.get("__packaged_model__")
    model = packaged_model if packaged_model is not None and not variant.startswith("rules") else (
        artifact.get("model", {}) if not variant.startswith("rules") else {}
    )
    if packaged_model is None and artifact.get("version", 1) >= 2 and not variant.startswith("rules"):
        model = {
            "version": int(artifact.get("version", 2)),
            "global": compact_model(artifact.get("model", {})),
            "heads": {name: compact_model(value) for name, value in artifact.get("head_models", {}).items()},
            "value": compact_model(artifact.get("value_model", {})) if variant in {
                "clone_value", "clone_search", "clone_combat_search", "clone_anti_mill_search",
            } else {},
            "confidence_thresholds": artifact.get("confidence_thresholds", {}),
            "calibration": artifact.get("calibration", {}),
            "router_calibration": artifact.get("router_calibration", {}),
            "selection_margin_thresholds": artifact.get("selection_margin_thresholds", {}),
            "residual_q": compact_model(artifact.get("residual_q_model", {})),
            "win_value": compact_model(artifact.get("win_value_model", {})),
            "advantage_ensemble": [
                compact_model(value) for value in artifact.get("advantage_ensemble", [])
            ],
            "advantage_calibration": artifact.get("advantage_calibration", {}),
        }
    elif packaged_model is None and model:
        model = compact_model(model)
    packaged_model_gzip = artifact.get("__packaged_model_gzip_b64__")
    if packaged_model_gzip is not None and not variant.startswith("rules"):
        model_bytes = base64.b64decode(packaged_model_gzip.encode("ascii"), validate=True)
    else:
        model_bytes = gzip.compress(json.dumps(model, separators=(",", ":")).encode("utf-8"), compresslevel=9)
    main_source = (
        "import gzip\n"
        "import json\n"
        "from pathlib import Path\n"
        "import meta_runtime\n"
        "from meta_runtime import run_agent\n"
        f"CONFIG = {config!r}\n"
        "BASE_DIR = Path(meta_runtime.__file__).resolve().parent\n"
        "MODEL = json.loads(gzip.open(BASE_DIR / 'model.json.gz', 'rt', encoding='utf-8').read())\n\n"
        "def agent(obs, configuration=None):\n"
        "    return run_agent(obs, CONFIG, MODEL)\n\n"
        "kaggle_agent = agent\n"
    ).encode("utf-8")
    entries: list[tuple[str, bytes]] = [
        ("main.py", main_source),
        ("meta_runtime.py", runtime.read_bytes()),
        ("deck.csv", deck_to_text(deck).encode("utf-8")),
        ("model.json.gz", model_bytes),
    ]
    for path in sorted(cg_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            if linux_only and path.name in {"cg.dll", "libcg.dylib", "libcg-arm64.so"}:
                continue
            entries.append(((Path("cg") / path.relative_to(cg_dir)).as_posix(), path.read_bytes()))
    metadata = {
        "family": family,
        "variant": variant,
        "lopunny_profile": lopunny_profile,
        "artifact": str(artifact_path),
        "validation_metrics": artifact.get("validation_metrics", {}),
        "reference_free_runtime": True,
        "linux_only": linux_only,
        "deck_sha256": hashlib.sha256("\n".join(map(str, deck)).encode("utf-8")).hexdigest(),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "artifact_version": int(artifact.get("version", 1)),
    }
    entries.append(("build_metadata.json", json.dumps(metadata, indent=2).encode("utf-8")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a replay-imitation meta submission.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "rules", "clone", "clone_fidelity", "clone_targetfix", "clone_endgame", "clone_value", "clone_search",
            "clone_combat", "clone_combat_search", "clone_anti_mill", "clone_anti_mill_search",
            "clone_residual", "clone_residual_search", "clone_residual_anti_mill", "clone_residual_anti_mill_search",
        ),
        default="clone",
    )
    parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--model-weight", type=float)
    parser.add_argument("--lopunny-profile", choices=("baseline", "attack", "cycle", "full"), default="baseline")
    parser.add_argument("--linux-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config-overrides", type=Path)
    args = parser.parse_args(argv)
    print(build(
        args.artifact,
        args.out,
        args.runtime,
        args.cg_dir,
        args.variant,
        args.model_weight,
        args.lopunny_profile,
        args.linux_only,
        json.loads(args.config_overrides.read_text(encoding="utf-8")) if args.config_overrides else None,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
