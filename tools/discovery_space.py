from __future__ import annotations

import csv
import hashlib
import json
import random
import tarfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tools.build_models import DEFAULT_BASE, BuildConfig
from tools.deck_rules import ACE_SPEC_IDS, BASIC_ENERGY_IDS, validate_deck_ids
from tools.prior_deck_space import (
    GT_PRIOR_PACKAGES,
    REFERENCE_BASES,
    SEARCH_PRIORS,
    great_tusk_prior_configs,
    metal_prior_configs,
    portfolio_seed_configs,
)


CARD_DATA = Path("pokemon-tcg-ai-battle/EN_Card_Data.csv")
FLEX_CUT_IDS = [
    1123,
    1182,
    1204,
    1213,
    1227,
    1194,
    1185,
    1122,
    1142,
]
GREAT_TUSK_CORE_IDS = {58, 344, 345, 607, 1152, 1197, 1121, 1097, 1147}


@dataclass(slots=True)
class Motif:
    name: str
    cards: list[int]
    note: str
    family_hint: str = "any"
    source: str = "manual"


def read_deck_from_tarball(path: Path, member: str = "deck.csv") -> list[int]:
    with tarfile.open(path, "r:gz") as tar:
        payload = tar.extractfile(member)
        if payload is None:
            raise ValueError(f"{path} has no {member}")
        deck = [int(line) for line in payload.read().decode("utf-8").splitlines() if line.strip()]
    validate_deck_ids(deck)
    return deck


def read_build_metadata(path: Path) -> dict[str, Any]:
    try:
        with tarfile.open(path, "r:gz") as tar:
            payload = tar.extractfile("build_metadata.json")
            if payload is None:
                return {}
            return json.loads(payload.read().decode("utf-8"))
    except Exception:
        return {}


def family_hint_from_path(path: Path) -> str:
    metadata = read_build_metadata(path)
    family = str(metadata.get("family") or "").strip()
    if family:
        return family
    lower = path.name.lower()
    for name in ("great_tusk", "lucario", "metal_tempo", "rahul_metal", "probabilistic"):
        if name in lower:
            return name
    if "steel" in lower or "metal" in lower:
        return "metal_tempo"
    return "great_tusk"


def deck_hash(deck: list[int]) -> str:
    text = ",".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deck_distance(a: list[int], b: list[int]) -> float:
    ca = Counter(a)
    cb = Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.0
    intersection = sum(min(ca[key], cb[key]) for key in keys)
    union = sum(max(ca[key], cb[key]) for key in keys)
    return 1.0 - intersection / max(1, union)


def card_index(path: Path = CARD_DATA) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            try:
                rows[int(row["Card ID"])] = row
            except Exception:
                continue
    return rows


def tag_cards(index: dict[int, dict[str, str]]) -> dict[str, list[int]]:
    tags: dict[str, list[int]] = {
        "draw": [],
        "search": [],
        "discard": [],
        "recovery": [],
        "switch": [],
        "gust": [],
        "energy": [],
        "tool": [],
        "stadium": [],
        "supporter": [],
        "item": [],
    }
    for card_id, row in index.items():
        stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").lower()
        category = row.get("Category", "").lower()
        name = row.get("Card Name", "").lower()
        text = f"{name} {stage} {category} {row.get('Effect Explanation', '').lower()}"
        if "draw" in text:
            tags["draw"].append(card_id)
        if "search your deck" in text or "look at the top" in text:
            tags["search"].append(card_id)
        if "discard" in text:
            tags["discard"].append(card_id)
        if "from your discard" in text or "put up to" in text and "discard pile" in text:
            tags["recovery"].append(card_id)
        if "switch" in text or "retreat" in text:
            tags["switch"].append(card_id)
        if "bench" in text and ("active" in text or "switch" in text):
            tags["gust"].append(card_id)
        if "energy" in text:
            tags["energy"].append(card_id)
        if "tool" in stage or "tool" in category:
            tags["tool"].append(card_id)
        if "stadium" in stage or "stadium" in category:
            tags["stadium"].append(card_id)
        if "supporter" in stage or "supporter" in category:
            tags["supporter"].append(card_id)
        if "item" in stage or "item" in category:
            tags["item"].append(card_id)
    return {key: sorted(set(values)) for key, values in tags.items()}


def manual_motifs() -> list[Motif]:
    motifs = [
        Motif("gt_loop_consistency", [1152, 1097, 1121], "Great Tusk loop: Pal Pad, Night Stretcher, Ultra Ball.", "great_tusk"),
        Motif("gt_hand_denial", [1087, 1186, 1197], "Hand disruption and Ancient supporter density.", "great_tusk"),
        Motif("gt_energy_denial", [1081, 1139, 1149], "Energy denial plus recursion.", "great_tusk"),
        Motif("gt_trap_stall", [1166, 1161, 1087], "Trap active, buy deckout turns.", "great_tusk"),
        Motif("gt_defensive_tools", [1177, 1174, 1147], "Defensive tools and healing pressure.", "great_tusk"),
        Motif("metal_boss_recovery", [1182, 1097, 8], "Metal pressure: Boss, recovery, energy.", "metal"),
        Motif("lucario_pressure", [1182, 1123, 6], "Lucario pressure and mobility package.", "lucario"),
    ]
    for name, swaps, note in GT_PRIOR_PACKAGES:
        motifs.append(Motif(f"prior_{name}", [add for add, _ in swaps], note, "great_tusk", "prior_deck_space"))
    return motifs


LOSS_DIGEST_MOTIFS = {
    "draw_recovery": ([1123, 1227, 1097], "Loss digest: restore draw and discard recursion after hand/deck collapse."),
    "hand_disruption": ([1087, 1186, 1197], "Loss digest: pressure opponent hand after hand surge or setup failure."),
    "defensive_tools": ([1177, 1174, 1147], "Loss digest: add defensive tools and healing after prize-race collapse."),
    "switch_pivot": ([1160, 1203, 1161], "Loss digest: add pivot protection after active disruption."),
    "resource_denial": ([1081, 1139, 1149], "Loss digest: deny opponent recovery and energy loops."),
    "resource_safety": ([1152, 1097, 1121], "Loss digest: protect recursion and repeated Land Collapse turns."),
}


def read_feedback(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def motifs_from_loss_digest(path: Path | None) -> list[Motif]:
    if path is None or not path.exists():
        return []
    try:
        digest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    motifs: list[Motif] = []
    for name in digest.get("recommended_motifs", []):
        spec = LOSS_DIGEST_MOTIFS.get(str(name))
        if not spec:
            continue
        cards, note = spec
        motifs.append(Motif(str(name), list(cards), note, "great_tusk", f"loss_digest:{name}"))
    return motifs


def motifs_from_feedback(path: Path | None) -> list[Motif]:
    feedback = read_feedback(path)
    motifs: list[Motif] = []
    seen: set[tuple[str, str]] = set()
    for item in feedback.get("pressure_items", []):
        if not isinstance(item, dict):
            continue
        pressure_id = str(item.get("pressure_id") or item.get("kind") or "pressure")
        family_hint = str(item.get("target_family") or "great_tusk")
        if family_hint == "unknown":
            family_hint = "great_tusk"
        for name in item.get("recommended_motifs", []):
            spec = LOSS_DIGEST_MOTIFS.get(str(name))
            if not spec:
                continue
            key = (pressure_id, str(name))
            if key in seen:
                continue
            seen.add(key)
            cards, note = spec
            motifs.append(
                Motif(
                    f"pressure_{str(name)}_{len(motifs)}",
                    list(cards),
                    f"Pressure {pressure_id}: {note}",
                    family_hint,
                    f"pressure:{pressure_id}",
                )
            )
    for name in feedback.get("recommended_motifs", []):
        spec = LOSS_DIGEST_MOTIFS.get(str(name))
        if not spec:
            continue
        key = ("summary", str(name))
        if key in seen:
            continue
        seen.add(key)
        cards, note = spec
        motifs.append(Motif(f"pressure_summary_{name}", list(cards), f"Feedback summary: {note}", "great_tusk", "pressure:summary"))
    return motifs


def motifs_from_decks(paths: list[Path], min_support: int = 2, max_motifs: int = 32) -> list[Motif]:
    baskets: list[set[int]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            baskets.append(set(read_deck_from_tarball(path)))
        except Exception:
            continue
    pair_counts: Counter[tuple[int, int]] = Counter()
    triple_counts: Counter[tuple[int, int, int]] = Counter()
    for basket in baskets:
        cards = sorted(card_id for card_id in basket if card_id not in BASIC_ENERGY_IDS)
        for i, a in enumerate(cards):
            for b in cards[i + 1 :]:
                pair_counts[(a, b)] += 1
        top_cards = cards[:80]
        for i, a in enumerate(top_cards):
            for j, b in enumerate(top_cards[i + 1 :], i + 1):
                for c in top_cards[j + 1 :]:
                    triple_counts[(a, b, c)] += 1
    motifs: list[Motif] = []
    for cards, support in triple_counts.most_common():
        if support < min_support:
            continue
        motifs.append(Motif(f"mined_{'_'.join(map(str, cards))}", list(cards), f"Mined triple support={support}.", source="hof"))
        if len(motifs) >= max_motifs:
            break
    if len(motifs) < max_motifs:
        for cards, support in pair_counts.most_common():
            if support < min_support:
                continue
            motifs.append(Motif(f"mined_{cards[0]}_{cards[1]}", list(cards), f"Mined pair support={support}.", source="hof"))
            if len(motifs) >= max_motifs:
                break
    return motifs


def _ace_conflicts(add_id: int, counts: Counter[int]) -> list[int]:
    if add_id not in ACE_SPEC_IDS:
        return []
    return [card_id for card_id, count in counts.items() if count > 0 and card_id in ACE_SPEC_IDS and card_id != add_id]


def choose_cut(counts: Counter[int], protected: set[int], rng: random.Random) -> int | None:
    for card_id in FLEX_CUT_IDS:
        if counts[card_id] > 0 and card_id not in protected:
            return card_id
    duplicated = [
        card_id
        for card_id, count in counts.items()
        if count > 1 and card_id not in protected and card_id not in BASIC_ENERGY_IDS
    ]
    if duplicated:
        duplicated.sort(key=lambda cid: (counts[cid], cid), reverse=True)
        return duplicated[0]
    basics = [card_id for card_id in BASIC_ENERGY_IDS if counts[card_id] > 0 and card_id not in protected]
    return rng.choice(basics) if basics else None


def insert_motif(deck: list[int], motif: Motif, rng: random.Random, protected: set[int] | None = None) -> list[int]:
    counts = Counter(deck)
    keep = set(protected or set())
    for add_id in motif.cards:
        if add_id not in BASIC_ENERGY_IDS and counts[add_id] >= 4:
            continue
        for conflict in _ace_conflicts(add_id, counts):
            if counts[conflict] > 0 and conflict not in keep:
                counts[conflict] -= 1
                break
        cut = choose_cut(counts, keep | set(motif.cards), rng)
        if cut is None:
            continue
        counts[cut] -= 1
        counts[add_id] += 1
    out: list[int] = []
    for card_id, count in counts.items():
        out.extend([card_id] * count)
    validate_deck_ids(out)
    return out


def strategy_variants(incumbent: BuildConfig, generation: int, out_dir: Path) -> list[BuildConfig]:
    prefix = f"d{generation:03d}"
    variants = [
        ("mill_extreme", {"opp_mill": 1250.0, "opp_deckout_bonus": 24000.0, "self_deckout_penalty": 15000.0}, "Push pure deckout valuation."),
        ("resource_safe", {"self_mill_penalty": 500.0, "deck_delta": 80.0, "hand_delta": 80.0}, "Prefer resource safety when local gains are noisy."),
        ("attack_ready", {"great_tusk_ready": 9000.0, "supporter_played_attack": 7000.0, "explorer_ready": 18000.0}, "Bias toward active Great Tusk attack turns."),
        ("prize_respect", {"prize_delta": 5200.0, "great_tusk_active": 1100.0}, "Punish prize-race failures more heavily."),
        ("late_override", {"opp_deckout_bonus": 30000.0, "opp_mill": 1050.0}, "Pair low search margin with near-terminal deckout bias."),
    ]
    configs: list[BuildConfig] = []
    for name, weights, note in variants:
        merged = dict(incumbent.strategy_weights)
        merged.update(weights)
        configs.append(
            replace(
                incumbent,
                name=f"{prefix}_strategy_{name}",
                out=out_dir / f"{prefix}_strategy_{name}.tar.gz",
                strategy_weights=merged,
                policy_variant=name,
                origin="strategy_variant",
                notes=f"{incumbent.notes} | {note}",
            )
        )
    for name, cand, budget, margin, rollout, note in SEARCH_PRIORS:
        cfg = replace(
            incumbent,
            name=f"{prefix}_searchshape_{name}",
            out=out_dir / f"{prefix}_searchshape_{name}.tar.gz",
            search_candidates=cand,
            search_budget_s=budget,
            search_margin=margin,
            search_rollout_steps=rollout,
            origin="search_shape",
            notes=f"{incumbent.notes} | {note}",
        )
        if name == "late_override":
            cfg.strategy_weights = {"opp_deckout_bonus": 30000.0, "opp_mill": 1050.0}
            cfg.policy_variant = "late_override"
        configs.append(cfg)
    return configs


def reference_seed_configs(out_dir: Path, generation: int) -> list[BuildConfig]:
    configs: list[BuildConfig] = []
    for family, base in REFERENCE_BASES.items():
        if not base.exists():
            continue
        configs.append(
            BuildConfig(
                name=f"d{generation:03d}_seed_{family}",
                family=family,
                base=base,
                out=out_dir / f"d{generation:03d}_seed_{family}.tar.gz",
                injection="great_tusk" if family == "great_tusk" else "none",
                enable_search=family == "great_tusk",
                deck_files=("deck.csv", "lucario_deck.csv") if family == "lucario" else ("deck.csv",),
                origin="reference_seed",
                notes=f"Reference seed copied from {base.name}.",
            )
        )
    return configs


def motif_configs(
    incumbent: BuildConfig,
    generation: int,
    out_dir: Path,
    motifs: list[Motif],
    seed_decks: list[tuple[str, str, Path, list[int]]],
    limit: int,
    rng: random.Random,
) -> list[BuildConfig]:
    configs: list[BuildConfig] = []
    protected = GREAT_TUSK_CORE_IDS if incumbent.family == "great_tusk" else set()
    motif_order = list(motifs)
    rng.shuffle(motif_order)
    bases = seed_decks[:]
    rng.shuffle(bases)
    for motif in motif_order:
        for family, label, base_path, deck in bases:
            if motif.family_hint not in {"any", family} and not (motif.family_hint == "metal" and "metal" in family):
                continue
            try:
                new_deck = insert_motif(deck, motif, rng, protected if family == "great_tusk" else set())
            except Exception:
                continue
            if deck_hash(new_deck) == deck_hash(deck):
                continue
            name = f"d{generation:03d}_motif_{label}_{motif.name}"[:90]
            configs.append(
                BuildConfig(
                    name=name,
                    family=family,
                    base=base_path,
                    out=out_dir / f"{name}.tar.gz",
                    injection="great_tusk" if family == "great_tusk" else "none",
                    enable_search=family == "great_tusk",
                    deck_override=new_deck,
                    deck_files=("deck.csv", "lucario_deck.csv") if family == "lucario" else ("deck.csv",),
                    origin=f"motif:{motif.source}",
                    notes=motif.note,
                )
            )
            if len(configs) >= limit:
                return configs
    return configs


def tag_sweep_configs(incumbent: BuildConfig, generation: int, out_dir: Path, limit: int, rng: random.Random) -> list[BuildConfig]:
    try:
        deck = read_deck_from_tarball(incumbent.base)
    except Exception:
        return []
    tags = tag_cards(card_index())
    chosen: list[tuple[str, int]] = []
    for tag in ("search", "draw", "recovery", "switch", "gust", "energy", "tool", "stadium"):
        cards = [card_id for card_id in tags.get(tag, []) if 1000 <= card_id <= 1300]
        rng.shuffle(cards)
        for card_id in cards[:4]:
            chosen.append((tag, card_id))
    rng.shuffle(chosen)
    configs: list[BuildConfig] = []
    for tag, card_id in chosen:
        if len(configs) >= limit:
            break
        motif = Motif(f"tag_{tag}_{card_id}", [card_id], f"Tag sweep from card data: {tag}.", "great_tusk", "card_data")
        try:
            new_deck = insert_motif(deck, motif, rng, GREAT_TUSK_CORE_IDS)
        except Exception:
            continue
        name = f"d{generation:03d}_tagsweep_{tag}_{card_id}"
        configs.append(
            replace(
                incumbent,
                name=name,
                out=out_dir / f"{name}.tar.gz",
                deck_override=new_deck,
                origin="tag_sweep",
                notes=motif.note,
            )
        )
    return configs


def tag_combo_configs(incumbent: BuildConfig, generation: int, out_dir: Path, limit: int, rng: random.Random) -> list[BuildConfig]:
    if limit <= 0:
        return []
    try:
        deck = read_deck_from_tarball(incumbent.base)
    except Exception:
        return []
    tags = tag_cards(card_index())
    tag_pairs = [
        ("search_draw", ("search", "draw")),
        ("search_recovery", ("search", "recovery")),
        ("draw_recovery", ("draw", "recovery")),
        ("switch_defense", ("switch", "tool")),
        ("gust_denial", ("gust", "energy")),
        ("stadium_resource", ("stadium", "recovery")),
        ("supporter_item", ("supporter", "item")),
        ("hand_energy", ("supporter", "energy")),
    ]
    candidates: list[tuple[str, list[int]]] = []
    for combo_name, combo_tags in tag_pairs:
        pools: list[list[int]] = []
        for tag in combo_tags:
            cards = [card_id for card_id in tags.get(tag, []) if 1000 <= card_id <= 1300]
            rng.shuffle(cards)
            pools.append(cards[:10])
        if len(pools) < 2 or not all(pools):
            continue
        for a in pools[0]:
            for b in pools[1]:
                if a == b:
                    continue
                candidates.append((combo_name, [a, b]))
    rng.shuffle(candidates)
    configs: list[BuildConfig] = []
    for index, (combo_name, cards) in enumerate(candidates):
        if len(configs) >= limit:
            break
        motif = Motif(
            f"tagcombo_{combo_name}_{'_'.join(map(str, cards))}",
            cards,
            f"Exploration tag combo from card data: {combo_name}.",
            "great_tusk",
            "card_data_combo",
        )
        try:
            new_deck = insert_motif(deck, motif, rng, GREAT_TUSK_CORE_IDS)
        except Exception:
            continue
        if deck_hash(new_deck) == deck_hash(deck):
            continue
        name = f"d{generation:03d}_tagcombo_{index:03d}_{combo_name}"
        configs.append(
            replace(
                incumbent,
                name=name,
                out=out_dir / f"{name}.tar.gz",
                deck_override=new_deck,
                origin="tag_combo",
                notes=motif.note,
            )
        )
    return configs


def motif_combo_configs(
    incumbent: BuildConfig,
    generation: int,
    out_dir: Path,
    motifs: list[Motif],
    seed_decks: list[tuple[str, str, Path, list[int]]],
    limit: int,
    rng: random.Random,
) -> list[BuildConfig]:
    if limit <= 0:
        return []
    motif_order = list(motifs)
    rng.shuffle(motif_order)
    bases = seed_decks[:]
    rng.shuffle(bases)
    pairs: list[tuple[Motif, Motif]] = []
    for i, first in enumerate(motif_order):
        for second in motif_order[i + 1 :]:
            if first.name == second.name:
                continue
            pairs.append((first, second))
    rng.shuffle(pairs)
    configs: list[BuildConfig] = []
    for index, (first, second) in enumerate(pairs):
        if len(configs) >= limit:
            break
        for family, label, base_path, deck in bases:
            if len(configs) >= limit:
                break
            if first.family_hint not in {"any", family} and not (first.family_hint == "metal" and "metal" in family):
                continue
            if second.family_hint not in {"any", family} and not (second.family_hint == "metal" and "metal" in family):
                continue
            protected = GREAT_TUSK_CORE_IDS if family == "great_tusk" else set()
            try:
                new_deck = insert_motif(deck, first, rng, protected)
                new_deck = insert_motif(new_deck, second, rng, protected | set(first.cards))
            except Exception:
                continue
            if deck_hash(new_deck) == deck_hash(deck):
                continue
            safe_first = "".join(ch if ch.isalnum() else "_" for ch in first.name)[-24:]
            safe_second = "".join(ch if ch.isalnum() else "_" for ch in second.name)[-24:]
            name = f"d{generation:03d}_motifcombo_{label}_{index:03d}_{safe_first}_{safe_second}"[:90]
            configs.append(
                BuildConfig(
                    name=name,
                    family=family,
                    base=base_path,
                    out=out_dir / f"{name}.tar.gz",
                    injection="great_tusk" if family == "great_tusk" else "none",
                    enable_search=family == "great_tusk",
                    deck_override=new_deck,
                    deck_files=("deck.csv", "lucario_deck.csv") if family == "lucario" else ("deck.csv",),
                    origin=f"motif_combo:{first.source}+{second.source}",
                    notes=f"Combined motifs: {first.note} / {second.note}",
                )
            )
    return configs


def search_strategy_matrix_configs(incumbent: BuildConfig, generation: int, out_dir: Path, limit: int) -> list[BuildConfig]:
    if limit <= 0:
        return []
    strategy_specs = [
        ("mill_extreme", {"opp_mill": 1250.0, "opp_deckout_bonus": 24000.0, "self_deckout_penalty": 15000.0}),
        ("resource_safe", {"self_mill_penalty": 500.0, "deck_delta": 80.0, "hand_delta": 80.0}),
        ("attack_ready", {"great_tusk_ready": 9000.0, "supporter_played_attack": 7000.0, "explorer_ready": 18000.0}),
        ("prize_respect", {"prize_delta": 5200.0, "great_tusk_active": 1100.0}),
        ("late_override", {"opp_deckout_bonus": 30000.0, "opp_mill": 1050.0}),
        ("anti_setup", {"hand_delta": 110.0, "opp_mill": 1080.0, "prize_delta": 4300.0}),
    ]
    configs: list[BuildConfig] = []
    for strat_name, weights in strategy_specs:
        for search_name, cand, budget, margin, rollout, note in SEARCH_PRIORS:
            if len(configs) >= limit:
                return configs
            merged = dict(incumbent.strategy_weights)
            merged.update(weights)
            name = f"d{generation:03d}_matrix_{strat_name}_{search_name}"
            configs.append(
                replace(
                    incumbent,
                    name=name,
                    out=out_dir / f"{name}.tar.gz",
                    search_candidates=cand,
                    search_budget_s=budget,
                    search_margin=margin,
                    search_rollout_steps=rollout,
                    strategy_weights=merged,
                    policy_variant=f"matrix_{strat_name}",
                    origin="search_strategy_matrix",
                    notes=f"Exploration matrix: {strat_name} with {search_name}. {note}",
                )
            )
    return configs


def opponent_search_matrix_configs(incumbent: BuildConfig, generation: int, out_dir: Path, limit: int) -> list[BuildConfig]:
    if limit <= 0:
        return []
    model_weights = [
        ("noisy", {"hand_delta": 70.0}),
        ("aggro_bias", {"prize_delta": 5700.0, "great_tusk_ready": 7600.0}),
        ("stall_bias", {"opp_mill": 1180.0, "opp_deckout_bonus": 24500.0}),
    ]
    configs: list[BuildConfig] = []
    for model, weights in model_weights:
        for search_name, cand, budget, margin, rollout, note in SEARCH_PRIORS:
            if len(configs) >= limit:
                return configs
            merged = dict(incumbent.strategy_weights)
            merged.update(weights)
            name = f"d{generation:03d}_oppmatrix_{model}_{search_name}"
            configs.append(
                replace(
                    incumbent,
                    name=name,
                    out=out_dir / f"{name}.tar.gz",
                    search_candidates=cand,
                    search_budget_s=budget,
                    search_margin=margin,
                    search_rollout_steps=rollout,
                    strategy_weights=merged,
                    policy_variant=f"opponent_model_{model}",
                    opponent_model=model,
                    origin="opponent_search_matrix",
                    notes=f"Opponent-model exploration: {model} with {search_name}. {note}",
                )
            )
    return configs


def pressure_strategy_configs(incumbent: BuildConfig, generation: int, out_dir: Path, feedback_path: Path | None, limit: int) -> list[BuildConfig]:
    feedback = read_feedback(feedback_path)
    configs: list[BuildConfig] = []
    for index, item in enumerate(feedback.get("pressure_items", [])):
        if len(configs) >= limit:
            break
        if not isinstance(item, dict):
            continue
        shifts = {str(k): float(v) for k, v in dict(item.get("strategy_shifts") or {}).items()}
        if not shifts:
            continue
        merged = dict(incumbent.strategy_weights)
        merged.update(shifts)
        pressure_id = str(item.get("pressure_id") or f"pressure_{index}")
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in pressure_id)[-48:]
        name = f"d{generation:03d}_pressure_strategy_{index}_{safe_id}"[:90]
        configs.append(
            replace(
                incumbent,
                name=name,
                out=out_dir / f"{name}.tar.gz",
                strategy_weights=merged,
                policy_variant=f"pressure_{str(item.get('kind') or 'shift')}"[:64],
                origin=f"pressure:{pressure_id}",
                notes=f"Pressure-driven strategy shift from {pressure_id}: {item.get('evidence', {})}",
            )
        )
    return configs


def pressure_opponent_model_configs(incumbent: BuildConfig, generation: int, out_dir: Path, feedback_path: Path | None, limit: int) -> list[BuildConfig]:
    feedback = read_feedback(feedback_path)
    if not feedback or limit <= 0:
        return []
    models = [
        ("noisy", {"hand_delta": 70.0}, "Rollout assumes occasional opponent mistakes."),
        ("aggro_bias", {"prize_delta": 5700.0, "great_tusk_ready": 7600.0}, "Stress test prize-race pressure."),
        ("stall_bias", {"opp_mill": 1180.0, "opp_deckout_bonus": 24500.0}, "Stress test stall/deckout pressure."),
    ]
    configs: list[BuildConfig] = []
    for index, (model, shifts, note) in enumerate(models[:limit]):
        merged = dict(incumbent.strategy_weights)
        merged.update(shifts)
        name = f"d{generation:03d}_pressure_oppmodel_{model}"
        configs.append(
            replace(
                incumbent,
                name=name,
                out=out_dir / f"{name}.tar.gz",
                strategy_weights=merged,
                policy_variant=f"opponent_model_{model}",
                opponent_model=model,
                origin="pressure:opponent_model",
                notes=note,
            )
        )
    return configs


def cfg_deck_signature(cfg: BuildConfig) -> list[int] | None:
    if cfg.deck_override:
        return [int(card_id) for card_id in cfg.deck_override]
    try:
        deck = read_deck_from_tarball(cfg.base)
    except Exception:
        return None
    if cfg.deck_swaps:
        counts = Counter(deck)
        for add_id, cut_id in cfg.deck_swaps:
            if counts[int(cut_id)] > 0:
                counts[int(cut_id)] -= 1
                counts[int(add_id)] += 1
        out: list[int] = []
        for card_id, count in counts.items():
            out.extend([card_id] * count)
        return out
    return deck


def same_strategy_shape(a: BuildConfig, b: BuildConfig) -> bool:
    return (
        a.strategy_weights == b.strategy_weights
        and a.policy_variant == b.policy_variant
        and a.opponent_model == b.opponent_model
        and a.injection == b.injection
        and a.search_candidates == b.search_candidates
        and a.search_budget_s == b.search_budget_s
        and a.search_margin == b.search_margin
        and a.search_rollout_steps == b.search_rollout_steps
    )


def incumbent_from_tarball(path: Path, out_dir: Path) -> BuildConfig:
    metadata = read_build_metadata(path)
    cfg = BuildConfig(
        name="incumbent",
        family=str(metadata.get("family", "great_tusk")),
        base=Path(metadata.get("base") or path),
        out=out_dir / "incumbent.tar.gz",
        enable_search=bool(metadata.get("enable_search", True)),
        injection=str(metadata.get("injection", "great_tusk")),
        search_candidates=int(metadata.get("search_candidates", 8)),
        search_budget_s=float(metadata.get("search_budget_s", 0.25)),
        search_margin=float(metadata.get("search_margin", 1200.0)),
        search_rollout_steps=int(metadata.get("search_rollout_steps", 16)),
        deck_swaps=[tuple(map(int, pair)) for pair in metadata.get("deck_swaps", [])],
        deck_override=metadata.get("deck_override"),
        deck_files=tuple(metadata.get("deck_files", ("deck.csv",))),
        strategy_weights={str(k): float(v) for k, v in dict(metadata.get("strategy_weights", {})).items()},
        policy_variant=str(metadata.get("policy_variant", "default")),
        opponent_model=str(metadata.get("opponent_model", "perfect")),
        origin=str(metadata.get("origin", "incumbent_tarball")),
        notes=str(metadata.get("notes", "")),
    )
    if not metadata:
        cfg.base = path
        cfg.deck_swaps = []
        cfg.notes = "Incumbent imported directly from tarball without build metadata."
    return cfg


def seed_decks_from_paths(paths: list[Path]) -> list[tuple[str, str, Path, list[int]]]:
    out: list[tuple[str, str, Path, list[int]]] = []
    for family, base in REFERENCE_BASES.items():
        if base.exists():
            try:
                out.append((family, family, base, read_deck_from_tarball(base)))
            except Exception:
                pass
    for path in paths:
        if not path.exists():
            continue
        label = path.name.removesuffix(".tar.gz").replace(".", "_")
        try:
            out.append((family_hint_from_path(path), label, path, read_deck_from_tarball(path)))
        except Exception:
            continue
    seen: set[str] = set()
    unique: list[tuple[str, str, Path, list[int]]] = []
    for row in out:
        digest = deck_hash(row[3])
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(row)
    return unique


def generate_discovery_configs(
    incumbent_tarball: Path,
    out_dir: Path,
    population: int,
    generation: int,
    seed: int,
    hof_paths: list[Path] | None = None,
    include_portfolio: bool = True,
    loss_digest_path: Path | None = None,
    feedback_path: Path | None = None,
    target_paths: list[Path] | None = None,
    pressure_only: bool = False,
    min_diversity_distance: float = 0.0,
) -> list[BuildConfig]:
    rng = random.Random(seed + generation * 1009)
    out_dir.mkdir(parents=True, exist_ok=True)
    hof_paths = [path for path in (hof_paths or []) if path.exists()]
    target_paths = [path for path in (target_paths or []) if path.exists()]
    incumbent = incumbent_from_tarball(incumbent_tarball, out_dir)
    incumbent = replace(incumbent, name=f"d{generation:03d}_incumbent", out=out_dir / f"d{generation:03d}_incumbent.tar.gz")

    seed_decks = seed_decks_from_paths([incumbent_tarball, *hof_paths, *target_paths])
    configs: list[BuildConfig] = [incumbent]
    pressure_motifs = motifs_from_feedback(feedback_path)
    has_feedback = bool(pressure_motifs or read_feedback(feedback_path).get("pressure_items"))
    pressure_budget = max(0, int(population * 0.70)) if has_feedback else 0
    if pressure_budget:
        configs.extend(motif_configs(incumbent, generation, out_dir, pressure_motifs, seed_decks, max(4, pressure_budget // 2), rng))
        configs.extend(motif_combo_configs(incumbent, generation, out_dir, pressure_motifs, seed_decks, max(4, pressure_budget // 5), rng))
        configs.extend(pressure_strategy_configs(incumbent, generation, out_dir, feedback_path, max(3, pressure_budget // 4)))
        configs.extend(pressure_opponent_model_configs(incumbent, generation, out_dir, feedback_path, max(1, pressure_budget // 12)))
        configs.extend(opponent_search_matrix_configs(incumbent, generation, out_dir, max(6, pressure_budget // 8)))
    if pressure_only and has_feedback:
        configs.extend(tag_combo_configs(incumbent, generation, out_dir, max(8, population // 3), rng))
        configs.extend(search_strategy_matrix_configs(incumbent, generation, out_dir, max(12, population // 2)))
        configs.extend(opponent_search_matrix_configs(incumbent, generation, out_dir, max(8, population // 4)))
    if not pressure_only:
        motifs = [*motifs_from_loss_digest(loss_digest_path), *manual_motifs(), *motifs_from_decks([incumbent_tarball, *hof_paths, *target_paths])]
        motif_limit = max(8, population // 3)
        configs.extend(reference_seed_configs(out_dir, generation))
        configs.extend(motif_configs(incumbent, generation, out_dir, motifs, seed_decks, motif_limit, rng))
        configs.extend(motif_combo_configs(incumbent, generation, out_dir, motifs, seed_decks, max(8, population // 4), rng))
        configs.extend(tag_sweep_configs(incumbent, generation, out_dir, max(4, population // 8), rng))
        configs.extend(tag_combo_configs(incumbent, generation, out_dir, max(8, population // 3), rng))
        configs.extend(strategy_variants(incumbent, generation, out_dir))
        configs.extend(search_strategy_matrix_configs(incumbent, generation, out_dir, max(12, population // 4)))
        configs.extend(opponent_search_matrix_configs(incumbent, generation, out_dir, max(8, population // 8)))
        configs.extend(great_tusk_prior_configs(incumbent, generation, out_dir, max(12, population // 4)))
        if include_portfolio:
            configs.extend(metal_prior_configs(generation, out_dir, max(8, population // 8)))
            configs.extend(portfolio_seed_configs(generation, out_dir))

    unique: list[BuildConfig] = []
    seen_name: set[str] = set()
    seen_shape: set[str] = set()
    deck_signatures: list[tuple[BuildConfig, list[int]]] = []
    diversity_deferred: list[tuple[BuildConfig, str, list[int] | None]] = []
    for cfg in configs:
        if cfg.name in seen_name:
            continue
        shape = json.dumps(
            {
                "base": str(cfg.base),
                "swaps": cfg.deck_swaps,
                "override_hash": deck_hash(cfg.deck_override) if cfg.deck_override else "",
                "strategy": cfg.strategy_weights,
                "search": [cfg.search_candidates, cfg.search_budget_s, cfg.search_margin, cfg.search_rollout_steps],
                "injection": cfg.injection,
                "policy_variant": cfg.policy_variant,
                "opponent_model": cfg.opponent_model,
            },
            sort_keys=True,
        )
        if shape in seen_shape:
            continue
        signature = cfg_deck_signature(cfg) if min_diversity_distance > 0 else None
        if signature is not None:
            too_close = False
            for prior_cfg, prior_signature in deck_signatures:
                if same_strategy_shape(cfg, prior_cfg) and deck_distance(signature, prior_signature) < min_diversity_distance:
                    too_close = True
                    break
            if too_close:
                diversity_deferred.append((cfg, shape, signature))
                continue
        seen_name.add(cfg.name)
        seen_shape.add(shape)
        cfg.origin = cfg.origin or "discovery_space"
        unique.append(cfg)
        if signature is not None:
            deck_signatures.append((cfg, signature))
        if len(unique) >= population:
            break
    for cfg, shape, signature in diversity_deferred:
        if len(unique) >= population:
            break
        if cfg.name in seen_name or shape in seen_shape:
            continue
        seen_name.add(cfg.name)
        seen_shape.add(shape)
        cfg.origin = cfg.origin or "discovery_space"
        unique.append(cfg)
        if signature is not None:
            deck_signatures.append((cfg, signature))
    return unique
