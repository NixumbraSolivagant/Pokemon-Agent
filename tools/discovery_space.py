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

from tools.build_submission import ACE_SPEC_IDS, BASIC_ENERGY_IDS, BuildConfig, DEFAULT_BASE, validate_deck_ids
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


def deck_hash(deck: list[int]) -> str:
    text = ",".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
            out.append(("great_tusk", label, path, read_deck_from_tarball(path)))
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
) -> list[BuildConfig]:
    rng = random.Random(seed + generation * 1009)
    out_dir.mkdir(parents=True, exist_ok=True)
    hof_paths = [path for path in (hof_paths or []) if path.exists()]
    incumbent = incumbent_from_tarball(incumbent_tarball, out_dir)
    incumbent = replace(incumbent, name=f"d{generation:03d}_incumbent", out=out_dir / f"d{generation:03d}_incumbent.tar.gz")

    seed_decks = seed_decks_from_paths([incumbent_tarball, *hof_paths])
    motifs = [*manual_motifs(), *motifs_from_decks([incumbent_tarball, *hof_paths])]
    motif_limit = max(8, population // 3)
    configs: list[BuildConfig] = [incumbent]
    configs.extend(reference_seed_configs(out_dir, generation))
    configs.extend(motif_configs(incumbent, generation, out_dir, motifs, seed_decks, motif_limit, rng))
    configs.extend(tag_sweep_configs(incumbent, generation, out_dir, max(4, population // 8), rng))
    configs.extend(strategy_variants(incumbent, generation, out_dir))
    configs.extend(great_tusk_prior_configs(incumbent, generation, out_dir, max(12, population // 4)))
    if include_portfolio:
        configs.extend(metal_prior_configs(generation, out_dir, max(8, population // 8)))
        configs.extend(portfolio_seed_configs(generation, out_dir))

    unique: list[BuildConfig] = []
    seen_name: set[str] = set()
    seen_shape: set[str] = set()
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
            },
            sort_keys=True,
        )
        if shape in seen_shape:
            continue
        seen_name.add(cfg.name)
        seen_shape.add(shape)
        cfg.origin = cfg.origin or "discovery_space"
        unique.append(cfg)
        if len(unique) >= population:
            break
    return unique
