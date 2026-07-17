from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tools.build_submission import BuildConfig, DEFAULT_BASE


REFERENCE_BASES = {
    "great_tusk": Path("outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
    "metal_tempo": Path("outputs/reference_submissions/pokemon-steel.tar.gz"),
    "rahul_metal": Path("outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz"),
    "lucario": Path("outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz"),
    "probabilistic": Path("outputs/reference_submissions/improved-probabilistic-agent.tar.gz"),
}


# Card IDs already observed in the strongest Great Tusk reference or in nearby public
# submissions. These are grouped by the job they perform in a TCG deck, not by name.
GT_PRIOR_PACKAGES: list[tuple[str, list[tuple[int, int]], str]] = [
    (
        "consistency_engine",
        [(1121, 1123), (1097, 1204), (1147, 1182)],
        "Human prior: increase search/recovery density, trim switch/boss/lisia.",
    ),
    (
        "hand_disruption",
        [(1087, 1182), (1186, 1204), (1197, 1123)],
        "Human prior: convert flex slots into hand disruption and item denial.",
    ),
    (
        "energy_denial",
        [(1081, 1123), (1149, 1182), (1139, 1204)],
        "Human prior: punish energy-heavy attackers while preserving recovery.",
    ),
    (
        "stall_loop",
        [(1161, 1182), (1166, 1204), (1097, 1123)],
        "Human prior: buy turns with tools and recovery loops.",
    ),
    (
        "supporter_density",
        [(1186, 1182), (1194, 1123), (1197, 1204)],
        "Human prior: more Ancient/supporter access for repeated Land Collapse turns.",
    ),
    (
        "wall_recovery",
        [(1147, 1182), (1097, 1204), (1129, 1123)],
        "Human prior: make Crustle/Great Tusk loops harder to exhaust.",
    ),
    (
        "anti_lucario",
        [(1161, 1182), (1081, 1123), (1147, 1204)],
        "Human prior: slow attacker setup and keep wall pieces alive.",
    ),
    (
        "anti_metal",
        [(1081, 1182), (1087, 1204), (1149, 1123)],
        "Human prior: attack metal tempo via energy and hand pressure.",
    ),
    (
        "supporter_loop",
        [(1152, 1123), (1097, 1204), (1129, 1182)],
        "Player prior: Pal Pad/Night Stretcher/Sacred Ash keep the Ancient Supporter mill loop online.",
    ),
    (
        "hard_mill_no_gust",
        [(1152, 1182), (1097, 1182), (1139, 1204)],
        "Player prior: trim gust effects for pure Great Tusk loop consistency and resource recursion.",
    ),
    (
        "setup_plus_loop",
        [(1121, 1123), (1152, 1204), (1097, 1182)],
        "Player prior: improve early setup while preserving late supporter recovery.",
    ),
    (
        "defensive_toolbox",
        [(1177, 1182), (1174, 1204), (1147, 1123)],
        "Player prior: use non-ACE defensive and pivot tools to buy extra Great Tusk turns.",
    ),
    (
        "trap_and_stall",
        [(1166, 1204), (1161, 1123), (1087, 1182)],
        "Player prior: trap low-energy active Pokemon, reduce hand size, and force inefficient attacks.",
    ),
    (
        "anti_setup_handlock",
        [(1186, 1182), (1087, 1204), (1213, 1123)],
        "Player prior: disrupt setup decks with Eri, Hand Trimmer, and Judge pressure.",
    ),
    (
        "energy_resource_loop",
        [(1139, 1204), (1097, 1123), (1149, 1182)],
        "Player prior: recover energy and deny opponent attachments in longer games.",
    ),
    (
        "prize_race_pivot",
        [(1174, 1123), (1177, 1182), (1147, 1204)],
        "Player prior: keep Great Tusk/Crustle alive and pivot cleanly under prize pressure without a second ACE SPEC.",
    ),
]


SEARCH_PRIORS: list[tuple[str, int, float, float, int, str]] = [
    ("fast_turn", 6, 0.16, 1800.0, 10, "Low compute, conservative search."),
    ("balanced_turn", 8, 0.25, 1200.0, 16, "Balanced turn-level rollout search."),
    ("wide_turn", 12, 0.38, 900.0, 18, "Wider candidate beam for high-impact turns."),
    ("deep_turn", 10, 0.45, 700.0, 24, "More complete own-turn rollout."),
    ("strict_override", 8, 0.25, 2400.0, 16, "Only override heuristic on very large value gap."),
    ("player_ordered", 10, 0.28, 850.0, 14, "Human-prior ordered search over supporter/search/attach/attack decisions."),
    ("mill_tactical", 14, 0.42, 650.0, 20, "Spend more budget on tactical mill and stall turning points."),
    ("late_override", 12, 0.34, 350.0, 16, "Lower override margin when search finds a near-terminal deckout line."),
]


def great_tusk_prior_configs(
    incumbent: BuildConfig,
    generation: int,
    out_dir: Path,
    limit: int,
) -> list[BuildConfig]:
    configs: list[BuildConfig] = []

    def add(cfg: BuildConfig) -> None:
        if len(configs) < limit:
            configs.append(cfg)

    prefix = f"g{generation:03d}"
    add(
        replace(
            incumbent,
            name=f"{prefix}_incumbent",
            family="great_tusk",
            out=out_dir / f"{prefix}_incumbent.tar.gz",
            notes=f"Frozen incumbent for generation {generation}.",
        )
    )

    for name, cand, budget, margin, rollout, note in SEARCH_PRIORS:
        add(
            replace(
                incumbent,
                name=f"{prefix}_search_{name}",
                family="great_tusk",
                out=out_dir / f"{prefix}_search_{name}.tar.gz",
                search_candidates=cand,
                search_budget_s=budget,
                search_margin=margin,
                search_rollout_steps=rollout,
                notes=f"{incumbent.notes} | {note}",
            )
        )

    existing = {tuple(pair) for pair in incumbent.deck_swaps}
    for name, swaps, note in GT_PRIOR_PACKAGES:
        merged = list(incumbent.deck_swaps)
        for swap in swaps:
            if swap not in existing and swap not in merged:
                merged.append(swap)
        add(
            replace(
                incumbent,
                name=f"{prefix}_prior_{name}",
                family="great_tusk",
                out=out_dir / f"{prefix}_prior_{name}.tar.gz",
                deck_swaps=merged[:5],
                notes=f"{incumbent.notes} | {note}",
            )
        )

    # Re-anchor from the original Great Tusk seed so the loop can escape harmful
    # inherited mutations while keeping the same archetype.
    for name, swaps, note in GT_PRIOR_PACKAGES[: max(1, limit // 4)]:
        add(
            BuildConfig(
                name=f"{prefix}_reseed_{name}",
                family="great_tusk",
                base=REFERENCE_BASES.get("great_tusk", DEFAULT_BASE),
                out=out_dir / f"{prefix}_reseed_{name}.tar.gz",
                injection="great_tusk",
                search_candidates=incumbent.search_candidates,
                search_budget_s=incumbent.search_budget_s,
                search_margin=incumbent.search_margin,
                search_rollout_steps=incumbent.search_rollout_steps,
                deck_swaps=swaps[:3],
                notes=f"Reseed from original Great Tusk. {note}",
            )
        )

    add(
        replace(
            incumbent,
            name=f"{prefix}_pure_heuristic",
            family="great_tusk",
            out=out_dir / f"{prefix}_pure_heuristic.tar.gz",
            enable_search=False,
            notes="Ablation: disable search wrapper.",
        )
    )
    return configs


def portfolio_seed_configs(generation: int, out_dir: Path) -> list[BuildConfig]:
    configs: list[BuildConfig] = []
    for family, base in REFERENCE_BASES.items():
        if family == "great_tusk" or not base.exists():
            continue
        configs.append(
            BuildConfig(
                name=f"g{generation:03d}_portfolio_{family}",
                family=family,
                base=base,
                out=out_dir / f"g{generation:03d}_portfolio_{family}.tar.gz",
                injection="none",
                enable_search=False,
                deck_files=("deck.csv", "lucario_deck.csv") if family == "lucario" else ("deck.csv",),
                notes=f"Portfolio seed copied from {base.name}; used to widen metagame pressure.",
            )
        )
    return configs


METAL_PRIOR_PACKAGES: list[tuple[str, list[tuple[int, int]], str]] = [
    (
        "boss_pressure",
        [(1182, 1213)],
        "Metal prior: maximize Boss pressure, cut Judge flex.",
    ),
    (
        "recovery",
        [(1097, 1213)],
        "Metal prior: more Night Stretcher recovery for attacker loops.",
    ),
    (
        "energy_density",
        [(8, 1213)],
        "Metal prior: add one Basic Metal Energy for tempo reliability.",
    ),
    (
        "relicanth_memory",
        [(57, 1213)],
        "Metal prior: improve Memory Dive access for evolved attackers.",
    ),
    (
        "draw_trim",
        [(1182, 1227)],
        "Metal prior: trade one draw supporter for closer pressure.",
    ),
]


def metal_prior_configs(generation: int, out_dir: Path, limit: int) -> list[BuildConfig]:
    configs: list[BuildConfig] = []

    def add(cfg: BuildConfig) -> None:
        if len(configs) < limit:
            configs.append(cfg)

    for family in ("metal_tempo", "rahul_metal"):
        base = REFERENCE_BASES.get(family)
        if base is None or not base.exists():
            continue
        add(
            BuildConfig(
                name=f"g{generation:03d}_{family}_base",
                family=family,
                base=base,
                out=out_dir / f"g{generation:03d}_{family}_base.tar.gz",
                injection="none",
                enable_search=False,
                deck_files=("deck.csv",),
                notes=f"Cross-archetype seed from {base.name}.",
            )
        )
        for name, swaps, note in METAL_PRIOR_PACKAGES:
            add(
                BuildConfig(
                    name=f"g{generation:03d}_{family}_{name}",
                    family=family,
                    base=base,
                    out=out_dir / f"g{generation:03d}_{family}_{name}.tar.gz",
                    injection="none",
                    enable_search=False,
                    deck_swaps=swaps,
                    notes=note,
                )
            )
    return configs
