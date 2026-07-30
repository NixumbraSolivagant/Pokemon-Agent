from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_eval.evaluator import run_candidate_pool, save_report, submission_name
from local_eval.models import EvalConfig, MatchReport
from tools.build_models import BuildConfig
from tools.build_submission import build_submission
from tools.deck_rules import validate_deck_ids
from tools.export_kaggle_submission import export_kaggle_submission


FIGHTING_ENERGY = 6
HARIYAMA = 674
SOLROCK = 676
RIOLU = 677
MEGA_LUCARIO_EX = 678
SWITCH = 1123
POKE_PAD = 1152
HERO_CAPE = 1159
BOSS_ORDERS = 1182
JUDGE = 1213
GRAVITY_MOUNTAIN = 1252
BATTLE_CAGE = 1264
LEGACY_ENERGY = 12

SEARCH_PROFILES = (
    ("heuristic", False, 0.0, 4, 1, 0.0, 999999.0),
    ("fast", True, 0.20, 4, 2, 0.18, 320.0),
    ("balanced", True, 0.48, 5, 3, 0.25, 180.0),
    ("guarded", True, 0.42, 5, 4, 0.30, 900.0),
    ("deep", True, 0.82, 6, 4, 0.32, 90.0),
)


@dataclass(slots=True)
class RankedCandidate:
    name: str
    tarball: str
    games: int
    wins: int
    losses: int
    no_results: int
    mean_rate: float
    worst_rate: float
    mean_lower: float
    worst_lower: float
    cvar_lower: float
    score: float
    matchups: dict[str, dict[str, float | int]]


def read_deck(path: Path) -> list[int]:
    with tarfile.open(path, "r:gz") as archive:
        payload = archive.extractfile("deck.csv")
        if payload is None:
            raise ValueError(f"Missing deck.csv in {path}")
        deck = [int(line) for line in payload.read().decode("utf-8").splitlines() if line.strip()]
    validate_deck_ids(deck)
    return deck


def replace_cards(deck: list[int], cuts: tuple[int, ...], additions: tuple[int, ...]) -> list[int] | None:
    result = list(deck)
    try:
        for card_id in cuts:
            result.remove(card_id)
    except ValueError:
        return None
    result.extend(additions)
    try:
        validate_deck_ids(result)
    except ValueError:
        return None
    return result


def deck_variants(base: list[int]) -> list[tuple[str, list[int]]]:
    specifications = [
        ("base", (), ()),
        ("cage1_energy", (FIGHTING_ENERGY,), (BATTLE_CAGE,)),
        ("cage1_pad", (POKE_PAD,), (BATTLE_CAGE,)),
        ("cage1_boss", (BOSS_ORDERS,), (BATTLE_CAGE,)),
        ("cage2_energy_pad", (FIGHTING_ENERGY, POKE_PAD), (BATTLE_CAGE, BATTLE_CAGE)),
        ("judge1_energy", (FIGHTING_ENERGY,), (JUDGE,)),
        ("judge1_pad", (POKE_PAD,), (JUDGE,)),
        ("judge2_energy_pad", (FIGHTING_ENERGY, POKE_PAD), (JUDGE, JUDGE)),
        ("cage_judge_energy_pad", (FIGHTING_ENERGY, POKE_PAD), (BATTLE_CAGE, JUDGE)),
        ("cage_judge_boss_pad", (BOSS_ORDERS, POKE_PAD), (BATTLE_CAGE, JUDGE)),
        ("legacy", (HERO_CAPE,), (LEGACY_ENERGY,)),
        ("legacy_cage", (HERO_CAPE, POKE_PAD), (LEGACY_ENERGY, BATTLE_CAGE)),
        ("legacy_judge", (HERO_CAPE, POKE_PAD), (LEGACY_ENERGY, JUDGE)),
        ("switch3", (FIGHTING_ENERGY,), (SWITCH,)),
        ("boss4", (FIGHTING_ENERGY,), (BOSS_ORDERS,)),
        ("hariyama3", (SOLROCK, FIGHTING_ENERGY), (HARIYAMA, HARIYAMA)),
        ("hariyama3_cage", (SOLROCK, FIGHTING_ENERGY, POKE_PAD), (HARIYAMA, HARIYAMA, BATTLE_CAGE)),
        ("energy15", (POKE_PAD,), (FIGHTING_ENERGY,)),
        ("no_mountain_cage", (GRAVITY_MOUNTAIN,), (BATTLE_CAGE,)),
    ]
    variants = []
    seen = set()
    for name, cuts, additions in specifications:
        deck = replace_cards(base, cuts, additions)
        if deck is None:
            continue
        signature = tuple(sorted(deck))
        if signature in seen:
            continue
        seen.add(signature)
        variants.append((name, deck))
    return variants


def render_runtime(source: str, profile: tuple[str, bool, float, int, int, float, float], models: dict[str, list[int]]) -> str:
    _, enabled, budget, candidates, worlds, risk, margin = profile
    replacements: dict[str, Any] = {
        "USE_SEARCH": enabled,
        "SEARCH_TIME_BUDGET": budget,
        "SEARCH_MAX_CANDIDATES": candidates,
        "SEARCH_BELIEF_WORLDS": worlds,
        "SEARCH_RISK_PENALTY": risk,
        "SEARCH_MARGIN": margin,
        "OPPONENT_DECK_MODELS": models,
    }
    rendered = source
    for name, value in replacements.items():
        rendered, count = re.subn(rf"^{name} = .*?$", f"{name} = {value!r}", rendered, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Could not replace {name}")
    return rendered


def opponent_models(paths: list[Path]) -> dict[str, list[int]]:
    models = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            models[submission_name(path)] = read_deck(path)
        except (OSError, ValueError, tarfile.TarError):
            continue
    return models


def build_candidates(args: argparse.Namespace, out: Path) -> list[Path]:
    source = args.runtime_source.read_text(encoding="utf-8")
    base_deck = read_deck(args.base)
    models = opponent_models([*args.train_pool, *args.holdout_pool, args.base])
    candidate_dir = out / "candidates"
    runtime_dir = out / "runtimes"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for deck_name, deck in deck_variants(base_deck):
        for profile in SEARCH_PROFILES:
            profile_name = profile[0]
            name = f"lucario_{deck_name}_{profile_name}"
            runtime = runtime_dir / f"{name}.py"
            runtime.write_text(render_runtime(source, profile, models), encoding="utf-8")
            candidates.append(
                build_submission(
                    BuildConfig(
                        name=name,
                        family="lucario_meta",
                        base=args.base,
                        out=candidate_dir / f"{name}.tar.gz",
                        runtime_source=runtime,
                        runtime_cg_dir=args.runtime_cg_dir,
                        enable_search=profile[1],
                        injection="none",
                        deck_override=deck,
                        origin="lucario_meta_search",
                        notes=f"Lucario meta variant {deck_name}; profile {profile_name}.",
                    )
                )
            )
    return candidates


def wilson_lower(wins: int, losses: int, draws: int, z: float = 1.2815515655446004) -> float:
    total = wins + losses + draws
    if total <= 0:
        return 0.0
    probability = (wins + 0.5 * draws) / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    spread = z * math.sqrt((probability * (1.0 - probability) + z * z / (4.0 * total)) / total)
    return max(0.0, (center - spread) / denominator)


def rank_report(report: MatchReport, candidates: list[Path], opponents: list[Path]) -> list[RankedCandidate]:
    candidate_names = {submission_name(path) for path in candidates}
    opponent_names = {submission_name(path) for path in opponents}
    rows = []
    for stats in report.standings:
        if stats.name not in candidate_names:
            continue
        rates = []
        lowers = []
        matchups = {}
        for name, values in stats.opponents.items():
            if name not in opponent_names:
                continue
            wins = int(values.get("wins", 0))
            losses = int(values.get("losses", 0))
            draws = int(values.get("draws", 0))
            total = wins + losses + draws
            rate = (wins + 0.5 * draws) / max(1, total)
            lower = wilson_lower(wins, losses, draws)
            rates.append(rate)
            lowers.append(lower)
            matchups[name] = {"wins": wins, "losses": losses, "draws": draws, "rate": rate, "lower": lower}
        if not rates:
            continue
        bottom = sorted(lowers)[: min(3, len(lowers))]
        no_result_rate = stats.no_results / max(1, stats.games)
        score = (
            0.40 * min(lowers)
            + 0.25 * (sum(lowers) / len(lowers))
            + 0.20 * (sum(bottom) / len(bottom))
            + 0.15 * (sum(rates) / len(rates))
            - min(0.35, 5.0 * no_result_rate)
        )
        rows.append(
            RankedCandidate(
                name=stats.name,
                tarball=stats.tarball,
                games=stats.games,
                wins=stats.wins,
                losses=stats.losses,
                no_results=stats.no_results,
                mean_rate=sum(rates) / len(rates),
                worst_rate=min(rates),
                mean_lower=sum(lowers) / len(lowers),
                worst_lower=min(lowers),
                cvar_lower=sum(bottom) / len(bottom),
                score=score,
                matchups=matchups,
            )
        )
    rows.sort(key=lambda row: (row.score, row.worst_lower, row.cvar_lower, row.mean_lower, -row.no_results), reverse=True)
    return rows


def evaluate(candidates: list[Path], opponents: list[Path], games: int, args: argparse.Namespace, out: Path, seed: int) -> MatchReport:
    config = EvalConfig(
        workers=args.workers,
        act_timeout_s=args.act_timeout,
        deck_timeout_s=8.0,
        import_timeout_s=8.0,
        overage_time_s=600.0,
        run_timeout_s=args.run_timeout,
        max_actions=args.max_actions,
        seed=seed,
        record_mode="none",
    )
    report = run_candidate_pool(candidates, opponents, games, config, args.project_root, peer_span=0)
    save_report(report, out)
    return report


def write_rows(path: Path, rows: list[RankedCandidate]) -> None:
    path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search Lucario meta decks with functioning belief-state lookahead.")
    parser.add_argument("--base", type=Path, default=Path("outputs/reference_submissions/improved-probabilistic-agent.tar.gz"))
    parser.add_argument("--runtime-source", type=Path, default=Path("agents/lucario_meta.py"))
    parser.add_argument("--runtime-cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-pool", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-pool", type=Path, nargs="+", required=True)
    parser.add_argument("--stage1-games", type=int, default=4)
    parser.add_argument("--stage2-games", type=int, default=32)
    parser.add_argument("--holdout-games", type=int, default=64)
    parser.add_argument("--stage2-candidates", type=int, default=16)
    parser.add_argument("--holdout-candidates", type=int, default=6)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--act-timeout", type=float, default=6.0)
    parser.add_argument("--run-timeout", type=float, default=240.0)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--champion", type=Path, default=Path("outputs/submissions/champion_lucario_meta.tar.gz"))
    parser.add_argument("--submission", type=Path, default=Path("outputs/submissions/submission_lucario_meta.tar.gz"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args, args.out)
    stage1 = rank_report(
        evaluate(candidates, args.train_pool, args.stage1_games, args, args.out / "stage1", args.seed),
        candidates,
        args.train_pool,
    )
    write_rows(args.out / "stage1_ranking.json", stage1)
    stage2_candidates = [Path(row.tarball) for row in stage1[: args.stage2_candidates]]
    stage2 = rank_report(
        evaluate(stage2_candidates, args.train_pool, args.stage2_games, args, args.out / "stage2", args.seed + 100000),
        stage2_candidates,
        args.train_pool,
    )
    write_rows(args.out / "stage2_ranking.json", stage2)
    holdout_candidates = [Path(row.tarball) for row in stage2[: args.holdout_candidates]]
    holdout = rank_report(
        evaluate(holdout_candidates, args.holdout_pool, args.holdout_games, args, args.out / "holdout", args.seed + 200000),
        holdout_candidates,
        args.holdout_pool,
    )
    write_rows(args.out / "holdout_ranking.json", holdout)
    stage2_map = {row.name: row for row in stage2}
    combined = []
    for row in holdout:
        train = stage2_map[row.name]
        score = 0.55 * min(train.score, row.score) + 0.25 * ((train.score + row.score) / 2.0) + 0.20 * min(train.worst_lower, row.worst_lower)
        combined.append({"name": row.name, "tarball": row.tarball, "score": score, "train": asdict(train), "holdout": asdict(row)})
    combined.sort(key=lambda item: (item["score"], min(item["train"]["worst_lower"], item["holdout"]["worst_lower"])), reverse=True)
    (args.out / "combined_ranking.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    if not combined:
        raise RuntimeError("No clean Lucario meta candidate survived")
    winner = Path(combined[0]["tarball"])
    args.champion.parent.mkdir(parents=True, exist_ok=True)
    args.champion.write_bytes(winner.read_bytes())
    export_kaggle_submission(args.champion, args.submission)
    result = {"winner": combined[0], "candidate_count": len(candidates), "stage2_count": len(stage2_candidates), "holdout_count": len(holdout_candidates)}
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
