from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_eval.evaluator import run_candidate_pool, save_report, submission_name
from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.build_models import BuildConfig
from tools.build_submission import build_submission
from tools.deck_rules import validate_deck_ids
from tools.export_kaggle_submission import export_kaggle_submission


BATTLE_CAGE = 1264
HABAN_BERRY = 1170

SEARCH_PROFILES = (
    ("fast", 4, 0.08, 2200.0, 6, 4, 0.25),
    ("balanced", 6, 0.14, 2000.0, 9, 6, 0.32),
    ("robust", 8, 0.17, 2300.0, 11, 8, 0.40),
)

DEFENSE_PACKAGES = {
    "c1": (BATTLE_CAGE,),
    "h1": (HABAN_BERRY,),
    "c1h1": (BATTLE_CAGE, HABAN_BERRY),
    "c2": (BATTLE_CAGE, BATTLE_CAGE),
    "h2": (HABAN_BERRY, HABAN_BERRY),
    "c2h1": (BATTLE_CAGE, BATTLE_CAGE, HABAN_BERRY),
    "c1h2": (BATTLE_CAGE, HABAN_BERRY, HABAN_BERRY),
    "c2h2": (BATTLE_CAGE, BATTLE_CAGE, HABAN_BERRY, HABAN_BERRY),
}

CUT_TEMPLATES = {
    1: ((1204,), (1182,), (1147,), (607,)),
    2: ((1204, 1182), (1204, 1123), (1204, 1152), (1182, 1197), (1147, 1139), (1182, 607)),
    3: ((1204, 1182, 1123), (1204, 1182, 1152), (1204, 1197, 1142), (1182, 1147, 1139)),
    4: (
        (1204, 1182, 1123, 1152),
        (1204, 1182, 1197, 1142),
        (1204, 1147, 1139, 607),
        (1182, 1123, 1152, 1197),
    ),
}


@dataclass(slots=True)
class RobustRow:
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
    robust_score: float
    matchups: dict[str, dict[str, float | int]]


def read_deck(path: Path) -> list[int]:
    with tarfile.open(path, "r:gz") as archive:
        payload = archive.extractfile("deck.csv")
        if payload is None:
            raise ValueError(f"Missing deck.csv in {path}")
        deck = [int(line) for line in payload.read().decode("utf-8").splitlines() if line.strip()]
    validate_deck_ids(deck)
    return deck


def mutate_deck(deck: list[int], additions: tuple[int, ...], cuts: tuple[int, ...]) -> list[int] | None:
    result = list(deck)
    try:
        for card_id in cuts:
            result.remove(card_id)
    except ValueError:
        return None
    result.extend(additions)
    validate_deck_ids(result)
    return result


def wilson_lower(wins: int, losses: int, draws: int, z: float = 1.2815515655446004) -> float:
    total = wins + losses + draws
    if total <= 0:
        return 0.0
    probability = (wins + 0.5 * draws) / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    spread = z * math.sqrt((probability * (1.0 - probability) + z * z / (4.0 * total)) / total)
    return max(0.0, (center - spread) / denominator)


def unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def build_candidates(args: argparse.Namespace, out: Path) -> list[Path]:
    candidate_dir = out / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    seen: set[tuple[tuple[int, ...], int, float, float, int, int, float]] = set()

    def add_candidate(base_name: str, deck: list[int], profile: tuple[str, int, float, float, int, int, float], tag: str) -> None:
        profile_name, search_candidates, budget, margin, rollout_steps, belief_worlds, risk_penalty = profile
        signature = (tuple(sorted(deck)), search_candidates, budget, margin, rollout_steps, belief_worlds, risk_penalty)
        if signature in seen:
            return
        seen.add(signature)
        name = f"{base_name}_{tag}_{profile_name}"
        config = BuildConfig(
            name=name,
            family="great_tusk",
            base=Path("/nonexistent/reference.tar.gz"),
            out=candidate_dir / f"{name}.tar.gz",
            runtime_source=args.runtime_source,
            runtime_cg_dir=args.runtime_cg_dir,
            enable_search=True,
            injection="great_tusk",
            search_candidates=search_candidates,
            search_budget_s=budget,
            search_margin=margin,
            search_rollout_steps=rollout_steps,
            belief_worlds=belief_worlds,
            risk_penalty=risk_penalty,
            deck_override=deck,
            origin="robust_gold_search",
            notes=f"Robust search candidate from {base_name}; package={tag}; profile={profile_name}.",
        )
        candidates.append(build_submission(config))

    for base in args.bases:
        deck = read_deck(base)
        base_name = submission_name(base).replace(".", "_")
        for profile in SEARCH_PROFILES:
            add_candidate(base_name, deck, profile, "base")
        for package_name, additions in DEFENSE_PACKAGES.items():
            for cuts in CUT_TEMPLATES[len(additions)]:
                mutated = mutate_deck(deck, additions, cuts)
                if mutated is None:
                    continue
                add_candidate(base_name, mutated, SEARCH_PROFILES[1], f"{package_name}_cut_{'_'.join(map(str, cuts))}")
                if package_name in {"c1h1", "c2h1", "c1h2"} and cuts == CUT_TEMPLATES[len(additions)][0]:
                    add_candidate(base_name, mutated, SEARCH_PROFILES[2], f"{package_name}_cut_{'_'.join(map(str, cuts))}")

    candidates.extend(path for path in args.external_candidates if path.exists())
    return unique_paths(candidates)


def evaluate(
    candidates: list[Path],
    opponents: list[Path],
    games_per_pair: int,
    args: argparse.Namespace,
    out: Path,
    seed: int,
) -> MatchReport:
    config = EvalConfig(
        workers=args.workers,
        run_timeout_s=args.run_timeout,
        max_actions=args.max_actions,
        seed=seed,
        record_mode=args.record_mode,
        record_sample_rate=args.record_sample_rate,
        progress=args.progress,
        progress_label=out.name,
        progress_mode="line",
    )
    report = run_candidate_pool(candidates, opponents, games_per_pair, config, args.project_root, peer_span=0)
    save_report(report, out)
    return report


def rank_report(report: MatchReport, candidates: list[Path], opponents: list[Path]) -> list[RobustRow]:
    candidate_names = {submission_name(path) for path in candidates}
    opponent_names = {submission_name(path) for path in opponents}
    rows: list[RobustRow] = []
    for stats in report.standings:
        if stats.name not in candidate_names:
            continue
        matchup_rows: dict[str, dict[str, float | int]] = {}
        rates: list[float] = []
        lowers: list[float] = []
        for opponent_name, values in stats.opponents.items():
            if opponent_name not in opponent_names:
                continue
            wins = int(values.get("wins", 0))
            losses = int(values.get("losses", 0))
            draws = int(values.get("draws", 0))
            total = wins + losses + draws
            rate = (wins + 0.5 * draws) / max(1, total)
            lower = wilson_lower(wins, losses, draws)
            rates.append(rate)
            lowers.append(lower)
            matchup_rows[opponent_name] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "rate": rate,
                "lower": lower,
            }
        if not rates:
            continue
        mean_rate = sum(rates) / len(rates)
        mean_lower = sum(lowers) / len(lowers)
        worst_rate = min(rates)
        worst_lower = min(lowers)
        no_result_rate = stats.no_results / max(1, stats.games)
        robust_score = 0.55 * worst_lower + 0.30 * mean_lower + 0.15 * mean_rate - min(0.25, 4.0 * no_result_rate)
        rows.append(
            RobustRow(
                name=stats.name,
                tarball=stats.tarball,
                games=stats.games,
                wins=stats.wins,
                losses=stats.losses,
                no_results=stats.no_results,
                mean_rate=mean_rate,
                worst_rate=worst_rate,
                mean_lower=mean_lower,
                worst_lower=worst_lower,
                robust_score=robust_score,
                matchups=matchup_rows,
            )
        )
    rows.sort(key=lambda row: (row.robust_score, row.worst_lower, row.mean_lower, -row.no_results), reverse=True)
    return rows


def write_ranking(path: Path, rows: list[RobustRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("name", "games", "wins", "losses", "no_results", "mean_rate", "worst_rate", "mean_lower", "worst_lower", "robust_score", "tarball"),
        )
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record.pop("matchups")
            writer.writerow(record)


def combine_rankings(train: list[RobustRow], holdout: list[RobustRow]) -> list[dict[str, Any]]:
    train_map = {row.name: row for row in train}
    holdout_map = {row.name: row for row in holdout}
    combined: list[dict[str, Any]] = []
    for name in train_map.keys() & holdout_map.keys():
        train_row = train_map[name]
        holdout_row = holdout_map[name]
        worst_lower = min(train_row.worst_lower, holdout_row.worst_lower)
        mean_lower = (train_row.mean_lower + holdout_row.mean_lower) / 2.0
        mean_rate = (train_row.mean_rate + holdout_row.mean_rate) / 2.0
        score = 0.60 * worst_lower + 0.30 * mean_lower + 0.10 * mean_rate
        combined.append(
            {
                "name": name,
                "tarball": train_row.tarball,
                "score": score,
                "worst_lower": worst_lower,
                "mean_lower": mean_lower,
                "mean_rate": mean_rate,
                "train": asdict(train_row),
                "holdout": asdict(holdout_row),
            }
        )
    combined.sort(key=lambda row: (row["score"], row["worst_lower"], row["mean_lower"]), reverse=True)
    return combined


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args, out)
    if not candidates:
        raise ValueError("No candidates were built")

    stage1_report = evaluate(candidates, args.train_pool, args.stage1_games, args, out / "stage1", args.seed)
    stage1_rows = rank_report(stage1_report, candidates, args.train_pool)
    write_ranking(out / "stage1_ranking.json", stage1_rows)
    stage2_candidates = [Path(row.tarball) for row in stage1_rows[: args.stage2_candidates]]

    stage2_report = evaluate(stage2_candidates, args.train_pool, args.stage2_games, args, out / "stage2", args.seed + 100000)
    stage2_rows = rank_report(stage2_report, stage2_candidates, args.train_pool)
    write_ranking(out / "stage2_ranking.json", stage2_rows)
    holdout_candidates = [Path(row.tarball) for row in stage2_rows[: args.holdout_candidates]]

    holdout_report = evaluate(holdout_candidates, args.holdout_pool, args.holdout_games, args, out / "holdout", args.seed + 200000)
    holdout_rows = rank_report(holdout_report, holdout_candidates, args.holdout_pool)
    write_ranking(out / "holdout_ranking.json", holdout_rows)
    combined = combine_rankings(stage2_rows, holdout_rows)
    (out / "combined_ranking.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    finalists = [Path(row["tarball"]) for row in combined[: args.finalists]]

    final_opponents = unique_paths([*args.train_pool, *args.holdout_pool])
    final_report = evaluate(finalists, final_opponents, args.final_games, args, out / "final", args.seed + 300000)
    final_rows = rank_report(final_report, finalists, final_opponents)
    write_ranking(out / "final_ranking.json", final_rows)
    if not final_rows:
        raise RuntimeError("Final evaluation produced no ranked candidates")

    winner = Path(final_rows[0].tarball)
    args.promote.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner, args.promote)
    export_kaggle_submission(args.promote, args.submission_out, strip_search_wrapper=args.strip_search_wrapper)
    result = {
        "winner": asdict(final_rows[0]),
        "promote": str(args.promote.resolve()),
        "submission": str(args.submission_out.resolve()),
        "candidate_count": len(candidates),
        "stage2_count": len(stage2_candidates),
        "holdout_count": len(holdout_candidates),
        "finalist_count": len(finalists),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust multi-pool search for a Pokémon TCG submission.")
    parser.add_argument("--out", type=Path, default=Path("outputs/robust_gold_search"))
    parser.add_argument("--bases", type=Path, nargs="+", required=True)
    parser.add_argument("--external-candidates", type=Path, nargs="*", default=[])
    parser.add_argument("--train-pool", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-pool", type=Path, nargs="+", required=True)
    parser.add_argument("--runtime-source", type=Path, default=Path("main.py"))
    parser.add_argument("--runtime-cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--promote", type=Path, default=Path("outputs/submissions/champion_robust_gold.tar.gz"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submissions/submission_robust_gold.tar.gz"))
    parser.add_argument("--stage1-games", type=int, default=6)
    parser.add_argument("--stage2-games", type=int, default=32)
    parser.add_argument("--holdout-games", type=int, default=64)
    parser.add_argument("--final-games", type=int, default=128)
    parser.add_argument("--stage2-candidates", type=int, default=14)
    parser.add_argument("--holdout-candidates", type=int, default=6)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--workers", type=int, default=72)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--record-mode", choices=("none", "losses", "sample", "all"), default="losses")
    parser.add_argument("--record-sample-rate", type=float, default=0.01)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip-search-wrapper", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
