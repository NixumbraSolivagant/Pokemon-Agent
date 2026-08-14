from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from cg.api import OptionType, SelectContext
from local_eval.evaluator import run_match, submission_name
from local_eval.models import EvalConfig


TERMINAL_TYPES = {int(OptionType.ATTACK), int(OptionType.END)}


def _option_type(row: dict[str, Any]) -> int:
    value = row.get("type", 0)
    if isinstance(value, str):
        try:
            return int(OptionType[value])
        except (KeyError, ValueError):
            return 0
    return int(value or 0)


def summarize_closed_loop(report: Any, candidate_name: str) -> dict[str, Any]:
    actions = Counter()
    main_actions = 0
    terminal_actions = 0
    terminal_turns = []
    actions_per_turn: Counter[tuple[str, int, int]] = Counter()
    failures = 0
    for game in report.games:
        if game.reason != "RESULT":
            failures += int(candidate_name in {game.p0, game.p1})
        candidate_seat = 0 if game.p0 == candidate_name else 1 if game.p1 == candidate_name else -1
        if candidate_seat < 0:
            continue
        for event in game.trace:
            if event.get("event") != "action" or int(event.get("actor_seat", -1)) != candidate_seat:
                continue
            observation = event.get("observation") or {}
            select = observation.get("select") or {}
            if int(select.get("context", -1)) != int(SelectContext.MAIN):
                continue
            selected = event.get("selected_options") or []
            selected_types = [_option_type(row) for row in selected]
            turn = int(((observation.get("current") or {}).get("turn", 0)) or 0)
            main_actions += 1
            actions_per_turn[(game.game_id, candidate_seat, turn)] += 1
            for option_type in selected_types:
                actions[str(option_type)] += 1
            if any(option_type in TERMINAL_TYPES for option_type in selected_types):
                terminal_actions += 1
                terminal_turns.append(turn)
    row = next(stats for stats in report.standings if stats.name == candidate_name)
    turn_counts = list(actions_per_turn.values())
    return {
        "games": row.games,
        "wins": row.wins,
        "losses": row.losses,
        "draws": row.draws,
        "score": (row.wins + 0.5 * row.draws) / max(1, row.games),
        "failures": failures,
        "main_actions": main_actions,
        "terminal_actions": terminal_actions,
        "terminal_fraction": terminal_actions / max(1, main_actions),
        "action_type_counts": dict(sorted(actions.items())),
        "actions_per_turn_mean": sum(turn_counts) / max(1, len(turn_counts)),
        "actions_per_turn_median": statistics.median(turn_counts) if turn_counts else 0.0,
        "terminal_turn_median": statistics.median(terminal_turns) if terminal_turns else None,
    }


def audit(candidate: Path, baseline: Path, games: int, workers: int, seed: int) -> dict[str, Any]:
    candidate_name = submission_name(candidate)
    report = run_match(
        candidate,
        baseline,
        games,
        EvalConfig(
            profile="kaggle",
            seed=seed,
            workers=max(1, workers),
            record_mode="training",
            record_focus=candidate_name,
            progress=False,
        ),
        Path.cwd(),
    )
    return {
        "candidate": str(candidate),
        "baseline": str(baseline),
        **summarize_closed_loop(report, candidate_name),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a clone under autonomous closed-loop play.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.candidate, args.baseline, args.games, args.workers, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
