from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from agents.meta_runtime import ATTACK_TABLE, CARD_TABLE, active, energy_count, option_card, option_target
from cg.api import OptionType, SelectContext, to_observation_class
from tools.replay_imitation import infer_target_deck, read_replays, target_index


FOCUS_ACTIONS = {
    "RETREAT",
    "ATTACK:Gale Thrust",
    "ATTACK:Spiky Hopper",
    "ATTACK:Resentful Refrain",
    "PLAY:Hand Trimmer",
    "PLAY:Wally's Compassion",
    "ABILITY:Fan Rotom",
    "ABILITY:Dudunsparce",
}


def card_name(card: object | None) -> str:
    card_id = int(getattr(card, "id", 0) or 0)
    definition = CARD_TABLE.get(card_id)
    return str(getattr(card, "name", "") or getattr(definition, "name", "") or card_id or "")


def option_label(obs, option) -> str:
    option_type = OptionType(option.type).name
    if option.type == OptionType.ATTACK:
        attack = ATTACK_TABLE.get(int(option.attackId or 0))
        name = getattr(attack, "name", "") or option.attackId
        return f"ATTACK:{name}"
    card = option_card(obs, option)
    if card is not None:
        return f"{option_type}:{card_name(card)}"
    return option_type


def snapshot(obs, label: str, selected_index: int, moved: bool) -> dict[str, Any]:
    state = obs.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    my_active = active(me)
    their_active = active(opponent)
    option = obs.select.option[selected_index]
    selected_card = option_card(obs, option)
    target = option_target(obs, option)
    return {
        "turn": int(state.turn or 0),
        "action_count": int(getattr(state, "turnActionCount", 0) or 0),
        "context": SelectContext(obs.select.context).name,
        "label": label,
        "card_id": int(getattr(selected_card, "id", 0) or 0),
        "moved": moved,
        "effect": int(getattr(getattr(obs.select, "effect", None), "id", 0) or 0),
        "active": int(getattr(my_active, "id", 0) or 0),
        "active_hp": int(getattr(my_active, "hp", 0) or 0),
        "active_energy": energy_count(my_active),
        "opponent_active": int(getattr(their_active, "id", 0) or 0),
        "opponent_hp": int(getattr(their_active, "hp", 0) or 0),
        "own_hand": int(getattr(me, "handCount", 0) or len(me.hand or [])),
        "opponent_hand": int(getattr(opponent, "handCount", 0) or len(opponent.hand or [])),
        "own_prizes": len(me.prize or []),
        "opponent_prizes": len(opponent.prize or []),
        "bench": [int(pokemon.id) for pokemon in me.bench or [] if pokemon is not None],
        "bench_energy": [energy_count(pokemon) for pokemon in me.bench or [] if pokemon is not None],
        "target": int(getattr(target, "id", 0) or 0),
        "target_hp": int(getattr(target, "hp", 0) or 0),
        "target_energy": energy_count(target),
        "option_count": len(obs.select.option),
        "available": [option_label(obs, candidate) for candidate in obs.select.option],
    }


def analyze_replay(replay: dict[str, Any], player_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_observation: dict[str, Any] | None = None
    first_lopunny = None
    first_gale = None
    first_froslass = None
    attacks = Counter()
    movement_turn = None
    previous_active_serial = None
    moved_this_turn = False
    for step in replay.get("steps", []):
        if player_index >= len(step):
            continue
        agent_step = step[player_index] or {}
        action = agent_step.get("action")
        observation = previous_observation
        if observation and isinstance(action, list) and len(action) != 60:
            try:
                obs = to_observation_class(observation)
            except Exception:
                obs = None
            if obs is not None and obs.select is not None:
                current_active = active(obs.current.players[obs.current.yourIndex])
                current_serial = getattr(current_active, "serial", None)
                current_turn = int(obs.current.turn or 0)
                if current_turn != movement_turn:
                    movement_turn = current_turn
                    previous_active_serial = current_serial
                    moved_this_turn = False
                else:
                    changed = current_serial != previous_active_serial and current_serial is not None
                    if changed:
                        previous_active_serial = current_serial
                        moved_this_turn = True
                field_ids = [
                    int(pokemon.id)
                    for pokemon in list(obs.current.players[obs.current.yourIndex].active or [])
                    + list(obs.current.players[obs.current.yourIndex].bench or [])
                    if pokemon is not None
                ]
                turn = int(obs.current.turn or 0)
                if 849 in field_ids and first_lopunny is None:
                    first_lopunny = turn
                if 861 in field_ids and first_froslass is None:
                    first_froslass = turn
                for selected_index in action:
                    if not isinstance(selected_index, int) or not 0 <= selected_index < len(obs.select.option):
                        continue
                    option = obs.select.option[selected_index]
                    label = option_label(obs, option)
                    row = snapshot(obs, label, selected_index, moved_this_turn)
                    rows.append(row)
                    if option.type == OptionType.ATTACK:
                        attacks[label] += 1
                        if label == "ATTACK:Gale Thrust" and first_gale is None:
                            first_gale = turn
        previous_observation = agent_step.get("observation") or None
    reward = int((replay.get("rewards") or [0, 0])[player_index] or 0)
    return rows, {
        "won": reward > 0,
        "first_lopunny": first_lopunny,
        "first_froslass": first_froslass,
        "first_gale": first_gale,
        "attacks": dict(attacks),
    }


def scalar_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize(rows: list[dict[str, Any]], games: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    focus = {}
    for label in sorted(FOCUS_ACTIONS):
        selected = by_label.get(label, [])
        if not selected:
            continue
        focus[label] = {
            "count": len(selected),
            "turn": scalar_summary(selected, "turn"),
            "active_hp": scalar_summary(selected, "active_hp"),
            "active_energy": scalar_summary(selected, "active_energy"),
            "opponent_hp": scalar_summary(selected, "opponent_hp"),
            "own_hand": scalar_summary(selected, "own_hand"),
            "opponent_hand": scalar_summary(selected, "opponent_hand"),
            "active_ids": Counter(row["active"] for row in selected).most_common(),
            "opponent_ids": Counter(row["opponent_active"] for row in selected).most_common(),
            "bench_contains_lopunny": sum(849 in row["bench"] for row in selected),
            "bench_contains_froslass": sum(861 in row["bench"] for row in selected),
            "moved": sum(bool(row["moved"]) for row in selected),
            "both_lopunny_attacks": sum(
                "ATTACK:Gale Thrust" in row["available"] and "ATTACK:Spiky Hopper" in row["available"]
                for row in selected
            ),
        }
    retreat_routes = Counter()
    for index, row in enumerate(rows):
        if row["label"] != "RETREAT":
            continue
        for following in rows[index + 1:index + 7]:
            if following["turn"] != row["turn"]:
                break
            if following["context"] in {"SWITCH", "TO_ACTIVE"} and following["label"].startswith("CARD:"):
                retreat_routes[(row["active"], following["card_id"])] += 1
                break
    searches = Counter()
    for row in rows:
        if row["context"] in {"TO_HAND", "TO_FIELD", "TO_BENCH", "ATTACH_TO"} and row["label"].startswith("CARD:"):
            searches[(row["effect"], row["card_id"])] += 1
    main_routes = Counter()
    for row in rows:
        if row["context"] != "MAIN":
            continue
        main_routes[
            (
                row["active"],
                row["active_energy"],
                int(row["moved"]),
                int(849 in row["bench"]),
                int(861 in row["bench"]),
                int("RETREAT" in row["available"]),
                row["label"],
            )
        ] += 1
    return {
        "games": len(games),
        "wins": sum(game["won"] for game in games),
        "actions": Counter(row["label"] for row in rows).most_common(),
        "timelines": {
            key: scalar_summary(games, key)
            for key in ("first_lopunny", "first_froslass", "first_gale")
        },
        "focus": focus,
        "retreat_routes": [
            {"source": source, "destination": destination, "count": count}
            for (source, destination), count in retreat_routes.most_common()
        ],
        "main_routes": [
            {
                "active": active_id,
                "energy": energies,
                "moved": bool(moved),
                "lopunny_bench": bool(lopunny_bench),
                "froslass_bench": bool(froslass_bench),
                "retreat_available": bool(retreat_available),
                "selected": selected,
                "count": count,
            }
            for (
                active_id,
                energies,
                moved,
                lopunny_bench,
                froslass_bench,
                retreat_available,
                selected,
            ), count in main_routes.most_common()
        ],
        "search_targets": [
            {"effect": effect, "target": target, "count": count}
            for (effect, target), count in searches.most_common()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze decision timing in Mega Lopunny Kaggle replays.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    replays = read_replays(args.path)
    deck = infer_target_deck(replays, args.target_name)
    rows: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for replay in replays:
        index = target_index(replay, args.target_name, deck)
        if index is None:
            continue
        replay_rows, game = analyze_replay(replay, index)
        rows.extend(replay_rows)
        games.append(game)
    result = summarize(rows, games)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
