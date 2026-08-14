from __future__ import annotations

import argparse
import atexit
import copy
import importlib.util
import json
import shutil
import statistics
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from agents.meta_runtime import energy_count, option_card, option_target
from tools.replay_imitation import infer_target_deck, read_replays, target_index


_AGENT_TEMP_DIRS: list[tempfile.TemporaryDirectory] = []
atexit.register(lambda: [entry.cleanup() for entry in _AGENT_TEMP_DIRS if entry])


def load_agent(package: Path) -> ModuleType:
    temp_dir = tempfile.TemporaryDirectory(prefix="behavior-audit-")
    _AGENT_TEMP_DIRS.append(temp_dir)
    root = Path(temp_dir.name)
    with tarfile.open(package, "r:gz") as archive:
        archive.extractall(root)
    sys.path.insert(0, str(root))
    sys.modules.pop("meta_runtime", None)
    for path in root.rglob("main.py"):
        spec = importlib.util.spec_from_file_location(f"audit_agent_{id(path)}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError(f"main.py missing from {package}")


def option_group(observation: dict[str, Any], action: list[int]) -> str:
    try:
        obs = to_observation_class(observation)
        if obs.select is None:
            return "reset"
        selected = [obs.select.option[index] for index in action if 0 <= index < len(obs.select.option)]
        if not selected:
            return SelectContext(obs.select.context).name
        option = selected[0]
        if option.type == OptionType.ATTACK:
            return f"attack:{option.attackId}"
        if option.type == OptionType.EVOLVE:
            return "evolve"
        if option.type == OptionType.ATTACH:
            return "attach"
        if option.type == OptionType.RETREAT:
            return "retreat"
        if option.type == OptionType.ABILITY:
            return "ability"
        if option.type == OptionType.PLAY:
            return "play"
        return OptionType(option.type).name.lower()
    except Exception:
        return "unknown"


def enum_value(value: Any) -> int | None:
    if value is None:
        return None
    return int(value.value if hasattr(value, "value") else value)


def card_signature(card: Any, area: Any = None, include_state: bool = True) -> tuple[Any, ...] | None:
    if card is None:
        return None
    signature: list[Any] = [int(getattr(card, "id", 0) or 0)]
    if hasattr(card, "hp") and include_state:
        signature.extend([
            enum_value(area),
            int(getattr(card, "hp", 0) or 0),
            int(getattr(card, "maxHp", 0) or 0),
            energy_count(card),
            tuple(sorted(int(getattr(tool, "id", 0) or 0) for tool in (getattr(card, "tools", None) or []))),
            tuple(sorted(int(value) for value in (getattr(card, "energies", None) or []))),
        ])
    return tuple(signature)


def option_signature(obs: Any, option: Any, include_state: bool = True) -> tuple[Any, ...]:
    option_type = enum_value(option.type)
    card = option_card(obs, option)
    target = option_target(obs, option)
    card_area = option.area
    target_area = option.inPlayArea
    player_index = option.playerIndex if option.playerIndex is not None else obs.current.yourIndex
    target_owner = player_index if target is not None else None
    return (
        option_type,
        card_signature(card, card_area, include_state),
        card_signature(target, target_area, include_state),
        target_owner,
        int(option.attackId or 0),
        int(option.cardId or 0),
        int(option.number or 0),
        int(option.count or 0),
        enum_value(option.specialConditionType),
    )


def action_signature(obs: Any, action: tuple[int, ...], include_state: bool = True) -> tuple[Any, ...]:
    signatures = [
        option_signature(obs, obs.select.option[index], include_state)
        for index in action
        if 0 <= index < len(obs.select.option)
    ]
    return tuple(sorted(signatures, key=repr))


def describe_card(card: Any, area: Any = None) -> dict[str, Any] | None:
    if card is None:
        return None
    result = {
        "id": int(getattr(card, "id", 0) or 0),
        "serial": int(getattr(card, "serial", 0) or 0),
        "area": enum_value(area),
    }
    if hasattr(card, "hp"):
        result.update({
            "hp": int(getattr(card, "hp", 0) or 0),
            "max_hp": int(getattr(card, "maxHp", 0) or 0),
            "energy_count": energy_count(card),
            "energies": [int(value) for value in (getattr(card, "energies", None) or [])],
            "tools": [int(getattr(tool, "id", 0) or 0) for tool in (getattr(card, "tools", None) or [])],
        })
    return result


def describe_option(obs: Any, index: int) -> dict[str, Any]:
    option = obs.select.option[index]
    return {
        "index": index,
        "type": OptionType(option.type).name,
        "card": describe_card(option_card(obs, option), option.area),
        "target": describe_card(option_target(obs, option), option.inPlayArea),
        "player_index": option.playerIndex,
        "attack_id": int(option.attackId or 0),
        "card_id": int(option.cardId or 0),
        "number": option.number,
        "count": option.count,
    }


def board_summary(obs: Any) -> dict[str, Any]:
    state = obs.current
    players = []
    for player_index, player in enumerate(state.players):
        players.append({
            "player_index": player_index,
            "active": [describe_card(card, AreaType.ACTIVE) for card in (player.active or []) if card is not None],
            "bench": [describe_card(card, AreaType.BENCH) for card in (player.bench or []) if card is not None],
            "hand_count": int(getattr(player, "handCount", 0) or len(player.hand or [])),
            "deck_count": int(player.deckCount or 0),
            "prize_count": int(getattr(player, "prizeCount", 0) or len(player.prize or [])),
        })
    return {
        "your_index": int(state.yourIndex),
        "turn_action_count": int(state.turnActionCount or 0),
        "supporter_played": bool(state.supporterPlayed),
        "energy_attached": bool(state.energyAttached),
        "retreated": bool(state.retreated),
        "players": players,
    }


def audit(
    package: Path,
    replay_dir: Path,
    target_name: str,
    episode_ids: set[str] | None = None,
    replays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    agent = load_agent(package)
    runtime = sys.modules.get("meta_runtime")
    if replays is None:
        replays = read_replays(replay_dir)
    if episode_ids is not None:
        replays = [replay for replay in replays if str(replay.get("_local_episode_id")) in episode_ids]
    target_deck = infer_target_deck(replays, target_name)
    totals = Counter()
    by_group: dict[str, Counter] = defaultdict(Counter)
    turn_actions: dict[tuple[str, int], dict[str, set[str]]] = defaultdict(
        lambda: {"actual": set(), "predicted": set()}
    )
    stopping = Counter()
    mismatches: list[dict[str, Any]] = []
    games = 0
    wins = 0
    for replay in replays:
        index = target_index(replay, target_name, target_deck)
        if index is None:
            continue
        games += 1
        if runtime is not None:
            reset = getattr(runtime, "_reset_policy_memory", None)
            if callable(reset):
                reset()
            getattr(runtime, "_LOPUNNY_MEMORY", {}).update(
                turn=None, active_serial=None, bench_serials=set(), moved=False
            )
            getattr(runtime, "_OGERPON_MEMORY", {}).update(turn=None, primary_serial=None)
        wins += int((replay.get("rewards") or [0, 0])[index] > 0)
        previous: dict[str, Any] | None = None
        for step_number, step in enumerate(replay.get("steps", [])):
            if index >= len(step):
                continue
            agent_step = step[index] or {}
            actual = agent_step.get("action")
            observation = previous
            previous = agent_step.get("observation") or None
            if observation is None or not isinstance(actual, list) or not actual:
                continue
            try:
                obs = to_observation_class(observation)
                if obs.select is None or len(actual) == 60 or len(obs.select.option) <= 1:
                    continue
                memory = copy.deepcopy(getattr(runtime, "_POLICY_MEMORY", {})) if runtime is not None else None
                predicted = agent.agent(observation)
                if not isinstance(predicted, list):
                    predicted = []
                if runtime is not None and memory is not None:
                    policy_memory = getattr(runtime, "_POLICY_MEMORY", None)
                    if isinstance(policy_memory, dict):
                        policy_memory.clear()
                        policy_memory.update(memory)
                    runtime_obs = runtime.to_observation_class(observation)
                    runtime._record_policy_action(runtime_obs, set(actual))
                actual_set = tuple(sorted(int(value) for value in actual if isinstance(value, int)))
                predicted_set = tuple(sorted(int(value) for value in predicted if isinstance(value, int)))
                group = option_group(observation, list(actual_set))
                semantic_actual = action_signature(obs, actual_set)
                semantic_predicted = action_signature(obs, predicted_set)
                coarse_actual = action_signature(obs, actual_set, include_state=False)
                coarse_predicted = action_signature(obs, predicted_set, include_state=False)
                exact_match = actual_set == predicted_set
                semantic_match = semantic_actual == semantic_predicted
                coarse_match = coarse_actual == coarse_predicted
                if SelectContext(obs.select.context) == SelectContext.MAIN:
                    actual_terminal = any(
                        obs.select.option[index].type in {OptionType.ATTACK, OptionType.END}
                        for index in actual_set if 0 <= index < len(obs.select.option)
                    )
                    predicted_terminal = any(
                        obs.select.option[index].type in {OptionType.ATTACK, OptionType.END}
                        for index in predicted_set if 0 <= index < len(obs.select.option)
                    )
                    stopping[(actual_terminal, predicted_terminal)] += 1
                totals["decisions"] += 1
                totals["exact"] += int(exact_match)
                totals["semantic"] += int(semantic_match)
                totals["coarse"] += int(coarse_match)
                by_group[group]["decisions"] += 1
                by_group[group]["exact"] += int(exact_match)
                by_group[group]["semantic"] += int(semantic_match)
                by_group[group]["coarse"] += int(coarse_match)
                if group in {"play", "attach", "evolve", "ability", "retreat", "end"} or group.startswith("attack:"):
                    key = (str(replay.get("id")), int(obs.current.turn or 0))
                    turn_actions[key]["actual"].add(repr(semantic_actual))
                    turn_actions[key]["predicted"].add(repr(semantic_predicted))
                if not semantic_match and len(mismatches) < 250:
                    mismatches.append({
                        "game": replay.get("id"),
                        "step": step_number,
                        "turn": int(obs.current.turn or 0),
                        "group": group,
                        "actual": list(actual_set),
                        "predicted": list(predicted_set),
                        "option_count": len(obs.select.option),
                        "actual_options": [describe_option(obs, index) for index in actual_set if 0 <= index < len(obs.select.option)],
                        "predicted_options": [describe_option(obs, index) for index in predicted_set if 0 <= index < len(obs.select.option)],
                        "options": [describe_option(obs, index) for index in range(len(obs.select.option))],
                        "board": board_summary(obs),
                    })
            except Exception:
                totals["errors"] += 1
    groups = {}
    for name, row in by_group.items():
        groups[name] = {
            "decisions": row["decisions"],
            "exact": row["exact"],
            "exact_rate": row["exact"] / max(1, row["decisions"]),
            "semantic": row["semantic"],
            "semantic_rate": row["semantic"] / max(1, row["decisions"]),
            "coarse": row["coarse"],
            "coarse_rate": row["coarse"] / max(1, row["decisions"]),
        }
    main_groups = {
        name: row for name, row in groups.items()
        if name in {"play", "attach", "evolve", "ability", "retreat", "end"} or name.startswith("attack:")
    }
    recalls = [float(row["semantic_rate"]) for row in main_groups.values() if int(row["decisions"]) > 0]
    attack_decisions = sum(int(row["decisions"]) for name, row in main_groups.items() if name.startswith("attack:"))
    attack_matches = sum(int(row["semantic"]) for name, row in main_groups.items() if name.startswith("attack:"))
    terminal_decisions = attack_decisions + int(main_groups.get("end", {}).get("decisions", 0))
    terminal_matches = attack_matches + int(main_groups.get("end", {}).get("semantic", 0))
    continue_recall = stopping[(False, False)] / max(1, stopping[(False, False)] + stopping[(False, True)])
    stop_recall = stopping[(True, True)] / max(1, stopping[(True, True)] + stopping[(True, False)])
    jaccards = []
    for row in turn_actions.values():
        union = row["actual"] | row["predicted"]
        jaccards.append(len(row["actual"] & row["predicted"]) / max(1, len(union)))
    return {
        "package": str(package),
        "episode_filter": sorted(episode_ids) if episode_ids is not None else None,
        "replays": games,
        "wins": wins,
        "decisions": totals["decisions"],
        "exact": totals["exact"],
        "exact_rate": totals["exact"] / max(1, totals["decisions"]),
        "semantic": totals["semantic"],
        "semantic_rate": totals["semantic"] / max(1, totals["decisions"]),
        "coarse": totals["coarse"],
        "coarse_rate": totals["coarse"] / max(1, totals["decisions"]),
        "errors": totals["errors"],
        "by_group": dict(sorted(groups.items(), key=lambda item: item[1]["decisions"], reverse=True)),
        "turn_fidelity": {
            "turns": len(jaccards),
            "action_jaccard_median": statistics.median(jaccards) if jaccards else 0.0,
            "main_macro_recall": sum(recalls) / max(1, len(recalls)),
            "attack_recall": attack_matches / max(1, attack_decisions),
            "attack_decisions": attack_decisions,
            "ability_recall": float(main_groups.get("ability", {}).get("semantic_rate", 1.0)),
            "terminal_recall": terminal_matches / max(1, terminal_decisions),
            "terminal_decisions": terminal_decisions,
            "stop_balanced_accuracy": 0.5 * (continue_recall + stop_recall),
            "stop_recall": stop_recall,
            "continue_recall": continue_recall,
        },
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit exact action agreement between a package and teacher replays.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--episode-ids", type=Path, help="Optional JSON list of locked episode ids.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    episode_ids = None
    if args.episode_ids:
        episode_ids = {str(value).split(":")[-1] for value in json.loads(args.episode_ids.read_text(encoding="utf-8"))}
    result = audit(args.package, args.replays, args.target_name, episode_ids)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
