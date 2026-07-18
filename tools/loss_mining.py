from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class Scenario:
    scenario_id: str
    source: str
    focus: str
    opponent: str
    focus_seat: int
    seed: int
    trigger_step: int
    trigger_turn: int
    prefix_actions: list[dict[str, Any]]
    p0_deck: list[int]
    p1_deck: list[int]
    value_before: float
    value_after: float
    drop: float
    tags: list[str]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_record_paths(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
            yield from sorted(path.rglob("*.json.gz"))
        elif path.exists():
            yield path


def read_record(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def player_value(obs: dict[str, Any], seat: int) -> float:
    current = obs.get("current") or obs
    players = current.get("players") or []
    if len(players) < 2:
        return 0.0
    me = players[seat] or {}
    opp = players[1 - seat] or {}
    my_deck = int(me.get("deckCount") or 0)
    opp_deck = int(opp.get("deckCount") or 0)
    my_prize = int(me.get("prizeCount") or 0)
    opp_prize = int(opp.get("prizeCount") or 0)
    my_hand = int(me.get("handCount") or 0)
    opp_hand = int(opp.get("handCount") or 0)
    my_active = (me.get("active") or [None])[0] or {}
    opp_active = (opp.get("active") or [None])[0] or {}
    my_hp = sum(int((pk or {}).get("hp") or 0) for pk in (me.get("active") or []) + (me.get("bench") or []))
    opp_hp = sum(int((pk or {}).get("hp") or 0) for pk in (opp.get("active") or []) + (opp.get("bench") or []))
    value = 0.0
    value += (opp_prize - my_prize) * 220.0
    value += (60 - opp_deck) * 80.0
    value -= (60 - my_deck) * 36.0
    value += (my_hand - opp_hand) * 8.0
    value += (my_hp - opp_hp) * 0.35
    if opp_deck <= 5:
        value += (6 - opp_deck) * 850.0
    if my_deck <= 5:
        value -= (6 - my_deck) * 850.0
    if my_active.get("id") == 58:
        value += 120.0 + 80.0 * len(my_active.get("energyCards") or [])
    return value


def ema(values: list[float], alpha: float = 0.35) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def event_tags(prev_obs: dict[str, Any], obs: dict[str, Any], seat: int) -> list[str]:
    prev = prev_obs.get("current") or prev_obs
    cur = obs.get("current") or obs
    prev_players = prev.get("players") or []
    cur_players = cur.get("players") or []
    if len(prev_players) < 2 or len(cur_players) < 2:
        return []
    me0, opp0 = prev_players[seat] or {}, prev_players[1 - seat] or {}
    me1, opp1 = cur_players[seat] or {}, cur_players[1 - seat] or {}
    tags: list[str] = []
    if int(me1.get("prizeCount") or 0) > int(me0.get("prizeCount") or 0):
        tags.append("focus_prize_regressed")
    if int(opp1.get("prizeCount") or 0) < int(opp0.get("prizeCount") or 0):
        tags.append("opponent_took_prize")
    if int(me1.get("deckCount") or 0) < int(me0.get("deckCount") or 0) - 2:
        tags.append("focus_deck_drop")
    if int(opp1.get("deckCount") or 0) > int(opp0.get("deckCount") or 0):
        tags.append("opponent_deck_recovered")
    active0 = (me0.get("active") or [None])[0] or {}
    active1 = (me1.get("active") or [None])[0] or {}
    if active0.get("id") != active1.get("id"):
        tags.append("focus_active_changed")
    if int(me1.get("handCount") or 0) < int(me0.get("handCount") or 0) - 2:
        tags.append("focus_hand_drop")
    if int(opp1.get("handCount") or 0) > int(opp0.get("handCount") or 0) + 2:
        tags.append("opponent_hand_surge")
    return tags


def mine_record(
    path: Path,
    record: dict[str, Any],
    focus: str = "",
    max_prefix_actions: int = 140,
    max_trigger_turn: int = 16,
    min_drop: float = 450.0,
) -> Scenario | None:
    game = record.get("game") or {}
    trace = game.get("trace") or []
    if not trace:
        return None
    loser = game.get("loser")
    if focus and loser != focus:
        return None
    if not loser:
        return None
    focus_seat = 0 if game.get("p0") == loser else 1
    deck_event = next((event for event in trace if event.get("event") == "deck"), None)
    if not deck_event:
        return None
    actions = [event for event in trace if event.get("event") == "action" and isinstance(event.get("action"), list)]
    if len(actions) < 4:
        return None
    values = [player_value(event.get("observation") or {}, focus_seat) for event in actions]
    smoothed = ema(values)
    best: tuple[float, int, list[str]] | None = None
    for idx in range(2, min(len(actions), max_prefix_actions)):
        obs = actions[idx].get("observation") or {}
        turn = int(obs.get("turn") or 0)
        if turn > max_trigger_turn:
            break
        drop = smoothed[idx - 2] - smoothed[idx]
        tags = event_tags(actions[idx - 1].get("observation") or {}, obs, focus_seat)
        gated = bool(tags) or drop >= min_drop * 1.6
        if drop >= min_drop and gated:
            if best is None or drop > best[0]:
                best = (drop, idx, tags)
    if best is None:
        return None
    drop, idx, tags = best
    trigger = actions[idx]
    prefix_actions = [
        {
            "actor_seat": int(event.get("actor_seat", 0)),
            "action": [int(x) for x in event.get("action", [])],
        }
        for event in actions[: idx + 1]
    ]
    scenario_id = f"{path.stem}_{game.get('game_id', 'game')}_{idx}"
    opponent = game.get("p1") if focus_seat == 0 else game.get("p0")
    return Scenario(
        scenario_id=scenario_id,
        source=str(path),
        focus=loser,
        opponent=str(opponent),
        focus_seat=focus_seat,
        seed=int(game.get("seed") or 0),
        trigger_step=int(trigger.get("step") or idx),
        trigger_turn=int((trigger.get("observation") or {}).get("turn") or 0),
        prefix_actions=prefix_actions,
        p0_deck=[int(x) for x in deck_event.get("p0_deck", [])],
        p1_deck=[int(x) for x in deck_event.get("p1_deck", [])],
        value_before=float(smoothed[max(0, idx - 2)]),
        value_after=float(smoothed[idx]),
        drop=float(drop),
        tags=tags,
        note="EMA value drop gated by concrete state-event tags.",
    )


def mine_losses(
    inputs: list[Path],
    out: Path,
    focus: str = "",
    limit: int = 64,
    max_prefix_actions: int = 140,
    max_trigger_turn: int = 16,
    min_drop: float = 450.0,
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in iter_record_paths(inputs):
        record = read_record(path)
        if not record:
            continue
        scenario = mine_record(path, record, focus, max_prefix_actions, max_trigger_turn, min_drop)
        if scenario is None:
            continue
        scenarios.append(scenario)
    scenarios.sort(key=lambda s: (s.drop, -s.trigger_turn, -s.trigger_step), reverse=True)
    scenarios = scenarios[: max(0, limit)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([scenario.to_dict() for scenario in scenarios], indent=2), encoding="utf-8")
    return scenarios


TAG_TO_MOTIFS = {
    "focus_hand_drop": ["draw_recovery", "hand_disruption"],
    "opponent_hand_surge": ["hand_disruption"],
    "opponent_took_prize": ["defensive_tools", "switch_pivot"],
    "focus_prize_regressed": ["defensive_tools"],
    "focus_deck_drop": ["draw_recovery", "resource_safety"],
    "opponent_deck_recovered": ["resource_denial"],
    "focus_active_changed": ["switch_pivot", "defensive_tools"],
}

TAG_TO_AVOID = {
    "focus_hand_drop": ["cut_draw_density"],
    "opponent_took_prize": ["cut_defensive_tools"],
    "focus_deck_drop": ["overdraw_packages"],
    "focus_active_changed": ["cut_switch_density"],
    "opponent_deck_recovered": ["cut_resource_denial"],
}


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    return [row for row in data if isinstance(row, dict)]


def digest_scenarios(scenarios: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    tags: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    focus: Counter[str] = Counter()
    drops_by_tag: dict[str, list[float]] = defaultdict(list)
    by_focus_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    drops_by_focus_opponent: dict[str, list[float]] = defaultdict(list)
    for scenario in scenarios:
        opponent = str(scenario.get("opponent", ""))
        focus_name = str(scenario.get("focus", ""))
        opponents.update([opponent])
        focus.update([focus_name])
        drop = float(scenario.get("drop") or 0.0)
        key = f"{focus_name}:::{opponent}"
        drops_by_focus_opponent[key].append(drop)
        for tag in scenario.get("tags") or []:
            tag = str(tag)
            tags[tag] += 1
            drops_by_tag[tag].append(drop)
            by_focus_opponent[key][tag] += 1
    motif_scores: Counter[str] = Counter()
    avoid_scores: Counter[str] = Counter()
    for tag, count in tags.items():
        for motif in TAG_TO_MOTIFS.get(tag, []):
            motif_scores[motif] += count
        for avoid in TAG_TO_AVOID.get(tag, []):
            avoid_scores[avoid] += count
    digest = {
        "scenario_count": len(scenarios),
        "warnings": [] if scenarios else ["no_scenarios_mined"],
        "top_tags": dict(tags.most_common(16)),
        "top_opponents": {k: v for k, v in opponents.most_common(12) if k},
        "top_focus": {k: v for k, v in focus.most_common(12) if k},
        "avg_drop_by_tag": {
            tag: sum(values) / len(values)
            for tag, values in sorted(drops_by_tag.items(), key=lambda item: len(item[1]), reverse=True)
        },
        "recommended_motifs": [name for name, _ in motif_scores.most_common(8)],
        "avoid_mutations": [name for name, _ in avoid_scores.most_common(8)],
        "failure_fingerprints": [
            {
                "focus": key.split(":::", 1)[0],
                "opponent": key.split(":::", 1)[1] if ":::" in key else "",
                "count": sum(counter.values()),
                "top_tags": dict(counter.most_common(8)),
                "avg_drop": sum(drops_by_focus_opponent[key]) / max(1, len(drops_by_focus_opponent[key])),
            }
            for key, counter in sorted(by_focus_opponent.items(), key=lambda item: sum(item[1].values()), reverse=True)[:24]
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    return digest


def digest_loss_files(scenario_paths: list[Path], out: Path) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for path in scenario_paths:
        scenarios.extend(load_scenarios(path))
    return digest_scenarios(scenarios, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine early/mid-game loss-collapse scenarios from local_eval game_records.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/discovery_gold/scenarios.json"))
    parser.add_argument("--focus", default="")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-prefix-actions", type=int, default=140)
    parser.add_argument("--max-trigger-turn", type=int, default=16)
    parser.add_argument("--min-drop", type=float, default=450.0)
    parser.add_argument("--digest-out", type=Path)
    args = parser.parse_args(argv)
    scenarios = mine_losses(
        args.inputs,
        args.out,
        args.focus,
        args.limit,
        args.max_prefix_actions,
        args.max_trigger_turn,
        args.min_drop,
    )
    result = {"out": str(args.out), "scenarios": len(scenarios)}
    if args.digest_out:
        result["digest"] = digest_scenarios([scenario.to_dict() for scenario in scenarios], args.digest_out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
