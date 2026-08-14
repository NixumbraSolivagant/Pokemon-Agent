from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ARCHETYPES: tuple[tuple[str, frozenset[int]], ...] = (
    ("archaludon_cinderace", frozenset({169, 190, 666})),
    ("hydrapple", frozenset({149, 150})),
    ("iono_box", frozenset({265, 266, 267, 268, 269, 270, 271})),
    ("mega_abomasnow", frozenset({722, 723})),
    ("mega_froslass", frozenset({860, 861})),
    ("mega_kangaskhan", frozenset({756})),
    ("hop_trevenant", frozenset({878, 879})),
    ("crustle", frozenset({344, 345})),
    ("alakazam", frozenset({741, 742, 743})),
    ("grimmsnarl", frozenset({646, 647, 648})),
    ("garchomp", frozenset({379, 380, 381})),
    ("mega_starmie", frozenset({1031})),
    ("mega_lucario", frozenset({677, 678})),
    ("dragapult", frozenset({119, 120, 121})),
    ("great_tusk", frozenset({58})),
)

ENERGY_IDS = frozenset(range(1, 21))
GREAT_TUSK = 58
LAND_COLLAPSE = 62


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def deck_hash(deck: Iterable[int]) -> str:
    payload = ",".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def classify_deck(deck: list[int]) -> str:
    cards = set(deck)
    matches = [(name, len(cards & core), len(core)) for name, core in ARCHETYPES if cards & core]
    if matches:
        matches.sort(key=lambda item: (item[1] / item[2], item[1]), reverse=True)
        return matches[0][0]
    counts = Counter(card_id for card_id in deck if card_id not in ENERGY_IDS)
    signature = "-".join(str(card_id) for card_id, _ in counts.most_common(3))
    return f"other:{signature or 'unknown'}"


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        digits, offset = suffix.split("+", 1)
        normalized = f"{prefix}.{digits[:6]:0<6}+{offset}"
    return datetime.fromisoformat(normalized)


def player_index(episode: dict[str, Any], submission_id: int) -> int:
    for index, agent in enumerate(episode.get("agents", [])):
        if int(agent.get("submissionId", -1)) == submission_id:
            return int(agent.get("index", index))
    raise ValueError(f"submission {submission_id} not found in episode {episode.get('id')}")


def full_deck(replay: dict[str, Any], index: int) -> list[int]:
    for step in replay.get("steps", [])[1:4]:
        action = step[index].get("action")
        if isinstance(action, list) and len(action) == 60 and all(isinstance(card_id, int) for card_id in action):
            return action
    return []


def terminal_current(replay: dict[str, Any], index: int) -> dict[str, Any]:
    for step in reversed(replay.get("steps", [])):
        observation = step[index].get("observation") or {}
        current = observation.get("current")
        if isinstance(current, dict):
            return current
    return {}


def remaining_overage(replay: dict[str, Any], index: int) -> tuple[float | None, float | None]:
    values = []
    for step in replay.get("steps", []):
        observation = step[index].get("observation") or {}
        value = observation.get("remainingOverageTime")
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None, None
    return values[-1], min(values)


def unique_events(replay: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    events: list[dict[str, Any]] = []
    for step in replay.get("steps", []):
        for agent in step:
            observation = agent.get("observation") or {}
            for event in observation.get("logs") or []:
                key = json.dumps(event, sort_keys=True, separators=(",", ":"))
                if key not in seen:
                    seen.add(key)
                    events.append(event)
    return events


def timeline_events(replay: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    events: list[dict[str, Any]] = []
    for step_index, step in enumerate(replay.get("steps", [])):
        for agent in step:
            observation = agent.get("observation") or {}
            current = observation.get("current") or {}
            for event in observation.get("logs") or []:
                key = json.dumps(event, sort_keys=True, separators=(",", ":"))
                if key in seen:
                    continue
                seen.add(key)
                events.append({**event, "step": step_index, "turn": current.get("turn")})
    return events


def attack_metrics(events: list[dict[str, Any]], player: int) -> dict[str, int | None]:
    attacks = [event for event in events if event.get("type") == 15 and int(event.get("playerIndex", -1)) == player]
    collapses = [event for event in attacks if int(event.get("attackId", -1)) == LAND_COLLAPSE]
    return {
        "attack_count": len(attacks),
        "land_collapse_count": len(collapses),
        "first_land_collapse_turn": collapses[0].get("turn") if collapses else None,
        "first_land_collapse_step": collapses[0].get("step") if collapses else None,
    }


def first_ready_tusk_turn(replay: dict[str, Any], player_index: int) -> int | None:
    for step in replay.get("steps", []):
        observation = step[player_index].get("observation") or {}
        current = observation.get("current") or {}
        players = current.get("players") or []
        if player_index >= len(players):
            continue
        player = players[player_index] or {}
        field = list(player.get("active") or []) + list(player.get("bench") or [])
        for pokemon in field:
            if not isinstance(pokemon, dict) or int(pokemon.get("id", -1)) != GREAT_TUSK:
                continue
            energies = pokemon.get("energyCards") or pokemon.get("energies") or []
            if len(energies) >= 2:
                return current.get("turn")
    return None


def classify_failure(row: dict[str, Any]) -> list[str]:
    if row.get("reward", 0) >= 0:
        return []
    labels: list[str] = []
    target_deck = row.get("target_deck_remaining")
    opponent_deck = row.get("opponent_deck_remaining")
    collapse_count = int(row.get("land_collapse_count") or 0)
    first_collapse = row.get("first_land_collapse_turn")
    if target_deck == 0 and isinstance(opponent_deck, int) and opponent_deck > 0:
        labels.append("self_deckout")
    if collapse_count == 0:
        labels.append("no_land_collapse")
    elif isinstance(first_collapse, int) and first_collapse >= 12:
        labels.append("late_land_collapse")
    if row.get("target_prizes_remaining") == 6:
        labels.append("no_prizes_taken")
    if row.get("opponent_archetype") in {"crustle", "great_tusk"}:
        labels.append("wall_or_mirror")
    if row.get("opponent_archetype") == "mega_abomasnow" and "self_deckout" in labels:
        labels.append("abomasnow_route_failure")
    if not labels:
        labels.append("prize_race_or_other")
    return labels


def last_attack(events: list[dict[str, Any]], attacker: int) -> tuple[int | None, int | None]:
    for event in reversed(events):
        if event.get("type") == 15 and int(event.get("playerIndex", -1)) == attacker:
            return event.get("cardId"), event.get("attackId")
    return None, None


def terminal_player_values(current: dict[str, Any], index: int) -> tuple[int | None, int | None, int | None]:
    players = current.get("players") or []
    if index >= len(players):
        return None, None, None
    player = players[index] or {}
    prize = player.get("prize")
    active = player.get("active") or []
    active_card = active[0].get("id") if active and isinstance(active[0], dict) else None
    return len(prize) if isinstance(prize, list) else None, player.get("deckCount"), active_card


def build_rows(log_root: Path, submission_ids: list[int]) -> list[dict[str, Any]]:
    team_by_id: dict[int, str] = {}
    episode_records: dict[tuple[int, int], dict[str, Any]] = {}
    for submission_id in submission_ids:
        listing = load_json(log_root / "probes" / f"ListEpisodes_{submission_id}.json")
        team_by_id.update({int(team["id"]): team.get("teamName", "") for team in listing.get("teams", [])})
        for episode in listing.get("episodes", []):
            episode_records[(submission_id, int(episode["id"]))] = episode

    scores = {}
    submissions_path = log_root / "submissions.json"
    if submissions_path.exists():
        scores = {int(row["ref"]): row.get("publicScore") for row in load_json(submissions_path)}

    rows = []
    for (submission_id, episode_id), episode in sorted(episode_records.items()):
        replay_path = log_root / "replays" / f"{episode_id}.json"
        if not replay_path.exists():
            continue
        replay = load_json(replay_path)
        target_index = player_index(episode, submission_id)
        opponent_index = 1 - target_index
        target_agent = episode["agents"][target_index]
        opponent_agent = episode["agents"][opponent_index]
        target_deck = full_deck(replay, target_index)
        opponent_deck = full_deck(replay, opponent_index)
        current = terminal_current(replay, target_index)
        first_player = current.get("firstPlayer")
        target_prizes, target_deck_count, target_active = terminal_player_values(current, target_index)
        opponent_prizes, opponent_deck_count, opponent_active = terminal_player_values(current, opponent_index)
        end_overage, min_overage = remaining_overage(replay, target_index)
        reward = int(target_agent.get("reward", replay.get("rewards", [0, 0])[target_index] or 0))
        events = unique_events(replay)
        timed_events = timeline_events(replay)
        metrics = attack_metrics(timed_events, target_index)
        finisher_card, finisher_attack = last_attack(events, opponent_index if reward < 0 else target_index)
        start = parse_time(episode.get("createTime"))
        end = parse_time(episode.get("endTime"))
        row = {
                "submission_id": submission_id,
                "public_score": scores.get(submission_id),
                "episode_id": episode_id,
                "reward": reward,
                "result": "win" if reward > 0 else "loss" if reward < 0 else "draw",
                "self_play": int(opponent_agent.get("submissionId", -1)) == submission_id,
                "initial_score": target_agent.get("initialScore"),
                "updated_score": target_agent.get("updatedScore"),
                "opponent_initial_score": opponent_agent.get("initialScore"),
                "opponent_updated_score": opponent_agent.get("updatedScore"),
                "target_index": target_index,
                "went_first": first_player == target_index,
                "steps": len(replay.get("steps", [])),
                "turn": current.get("turn"),
                "duration_seconds": (end - start).total_seconds() if start and end else None,
                "target_status": replay.get("statuses", [None, None])[target_index],
                "opponent_status": replay.get("statuses", [None, None])[opponent_index],
                "target_team": team_by_id.get(int(target_agent.get("teamId", -1)), ""),
                "opponent_team": team_by_id.get(int(opponent_agent.get("teamId", -1)), ""),
                "opponent_submission_id": opponent_agent.get("submissionId"),
                "target_deck_hash": deck_hash(target_deck),
                "opponent_deck_hash": deck_hash(opponent_deck),
                "opponent_archetype": classify_deck(opponent_deck),
                "opponent_deck": json.dumps(Counter(opponent_deck), sort_keys=True),
                "target_prizes_remaining": target_prizes,
                "opponent_prizes_remaining": opponent_prizes,
                "target_deck_remaining": target_deck_count,
                "opponent_deck_remaining": opponent_deck_count,
                "target_active_card": target_active,
                "opponent_active_card": opponent_active,
                "finisher_card": finisher_card,
                "finisher_attack": finisher_attack,
                "end_overage_seconds": end_overage,
                "min_overage_seconds": min_overage,
                "first_ready_tusk_turn": first_ready_tusk_turn(replay, target_index),
                **metrics,
            }
        row["rating_delta"] = (
            float(row["updated_score"]) - float(row["initial_score"])
            if isinstance(row.get("updated_score"), (int, float)) and isinstance(row.get("initial_score"), (int, float))
            else None
        )
        failures = classify_failure(row)
        row["failure_class"] = failures[0] if failures else ""
        row["failure_tags"] = ",".join(failures)
        rows.append(row)
    return rows


def ratio(wins: int, games: int) -> float:
    return wins / games if games else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"submissions": {}}
    for submission_id in sorted({row["submission_id"] for row in rows}):
        all_rows = [row for row in rows if row["submission_id"] == submission_id]
        subset = [row for row in all_rows if not row["self_play"]]
        opponent_scores = [
            float(row["opponent_initial_score"])
            for row in subset
            if isinstance(row.get("opponent_initial_score"), (int, float))
        ]
        summary: dict[str, Any] = {
            "public_score": all_rows[0].get("public_score"),
            "games": len(subset),
            "self_play_games_excluded": len(all_rows) - len(subset),
            "wins": sum(row["reward"] > 0 for row in subset),
            "losses": sum(row["reward"] < 0 for row in subset),
            "draws": sum(row["reward"] == 0 for row in subset),
            "by_archetype": {},
            "by_first_player": {},
            "by_archetype_and_order": {},
            "failure_classes": Counter(
                tag
                for row in subset
                if row["reward"] < 0
                for tag in str(row.get("failure_tags", "")).split(",")
                if tag
            ).most_common(),
            "loss_finishers": Counter(
                str(row["finisher_card"]) for row in subset if row["reward"] < 0 and row["finisher_card"] is not None
            ).most_common(),
            "mean_opponent_initial_score": sum(opponent_scores) / len(opponent_scores) if opponent_scores else None,
        }
        summary["win_rate"] = ratio(summary["wins"], summary["games"])
        for archetype in sorted({row["opponent_archetype"] for row in subset}):
            group = [row for row in subset if row["opponent_archetype"] == archetype]
            wins = sum(row["reward"] > 0 for row in group)
            summary["by_archetype"][archetype] = {
                "games": len(group),
                "wins": wins,
                "losses": sum(row["reward"] < 0 for row in group),
                "win_rate": ratio(wins, len(group)),
                "mean_rating_delta": (
                    sum(float(row["rating_delta"]) for row in group if isinstance(row.get("rating_delta"), (int, float)))
                    / sum(isinstance(row.get("rating_delta"), (int, float)) for row in group)
                    if any(isinstance(row.get("rating_delta"), (int, float)) for row in group)
                    else None
                ),
            }
        for label, went_first in (("first", True), ("second", False)):
            group = [row for row in subset if row["went_first"] is went_first]
            wins = sum(row["reward"] > 0 for row in group)
            summary["by_first_player"][label] = {
                "games": len(group),
                "wins": wins,
                "losses": sum(row["reward"] < 0 for row in group),
                "win_rate": ratio(wins, len(group)),
            }
        for archetype in sorted({row["opponent_archetype"] for row in subset}):
            summary["by_archetype_and_order"][archetype] = {}
            for label, went_first in (("first", True), ("second", False)):
                group = [
                    row for row in subset
                    if row["opponent_archetype"] == archetype and row["went_first"] is went_first
                ]
                wins = sum(row["reward"] > 0 for row in group)
                summary["by_archetype_and_order"][archetype][label] = {
                    "games": len(group),
                    "wins": wins,
                    "losses": sum(row["reward"] < 0 for row in group),
                    "win_rate": ratio(wins, len(group)),
                }
        report["submissions"][str(submission_id)] = summary
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine downloaded Kaggle episode replays without running local matches.")
    parser.add_argument("--log-root", type=Path, default=Path("output/kaggle_logs"))
    parser.add_argument("--submission-ids", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/kaggle_logs/report"))
    args = parser.parse_args()

    rows = build_rows(args.log_root, args.submission_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "episodes.csv", rows)
    (args.out / "summary.json").write_text(json.dumps(aggregate(rows), indent=2, ensure_ascii=False))
    print(json.dumps(aggregate(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
