from __future__ import annotations

import multiprocessing as mp
import random
import json
import hashlib
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .archive import safe_extract_tar_gz
from .models import EvalConfig, GameResult
from .player import PlayerProcess


class InvalidAction(ValueError):
    pass


BASIC_ENERGY_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
ACE_SPEC_IDS = {
    10,
    12,
    13,
    1080,
    1082,
    1085,
    1088,
    1089,
    1092,
    1093,
    1095,
    1096,
    1100,
    1104,
    1107,
    1109,
    1110,
    1111,
    1125,
    1126,
    1128,
    1155,
    1158,
    1159,
    1165,
    1167,
    1169,
    1247,
    1249,
}


def validate_action(obs: dict[str, Any], action: Any) -> list[int]:
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        raise InvalidAction("action must be list[int]")
    select = obs.get("select") or {}
    options = select.get("option") or []
    min_count = int(select.get("minCount", 0) or 0)
    max_count = int(select.get("maxCount", len(options)) or 0)
    if len(action) < min_count or len(action) > max_count:
        raise InvalidAction(f"action length {len(action)} outside [{min_count}, {max_count}]")
    if len(set(action)) != len(action):
        raise InvalidAction("action contains duplicate indices")
    if any(i < 0 or i >= len(options) for i in action):
        raise InvalidAction(f"action index outside [0, {len(options)})")
    return action


def validate_deck(deck: Any) -> list[int]:
    if not isinstance(deck, list) or len(deck) != 60 or not all(type(i) is int for i in deck):
        raise InvalidAction("deck selection must return exactly 60 ints")
    if any(card_id <= 0 for card_id in deck):
        raise InvalidAction("deck card ids must be positive ints")
    counts = Counter(deck)
    over_limit = [card_id for card_id, count in counts.items() if card_id not in BASIC_ENERGY_IDS and count > 4]
    if over_limit:
        raise InvalidAction(f"non-basic cards exceed four-copy limit: {sorted(over_limit)}")
    ace_specs = [card_id for card_id in deck if card_id in ACE_SPEC_IDS]
    if len(ace_specs) > 1:
        raise InvalidAction(f"deck cannot contain more than one ACE SPEC card: {sorted(ace_specs)}")
    return deck


def play_game(
    p0_tarball: str,
    p1_tarball: str,
    p0_name: str,
    p1_name: str,
    seed: int,
    game_id: str,
    config: EvalConfig,
    project_root: str | Path,
) -> GameResult:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    with tempfile.TemporaryDirectory(prefix="pokemon_local_eval_trace_") as trace_tmp:
        trace_path = Path(trace_tmp) / f"{game_id}.jsonl"
        proc = ctx.Process(
            target=_play_game_worker,
            args=(
                p0_tarball,
                p1_tarball,
                p0_name,
                p1_name,
                seed,
                game_id,
                config,
                str(project_root),
                str(trace_path),
                queue,
            ),
        )
        started = time.perf_counter()
        proc.start()
        proc.join(timeout=config.run_timeout_s + 10.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
            trace = _read_trace(trace_path)
            result = GameResult(
                game_id=game_id,
                p0=p0_name,
                p1=p1_name,
                seed=seed,
                result=None,
                outcome="NO_RESULT",
                winner=None,
                loser=None,
                reason="RUN_TIMEOUT",
                actions=max(0, len([e for e in trace if e.get("event") == "action"])),
                duration_s=time.perf_counter() - started,
                error=f"Game exceeded run timeout {config.run_timeout_s}s",
                trace=trace,
            )
            if not _should_keep_trace(result, config):
                result.trace = []
            return result
        if queue.empty():
            result = GameResult(
                game_id=game_id,
                p0=p0_name,
                p1=p1_name,
                seed=seed,
                result=None,
                outcome="NO_RESULT",
                winner=None,
                loser=None,
                reason="REFEREE_CRASH",
                actions=0,
                duration_s=time.perf_counter() - started,
                error=f"Worker exited with code {proc.exitcode}",
                trace=_read_trace(trace_path),
            )
            if not _should_keep_trace(result, config):
                result.trace = []
            return result
        result = queue.get()
        result.trace = _read_trace(trace_path) if _should_keep_trace(result, config) else []
        return result


def _play_game_worker(
    p0_tarball: str,
    p1_tarball: str,
    p0_name: str,
    p1_name: str,
    seed: int,
    game_id: str,
    config: EvalConfig,
    project_root: str,
    trace_path: str,
    queue: mp.Queue,
) -> None:
    started = time.perf_counter()
    p0: PlayerProcess | None = None
    p1: PlayerProcess | None = None
    actions = 0
    trace_file = Path(trace_path)
    try:
        random.seed(seed)
        with tempfile.TemporaryDirectory(prefix="pokemon_local_eval_") as tmp:
            tmp_path = Path(tmp)
            p0_dir = safe_extract_tar_gz(p0_tarball, tmp_path / "p0")
            p1_dir = safe_extract_tar_gz(p1_tarball, tmp_path / "p1")

            sys.path.insert(0, str(p0_dir))
            from cg.game import battle_finish, battle_select, battle_start

            p0 = PlayerProcess(p0_name, p0_dir, Path(project_root), config.act_timeout_s, config.import_timeout_s)
            p1 = PlayerProcess(p1_name, p1_dir, Path(project_root), config.act_timeout_s, config.import_timeout_s)
            try:
                p0.start()
            except Exception as exc:
                queue.put(_forfeit(game_id, p0_name, p1_name, seed, 0, "IMPORT_ERROR", actions, started, str(exc)))
                return
            try:
                p1.start()
            except Exception as exc:
                queue.put(_forfeit(game_id, p0_name, p1_name, seed, 1, "IMPORT_ERROR", actions, started, str(exc)))
                return
            try:
                deck0 = validate_deck(p0.request({"kind": "deck"}, config.deck_timeout_s)[0])
            except Exception as exc:
                queue.put(_forfeit(game_id, p0_name, p1_name, seed, 0, "DECK_ERROR", actions, started, str(exc)))
                return
            try:
                deck1 = validate_deck(p1.request({"kind": "deck"}, config.deck_timeout_s)[0])
            except Exception as exc:
                queue.put(_forfeit(game_id, p0_name, p1_name, seed, 1, "DECK_ERROR", actions, started, str(exc)))
                return
            _write_trace(
                trace_file,
                {
                    "event": "deck",
                    "p0": p0_name,
                    "p1": p1_name,
                    "p0_deck": deck0,
                    "p1_deck": deck1,
                },
            )
            overage = {0: config.overage_time_s, 1: config.overage_time_s}
            obs, _ = battle_start(deck0, deck1)
            if obs is None:
                raise RuntimeError("cg.battle_start returned None")

            try:
                while actions < config.max_actions:
                    cur = obs.get("current") or {}
                    result = cur.get("result", -1)
                    if result in (0, 1, 2):
                        queue.put(_finished(game_id, p0_name, p1_name, seed, int(result), actions, started))
                        return
                    player_idx = int(cur.get("yourIndex", 0))
                    actor = p0 if player_idx == 0 else p1
                    assert actor is not None
                    step_record: dict[str, Any] = {
                        "event": "action",
                        "step": actions,
                        "actor_seat": player_idx,
                        "actor": p0_name if player_idx == 0 else p1_name,
                        "observation": _summarize_observation(obs),
                    }
                    try:
                        allowed_s = config.act_timeout_s + max(0.0, overage[player_idx])
                        raw_action, duration = actor.request({"kind": "act", "obs": obs}, allowed_s)
                        overage[player_idx] -= max(0.0, duration - config.act_timeout_s)
                        if overage[player_idx] < 0:
                            raise TimeoutError(
                                f"remainingOverageTime below 0 after {duration:.3f}s action"
                            )
                        action = validate_action(obs, raw_action)
                        step_record["raw_action"] = raw_action
                        step_record["action"] = action
                        step_record["duration_s"] = duration
                        step_record["remaining_overage_s"] = overage[player_idx]
                        step_record["selected_options"] = _selected_options(obs, action)
                    except TimeoutError as exc:
                        step_record["error"] = str(exc)
                        _write_trace(trace_file, step_record)
                        queue.put(_forfeit(game_id, p0_name, p1_name, seed, player_idx, "TIMEOUT", actions, started, str(exc)))
                        return
                    except Exception as exc:
                        step_record["raw_action"] = locals().get("raw_action")
                        step_record["error"] = str(exc)
                        _write_trace(trace_file, step_record)
                        queue.put(_forfeit(game_id, p0_name, p1_name, seed, player_idx, "INVALID_ACTION", actions, started, str(exc)))
                        return
                    actions += 1
                    try:
                        obs = battle_select(action)
                        step_record["next_result"] = (obs.get("current") or {}).get("result", -1)
                        _write_trace(trace_file, step_record)
                    except Exception as exc:
                        step_record["error"] = str(exc)
                        _write_trace(trace_file, step_record)
                        queue.put(_forfeit(game_id, p0_name, p1_name, seed, player_idx, "ENGINE_REJECTED_ACTION", actions, started, str(exc)))
                        return
                queue.put(
                    GameResult(
                        game_id=game_id,
                        p0=p0_name,
                        p1=p1_name,
                        seed=seed,
                        result=None,
                        outcome="NO_RESULT",
                        winner=None,
                        loser=None,
                        reason="MAX_ACTIONS",
                        actions=actions,
                        duration_s=time.perf_counter() - started,
                    )
                )
            finally:
                try:
                    battle_finish()
                except Exception:
                    pass
    except Exception as exc:
        queue.put(
            GameResult(
                game_id=game_id,
                p0=p0_name,
                p1=p1_name,
                seed=seed,
                result=None,
                outcome="NO_RESULT",
                winner=None,
                loser=None,
                reason="REFEREE_CRASH",
                actions=actions,
                duration_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
    finally:
        if p0 is not None:
            p0.close()
        if p1 is not None:
            p1.close()


def _finished(
    game_id: str,
    p0: str,
    p1: str,
    seed: int,
    result: int,
    actions: int,
    started: float,
    trace: list[dict[str, Any]] | None = None,
) -> GameResult:
    if result == 0:
        return GameResult(game_id, p0, p1, seed, result, "P0_WIN", p0, p1, "RESULT", actions, time.perf_counter() - started, trace=trace or [])
    if result == 1:
        return GameResult(game_id, p0, p1, seed, result, "P1_WIN", p1, p0, "RESULT", actions, time.perf_counter() - started, trace=trace or [])
    return GameResult(game_id, p0, p1, seed, result, "DRAW", None, None, "RESULT", actions, time.perf_counter() - started, trace=trace or [])


def _forfeit(
    game_id: str,
    p0: str,
    p1: str,
    seed: int,
    bad_seat: int,
    reason: str,
    actions: int,
    started: float,
    error: str,
    trace: list[dict[str, Any]] | None = None,
) -> GameResult:
    if bad_seat == 0:
        return GameResult(game_id, p0, p1, seed, 1, "P0_LOSS", p1, p0, reason, actions, time.perf_counter() - started, error=error, trace=trace or [])
    return GameResult(game_id, p0, p1, seed, 0, "P1_LOSS", p0, p1, reason, actions, time.perf_counter() - started, error=error, trace=trace or [])


def _summarize_observation(obs: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    players = current.get("players") or []
    return {
        "turn": current.get("turn"),
        "turnActionCount": current.get("turnActionCount"),
        "yourIndex": current.get("yourIndex"),
        "firstPlayer": current.get("firstPlayer"),
        "result": current.get("result"),
        "supporterPlayed": current.get("supporterPlayed"),
        "stadiumPlayed": current.get("stadiumPlayed"),
        "energyAttached": current.get("energyAttached"),
        "retreated": current.get("retreated"),
        "stadium": _cards(current.get("stadium") or []),
        "players": [_summarize_player(player) for player in players],
        "logs_tail": (obs.get("logs") or [])[-20:],
        "select": {
            "type": select.get("type"),
            "context": select.get("context"),
            "minCount": select.get("minCount"),
            "maxCount": select.get("maxCount"),
            "remainDamageCounter": select.get("remainDamageCounter"),
            "remainEnergyCost": select.get("remainEnergyCost"),
            "option_count": len(select.get("option") or []),
            "deck_count": len(select.get("deck") or []),
            "contextCard": select.get("contextCard"),
            "effect": select.get("effect"),
        },
    }


def _summarize_player(player: dict[str, Any]) -> dict[str, Any]:
    discard = player.get("discard") or []
    hand = player.get("hand") or []
    return {
        "active": _pokemon(player.get("active") or []),
        "bench": _pokemon(player.get("bench") or []),
        "benchMax": player.get("benchMax"),
        "deckCount": player.get("deckCount"),
        "discardCount": len(discard),
        "discardTail": _cards(discard[-20:]),
        "prizeCount": len(player.get("prize") or []),
        "handCount": player.get("handCount"),
        "hand": _cards(hand[:20]),
        "conditions": {
            "poisoned": player.get("poisoned"),
            "burned": player.get("burned"),
            "asleep": player.get("asleep"),
            "paralyzed": player.get("paralyzed"),
            "confused": player.get("confused"),
        },
    }


def _cards(cards: list[Any]) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    for card in cards:
        if card is None:
            out.append(None)
        else:
            out.append({"id": card.get("id"), "serial": card.get("serial"), "playerIndex": card.get("playerIndex")})
    return out


def _pokemon(pokemon_list: list[Any]) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    for pokemon in pokemon_list:
        if pokemon is None:
            out.append(None)
            continue
        out.append(
            {
                "id": pokemon.get("id"),
                "serial": pokemon.get("serial"),
                "hp": pokemon.get("hp"),
                "maxHp": pokemon.get("maxHp"),
                "appearThisTurn": pokemon.get("appearThisTurn"),
                "energies": pokemon.get("energies") or [],
                "energyCards": _cards(pokemon.get("energyCards") or []),
                "tools": _cards(pokemon.get("tools") or []),
                "preEvolution": _cards(pokemon.get("preEvolution") or []),
            }
        )
    return out


def _summarize_option(option: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "type",
        "number",
        "area",
        "index",
        "playerIndex",
        "toolIndex",
        "energyIndex",
        "count",
        "inPlayArea",
        "inPlayIndex",
        "attackId",
        "cardId",
        "serial",
        "specialConditionType",
    ]
    return {key: option.get(key) for key in keys if key in option}


def _selected_options(obs: dict[str, Any], action: list[int]) -> list[dict[str, Any]]:
    options = ((obs.get("select") or {}).get("option") or [])
    return [_summarize_option(options[index]) for index in action if 0 <= index < len(options)]


def _write_trace(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"event": "trace_decode_error", "raw": line})
    return out


def _should_keep_trace(result: GameResult, config: EvalConfig) -> bool:
    mode = (config.record_mode or "all").lower()
    if mode == "all":
        return True
    if mode == "none":
        return False
    focus = (config.record_focus or "").strip()
    if mode == "losses":
        if focus:
            return result.loser == focus or (result.reason != "RESULT" and focus in {result.p0, result.p1})
        return result.reason != "RESULT" or result.outcome == "NO_RESULT"
    if mode == "sample":
        if result.reason != "RESULT" or result.outcome == "NO_RESULT":
            return True
        rate = max(0.0, min(1.0, float(config.record_sample_rate)))
        if rate <= 0:
            return False
        digest = hashlib.sha256(f"{result.game_id}:{result.seed}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return bucket < rate
    return True
