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

from .archive import cached_extract_tar_gz, safe_extract_tar_gz
from .models import EvalConfig, GameResult
from .player import PlayerProcess, PlayerProtocolError


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


def validate_kaggle_deck_action(deck: Any) -> list[int]:
    if not isinstance(deck, list):
        raise InvalidAction("deck selection must be an array")
    if len(deck) != 60:
        raise InvalidAction("deck selection must return exactly 60 ints")
    return deck


def _validate_deck_for_profile(deck: Any, config: EvalConfig) -> list[int]:
    if config.profile == "kaggle":
        return validate_kaggle_deck_action(deck)
    return validate_deck(deck)


def _validate_action_for_profile(obs: dict[str, Any], action: Any, config: EvalConfig) -> list[Any]:
    if config.profile == "kaggle":
        if not isinstance(action, list):
            raise InvalidAction("action must be an array")
        return action
    return validate_action(obs, action)


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
    record_trace = _should_record_before_game(game_id, seed, config)
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
                record_trace,
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
                reason="HARNESS_FAILURE" if config.profile == "kaggle" else "RUN_TIMEOUT",
                actions=max(0, len([e for e in trace if e.get("event") == "action"])),
                duration_s=time.perf_counter() - started,
                error=f"Game exceeded run timeout {config.run_timeout_s}s",
                trace=trace,
                failure_class="HARNESS_FAILURE",
                ranking_eligible=False,
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
                reason="HARNESS_FAILURE" if config.profile == "kaggle" else "REFEREE_CRASH",
                actions=0,
                duration_s=time.perf_counter() - started,
                error=f"Worker exited with code {proc.exitcode}",
                trace=_read_trace(trace_path),
                failure_class="HARNESS_FAILURE",
                ranking_eligible=False,
            )
            if not _should_keep_trace(result, config):
                result.trace = []
            return result
        result = queue.get()
        result.trace = _read_trace(trace_path) if _should_keep_trace(result, config) else []
        if _should_keep_trace(result, config) and not result.trace and result.reason != "RESULT":
            result.trace = [{"event": "diagnostic", "reason": result.reason, "error": result.error}]
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
    record_trace: bool,
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
            if config.archive_cache_dir:
                p0_dir = cached_extract_tar_gz(p0_tarball, config.archive_cache_dir)
                p1_dir = cached_extract_tar_gz(p1_tarball, config.archive_cache_dir)
            else:
                p0_dir = safe_extract_tar_gz(p0_tarball, tmp_path / "p0")
                p1_dir = safe_extract_tar_gz(p1_tarball, tmp_path / "p1")

            sys.path.insert(0, str(p0_dir))
            from cg.game import battle_finish, battle_select, battle_start

            player_logs = tmp_path / "player_logs"
            p0 = PlayerProcess(p0_name, p0_dir, Path(project_root), config.act_timeout_s, config.import_timeout_s, player_logs)
            p1 = PlayerProcess(p1_name, p1_dir, Path(project_root), config.act_timeout_s, config.import_timeout_s, player_logs)
            try:
                p0.start()
            except Exception as exc:
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 0, "IMPORT_ERROR", "ERROR", actions, started, str(exc), initial=True))
                return
            try:
                p1.start()
            except Exception as exc:
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 1, "IMPORT_ERROR", "ERROR", actions, started, str(exc), initial=True))
                return
            overage = {0: config.overage_time_s, 1: config.overage_time_s}
            try:
                deck_timeout0 = config.act_timeout_s + overage[0] if config.profile == "kaggle" else config.deck_timeout_s
                raw_deck0, deck_duration0 = p0.request({"kind": "deck"}, deck_timeout0)
                if config.profile == "kaggle":
                    overage[0] -= max(0.0, deck_duration0 - config.act_timeout_s)
                    if overage[0] < 0:
                        raise TimeoutError("remainingOverageTime below 0 after deck action")
                deck0 = _validate_deck_for_profile(raw_deck0, config)
            except TimeoutError as exc:
                reason = "DECK_TIMEOUT" if config.profile == "kaggle" else "DECK_ERROR"
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 0, reason, "TIMEOUT", actions, started, str(exc), initial=True))
                return
            except PlayerProtocolError as exc:
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 0, "DECK_ERROR", "ERROR", actions, started, str(exc), initial=True))
                return
            except Exception as exc:
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 0, "DECK_ERROR", "INVALID", actions, started, str(exc), initial=True))
                return
            try:
                deck_timeout1 = config.act_timeout_s + overage[1] if config.profile == "kaggle" else config.deck_timeout_s
                raw_deck1, deck_duration1 = p1.request({"kind": "deck"}, deck_timeout1)
                if config.profile == "kaggle":
                    overage[1] -= max(0.0, deck_duration1 - config.act_timeout_s)
                    if overage[1] < 0:
                        raise TimeoutError("remainingOverageTime below 0 after deck action")
                deck1 = _validate_deck_for_profile(raw_deck1, config)
            except TimeoutError as exc:
                reason = "DECK_TIMEOUT" if config.profile == "kaggle" else "DECK_ERROR"
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 1, reason, "TIMEOUT", actions, started, str(exc), initial=True))
                return
            except PlayerProtocolError as exc:
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 1, "DECK_ERROR", "ERROR", actions, started, str(exc), initial=True))
                return
            except Exception as exc:
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, 1, "DECK_ERROR", "INVALID", actions, started, str(exc), initial=True))
                return
            if record_trace:
                _write_trace(trace_file, {
                    "event": "deck",
                    "p0": p0_name,
                    "p1": p1_name,
                    "p0_deck": deck0,
                    "p1_deck": deck1,
                })
            obs, start_data = battle_start(deck0, deck1)
            if config.profile == "kaggle" and start_data.errorPlayer >= 0:
                bad_seat = int(start_data.errorPlayer)
                queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, bad_seat, "ENGINE_DECK_ERROR", "INVALID", actions, started, f"Player {bad_seat}'s deck error.", initial=True))
                return
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
                    actor_name = p0_name if player_idx == 0 else p1_name
                    training_focus = (config.record_mode or "").lower() == "training" and (
                        not config.record_focus or config.record_focus == actor_name
                    )
                    step_record: dict[str, Any] = {
                        "event": "action",
                        "step": actions,
                        "actor_seat": player_idx,
                        "actor": actor_name,
                        "observation": obs if training_focus else _summarize_observation(obs),
                        "training_state": training_focus,
                    }
                    try:
                        allowed_s = config.act_timeout_s + max(0.0, overage[player_idx])
                        raw_action, duration = actor.request({"kind": "act", "obs": obs}, allowed_s)
                        overage[player_idx] -= max(0.0, duration - config.act_timeout_s)
                        if overage[player_idx] < 0:
                            raise TimeoutError(
                                f"remainingOverageTime below 0 after {duration:.3f}s action"
                            )
                        action = _validate_action_for_profile(obs, raw_action, config)
                        step_record["raw_action"] = raw_action
                        step_record["action"] = action
                        step_record["duration_s"] = duration
                        step_record["remaining_overage_s"] = overage[player_idx]
                        step_record["selected_options"] = _selected_options_safe(obs, action)
                    except TimeoutError as exc:
                        step_record["error"] = str(exc)
                        if record_trace:
                            _write_trace(trace_file, step_record)
                        queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, player_idx, "TIMEOUT", "TIMEOUT", actions, started, str(exc)))
                        return
                    except PlayerProtocolError as exc:
                        step_record["error"] = str(exc)
                        if record_trace:
                            _write_trace(trace_file, step_record)
                        queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, player_idx, "AGENT_ERROR", "ERROR", actions, started, str(exc)))
                        return
                    except Exception as exc:
                        step_record["raw_action"] = locals().get("raw_action")
                        step_record["error"] = str(exc)
                        if record_trace:
                            _write_trace(trace_file, step_record)
                        queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, player_idx, "INVALID_ACTION", "INVALID", actions, started, str(exc)))
                        return
                    actions += 1
                    try:
                        obs = battle_select(action)
                        step_record["next_result"] = (obs.get("current") or {}).get("result", -1)
                        if record_trace:
                            _write_trace(trace_file, step_record)
                    except Exception as exc:
                        step_record["error"] = str(exc)
                        if record_trace:
                            _write_trace(trace_file, step_record)
                        queue.put(_agent_failure(config, game_id, p0_name, p1_name, seed, player_idx, "ENGINE_REJECTED_ACTION", "INVALID", actions, started, str(exc)))
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
                        reason="EPISODE_STEP_LIMIT" if config.profile == "kaggle" else "MAX_ACTIONS",
                        actions=actions,
                        duration_s=time.perf_counter() - started,
                        p0_status="DONE",
                        p1_status="DONE",
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
                reason="HARNESS_FAILURE" if config.profile == "kaggle" else "REFEREE_CRASH",
                actions=actions,
                duration_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
                failure_class="HARNESS_FAILURE",
                ranking_eligible=False,
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
        return GameResult(game_id, p0, p1, seed, result, "P0_WIN", p0, p1, "RESULT", actions, time.perf_counter() - started, trace=trace or [], p0_reward=1, p1_reward=-1)
    if result == 1:
        return GameResult(game_id, p0, p1, seed, result, "P1_WIN", p1, p0, "RESULT", actions, time.perf_counter() - started, trace=trace or [], p0_reward=-1, p1_reward=1)
    return GameResult(game_id, p0, p1, seed, result, "DRAW", None, None, "RESULT", actions, time.perf_counter() - started, trace=trace or [], p0_reward=0, p1_reward=0)


def _agent_failure(
    config: EvalConfig,
    game_id: str,
    p0: str,
    p1: str,
    seed: int,
    bad_seat: int,
    reason: str,
    status: str,
    actions: int,
    started: float,
    error: str,
    trace: list[dict[str, Any]] | None = None,
    initial: bool = False,
) -> GameResult:
    if config.profile != "kaggle":
        return _forfeit(game_id, p0, p1, seed, bad_seat, reason, actions, started, error, trace)

    statuses = ["DONE", "DONE"]
    statuses[bad_seat] = status
    if initial:
        return GameResult(
            game_id=game_id,
            p0=p0,
            p1=p1,
            seed=seed,
            result=None,
            outcome="NO_RESULT",
            winner=None,
            loser=p0 if bad_seat == 0 else p1,
            reason=reason,
            actions=actions,
            duration_s=time.perf_counter() - started,
            error=error,
            trace=trace or [],
            p0_status=statuses[0],
            p1_status=statuses[1],
            failure_class="AGENT_FAILURE",
        )

    rewards: list[float | None] = [1, 1]
    rewards[bad_seat] = None
    winner = p1 if bad_seat == 0 else p0
    loser = p0 if bad_seat == 0 else p1
    return GameResult(
        game_id=game_id,
        p0=p0,
        p1=p1,
        seed=seed,
        result=1 if bad_seat == 0 else 0,
        outcome="P0_LOSS" if bad_seat == 0 else "P1_LOSS",
        winner=winner,
        loser=loser,
        reason=reason,
        actions=actions,
        duration_s=time.perf_counter() - started,
        error=error,
        trace=trace or [],
        p0_status=statuses[0],
        p1_status=statuses[1],
        p0_reward=rewards[0],
        p1_reward=rewards[1],
        failure_class="AGENT_FAILURE",
    )


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


def _selected_options_safe(obs: dict[str, Any], action: list[Any]) -> list[dict[str, Any]]:
    if not all(type(index) is int for index in action):
        return []
    return _selected_options(obs, action)


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
        return bool(result.loser) or result.reason != "RESULT" or result.outcome == "NO_RESULT"
    if mode == "sample":
        if result.reason != "RESULT" or result.outcome == "NO_RESULT":
            return True
        return _sample_trace_selected(result.game_id, result.seed, config.record_sample_rate)
    if mode == "training":
        return not focus or focus in {result.p0, result.p1}
    return True


def _should_record_before_game(game_id: str, seed: int, config: EvalConfig) -> bool:
    mode = (config.record_mode or "all").lower()
    if mode == "none":
        return False
    if mode == "sample":
        return _sample_trace_selected(game_id, seed, config.record_sample_rate)
    if mode == "training":
        return True
    return True


def _sample_trace_selected(game_id: str, seed: int, rate: float) -> bool:
    normalized = max(0.0, min(1.0, float(rate)))
    if normalized <= 0:
        return False
    digest = hashlib.sha256(f"{game_id}:{seed}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return bucket < normalized
