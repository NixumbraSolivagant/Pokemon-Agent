from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from tools.build_models import DEFAULT_BASE, DEFAULT_OUT, BuildConfig
from tools.deck_rules import ACE_SPEC_IDS, BASIC_ENERGY_IDS, apply_deck_swaps, deck_to_text, validate_deck_ids
from tools.export_kaggle_submission import export_kaggle_submission


SEARCH_INJECTION = r'''

# --- Champion Great Tusk search wrapper injected by tools.build_submission ---
from collections import Counter as _gt_Counter
import random as _gt_random
import time as _gt_time

_GT_SEARCH_OK = False
try:
    from cg.api import search_begin as _gt_search_begin, search_step as _gt_search_step, search_end as _gt_search_end
    _GT_SEARCH_OK = True
except Exception:
    _GT_SEARCH_OK = False

GT_SEARCH_ENABLED = __GT_SEARCH_ENABLED__
GT_SEARCH_CANDIDATES = __GT_SEARCH_CANDIDATES__
GT_SEARCH_BUDGET_S = __GT_SEARCH_BUDGET_S__
GT_SEARCH_MARGIN = __GT_SEARCH_MARGIN__
GT_SEARCH_ROLLOUT_STEPS = __GT_SEARCH_ROLLOUT_STEPS__
GT_BELIEF_WORLDS = __GT_BELIEF_WORLDS__
GT_RISK_PENALTY = __GT_RISK_PENALTY__
GT_STRATEGY_WEIGHTS = __GT_STRATEGY_WEIGHTS__
GT_POLICY_VARIANT = "__GT_POLICY_VARIANT__"
GT_OPPONENT_MODEL = "__GT_OPPONENT_MODEL__"
GT_OPPONENT_DECKS = __GT_OPPONENT_DECKS__


def _gt_weight(name, default):
    try:
        return float(GT_STRATEGY_WEIGHTS.get(name, default))
    except Exception:
        return default


def _gt_model_noise(index):
    if GT_OPPONENT_MODEL != "noisy":
        return 0.0
    return ((index * 1103515245 + 12345) % 997) - 498.0


def _gt_to_builtin(value):
    if isinstance(value, dict):
        return {k: _gt_to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_gt_to_builtin(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            k: _gt_to_builtin(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return value


def _gt_energy_count(pk):
    if pk is None:
        return 0
    return len(getattr(pk, "energyCards", None) or getattr(pk, "energies", []) or [])


def _gt_seed(obs, salt=0):
    current = getattr(obs, "current", None)
    value = int(getattr(current, "turn", 0) or 0) * 1000003 + int(salt)
    players = getattr(current, "players", []) if current is not None else []
    for player in players:
        for zone in (getattr(player, "active", []), getattr(player, "bench", []), getattr(player, "discard", [])):
            for card in zone or []:
                if card is not None:
                    value = (value * 1009 + int(getattr(card, "id", 0) or 0)) & 0x7FFFFFFF
    return value


def _gt_visible_ids(player):
    result = []
    for zone in (getattr(player, "active", []), getattr(player, "bench", []), getattr(player, "discard", [])):
        result.extend(getattr(card, "id", 0) for card in zone or [] if card is not None)
    return result


def _gt_partition_cards(deck, counts, visible_ids, rng, fallback):
    available = _gt_Counter(deck)
    for card_id in visible_ids:
        if available[card_id] > 0:
            available[card_id] -= 1
    pool = [card_id for card_id, quantity in available.items() for _ in range(max(0, quantity))]
    rng.shuffle(pool)
    total = sum(max(0, int(count or 0)) for count in counts)
    while len(pool) < total:
        pool.append(fallback)
    result = []
    cursor = 0
    for count in counts:
        count = max(0, int(count or 0))
        result.append(pool[cursor : cursor + count])
        cursor += count
    return result


def _gt_sample_hidden_worlds(obs, me, opp, deck, fallback):
    models = {str(name): list(cards) for name, cards in GT_OPPONENT_DECKS.items() if cards}
    visible = _gt_visible_ids(opp)
    ranked = sorted(
        ((len(set(visible).intersection(cards)), name) for name, cards in models.items()),
        reverse=True,
    )
    selected = [name for score, name in ranked if ranked and score == ranked[0][0] and score > 0]
    if not selected:
        selected = ["unknown"]
    worlds = []
    for index in range(max(1, int(GT_BELIEF_WORLDS))):
        rng = _gt_random.Random(_gt_seed(obs, index))
        opponent_deck = list(models.get(rng.choice(selected), deck))
        own_hidden, own_prize = _gt_partition_cards(
            deck,
            (me.deckCount, len(me.prize)),
            _gt_visible_ids(me),
            rng,
            fallback,
        )
        opp_deck_hidden, opp_prize, opp_hand = _gt_partition_cards(
            opponent_deck,
            (opp.deckCount, len(opp.prize), opp.handCount),
            visible,
            rng,
            fallback,
        )
        worlds.append(
            (own_hidden, own_prize, opp_deck_hidden, opp_prize, opp_hand)
        )
    return worlds


def _gt_eval_state(obs, me_idx):
    st = getattr(obs, "current", None)
    if st is None:
        return 0.0
    result = getattr(st, "result", -1)
    if result == me_idx:
        return 1.0e9
    if result == 1 - me_idx:
        return -1.0e9
    me = st.players[me_idx]
    opp = st.players[1 - me_idx]
    value = 0.0
    opp_deck = getattr(opp, "deckCount", 60)
    my_deck = getattr(me, "deckCount", 60)
    value += (len(getattr(opp, "prize", []) or []) - len(getattr(me, "prize", []) or [])) * _gt_weight("prize_delta", 3500.0)
    value += (60 - opp_deck) * _gt_weight("opp_mill", 950.0)
    value -= (60 - my_deck) * _gt_weight("self_mill_penalty", 280.0)
    if opp_deck <= 4:
        value += (5 - opp_deck) * _gt_weight("opp_deckout_bonus", 18000.0)
    if my_deck <= 5:
        value -= (6 - my_deck) * _gt_weight("self_deckout_penalty", 12000.0)
    value += (getattr(me, "handCount", 0) - getattr(opp, "handCount", 0)) * _gt_weight("hand_delta", 45.0)
    value += (my_deck - opp_deck) * _gt_weight("deck_delta", 30.0)

    hand = list(getattr(me, "hand", []) or [])
    hand_ids = [getattr(card, "id", None) for card in hand]
    active = active_pokemon(me)
    if active is not None and active.id == GREAT_TUSK:
        value += _gt_weight("great_tusk_active", 1800.0)
        if _gt_energy_count(active) >= 2:
            value += _gt_weight("great_tusk_ready", 6500.0)
            if not getattr(st, "supporterPlayed", False) and EXPLORER_GUIDANCE in hand_ids:
                value += _gt_weight("explorer_ready", 14000.0)
            if getattr(st, "supporterPlayed", False):
                value += _gt_weight("supporter_played_attack", 4500.0)
    if EXPLORER_GUIDANCE in hand_ids:
        value += 3500.0
    if POKEGEAR_30 in hand_ids or POKE_PAD in hand_ids:
        value += 1100.0
    if NIGHT_STRETCHER in hand_ids or SACRED_ASH in hand_ids:
        value += 900.0

    my_field = list(getattr(me, "active", []) or []) + list(getattr(me, "bench", []) or [])
    opp_field = list(getattr(opp, "active", []) or []) + list(getattr(opp, "bench", []) or [])
    for pk in my_field:
        if pk is None:
            continue
        if pk.id == GREAT_TUSK:
            value += 1600.0 + 900.0 * min(2, _gt_energy_count(pk))
            if getattr(pk, "hp", 0) <= 60:
                value -= 1200.0
        elif pk.id == CRUSTLE:
            value += 2200.0 + 8.0 * getattr(pk, "hp", 0)
        elif pk.id == DWEBBLE:
            value += 750.0
        elif pk.id == TATSUGIRI:
            value += 400.0
        value += 2.0 * getattr(pk, "hp", 0)
    for pk in opp_field:
        if pk is None:
            continue
        value -= 1.1 * getattr(pk, "hp", 0)
        if is_ex_pokemon(pk):
            value += 650.0
    if GT_OPPONENT_MODEL == "aggro_bias":
        value += (len(getattr(opp, "prize", []) or []) - len(getattr(me, "prize", []) or [])) * 1700.0
        value += 600.0 if active_pokemon(opp) is not None else 0.0
    elif GT_OPPONENT_MODEL == "stall_bias":
        value += (60 - opp_deck) * 180.0
        value -= max(0, 7 - my_deck) * 1600.0
    elif GT_OPPONENT_MODEL == "noisy":
        value -= 350.0
    return value


def _gt_option_tactical_bonus(obs, option, score):
    state = obs.current
    select = obs.select
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    active = active_pokemon(me)
    bonus = 0
    try:
        route = GT_POLICY_VARIANT.replace("synthetic_", "")
        route = {
            "fast_ko": "ko_first",
            "slow_ko": "ko_first",
            "bench_pressure": "ko_first",
            "wall": "wall_first",
            "mill": "mill_first",
            "denial": "resource_denial",
        }.get(route, route)
        if select.context == SelectContext.MAIN:
            if option.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, option.index, state.yourIndex)
                cid = getattr(card, "id", None)
                if cid == EXPLORER_GUIDANCE and active is not None and active.id == GREAT_TUSK and can_pay_attack(active, LAND_COLLAPSE):
                    bonus += 900000
                    if route == "mill_first":
                        bonus += 240000
                elif cid in (POKEGEAR_30, POKE_PAD, ULTRA_BALL, FIGHT_GONG, BUDDY_BUDDY_POFFIN):
                    bonus += 220000
                elif cid in (ERI, XEROSIC_SCHEME, HAND_TRIMMER, ENHANCED_HAMMER, ENERGY_LASSO, FLUTE):
                    bonus += 160000
                    if route == "resource_denial":
                        bonus += int(260000 * _gt_weight("disruption_priority", 1.0))
                elif cid in (NIGHT_STRETCHER, SACRED_ASH, ENERGY_RECYCLER, JUMBO_ICE_CREAM):
                    bonus += 90000
                elif cid in (HERO_CAPE, AIR_BALLOON, SACRED_CHARM, HANDY_CIRCULATOR, GRAVITY_GEM):
                    bonus += 85000
                elif cid in (BOSS_ORDERS, LISIA_APPEAL):
                    bonus += 75000 if opponent.deckCount <= 16 else 25000
            elif option.type == OptionType.ATTACH:
                bonus += 180000
            elif option.type == OptionType.EVOLVE:
                bonus += 160000
                if route == "wall_first":
                    bonus += 200000
            elif option.type == OptionType.ABILITY:
                bonus += 150000
            elif option.type == OptionType.RETREAT:
                bonus += 100000
                if route == "wall_first":
                    bonus += int(150000 * _gt_weight("switch_priority", 1.0))
            elif option.type == OptionType.ATTACK:
                bonus += 260000
                if route == "ko_first":
                    bonus += 220000
                if active is not None and active.id == GREAT_TUSK and can_pay_attack(active, LAND_COLLAPSE):
                    bonus += 220000
                    if route == "mill_first":
                        bonus += 300000
                    if state.supporterPlayed:
                        bonus += 320000
            elif option.type == OptionType.END:
                bonus -= 350000
        elif option.type == OptionType.CARD:
            card = get_card(obs, option.area, option.index, option.playerIndex)
            cid = getattr(card, "id", None)
            if select.context == SelectContext.TO_HAND:
                if cid == EXPLORER_GUIDANCE:
                    bonus += 500000
                elif cid in (POKEGEAR_30, POKE_PAD, ULTRA_BALL, FIGHT_GONG, BUDDY_BUDDY_POFFIN):
                    bonus += 260000
                elif cid in (GREAT_TUSK, DWEBBLE, CRUSTLE):
                    bonus += 220000
                elif cid in (NIGHT_STRETCHER, SACRED_ASH, ENERGY_RECYCLER):
                    bonus += 140000
            elif select.context in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
                if cid in (EXPLORER_GUIDANCE, GREAT_TUSK, CRUSTLE):
                    bonus -= 260000
            elif select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                if option.playerIndex == state.yourIndex and cid in (GREAT_TUSK, CRUSTLE):
                    bonus += 180000
        return score + bonus
    except Exception:
        return score


def _gt_candidate_order(obs, hidx):
    scores = _gt_score_options_from_observation(obs)
    if not scores:
        return [hidx]
    ranked = sorted(
        range(len(scores)),
        key=lambda i: (_gt_option_tactical_bonus(obs, obs.select.option[i], scores[i]) + _gt_model_noise(i), scores[i]),
        reverse=True,
    )
    order = []
    for idx in [hidx] + ranked:
        if 0 <= idx < len(scores) and idx not in order:
            order.append(idx)
    return order


def _gt_score_options_from_observation(obs):
    state = obs.current
    select = obs.select
    if select is None or not select.option:
        return []
    context = select.context
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    try:
        wall_mode = should_wall_mode(me, opponent, state)
    except Exception:
        wall_mode = False
    try:
        ko_mode = should_ko_mode(me, opponent, state)
    except Exception:
        ko_mode = False
    active = active_pokemon(me)
    scores = []
    for option in select.option:
        score = 0
        try:
            if context == SelectContext.MAIN:
                if option.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, option.index, state.yourIndex)
                    if card is not None:
                        score = play_score(card.id, me, opponent, state, wall_mode, ko_mode)
                elif option.type == OptionType.EVOLVE:
                    target = get_card(obs, option.inPlayArea, option.inPlayIndex, state.yourIndex)
                    score = 90000 if target is not None and target.id == DWEBBLE else 2000
                    if wall_mode:
                        score += 40000
                elif option.type == OptionType.ATTACH:
                    card = get_card(obs, option.area, option.index, state.yourIndex)
                    target = get_card(obs, option.inPlayArea, option.inPlayIndex, state.yourIndex)
                    if card is not None:
                        score = attach_score(card.id, target, option.inPlayArea, me, opponent, wall_mode, ko_mode)
                elif option.type == OptionType.ABILITY:
                    card = get_card(obs, option.area, option.index, state.yourIndex)
                    if card is not None and card.id == TATSUGIRI:
                        score = 42000 if not state.supporterPlayed and count_in_hand(me, EXPLORER_GUIDANCE) == 0 else -10000
                    elif card is not None and card.id == DURANT_EX:
                        score = 80000
                    else:
                        score = 12000
                elif option.type == OptionType.RETREAT:
                    neutral_tusk = (
                        active is not None
                        and active.id == GREAT_TUSK
                        and state.stadium
                        and state.stadium[0].id == NEUTRAL_CENTER
                    )
                    if wall_mode and any(p.id == CRUSTLE for p in me.bench) and not neutral_tusk:
                        score = 130000
                    elif ready_tusk_on_bench(me):
                        score = 125000
                    elif active is not None and active.id == GREAT_TUSK and not can_pay_attack(active, LAND_COLLAPSE):
                        score = 70000
                    elif active is not None and active.id == TATSUGIRI and (state.supporterPlayed or count_in_hand(me, EXPLORER_GUIDANCE) > 0):
                        score = 36000
                    else:
                        score = -10000
                elif option.type == OptionType.ATTACK:
                    score = attack_score(option.attackId, me, opponent, state, wall_mode, ko_mode)
                    if option.attackId is None:
                        if active_tusk_ready(me):
                            score = 200000 + (70000 if state.supporterPlayed else 0)
                        elif active is not None and active.id == DWEBBLE:
                            score = 90000
                        elif active is not None and active.id == CRUSTLE and can_pay_attack(active, SUPERB_SCISSORS):
                            score = 325000 if ko_mode else (65000 if wall_mode else 9000)
                        else:
                            score = 4000 if ko_mode else 1000
                elif option.type == OptionType.END:
                    score = -100
                else:
                    score = 1000
            elif option.type == OptionType.CARD:
                card = get_card(obs, option.area, option.index, option.playerIndex)
                score = select_card_score(card, option.playerIndex, context, me, opponent, state, wall_mode, ko_mode)
            elif option.type == OptionType.YES:
                effect = select.effect or select.contextCard
                score = 100
                if effect is not None and effect.id in (FIGHT_GONG, ULTRA_BALL, BUG_CATCHING_SET, POKEGEAR_30, ROTO_STICK, EXPLORER_GUIDANCE, TATSUGIRI):
                    score = 2000
            elif option.type == OptionType.NO:
                score = 0
            elif option.type == OptionType.NUMBER:
                n = option.number or 0
                if context == SelectContext.DRAW_COUNT:
                    score = -10 * n
                    if not has_ready_tusk(me) and me.deckCount > 8:
                        score += 18 * n
                elif context in (SelectContext.DAMAGE_COUNTER_COUNT, SelectContext.REMOVE_DAMAGE_COUNTER_COUNT):
                    score = n if ko_mode else -n
                else:
                    score = n
            elif option.type in (OptionType.ENERGY, OptionType.ENERGY_CARD, OptionType.TOOL_CARD):
                score = option.count or 0
            elif option.type == OptionType.ATTACK:
                score = attack_score(option.attackId, me, opponent, state, wall_mode, ko_mode)
            elif option.type == OptionType.SKILL:
                score = 100
            else:
                score = 0
        except Exception:
            score = -1000000
        scores.append(score)
    return scores


def _gt_greedy_action_from_observation(obs):
    select = obs.select
    if select is None or not select.option:
        return []
    scores = _gt_score_options_from_observation(obs)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    result = []
    for index in order:
        if len(result) >= select.maxCount:
            break
        if scores[index] >= 0 or len(result) < select.minCount:
            result.append(index)
    if len(result) < select.minCount:
        result = order[:select.minCount]
    return result


def _gt_rollout_to_turn_boundary(search_state, me_idx, deadline):
    current = search_state
    start_turn = current.observation.current.turn if current.observation.current is not None else None
    steps = 0
    while steps < GT_SEARCH_ROLLOUT_STEPS and _gt_time.perf_counter() < deadline:
        obs = current.observation
        if obs is None or obs.current is None or obs.select is None:
            break
        if obs.current.result != -1:
            break
        if obs.current.yourIndex != me_idx or obs.current.turn != start_turn:
            break
        action = _gt_greedy_action_from_observation(obs)
        if not action and obs.select.minCount > 0:
            break
        try:
            current = _gt_search_step(current.searchId, action)
        except Exception:
            break
        steps += 1
    return current.observation


def _gt_search_action(obs_dict, obs):
    if not (_GT_SEARCH_OK and GT_SEARCH_ENABLED):
        return None
    if obs.select is None or obs.current is None:
        return None
    if obs.select.context != SelectContext.MAIN or obs.select.minCount != 1 or obs.select.maxCount != 1:
        return None
    if not obs.select.option or len(obs.select.option) < 2:
        return None
    sbi = obs_dict.get("search_begin_input")
    if not sbi:
        return None

    heuristic = _agent(obs_dict)
    if not heuristic:
        heuristic = [0]
    hidx = int(heuristic[0])
    if hidx < 0 or hidx >= len(obs.select.option):
        hidx = 0

    me_idx = obs.current.yourIndex
    me = obs.current.players[me_idx]
    opp = obs.current.players[1 - me_idx]
    deck = read_deck_csv()
    fallback_basic = GREAT_TUSK
    order = _gt_candidate_order(obs, hidx)
    order = order[: max(1, GT_SEARCH_CANDIDATES)]
    started = _gt_time.perf_counter()
    deadline = started + GT_SEARCH_BUDGET_S
    values = {idx: [] for idx in order}
    worlds = _gt_sample_hidden_worlds(obs, me, opp, deck, fallback_basic)
    for own_deck, own_prize, opp_deck, opp_prize, opp_hand in worlds:
        if _gt_time.perf_counter() >= deadline:
            break
        try:
            root = _gt_search_begin(
                obs,
                own_deck,
                own_prize,
                opp_deck,
                opp_prize,
                opp_hand,
                [fallback_basic] if getattr(opp, "active", None) and opp.active and opp.active[0] is None else [],
                False,
            )
        except Exception:
            continue
        try:
            for idx in order:
                if _gt_time.perf_counter() >= deadline:
                    break
                try:
                    nxt = _gt_search_step(root.searchId, [idx])
                    end_obs = _gt_rollout_to_turn_boundary(nxt, me_idx, deadline)
                    values[idx].append(_gt_eval_state(end_obs, me_idx))
                except Exception:
                    continue
        finally:
            try:
                _gt_search_end()
            except Exception:
                pass
    aggregates = {}
    for idx, samples in values.items():
        if not samples:
            continue
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / len(samples)
        aggregates[idx] = mean - float(GT_RISK_PENALTY) * variance ** 0.5
    if not aggregates:
        return None
    hval = aggregates.get(hidx, -1.0e18)
    best = max(aggregates, key=lambda i: aggregates[i])
    if best != hidx and aggregates[best] > hval + GT_SEARCH_MARGIN:
        return [best]
    return heuristic


def agent(obs_dict: dict, configuration=None) -> list[int]:
    obs_dict = _gt_to_builtin(obs_dict)
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()
        selected = _gt_search_action(obs_dict, obs)
        if selected is not None:
            return selected
        return _agent(obs_dict)
    except Exception:
        if os.environ.get("DEBUG_AGENT") == "1":
            import traceback
            traceback.print_exc()
        select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if select is None:
            return read_deck_csv()
        options = select.get("option") or []
        min_count = max(0, int(select.get("minCount", 0) or 0))
        return list(range(min(min_count, len(options))))


# Kaggle executes the file and picks the last callable in insertion order.
# Rebinding the existing agent under a new final name keeps the search helpers
# private to our wrapper and prevents Kaggle from calling them directly.
kaggle_agent = agent
'''


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_deck_from_archive(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        with tarfile.open(path, "r:gz") as tar:
            member = tar.getmember("deck.csv")
            payload = tar.extractfile(member)
            if payload is None:
                return []
            return [int(line) for line in payload.read().decode("utf-8").splitlines() if line.strip()]
    except (KeyError, OSError, ValueError, tarfile.TarError):
        return []


def _resolved_opponent_decks(cfg: BuildConfig) -> dict[str, list[int]]:
    return {str(name): list(deck) for name, deck in cfg.opponent_decks.items() if deck}


def make_config(args: argparse.Namespace) -> BuildConfig:
    data = load_config(args.config)
    deck_swaps = [tuple(map(int, pair)) for pair in data.get("deck_swaps", [])]
    deck_override = data.get("deck_override")
    if deck_override is not None:
        deck_override = [int(card_id) for card_id in deck_override]
    strategy_weights = {
        str(key): float(value)
        for key, value in dict(data.get("strategy_weights", {})).items()
    }
    return BuildConfig(
        name=str(data.get("name") or args.name),
        family=str(data.get("family", "great_tusk")),
        base=Path(data.get("base") or args.base),
        out=Path(data.get("out") or args.out),
        runtime_source=Path(data.get("runtime_source", "main.py")),
        runtime_cg_dir=Path(data.get("runtime_cg_dir", "cg")),
        enable_search=bool(data.get("enable_search", args.enable_search)),
        injection=str(data.get("injection", args.injection)),
        search_candidates=int(data.get("search_candidates", args.search_candidates)),
        search_budget_s=float(data.get("search_budget_s", args.search_budget_s)),
        search_margin=float(data.get("search_margin", args.search_margin)),
        search_rollout_steps=int(data.get("search_rollout_steps", args.search_rollout_steps)),
        belief_worlds=int(data.get("belief_worlds", args.belief_worlds)),
        risk_penalty=float(data.get("risk_penalty", args.risk_penalty)),
        opponent_decks={
            str(name): [int(card_id) for card_id in deck]
            for name, deck in dict(data.get("opponent_decks", {})).items()
        },
        deck_swaps=deck_swaps,
        deck_override=deck_override,
        deck_files=tuple(data.get("deck_files", ("deck.csv",))),
        strategy_weights=strategy_weights,
        strategy_genome=dict(data.get("strategy_genome", {})),
        policy_variant=str(data.get("policy_variant", "default")),
        opponent_model=str(data.get("opponent_model", "perfect")),
        origin=str(data.get("origin", "")),
        notes=str(data.get("notes", "")),
        include_build_metadata=bool(data.get("include_build_metadata", args.include_build_metadata)),
    )


def build_main(original: str, cfg: BuildConfig) -> str:
    if cfg.injection == "none":
        return original.rstrip() + "\n"
    marker = "# --- Champion Great Tusk search wrapper injected by tools.build_submission ---"
    existing = original.find(marker)
    if existing >= 0:
        original = original[:existing].rstrip() + "\n"
    injection = (
        SEARCH_INJECTION.replace("__GT_SEARCH_ENABLED__", "True" if cfg.enable_search else "False")
        .replace("__GT_SEARCH_CANDIDATES__", str(cfg.search_candidates))
        .replace("__GT_SEARCH_BUDGET_S__", repr(cfg.search_budget_s))
        .replace("__GT_SEARCH_MARGIN__", repr(cfg.search_margin))
        .replace("__GT_SEARCH_ROLLOUT_STEPS__", str(cfg.search_rollout_steps))
        .replace("__GT_BELIEF_WORLDS__", str(cfg.belief_worlds))
        .replace("__GT_RISK_PENALTY__", repr(cfg.risk_penalty))
        .replace("__GT_STRATEGY_WEIGHTS__", repr(dict(cfg.strategy_weights)))
        .replace("__GT_POLICY_VARIANT__", cfg.policy_variant.replace("\\", "\\\\").replace('"', '\\"'))
        .replace("__GT_OPPONENT_MODEL__", cfg.opponent_model.replace("\\", "\\\\").replace('"', '\\"'))
        .replace("__GT_OPPONENT_DECKS__", repr(_resolved_opponent_decks(cfg)))
    )
    return original.rstrip() + "\n" + injection.lstrip()


def build_submission(cfg: BuildConfig) -> Path:
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.runtime_source.is_file():
        raise FileNotFoundError(f"Missing canonical runtime source: {cfg.runtime_source}")
    if not cfg.runtime_cg_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical cg runtime: {cfg.runtime_cg_dir}")
    deck = list(cfg.deck_override or _read_deck_from_archive(cfg.base))
    if not deck:
        deck_path = Path("deck.csv")
        deck = [int(line) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if cfg.deck_swaps and cfg.deck_override is None:
        deck = [int(line) for line in apply_deck_swaps(deck_to_text(deck), cfg.deck_swaps).splitlines()]
    validate_deck_ids(deck)
    main_data = build_main(cfg.runtime_source.read_text(encoding="utf-8"), cfg).encode("utf-8")
    deck_data = deck_to_text(deck).encode("utf-8")
    metadata = {
        "name": cfg.name,
        "family": cfg.family,
        "base": str(cfg.base),
        "runtime_source": str(cfg.runtime_source),
        "runtime_cg_dir": str(cfg.runtime_cg_dir),
        "reference_free_runtime": True,
        "enable_search": cfg.enable_search,
        "injection": cfg.injection,
        "search_candidates": cfg.search_candidates,
        "search_budget_s": cfg.search_budget_s,
        "search_margin": cfg.search_margin,
        "search_rollout_steps": cfg.search_rollout_steps,
        "belief_worlds": cfg.belief_worlds,
        "risk_penalty": cfg.risk_penalty,
        "opponent_deck_models": sorted(_resolved_opponent_decks(cfg)),
        "deck_swaps": cfg.deck_swaps,
        "deck_override": cfg.deck_override,
        "deck_files": list(cfg.deck_files),
        "strategy_weights": cfg.strategy_weights,
        "strategy_genome": cfg.strategy_genome,
        "policy_variant": cfg.policy_variant,
        "opponent_model": cfg.opponent_model,
        "origin": cfg.origin,
        "notes": cfg.notes,
    }
    with tarfile.open(cfg.out, "w:gz") as dst:
        entries: list[tuple[str, bytes, int]] = [("main.py", main_data, 0o644)]
        entries.extend((name, deck_data, 0o644) for name in dict.fromkeys(cfg.deck_files))
        for path in sorted(cfg.runtime_cg_dir.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            entries.append(((Path("cg") / path.relative_to(cfg.runtime_cg_dir)).as_posix(), path.read_bytes(), 0o644))
        for name, data, mode in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            dst.addfile(info, io.BytesIO(data))
        if cfg.include_build_metadata:
            meta = json.dumps(metadata, indent=2).encode("utf-8")
            info = tarfile.TarInfo("build_metadata.json")
            info.size = len(meta)
            info.mode = 0o644
            dst.addfile(info, io.BytesIO(meta))
    return cfg.out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Great Tusk champion submission tarball.")
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--name", default="champion_gt_search")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--enable-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--injection", choices=["great_tusk", "none"], default="great_tusk")
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-budget-s", type=float, default=0.25)
    parser.add_argument("--search-margin", type=float, default=1200.0)
    parser.add_argument("--search-rollout-steps", type=int, default=16)
    parser.add_argument("--belief-worlds", type=int, default=4)
    parser.add_argument("--risk-penalty", type=float, default=0.20)
    parser.add_argument("--include-build-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-kaggle-submission", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submissions/submission.tar.gz"))
    parser.add_argument("--strip-search-wrapper-for-submission", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    out = build_submission(make_config(args))
    kaggle_submission = None
    if args.export_kaggle_submission:
        kaggle_submission = export_kaggle_submission(
            out,
            args.submission_out,
            strip_search_wrapper=args.strip_search_wrapper_for_submission,
        )
    print(json.dumps({"internal_submission": str(out), "kaggle_submission": str(kaggle_submission) if kaggle_submission else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
