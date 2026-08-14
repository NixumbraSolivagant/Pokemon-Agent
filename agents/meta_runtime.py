from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

from cg.api import AreaType, CardType, OptionType, SelectContext, all_attack, all_card_data, to_observation_class


CARD_TABLE = {card.cardId: card for card in all_card_data()}
ATTACK_TABLE = {attack.attackId: attack for attack in all_attack()}

GRIMMSNARL_EX = 648
MORGREM = 647
IMPIDIMP = 646
FROSLASS = 104
MUNKIDORI = 112
DARKNESS_ENERGY = 7
SHADOW_BULLET = 937
FILCH = 934
BUDDY_BUDDY_POFFIN = 1086
RARE_CANDY = 1079
NIGHT_STRETCHER = 1097
UNFAIR_STAMP = 1080
POKEGEAR_30 = 1122
TOOL_SCRAPPER = 1137
POKE_PAD = 1152
BOSS_ORDERS = 1182
TEAM_ROCKETS_PETREL = 1219
LILLIES_DETERMINATION = 1227
DAWN = 1231
SPIKEMUTH_GYM = 1259
SNORUNT = 860
MEGA_KANGASKHAN_EX = 756
TEAL_MASK_OGERPON_EX = 96
WELLSPRING_MASK_OGERPON_EX = 108
RAGING_BOLT_EX = 63
LATIAS_EX = 184
MEOWTH_EX = 1071
PASSIMIAN = 978
GRASS_ENERGY = 1
DURALUDON = 169
MEGA_LOPUNNY_EX = 849
MEGA_FROSLASS_EX = 861
DUNSPARCE = 305
DUDUNSPARCE = 66
BUNEARY = 848
FAN_ROTOM = 174
GALE_THRUST = 1225
SPIKY_HOPPER = 1226
RESENTFUL_REFRAIN = 1240
ABSOLUTE_SNOW = 1241
MYRIAD_LEAF_SHOWER = 120
WALLYS_COMPASSION = 1229
HAND_TRIMMER = 1087
AIR_BALLOON = 1174
HEROES_CAPE = 1159
GROW_GRASS_ENERGY = 18
GREAT_TUSK = 58
DWEBBLE = 344
CRUSTLE = 345
TERRAKION = 607
CINDERACE = 666
ARCHALUDON_EX = 190
BUG_CATCHING_SET = 1094
ENERGY_SEARCH = 1119
TERA_ORB = 1127
JUDGE = 1213
HARLEQUIN = 1223

_LOPUNNY_MEMORY = {"turn": None, "active_serial": None, "bench_serials": set(), "moved": False}
_OGERPON_MEMORY = {"turn": None, "primary_serial": None}
_POLICY_ROUTER_MEMORY = {"seen_opponent_cards": set(), "expert": None}
POLICY_HISTORY_LENGTH = 5
POLICY_CARD_HASH_BUCKETS = 64
_POLICY_MEMORY = {
    "turn": None,
    "counts": [0] * 17,
    "last_type": -1,
    "last_card_id": 0,
    "recent_types": [],
    "recent_card_ids": [],
    "card_counts": [0] * POLICY_CARD_HASH_BUCKETS,
    "start_hand": 0,
    "start_deck": 0,
    "start_bench": 0,
    "start_energy": 0,
    "start_active_serial": -1,
}
_SEARCH_STATS = {"attempts": 0, "successes": 0, "changes": 0, "failures": 0}
_ADVANTAGE_MEMORY = {"overrides": 0}

ID_BITS = 12
OPTION_TYPE_COUNT = 17
SELECT_CONTEXT_COUNT = 49
VISIBLE_HASH_BUCKETS = 64
BOARD_SLOT_COUNT = 6

BASE_FEATURE_NAMES = (
    "context",
    "option_type",
    "card_id",
    "attack_id",
    "target_card_id",
    "target_hp",
    "target_energy",
    "target_is_opponent",
    "turn",
    "turn_action_count",
    "own_deck",
    "opponent_deck",
    "own_prizes",
    "opponent_prizes",
    "own_hand",
    "opponent_hand",
    "own_bench",
    "opponent_bench",
    "own_active_id",
    "opponent_active_id",
    "own_active_hp",
    "opponent_active_hp",
    "own_active_energy",
    "opponent_active_energy",
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "went_first",
    "option_count",
    "option_number",
)

KEY_CARD_IDS = (
    1, 3, 4, 5, 6, 7, 11, 13, 63, 66, 96, 104, 108, 112, 140, 174, 184, 305, 646, 647, 648, 673, 674, 675,
    676, 677, 678, 756, 848, 849, 860, 861, 978,
    1071, 1079, 1080, 1086, 1088, 1094, 1097, 1098, 1116, 1118, 1119, 1120, 1121, 1122, 1127,
    1123, 1137, 1141, 1142, 1147, 1152, 1159, 1174, 1182, 1197, 1198, 1201, 1205, 1213, 1219, 1221, 1223, 1225,
    1227, 1229, 1231, 1250, 1251, 1259, 1264,
)

FIELD_CARD_IDS = (58, 63, 66, 96, 104, 108, 112, 119, 120, 121, 174, 184, 305, 344, 345, 646, 647, 648, 677, 678, 722, 723, 741, 742, 743, 756, 848, 849, 860, 861, 978, 1031, 1071)

FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(f"hand_{card_id}" for card_id in KEY_CARD_IDS) + tuple(
    f"discard_{card_id}" for card_id in KEY_CARD_IDS
) + tuple(f"own_field_{card_id}" for card_id in FIELD_CARD_IDS) + tuple(
    f"opponent_field_{card_id}" for card_id in FIELD_CARD_IDS
) + (
    "effect_card_id",
    "context_card_id",
    "option_area",
    "option_in_play_area",
    "active_moved_this_turn",
    "active_damage",
    "opponent_active_damage",
    "active_has_air_balloon",
    "bench_air_balloon_count",
    "retreated_this_turn",
    "effective_attack_damage",
    "attack_would_ko",
    "option_source_index",
    "option_target_index",
    "option_card_hp",
    "option_card_max_hp",
    "option_card_damage",
    "option_card_energy",
    "option_card_basic_energy",
    "option_card_grow_energy",
    "option_card_has_cape",
    "option_card_appeared",
    "target_max_hp",
    "target_damage",
    "target_basic_energy",
    "target_grow_energy",
    "target_has_cape",
    "own_ogerpon_total_energy",
    "own_ogerpon_max_energy",
    "own_ogerpon_min_energy",
    "own_ogerpon_damaged_count",
    "own_ogerpon_cape_count",
    "ogerpon_primary_energy",
    "ogerpon_ready_count",
    "ogerpon_active_is_primary",
) + tuple(f"context_onehot_{index}" for index in range(SELECT_CONTEXT_COUNT)) + tuple(
    f"option_type_onehot_{index}" for index in range(OPTION_TYPE_COUNT)
) + tuple(
    f"{field}_bit_{bit}" for field in (
        "card_id", "attack_id", "target_card_id", "own_active_id", "opponent_active_id", "effect_card_id", "context_card_id",
    ) for bit in range(ID_BITS)
) + tuple(f"turn_option_count_{index}" for index in range(OPTION_TYPE_COUNT)) + tuple(
    f"last_option_type_{index}" for index in range(OPTION_TYPE_COUNT)
) + tuple(f"last_card_id_bit_{bit}" for bit in range(ID_BITS)) + ("turn_recorded_actions",) + tuple(
    f"recent_option_type_{position}_{index}"
    for position in range(POLICY_HISTORY_LENGTH)
    for index in range(OPTION_TYPE_COUNT)
) + tuple(
    f"recent_card_{position}_bit_{bit}"
    for position in range(POLICY_HISTORY_LENGTH)
    for bit in range(ID_BITS)
) + tuple(f"turn_card_hash_{bucket}" for bucket in range(POLICY_CARD_HASH_BUCKETS)) + (
    "turn_hand_delta",
    "turn_deck_delta",
    "turn_bench_delta",
    "turn_energy_delta",
    "turn_active_changed",
)
FEATURE_NAMES += tuple(
    f"{zone}_visible_hash_{bucket}"
    for zone in ("own_hand", "own_discard", "opponent_visible")
    for bucket in range(VISIBLE_HASH_BUCKETS)
)
FEATURE_NAMES += tuple(
    f"{side}_slot_{slot}_{field}"
    for side in ("own", "opponent")
    for slot in range(BOARD_SLOT_COUNT)
    for field in (
        *(f"id_bit_{bit}" for bit in range(ID_BITS)),
        "hp", "max_hp", "damage", "energy", "has_cape",
    )
)

_TYPE_OPTION_FIELDS = {
    "option_type", "card_id", "attack_id", "target_card_id", "target_hp", "target_energy",
    "target_is_opponent", "option_count", "option_number", "option_area", "option_in_play_area", "effective_attack_damage",
    "attack_would_ko", "option_source_index", "option_target_index", "option_card_hp", "option_card_max_hp",
    "option_card_damage", "option_card_energy", "option_card_basic_energy", "option_card_grow_energy",
    "option_card_has_cape", "option_card_appeared", "target_max_hp", "target_damage", "target_basic_energy",
    "target_grow_energy", "target_has_cape",
}
_TYPE_OPTION_PREFIXES = (
    "option_type_onehot_", "card_id_bit_", "attack_id_bit_", "target_card_id_bit_",
)
TYPE_STATE_FEATURE_INDICES = tuple(
    index
    for index, name in enumerate(FEATURE_NAMES)
    if name not in _TYPE_OPTION_FIELDS and not name.startswith(_TYPE_OPTION_PREFIXES)
)
TYPE_FEATURE_NAMES = tuple(FEATURE_NAMES[index] for index in TYPE_STATE_FEATURE_INDICES) + (
    "candidate_type",
    *(f"candidate_type_onehot_{index}" for index in range(OPTION_TYPE_COUNT)),
    "candidate_count",
    "total_candidate_count",
    "base_score_max",
    "base_score_mean",
    "base_score_std",
    "base_score_second",
    "base_score_third",
    "attack_damage_max",
    "attack_ko_any",
    "target_hp_max",
    "target_energy_max",
    "option_energy_max",
    "option_energy_mean",
    "option_hp_max",
    "option_damage_max",
    "option_appeared_any",
) + tuple(f"candidate_card_hash_{bucket}" for bucket in range(POLICY_CARD_HASH_BUCKETS)) + (
    "candidate_has_play_card",
    "candidate_has_attack",
    "candidate_has_ko",
)
_ROUTER_EXACT_ID_NAMES = {
    "own_active_id", "opponent_active_id", "effect_card_id", "context_card_id",
}
_ROUTER_EXCLUDED_PREFIXES = (
    "hand_", "discard_", "own_field_", "opponent_field_", "candidate_card_hash_",
    "card_id_bit_", "attack_id_bit_", "target_card_id_bit_", "own_active_id_bit_",
    "opponent_active_id_bit_", "effect_card_id_bit_", "context_card_id_bit_",
)
ROUTER_FEATURE_INDICES = tuple(
    index
    for index, name in enumerate(TYPE_FEATURE_NAMES)
    if name not in _ROUTER_EXACT_ID_NAMES
    and not name.startswith(_ROUTER_EXCLUDED_PREFIXES)
    and not ("_slot_" in name and "_id_bit_" in name)
)
ROUTER_FEATURE_NAMES = tuple(TYPE_FEATURE_NAMES[index] for index in ROUTER_FEATURE_INDICES)
STOP_EXTRA_FEATURE_NAMES = (
    "continue_type_count",
    "continue_candidate_count",
    "play_candidate_count",
    "ability_candidate_count",
    "attach_candidate_count",
    "evolve_candidate_count",
    "retreat_candidate_count",
    "unseen_continue_type_count",
    "turn_action_count",
    "attack_damage_max",
    "attack_ko_any",
    "terminal_continue_base_margin",
    "turn_energy_delta",
    "turn_hand_delta",
    "turn_bench_delta",
)


def apply_stop_feature_ablation(features: list[float], mode: str) -> list[float]:
    if mode == "full":
        return features
    names = (
        *(f"continue::{name}" for name in TYPE_FEATURE_NAMES),
        *(f"terminal::{name}" for name in TYPE_FEATURE_NAMES),
        *(f"delta::{name}" for name in TYPE_FEATURE_NAMES),
        *STOP_EXTRA_FEATURE_NAMES,
    )
    result = list(features)
    for index, name in enumerate(names):
        if "::base_score_" not in name:
            continue
        if mode == "no_base" or (mode == "delta_only" and not name.startswith("delta::")):
            result[index] = 0.0
        elif mode == "damped_base":
            result[index] *= 0.25
    return result
MAIN_TYPE_HEADS_RUNTIME = {
    int(getattr(OptionType.PLAY, "value", OptionType.PLAY)): "main_play",
    int(getattr(OptionType.ATTACH, "value", OptionType.ATTACH)): "main_attach",
    int(getattr(OptionType.EVOLVE, "value", OptionType.EVOLVE)): "main_evolve",
    int(getattr(OptionType.ABILITY, "value", OptionType.ABILITY)): "main_ability",
    int(getattr(OptionType.DISCARD, "value", OptionType.DISCARD)): "main_discard",
    int(getattr(OptionType.RETREAT, "value", OptionType.RETREAT)): "main_retreat",
    int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)): "main_attack",
    int(getattr(OptionType.END, "value", OptionType.END)): "main_end",
}
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def read_deck_csv() -> list[int]:
    candidates = []
    module_file = globals().get("__file__")
    if module_file:
        candidates.append(Path(module_file).resolve().parent / "deck.csv")
    candidates.extend((Path("/kaggle_simulations/agent/deck.csv"), Path("deck.csv")))
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines()[:60]]


def _update_ogerpon_memory(obs) -> None:
    state = obs.current
    if state is None:
        _OGERPON_MEMORY.update(turn=None, primary_serial=None)
        return
    me = state.players[state.yourIndex]
    ogerpons = [
        pokemon for pokemon in list(me.active or []) + list(me.bench or [])
        if pokemon is not None and pokemon.id == TEAL_MASK_OGERPON_EX
    ]
    if _OGERPON_MEMORY["turn"] != state.turn or not any(
        pokemon.serial == _OGERPON_MEMORY["primary_serial"] for pokemon in ogerpons
    ):
        primary = max(
            ogerpons,
            key=lambda pokemon: (energy_count(pokemon), int(getattr(pokemon, "hp", 0) or 0), -int(pokemon.serial or 0)),
            default=None,
        )
        _OGERPON_MEMORY.update(turn=state.turn, primary_serial=getattr(primary, "serial", None))


def _policy_turn_snapshot(obs) -> dict:
    state = getattr(obs, "current", None)
    if state is None or not hasattr(state, "players") or not hasattr(state, "yourIndex"):
        return {"start_hand": 0, "start_deck": 0, "start_bench": 0, "start_energy": 0, "start_active_serial": -1}
    me = state.players[state.yourIndex]
    my_active = active(me)
    board = [pokemon for pokemon in list(me.active or []) + list(me.bench or []) if pokemon is not None]
    return {
        "start_hand": int(getattr(me, "handCount", 0) or len(me.hand or [])),
        "start_deck": int(getattr(me, "deckCount", 0) or 0),
        "start_bench": len(me.bench or []),
        "start_energy": sum(energy_count(pokemon) for pokemon in board),
        "start_active_serial": int(getattr(my_active, "serial", -1) or -1),
    }


def _reset_policy_memory() -> None:
    _POLICY_MEMORY.clear()
    _POLICY_MEMORY.update(
        turn=None,
        counts=[0] * OPTION_TYPE_COUNT,
        last_type=-1,
        last_card_id=0,
        recent_types=[],
        recent_card_ids=[],
        card_counts=[0] * POLICY_CARD_HASH_BUCKETS,
        start_hand=0,
        start_deck=0,
        start_bench=0,
        start_energy=0,
        start_active_serial=-1,
    )


def _sync_policy_memory(obs) -> None:
    turn = int(getattr(getattr(obs, "current", None), "turn", 0) or 0)
    if _POLICY_MEMORY.get("turn") != turn:
        _POLICY_MEMORY.update(
            turn=turn,
            counts=[0] * OPTION_TYPE_COUNT,
            last_type=-1,
            last_card_id=0,
            recent_types=[],
            recent_card_ids=[],
            card_counts=[0] * POLICY_CARD_HASH_BUCKETS,
            **_policy_turn_snapshot(obs),
        )


def _record_policy_action(obs, selected: list[int] | set[int]) -> None:
    _sync_policy_memory(obs)
    options = list(getattr(getattr(obs, "select", None), "option", None) or [])
    counts = list(_POLICY_MEMORY.get("counts") or [0] * OPTION_TYPE_COUNT)
    recent_types = list(_POLICY_MEMORY.get("recent_types") or [])
    recent_card_ids = list(_POLICY_MEMORY.get("recent_card_ids") or [])
    card_counts = list(_POLICY_MEMORY.get("card_counts") or [0] * POLICY_CARD_HASH_BUCKETS)
    for index in selected:
        if not isinstance(index, int) or not 0 <= index < len(options):
            continue
        option = options[index]
        option_type = int(getattr(getattr(option, "type", 0), "value", getattr(option, "type", 0)) or 0)
        if 0 <= option_type < OPTION_TYPE_COUNT:
            counts[option_type] += 1
        try:
            card = option_card(obs, option)
            card_id = int(getattr(card, "id", 0) or 0)
        except (AttributeError, TypeError):
            card_id = int(option.get("cardId", 0) or 0) if isinstance(option, dict) else 0
        recent_types = (recent_types + [option_type])[-POLICY_HISTORY_LENGTH:]
        recent_card_ids = (recent_card_ids + [card_id])[-POLICY_HISTORY_LENGTH:]
        if card_id:
            card_counts[card_id % POLICY_CARD_HASH_BUCKETS] += 1
        _POLICY_MEMORY.update(last_type=option_type, last_card_id=card_id)
    _POLICY_MEMORY["counts"] = counts
    _POLICY_MEMORY["recent_types"] = recent_types
    _POLICY_MEMORY["recent_card_ids"] = recent_card_ids
    _POLICY_MEMORY["card_counts"] = card_counts


def _one_hot(value: int, size: int) -> list[float]:
    return [float(index == value) for index in range(size)]


def _id_bits(value: int) -> list[float]:
    normalized = max(0, int(value or 0))
    return [float((normalized >> bit) & 1) for bit in range(ID_BITS)]


def _visible_hash_counts(cards) -> list[float]:
    counts = [0.0] * VISIBLE_HASH_BUCKETS
    for card in cards or []:
        card_id = int(getattr(card, "id", 0) or 0)
        counts[(card_id * 2654435761) % VISIBLE_HASH_BUCKETS] += 1.0
    return counts


def _board_slot_features(player) -> list[float]:
    cards = list(getattr(player, "active", None) or []) + list(getattr(player, "bench", None) or [])
    cards = cards[:BOARD_SLOT_COUNT] + [None] * max(0, BOARD_SLOT_COUNT - len(cards))
    features = []
    for card in cards:
        card_id = int(getattr(card, "id", 0) or 0)
        hp = int(getattr(card, "hp", 0) or 0)
        max_hp = int(getattr(card, "maxHp", hp) or hp)
        features.extend(_id_bits(card_id))
        features.extend((
            float(hp),
            float(max_hp),
            float(max(0, max_hp - hp)),
            float(energy_count(card)),
            float(has_tool(card, HEROES_CAPE)),
        ))
    return features


def _zone(obs, area, player_index):
    state = obs.current
    player = state.players[player_index]
    zones = {
        AreaType.ACTIVE: player.active,
        AreaType.BENCH: player.bench,
        AreaType.HAND: player.hand,
        AreaType.DISCARD: player.discard,
        AreaType.PRIZE: player.prize,
        AreaType.STADIUM: state.stadium,
        AreaType.LOOKING: state.looking,
        AreaType.DECK: obs.select.deck if obs.select is not None else [],
    }
    return zones.get(area, []) or []


def get_card(obs, area, index, player_index):
    if area is None or index is None or player_index is None:
        return None
    zone = _zone(obs, area, player_index)
    return zone[index] if 0 <= index < len(zone) else None


def active(player):
    return player.active[0] if player.active and player.active[0] is not None else None


def _update_lopunny_memory(obs) -> None:
    me = obs.current.players[obs.current.yourIndex]
    current = active(me)
    turn = int(obs.current.turn or 0)
    serial = getattr(current, "serial", None)
    bench_serials = {getattr(pokemon, "serial", None) for pokemon in me.bench or [] if pokemon is not None}
    if _LOPUNNY_MEMORY["turn"] != turn:
        _LOPUNNY_MEMORY.update(turn=turn, active_serial=serial, bench_serials=bench_serials, moved=False)
        return
    previous = _LOPUNNY_MEMORY.get("active_serial")
    if serial != previous:
        if getattr(current, "id", 0) == MEGA_LOPUNNY_EX and serial in _LOPUNNY_MEMORY.get("bench_serials", set()):
            _LOPUNNY_MEMORY["moved"] = True
        _LOPUNNY_MEMORY["active_serial"] = serial
    _LOPUNNY_MEMORY["bench_serials"] = bench_serials


def energy_count(pokemon) -> int:
    if pokemon is None:
        return 0
    return len(getattr(pokemon, "energyCards", None) or getattr(pokemon, "energies", []) or [])


def _pokemon_cards(player) -> list:
    return [
        pokemon
        for pokemon in list(getattr(player, "active", []) or []) + list(getattr(player, "bench", []) or [])
        if pokemon is not None
    ]


def _prize_value(pokemon) -> int:
    card = CARD_TABLE.get(int(getattr(pokemon, "id", 0) or 0))
    if card is None:
        return 1
    if getattr(card, "megaEx", False):
        return 3
    if getattr(card, "ex", False):
        return 2
    return 1


def _opponent_is_mill(obs) -> bool:
    state = obs.current
    opponent = state.players[1 - state.yourIndex]
    visible = _pokemon_cards(opponent) + list(getattr(opponent, "discard", []) or [])
    return any(getattr(card, "id", 0) in {GREAT_TUSK, DWEBBLE, CRUSTLE, TERRAKION} for card in visible)


def _ogerpon_attack_damage(obs, attack_id: int, target=None, perspective_index: int | None = None) -> float:
    attack = ATTACK_TABLE.get(attack_id)
    damage = float(getattr(attack, "damage", 0) or 0)
    if attack_id != MYRIAD_LEAF_SHOWER:
        return damage
    state = obs.current
    player_index = state.yourIndex if perspective_index is None else perspective_index
    me = state.players[player_index]
    opponent = state.players[1 - player_index]
    damage += 30.0 * (energy_count(active(me)) + energy_count(active(opponent)))
    target = target or active(opponent)
    target_card = CARD_TABLE.get(int(getattr(target, "id", 0) or 0))
    if target_card is not None and getattr(target_card, "weakness", None) == GRASS_ENERGY:
        damage *= 2.0
    return damage


def search_stats() -> dict[str, int]:
    return dict(_SEARCH_STATS)


def _visible_ids(player, include_hand: bool) -> list[int]:
    zones = [
        getattr(player, "active", []) or [],
        getattr(player, "bench", []) or [],
        getattr(player, "discard", []) or [],
    ]
    if include_hand:
        zones.append(getattr(player, "hand", []) or [])
    return [int(getattr(card, "id", 0) or 0) for zone in zones for card in zone if card is not None]


def _reset_policy_router() -> None:
    _POLICY_ROUTER_MEMORY["seen_opponent_cards"].clear()
    _POLICY_ROUTER_MEMORY["expert"] = None


def _merge_policy_overrides(config: dict, overrides: dict) -> dict:
    routed = dict(config)
    routed.pop("policy_router", None)
    for key, value in overrides.items():
        if key == "search" and isinstance(value, dict):
            routed["search"] = {**config.get("search", {}), **value}
        else:
            routed[key] = value
    return routed


def _select_policy_config(obs, config: dict) -> dict:
    router = config.get("policy_router") or {}
    if not router.get("enabled") or _POLICY_ROUTER_MEMORY["expert"] == "default":
        return config
    state = getattr(obs, "current", None)
    if state is None:
        return config
    opponent = state.players[1 - state.yourIndex]
    _POLICY_ROUTER_MEMORY["seen_opponent_cards"].update(_visible_ids(opponent, include_hand=False))
    selected = _POLICY_ROUTER_MEMORY["expert"]
    experts = router.get("experts") or {}
    if selected is None:
        turn = int(getattr(state, "turn", 0) or 0)
        maximum_turn = int(router.get("max_turn", 10**9) or 0)
        if turn > maximum_turn:
            _POLICY_ROUTER_MEMORY["expert"] = "default"
            return config
        best = None
        for name, expert in experts.items():
            if turn < int(expert.get("min_turn", router.get("min_turn", 0)) or 0):
                continue
            minimum_evidence = int(expert.get("min_evidence", router.get("min_evidence", 2)) or 0)
            profiles = expert.get("deck_profiles") or []
            if profiles:
                profile_matches = [
                    (
                        len(_POLICY_ROUTER_MEMORY["seen_opponent_cards"] & {int(card_id) for card_id in profile.get("cards", [])}),
                        float(profile.get("score", 0.0)),
                    )
                    for profile in profiles
                ]
                evidence_count, score = max(profile_matches, default=(0, float("-inf")))
            else:
                card_scores = {int(card_id): float(value) for card_id, value in (expert.get("card_scores") or {}).items()}
                evidence = [card_scores[card_id] for card_id in _POLICY_ROUTER_MEMORY["seen_opponent_cards"] if card_id in card_scores]
                evidence_count = len(evidence)
                score = float(expert.get("intercept", 0.0)) + sum(evidence)
            if evidence_count < minimum_evidence:
                continue
            threshold = float(expert.get("threshold", router.get("threshold", 0.0)) or 0.0)
            margin = score - threshold
            if margin >= 0.0 and (best is None or margin > best[0]):
                best = (margin, str(name))
        if best is not None:
            selected = best[1]
            _POLICY_ROUTER_MEMORY["expert"] = selected
    expert = experts.get(selected) if selected is not None else None
    if not isinstance(expert, dict):
        return config
    return _merge_policy_overrides(config, expert.get("overrides") or {})


def _fallback_basic(deck: list[int]) -> int:
    for card_id in deck:
        card = CARD_TABLE.get(card_id)
        if card is not None and card.cardType == CardType.POKEMON and getattr(card, "basic", False):
            return card_id
    return TEAL_MASK_OGERPON_EX


def _partition_hidden_cards(
    deck: list[int],
    counts: tuple[int, ...],
    visible_ids: list[int],
    rng: random.Random,
    fallback: int,
) -> list[list[int]]:
    available = Counter(deck)
    for card_id in visible_ids:
        if available[card_id] > 0:
            available[card_id] -= 1
    pool = [card_id for card_id, quantity in available.items() for _ in range(max(0, quantity))]
    rng.shuffle(pool)
    required = sum(max(0, int(count or 0)) for count in counts)
    while len(pool) < required:
        pool.append(fallback)
    result: list[list[int]] = []
    cursor = 0
    for count in counts:
        count = max(0, int(count or 0))
        result.append(pool[cursor:cursor + count])
        cursor += count
    return result


def _belief_seed(obs, world_index: int) -> int:
    state = obs.current
    values = [int(state.turn or 0), int(state.yourIndex), world_index]
    for player in state.players:
        values.extend((int(getattr(player, "deckCount", 0) or 0), len(getattr(player, "prize", []) or [])))
        values.extend(sorted(_visible_ids(player, include_hand=False)))
    seed = 2166136261
    for value in values:
        seed = ((seed ^ value) * 16777619) & 0xFFFFFFFF
    return seed


def _sample_hidden_worlds(obs, config: dict, world_count: int) -> list[tuple[list[int], list[int], list[int], list[int], list[int], list[int]]]:
    state = obs.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    configured_own_deck = list(map(int, config.get("own_deck", [])))
    own_model = configured_own_deck if len(configured_own_deck) == 60 else read_deck_csv()
    opponent_models = [list(map(int, deck)) for deck in config.get("opponent_decks", []) if len(deck) == 60]
    if not opponent_models:
        opponent_models = [own_model]
    visible_opponent = _visible_ids(opponent, include_hand=False)
    opponent_models.sort(key=lambda deck: sum((Counter(deck) & Counter(visible_opponent)).values()), reverse=True)
    best_overlap = sum((Counter(opponent_models[0]) & Counter(visible_opponent)).values())
    plausible = [
        deck for deck in opponent_models
        if sum((Counter(deck) & Counter(visible_opponent)).values()) == best_overlap
    ] or opponent_models
    worlds = []
    for world_index in range(max(1, world_count)):
        rng = random.Random(_belief_seed(obs, world_index))
        opponent_model = plausible[world_index % len(plausible)]
        own_fallback = _fallback_basic(own_model)
        opponent_fallback = _fallback_basic(opponent_model)
        own_deck, own_prize = _partition_hidden_cards(
            own_model,
            (int(getattr(me, "deckCount", 0) or 0), len(getattr(me, "prize", []) or [])),
            _visible_ids(me, include_hand=True),
            rng,
            own_fallback,
        )
        opponent_deck, opponent_prize, opponent_hand = _partition_hidden_cards(
            opponent_model,
            (
                int(getattr(opponent, "deckCount", 0) or 0),
                len(getattr(opponent, "prize", []) or []),
                int(getattr(opponent, "handCount", 0) or 0),
            ),
            visible_opponent,
            rng,
            opponent_fallback,
        )
        hidden_active = []
        if getattr(opponent, "active", None) and opponent.active and opponent.active[0] is None:
            hidden_active = [opponent_fallback]
        worlds.append((own_deck, own_prize, opponent_deck, opponent_prize, opponent_hand, hidden_active))
    return worlds


def _ogerpon_target_score(obs, target) -> float:
    if target is None:
        return 0.0
    card_id = int(getattr(target, "id", 0) or 0)
    if card_id == CRUSTLE:
        return -900000.0
    damage = _ogerpon_attack_damage(obs, MYRIAD_LEAF_SHOWER, target)
    hp = float(getattr(target, "hp", 0) or 0)
    score = 90000.0 * _prize_value(target) + 1200.0 * energy_count(target)
    if hp > 0 and damage >= hp:
        score += 520000.0 + 90000.0 * _prize_value(target)
    else:
        score += max(0.0, damage - hp) * 120.0 - hp * 80.0
    score += {
        DWEBBLE: 360000.0,
        GREAT_TUSK: 260000.0,
        TERRAKION: 220000.0,
        DURALUDON: 340000.0,
        ARCHALUDON_EX: 180000.0,
        CINDERACE: 260000.0,
    }.get(card_id, 0.0)
    return score


def has_tool(pokemon, card_id: int) -> bool:
    return any(getattr(tool, "id", 0) == card_id for tool in getattr(pokemon, "tools", []) or [])


def _option_index(obs, option_type: OptionType, attack_id: int = 0) -> int | None:
    for index, option in enumerate(obs.select.option):
        if option.type == option_type and (not attack_id or option.attackId == attack_id):
            return index
    return None


def _only_terminal_main_actions(obs) -> bool:
    return not any(
        option.type in (OptionType.PLAY, OptionType.ABILITY, OptionType.ATTACH, OptionType.EVOLVE)
        for option in obs.select.option
    )


def _lopunny_pivot_available(me) -> bool:
    for pokemon in me.bench or []:
        if pokemon is None:
            continue
        if pokemon.id in (MEGA_LOPUNNY_EX, DUDUNSPARCE):
            return True
    return False


def _lopunny_attack_action(obs) -> list[int] | None:
    me = obs.current.players[obs.current.yourIndex]
    opponent = obs.current.players[1 - obs.current.yourIndex]
    my_active = active(me)
    opponent_active = active(opponent)
    active_id = getattr(my_active, "id", 0)
    opponent_hp = int(getattr(opponent_active, "hp", 0) or 0)
    gale = _option_index(obs, OptionType.ATTACK, GALE_THRUST)
    spiky = _option_index(obs, OptionType.ATTACK, SPIKY_HOPPER)
    if active_id == MEGA_LOPUNNY_EX:
        if _LOPUNNY_MEMORY.get("moved") and gale is not None:
            if spiky is not None and 0 < opponent_hp <= 160:
                return [spiky]
            return [gale]
        if spiky is not None:
            return [spiky]
        if gale is not None:
            return [gale]
    return None


def option_card(obs, option):
    state = obs.current
    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    area = option.area
    index = option.index
    if area is None and option.type == OptionType.PLAY:
        area = AreaType.HAND
    return get_card(obs, area, index, player_index)


def option_target(obs, option):
    state = obs.current
    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    return get_card(obs, option.inPlayArea, option.inPlayIndex, player_index)


def _effect_card_id(obs) -> int:
    select = obs.select
    effect = getattr(select, "effect", None)
    context_card = getattr(select, "contextCard", None)
    return int(getattr(effect, "id", 0) or getattr(context_card, "id", 0) or 0)


def _own_pokemon_options(obs) -> list[tuple[int, object]]:
    state = obs.current
    result = []
    for index, option in enumerate(obs.select.option):
        player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
        if option.type != OptionType.CARD or player_index != state.yourIndex:
            continue
        card = option_card(obs, option)
        if card is not None and hasattr(card, "hp"):
            result.append((index, card))
    return result


def _ready_grim_on_bench(obs):
    me = obs.current.players[obs.current.yourIndex]
    candidates = [pokemon for pokemon in me.bench or [] if pokemon.id == GRIMMSNARL_EX]
    return max(candidates, key=lambda pokemon: (energy_count(pokemon) >= 2, energy_count(pokemon), pokemon.hp), default=None)


def _card_counts(cards) -> dict[int, int]:
    counts: dict[int, int] = {}
    for card in cards or []:
        if card is not None:
            counts[card.id] = counts.get(card.id, 0) + 1
    return counts


def _grim_position(obs) -> tuple[dict[int, int], dict[int, int]]:
    me = obs.current.players[obs.current.yourIndex]
    field = _card_counts(list(me.active or []) + list(me.bench or []))
    hand = _card_counts(me.hand or [])
    return field, hand


def _lopunny_position(obs) -> tuple[dict[int, int], dict[int, int]]:
    me = obs.current.players[obs.current.yourIndex]
    field = _card_counts(list(me.active or []) + list(me.bench or []))
    hand = _card_counts(me.hand or [])
    return field, hand


def _lopunny_search_score(obs, card_id: int, effect_card_id: int) -> float:
    field, hand = _lopunny_position(obs)
    bunny_line = field.get(BUNEARY, 0) + field.get(MEGA_LOPUNNY_EX, 0)
    duns_line = field.get(DUNSPARCE, 0) + field.get(DUDUNSPARCE, 0)
    snow_line = field.get(SNORUNT, 0) + field.get(MEGA_FROSLASS_EX, 0)
    if effect_card_id in (BUDDY_BUDDY_POFFIN, FAN_ROTOM):
        if card_id == BUNEARY:
            return 1200000.0 - 500000.0 * bunny_line
        if card_id == DUNSPARCE:
            return 1100000.0 - 260000.0 * min(3, duns_line)
        if card_id == SNORUNT:
            return 760000.0 - 320000.0 * snow_line
        if card_id == FAN_ROTOM:
            return 620000.0 if int(obs.current.turn or 0) <= 2 and not field.get(FAN_ROTOM, 0) else -800000.0
        return -900000.0
    if effect_card_id in (POKE_PAD, 1121):
        if card_id == MEGA_LOPUNNY_EX and field.get(BUNEARY, 0) > hand.get(MEGA_LOPUNNY_EX, 0):
            return 1300000.0
        if card_id == DUDUNSPARCE and field.get(DUNSPARCE, 0) > hand.get(DUDUNSPARCE, 0):
            return 1220000.0
        if card_id == MEGA_FROSLASS_EX and field.get(SNORUNT, 0) > hand.get(MEGA_FROSLASS_EX, 0):
            return 1000000.0
        if card_id == BUNEARY and bunny_line == 0:
            return 920000.0
        if card_id == DUNSPARCE and duns_line < 2:
            return 850000.0
    if effect_card_id == 1225:
        if card_id == MEGA_LOPUNNY_EX and field.get(BUNEARY, 0):
            return 1350000.0
        if card_id == DUDUNSPARCE and field.get(DUNSPARCE, 0):
            return 1200000.0
        if card_id == MEGA_FROSLASS_EX and field.get(SNORUNT, 0):
            return 1080000.0
        if card_id == 13:
            return 1150000.0
        if card_id in (3, 11):
            return 900000.0
    return 0.0


def _lopunny_main_adjustment(obs, option, card, target) -> float:
    state = obs.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    my_active = active(me)
    card_id = int(getattr(card, "id", 0) or 0)
    target_id = int(getattr(target, "id", 0) or 0)
    field, _ = _lopunny_position(obs)
    if option.type == OptionType.ATTACK:
        if option.attackId == GALE_THRUST:
            return 1500000.0 if _LOPUNNY_MEMORY.get("moved") else -1200000.0
        if option.attackId == SPIKY_HOPPER:
            return 1100000.0 if not _LOPUNNY_MEMORY.get("moved") else 400000.0
        if option.attackId == RESENTFUL_REFRAIN:
            refrain_damage = 50 * int(opponent.handCount or len(opponent.hand or []))
            return 1450000.0 + 1500.0 * (refrain_damage - 150)
        if option.attackId == ABSOLUTE_SNOW:
            return 1450000.0
    if option.type == OptionType.RETREAT:
        active_id = getattr(my_active, "id", 0)
        if active_id == MEGA_LOPUNNY_EX and not _LOPUNNY_MEMORY.get("moved") and _lopunny_pivot_available(me):
            return 1500000.0
        if active_id != MEGA_LOPUNNY_EX and field.get(MEGA_LOPUNNY_EX, 0):
            return 1200000.0
        return -500000.0
    if option.type == OptionType.ATTACH:
        energies = energy_count(target)
        if target_id == MEGA_LOPUNNY_EX:
            if getattr(active(opponent), "id", 0) == 345 and energies == 1:
                return 1800000.0
            return 1450000.0 if energies == 0 else (550000.0 if energies == 1 else -1000000.0)
        if target_id == MEGA_FROSLASS_EX:
            return 1200000.0 if energies == 0 else -500000.0
        if target_id in (DUDUNSPARCE, FAN_ROTOM) and has_tool(target, AIR_BALLOON):
            return -900000.0
    if option.type == OptionType.PLAY:
        if card_id == HAND_TRIMMER:
            opponent_hand = int(opponent.handCount or len(opponent.hand or []))
            if opponent_hand <= 5 or getattr(my_active, "id", 0) == MEGA_FROSLASS_EX:
                return -1500000.0
            return 220000.0 * (opponent_hand - 5)
        if card_id == WALLYS_COMPASSION:
            damaged = [
                int(getattr(pokemon, "maxHp", pokemon.hp) or pokemon.hp) - int(pokemon.hp or 0)
                for pokemon in list(me.active or []) + list(me.bench or [])
                if pokemon is not None and pokemon.id in (MEGA_LOPUNNY_EX, MEGA_FROSLASS_EX)
            ]
            maximum_damage = max(damaged, default=0)
            if maximum_damage <= 0:
                return -1800000.0
            return 9000.0 * maximum_damage
        if card_id == BUNEARY and not field.get(BUNEARY, 0) and not field.get(MEGA_LOPUNNY_EX, 0):
            return 1400000.0
        if card_id == DUNSPARCE and field.get(DUNSPARCE, 0) + field.get(DUDUNSPARCE, 0) >= 3:
            return -1000000.0
        if card_id == SNORUNT and field.get(SNORUNT, 0) + field.get(MEGA_FROSLASS_EX, 0) >= 2:
            return -800000.0
    if option.type == OptionType.EVOLVE:
        if card_id == MEGA_LOPUNNY_EX:
            return 1500000.0
        if card_id == DUDUNSPARCE:
            return 1350000.0
        if card_id == MEGA_FROSLASS_EX:
            return 1100000.0
    return 0.0


def _grim_search_score(obs, card_id: int, effect_card_id: int) -> float:
    field, hand = _grim_position(obs)
    imp_line = field.get(IMPIDIMP, 0) + field.get(MORGREM, 0) + field.get(GRIMMSNARL_EX, 0)
    snow_line = field.get(SNORUNT, 0) + field.get(FROSLASS, 0)

    if effect_card_id == BUDDY_BUDDY_POFFIN:
        if card_id == IMPIDIMP:
            return 900000.0 - 120000.0 * imp_line
        if card_id == SNORUNT:
            return 800000.0 - 120000.0 * snow_line
        return 100000.0 if card_id == MUNKIDORI else -500000.0
    if effect_card_id == SPIKEMUTH_GYM:
        if field.get(MORGREM, 0):
            return 950000.0 if card_id == GRIMMSNARL_EX else -100000.0
        if field.get(IMPIDIMP, 0):
            if hand.get(RARE_CANDY, 0):
                return 950000.0 if card_id == GRIMMSNARL_EX else 700000.0 if card_id == MORGREM else 0.0
            return 900000.0 if card_id == MORGREM else 720000.0 if card_id == GRIMMSNARL_EX else 0.0
        return 900000.0 if card_id == IMPIDIMP else 0.0
    if effect_card_id == POKE_PAD:
        if imp_line == 0 and card_id == IMPIDIMP:
            return 990000.0
        if field.get(SNORUNT, 0) > field.get(FROSLASS, 0) and card_id == FROSLASS:
            return 920000.0
        if field.get(IMPIDIMP, 0) > field.get(MORGREM, 0) + field.get(GRIMMSNARL_EX, 0) and card_id == MORGREM:
            return 900000.0
        if field.get(MUNKIDORI, 0) < 1 and card_id == MUNKIDORI:
            return 820000.0
    if effect_card_id == TEAM_ROCKETS_PETREL:
        if imp_line == 0 and card_id == BUDDY_BUDDY_POFFIN:
            return 990000.0
        if field.get(IMPIDIMP, 0) and hand.get(GRIMMSNARL_EX, 0) and card_id == RARE_CANDY:
            return 980000.0
        priorities = (DAWN, BUDDY_BUDDY_POFFIN, POKE_PAD, LILLIES_DETERMINATION, RARE_CANDY, NIGHT_STRETCHER)
        if card_id in priorities:
            return 850000.0 - 60000.0 * priorities.index(card_id)
    if effect_card_id == POKEGEAR_30:
        priorities = (
            (TEAM_ROCKETS_PETREL, LILLIES_DETERMINATION, DAWN, BOSS_ORDERS)
            if imp_line == 0
            else (DAWN, TEAM_ROCKETS_PETREL, LILLIES_DETERMINATION, BOSS_ORDERS)
        )
        if card_id in priorities:
            return 850000.0 - 70000.0 * priorities.index(card_id)
    if effect_card_id == NIGHT_STRETCHER:
        priorities = (GRIMMSNARL_EX, IMPIDIMP, FROSLASS, SNORUNT, MUNKIDORI, DARKNESS_ENERGY)
        if card_id in priorities:
            return 850000.0 - 60000.0 * priorities.index(card_id)
    if effect_card_id == DAWN:
        priorities = (GRIMMSNARL_EX, MORGREM, IMPIDIMP, FROSLASS, SNORUNT, MUNKIDORI)
        if card_id in priorities:
            return 900000.0 - 50000.0 * priorities.index(card_id)
    return 0.0


def _grim_main_adjustment(obs, option, card, target) -> float:
    me = obs.current.players[obs.current.yourIndex]
    opponent = obs.current.players[1 - obs.current.yourIndex]
    field, hand = _grim_position(obs)
    card_id = int(getattr(card, "id", 0) or 0)
    target_id = int(getattr(target, "id", 0) or 0)
    bench_space = max(0, int(me.benchMax or 5) - len(me.bench or []))
    imp_line = field.get(IMPIDIMP, 0) + field.get(MORGREM, 0) + field.get(GRIMMSNARL_EX, 0)
    snow_line = field.get(SNORUNT, 0) + field.get(FROSLASS, 0)

    if option.type == OptionType.PLAY:
        if card_id == IMPIDIMP and imp_line < 2:
            return 950000.0
        if card_id == SNORUNT and snow_line < 2:
            return 900000.0
        if card_id == MUNKIDORI and field.get(MUNKIDORI, 0) < 2:
            return 760000.0
        if card_id == BUDDY_BUDDY_POFFIN and bench_space >= 1 and (imp_line < 2 or snow_line < 2):
            return 930000.0
        if card_id == RARE_CANDY:
            return 1000000.0
        if card_id == DAWN and (not hand.get(GRIMMSNARL_EX, 0) or imp_line < 1):
            return 880000.0
        if card_id == SPIKEMUTH_GYM and not hand.get(GRIMMSNARL_EX, 0):
            return 840000.0
        if card_id == POKE_PAD and (field.get(SNORUNT, 0) > field.get(FROSLASS, 0) or field.get(IMPIDIMP, 0) > field.get(MORGREM, 0)):
            return 800000.0
        if card_id == TEAM_ROCKETS_PETREL:
            return 720000.0
        if card_id == BOSS_ORDERS:
            opponent_active = active(opponent)
            if (
                getattr(active(me), "id", 0) == GRIMMSNARL_EX
                and energy_count(active(me)) >= 2
                and getattr(opponent_active, "id", 0) != DURALUDON
                and any(pokemon.id == DURALUDON for pokemon in opponent.bench or [])
            ):
                return 980000.0
        if card_id == UNFAIR_STAMP:
            return 780000.0
        if card_id == LILLIES_DETERMINATION:
            if imp_line == 0:
                return 850000.0
            if not hand.get(GRIMMSNARL_EX, 0) and not hand.get(MORGREM, 0):
                return 650000.0
            return 300000.0 - 30000.0 * len(me.hand or [])
    if option.type == OptionType.ATTACH:
        energies = energy_count(target)
        if target_id == GRIMMSNARL_EX and energies < 2:
            return 900000.0
        if target_id == MUNKIDORI and energies == 0:
            return 650000.0
        if target_id in (MORGREM, IMPIDIMP) and getattr(active(me), "serial", None) == getattr(target, "serial", None):
            return 500000.0
    if option.type == OptionType.EVOLVE:
        if card_id == GRIMMSNARL_EX:
            return 1000000.0
        if card_id in (MORGREM, FROSLASS):
            return 850000.0
    return 0.0


def forced_action(obs, config: dict) -> list[int] | None:
    """Handle mandatory family routes before any learned option ranking."""
    select = obs.select
    if select is None:
        return None

    if select.context == SelectContext.SKILL_ORDER:
        priority = {FROSLASS: 300, MUNKIDORI: 200, GRIMMSNARL_EX: 100}
        order = sorted(
            range(len(select.option)),
            key=lambda index: priority.get(int(select.option[index].cardId or 0), 0),
            reverse=True,
        )
        count = min(select.maxCount, len(order))
        if count >= select.minCount:
            return order[:count]

    family = config.get("family")
    if family == "teal_ogerpon":
        if select.context == SelectContext.IS_FIRST:
            for index, option in enumerate(select.option):
                if option.type == OptionType.YES:
                    return [index]
        if select.context == SelectContext.ATTACH_TO and _effect_card_id(obs) == TEAL_MASK_OGERPON_EX:
            grass = [
                index for index, option in enumerate(select.option)
                if getattr(option_card(obs, option), "id", 0) == GRASS_ENERGY
            ]
            if grass:
                return [grass[0]]
        if config.get("ogerpon_force_promotion") and select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            ogerpon = [(index, pokemon) for index, pokemon in _own_pokemon_options(obs) if pokemon.id == TEAL_MASK_OGERPON_EX]
            if ogerpon:
                return [max(ogerpon, key=lambda item: (energy_count(item[1]), item[1].hp))[0]]
        if select.context == SelectContext.MAIN:
            if config.get("ogerpon_force_ability"):
                for index, option in enumerate(select.option):
                    if option.type == OptionType.ABILITY and getattr(option_card(obs, option), "id", 0) == TEAL_MASK_OGERPON_EX:
                        return [index]
            if config.get("ogerpon_force_terminal_attack") and _only_terminal_main_actions(obs):
                shower = _option_index(obs, OptionType.ATTACK, MYRIAD_LEAF_SHOWER)
                if shower is not None:
                    return [shower]
        return None

    if family == "mega_lopunny":
        if select.context == SelectContext.IS_FIRST:
            for index, option in enumerate(select.option):
                if option.type == OptionType.YES:
                    return [index]
        if select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            own_options = _own_pokemon_options(obs)
            lopunny = [(index, pokemon) for index, pokemon in own_options if pokemon.id == MEGA_LOPUNNY_EX]
            if lopunny:
                return [max(lopunny, key=lambda item: (energy_count(item[1]), item[1].hp))[0]]
        if select.context == SelectContext.MAIN:
            me = obs.current.players[obs.current.yourIndex]
            my_active = active(me)
            for index, option in enumerate(select.option):
                if option.type == OptionType.ABILITY:
                    card = option_card(obs, option)
                    card_id = getattr(card, "id", 0)
                    if card_id == FAN_ROTOM or (
                        card_id == DUDUNSPARCE
                        and (not config.get("lopunny_deck_guard") or int(me.deckCount or 0) > 3)
                    ):
                        return [index]
            if _only_terminal_main_actions(obs) and (config.get("lopunny_attack_fix") or config.get("lopunny_cycle")):
                retreat = _option_index(obs, OptionType.RETREAT)
                active_id = getattr(my_active, "id", 0)
                if (
                    config.get("lopunny_cycle")
                    and retreat is not None
                    and active_id == MEGA_LOPUNNY_EX
                    and not _LOPUNNY_MEMORY.get("moved")
                    and _lopunny_pivot_available(me)
                    and not (getattr(active(obs.current.players[1 - obs.current.yourIndex]), "id", 0) == 345 and energy_count(my_active) < 2)
                ):
                    return [retreat]
                attack = _lopunny_attack_action(obs)
                if (
                    config.get("lopunny_cycle")
                    and retreat is not None
                    and active_id not in (MEGA_LOPUNNY_EX, MEGA_FROSLASS_EX)
                    and any(pokemon.id == MEGA_LOPUNNY_EX for pokemon in me.bench or [] if pokemon is not None)
                ):
                    return [retreat]
                if attack is not None:
                    return attack
            if getattr(my_active, "id", 0) == MEGA_LOPUNNY_EX and _LOPUNNY_MEMORY.get("moved"):
                attacks = [(index, option) for index, option in enumerate(select.option) if option.type == OptionType.ATTACK]
                if attacks and _only_terminal_main_actions(obs):
                    gale = [index for index, option in attacks if option.attackId == GALE_THRUST]
                    if gale:
                        return [gale[0]]
        return None

    if family == "mega_kangaskhan":
        effect_card_id = _effect_card_id(obs)
        if select.context == SelectContext.ACTIVATE:
            for index, option in enumerate(select.option):
                if option.type == OptionType.YES:
                    return [index]
        if select.context == SelectContext.ATTACH_TO and effect_card_id == TEAL_MASK_OGERPON_EX:
            grass = [
                index for index, option in enumerate(select.option)
                if getattr(option_card(obs, option), "id", 0) == GRASS_ENERGY
            ]
            if grass:
                return [grass[0]]
        if select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            own_options = _own_pokemon_options(obs)
            requirements = {
                MEGA_KANGASKHAN_EX: 3,
                RAGING_BOLT_EX: 2,
                TEAL_MASK_OGERPON_EX: 3,
                WELLSPRING_MASK_OGERPON_EX: 3,
                LATIAS_EX: 3,
                PASSIMIAN: 2,
            }
            ready = [item for item in own_options if energy_count(item[1]) >= requirements.get(item[1].id, 99)]
            if ready:
                priority = {MEGA_KANGASKHAN_EX: 600, RAGING_BOLT_EX: 500, WELLSPRING_MASK_OGERPON_EX: 400, TEAL_MASK_OGERPON_EX: 300}
                return [max(ready, key=lambda item: (priority.get(item[1].id, 0), energy_count(item[1]), item[1].hp))[0]]
        return None

    if family != "grimmsnarl":
        return None

    effect_card_id = _effect_card_id(obs)
    if select.context == SelectContext.ACTIVATE and effect_card_id == GRIMMSNARL_EX:
        for index, option in enumerate(select.option):
            if option.type == OptionType.YES:
                return [index]

    if select.context == SelectContext.ATTACH_FROM and effect_card_id == GRIMMSNARL_EX:
        source_serial = int(getattr(getattr(select, "effect", None), "serial", 0) or 0)

        def target_score(item: tuple[int, object]) -> tuple[int, int, int]:
            _, pokemon = item
            energies = energy_count(pokemon)
            is_source = pokemon.id == GRIMMSNARL_EX and pokemon.serial == source_serial
            if is_source and energies < 2:
                tier = 1500
            elif pokemon.id == GRIMMSNARL_EX and energies < 2:
                tier = 1100
            elif pokemon.id == MUNKIDORI and energies == 0:
                tier = 900
            elif pokemon.id in (MORGREM, IMPIDIMP) and energies < 2:
                tier = 700
            elif pokemon.id == GRIMMSNARL_EX:
                tier = 200
            else:
                tier = 0
            return tier, -energies, int(pokemon.hp or 0)

        options = _own_pokemon_options(obs)
        if options:
            return [max(options, key=target_score)[0]]

    if select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        own_options = _own_pokemon_options(obs)
        ready = [(index, pokemon) for index, pokemon in own_options if pokemon.id == GRIMMSNARL_EX and energy_count(pokemon) >= 2]
        if ready:
            return [max(ready, key=lambda item: (energy_count(item[1]), item[1].hp))[0]]
        grim = [(index, pokemon) for index, pokemon in own_options if pokemon.id == GRIMMSNARL_EX]
        if grim:
            return [max(grim, key=lambda item: (energy_count(item[1]), item[1].hp))[0]]

    if select.context == SelectContext.MAIN:
        me = obs.current.players[obs.current.yourIndex]
        my_active = active(me)
        if getattr(my_active, "id", 0) == GRIMMSNARL_EX:
            attacks = [(index, option) for index, option in enumerate(select.option) if option.type == OptionType.ATTACK]
            if attacks and not any(option.type in (OptionType.PLAY, OptionType.ABILITY, OptionType.ATTACH, OptionType.EVOLVE) for option in select.option):
                shadow = [index for index, option in attacks if option.attackId == SHADOW_BULLET]
                if shadow:
                    return [shadow[0]]
    return None


def ensure_mandatory_minimums(obs, action: list[int], scores: list[float]) -> list[int]:
    select = obs.select
    if select.context != SelectContext.ATTACH_TO or _effect_card_id(obs) != GRIMMSNARL_EX:
        return action
    me = obs.current.players[obs.current.yourIndex]
    source_serial = int(getattr(getattr(select, "effect", None), "serial", 0) or 0)
    source = next(
        (pokemon for pokemon in list(me.active or []) + list(me.bench or []) if pokemon is not None and pokemon.serial == source_serial),
        None,
    )
    required = max(select.minCount, 2 - energy_count(source))
    if len(action) >= required:
        return action
    energy_options = [
        index for index, option in enumerate(select.option)
        if getattr(option_card(obs, option), "id", 0) == DARKNESS_ENERGY
    ]
    ranked = sorted(energy_options, key=lambda index: scores[index], reverse=True)
    result = list(action)
    for index in ranked:
        if index not in result:
            result.append(index)
        if len(result) >= min(required, select.maxCount):
            break
    return result


def feature_vector(obs, option) -> list[float]:
    _sync_policy_memory(obs)
    state = obs.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    my_active = active(me)
    opponent_active = active(opponent)
    card = option_card(obs, option)
    target = option_target(obs, option)
    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    features = [
        float(obs.select.context.value if hasattr(obs.select.context, "value") else obs.select.context or 0),
        float(option.type.value if hasattr(option.type, "value") else option.type or 0),
        float(getattr(card, "id", 0) or 0),
        float(option.attackId or 0),
        float(getattr(target, "id", 0) or 0),
        float(getattr(target, "hp", 0) or 0),
        float(energy_count(target)),
        float(player_index != state.yourIndex),
        float(state.turn or 0),
        float(state.turnActionCount or 0),
        float(me.deckCount or 0),
        float(opponent.deckCount or 0),
        float(len(me.prize or [])),
        float(len(opponent.prize or [])),
        float(me.handCount or len(me.hand or [])),
        float(opponent.handCount or len(opponent.hand or [])),
        float(len(me.bench or [])),
        float(len(opponent.bench or [])),
        float(getattr(my_active, "id", 0) or 0),
        float(getattr(opponent_active, "id", 0) or 0),
        float(getattr(my_active, "hp", 0) or 0),
        float(getattr(opponent_active, "hp", 0) or 0),
        float(energy_count(my_active)),
        float(energy_count(opponent_active)),
        float(bool(state.supporterPlayed)),
        float(bool(state.stadiumPlayed)),
        float(bool(state.energyAttached)),
        float(state.firstPlayer == state.yourIndex),
        float(option.count or 0),
        float(option.number or 0),
    ]
    hand_counts = {card_id: 0 for card_id in KEY_CARD_IDS}
    discard_counts = {card_id: 0 for card_id in KEY_CARD_IDS}
    own_field_counts = {card_id: 0 for card_id in FIELD_CARD_IDS}
    opponent_field_counts = {card_id: 0 for card_id in FIELD_CARD_IDS}
    for value in me.hand or []:
        if value.id in hand_counts:
            hand_counts[value.id] += 1
    for value in me.discard or []:
        if value.id in discard_counts:
            discard_counts[value.id] += 1
    for value in list(me.active or []) + list(me.bench or []):
        if value is not None and value.id in own_field_counts:
            own_field_counts[value.id] += 1
    for value in list(opponent.active or []) + list(opponent.bench or []):
        if value is not None and value.id in opponent_field_counts:
            opponent_field_counts[value.id] += 1
    features.extend(float(hand_counts[card_id]) for card_id in KEY_CARD_IDS)
    features.extend(float(discard_counts[card_id]) for card_id in KEY_CARD_IDS)
    features.extend(float(own_field_counts[card_id]) for card_id in FIELD_CARD_IDS)
    features.extend(float(opponent_field_counts[card_id]) for card_id in FIELD_CARD_IDS)
    features.extend(
        (
            float(getattr(getattr(obs.select, "effect", None), "id", 0) or 0),
            float(getattr(getattr(obs.select, "contextCard", None), "id", 0) or 0),
            float(option.area.value if hasattr(option.area, "value") else option.area or 0),
            float(option.inPlayArea.value if hasattr(option.inPlayArea, "value") else option.inPlayArea or 0),
        )
    )
    attack_damage = float(getattr(ATTACK_TABLE.get(option.attackId), "damage", 0) or 0)
    if option.attackId == GALE_THRUST and _LOPUNNY_MEMORY.get("moved"):
        attack_damage += 170.0
    elif option.attackId == RESENTFUL_REFRAIN:
        attack_damage = 50.0 * float(opponent.handCount or len(opponent.hand or []))
    opponent_hp = float(getattr(opponent_active, "hp", 0) or 0)
    card_energies = list(getattr(card, "energies", None) or [])
    target_energies = list(getattr(target, "energies", None) or [])
    own_ogerpon = [
        pokemon for pokemon in list(me.active or []) + list(me.bench or [])
        if pokemon is not None and pokemon.id == TEAL_MASK_OGERPON_EX
    ]
    own_ogerpon_energies = [energy_count(pokemon) for pokemon in own_ogerpon]
    primary_ogerpon = next(
        (pokemon for pokemon in own_ogerpon if pokemon.serial == _OGERPON_MEMORY.get("primary_serial")),
        None,
    )
    features.extend(
        (
            float(bool(_LOPUNNY_MEMORY.get("moved"))),
            float(max(0, int(getattr(my_active, "maxHp", getattr(my_active, "hp", 0)) or 0) - int(getattr(my_active, "hp", 0) or 0))),
            float(max(0, int(getattr(opponent_active, "maxHp", getattr(opponent_active, "hp", 0)) or 0) - int(getattr(opponent_active, "hp", 0) or 0))),
            float(has_tool(my_active, AIR_BALLOON)),
            float(sum(has_tool(pokemon, AIR_BALLOON) for pokemon in me.bench or [] if pokemon is not None)),
            float(bool(getattr(state, "retreated", False))),
            attack_damage,
            float(bool(option.type == OptionType.ATTACK and opponent_hp > 0 and attack_damage >= opponent_hp)),
            float(option.index if option.index is not None else -1),
            float(option.inPlayIndex if option.inPlayIndex is not None else -1),
            float(getattr(card, "hp", 0) or 0),
            float(getattr(card, "maxHp", 0) or 0),
            float(max(0, int(getattr(card, "maxHp", getattr(card, "hp", 0)) or 0) - int(getattr(card, "hp", 0) or 0))),
            float(energy_count(card)),
            float(card_energies.count(GRASS_ENERGY)),
            float(card_energies.count(GROW_GRASS_ENERGY)),
            float(has_tool(card, HEROES_CAPE)),
            float(bool(getattr(card, "appearThisTurn", False))),
            float(getattr(target, "maxHp", 0) or 0),
            float(max(0, int(getattr(target, "maxHp", getattr(target, "hp", 0)) or 0) - int(getattr(target, "hp", 0) or 0))),
            float(target_energies.count(GRASS_ENERGY)),
            float(target_energies.count(GROW_GRASS_ENERGY)),
            float(has_tool(target, HEROES_CAPE)),
            float(sum(own_ogerpon_energies)),
            float(max(own_ogerpon_energies, default=0)),
            float(min(own_ogerpon_energies, default=0)),
            float(sum(int(pokemon.hp < pokemon.maxHp) for pokemon in own_ogerpon)),
            float(sum(has_tool(pokemon, HEROES_CAPE) for pokemon in own_ogerpon)),
            float(energy_count(primary_ogerpon)),
            float(sum(energy_count(pokemon) >= 3 for pokemon in own_ogerpon)),
            float(getattr(my_active, "serial", None) == _OGERPON_MEMORY.get("primary_serial")),
        )
    )
    context_value = int(getattr(obs.select.context, "value", obs.select.context) or 0)
    option_type_value = int(getattr(option.type, "value", option.type) or 0)
    features.extend(_one_hot(context_value, SELECT_CONTEXT_COUNT))
    features.extend(_one_hot(option_type_value, OPTION_TYPE_COUNT))
    for value in (
        getattr(card, "id", 0),
        option.attackId or 0,
        getattr(target, "id", 0),
        getattr(my_active, "id", 0),
        getattr(opponent_active, "id", 0),
        getattr(getattr(obs.select, "effect", None), "id", 0),
        getattr(getattr(obs.select, "contextCard", None), "id", 0),
    ):
        features.extend(_id_bits(int(value or 0)))
    counts = list(_POLICY_MEMORY.get("counts") or [0] * OPTION_TYPE_COUNT)
    features.extend(float(counts[index] if index < len(counts) else 0) for index in range(OPTION_TYPE_COUNT))
    features.extend(_one_hot(int(_POLICY_MEMORY.get("last_type", -1)), OPTION_TYPE_COUNT))
    features.extend(_id_bits(int(_POLICY_MEMORY.get("last_card_id", 0) or 0)))
    features.append(float(sum(counts)))
    recent_types = list(_POLICY_MEMORY.get("recent_types") or [])[-POLICY_HISTORY_LENGTH:]
    recent_card_ids = list(_POLICY_MEMORY.get("recent_card_ids") or [])[-POLICY_HISTORY_LENGTH:]
    recent_types = [-1] * (POLICY_HISTORY_LENGTH - len(recent_types)) + recent_types
    recent_card_ids = [0] * (POLICY_HISTORY_LENGTH - len(recent_card_ids)) + recent_card_ids
    for option_type in recent_types:
        features.extend(_one_hot(int(option_type), OPTION_TYPE_COUNT))
    for card_id in recent_card_ids:
        features.extend(_id_bits(int(card_id or 0)))
    card_counts = list(_POLICY_MEMORY.get("card_counts") or [0] * POLICY_CARD_HASH_BUCKETS)
    features.extend(float(card_counts[index] if index < len(card_counts) else 0) for index in range(POLICY_CARD_HASH_BUCKETS))
    board_energy = sum(
        energy_count(pokemon)
        for pokemon in list(me.active or []) + list(me.bench or [])
        if pokemon is not None
    )
    features.extend((
        float((getattr(me, "handCount", 0) or len(me.hand or [])) - int(_POLICY_MEMORY.get("start_hand", 0))),
        float(int(_POLICY_MEMORY.get("start_deck", 0)) - int(getattr(me, "deckCount", 0) or 0)),
        float(len(me.bench or []) - int(_POLICY_MEMORY.get("start_bench", 0))),
        float(board_energy - int(_POLICY_MEMORY.get("start_energy", 0))),
        float(int(getattr(my_active, "serial", -1) or -1) != int(_POLICY_MEMORY.get("start_active_serial", -1))),
    ))
    features.extend(_visible_hash_counts(me.hand))
    features.extend(_visible_hash_counts(me.discard))
    opponent_visible = list(opponent.active or []) + list(opponent.bench or []) + list(opponent.discard or [])
    features.extend(_visible_hash_counts(opponent_visible))
    features.extend(_board_slot_features(me))
    features.extend(_board_slot_features(opponent))
    return features


def _tree_value(node: dict, features: list[float]) -> float:
    while "leaf_value" not in node:
        feature = int(node.get("split_feature", 0))
        threshold = node.get("threshold", 0.0)
        value = features[feature] if feature < len(features) else 0.0
        if node.get("decision_type") in ("==", "in"):
            choices = {float(item) for item in str(threshold).split("||") if item != ""}
            go_left = value in choices
        else:
            go_left = value <= float(threshold)
        node = node["left_child"] if go_left else node["right_child"]
    return float(node.get("leaf_value", 0.0))


def _model_has_trees(model: dict | None) -> bool:
    return bool(model) and ("tree_info" in model or "oblivious_trees" in model)


def model_score(model: dict, features: list[float]) -> float:
    if "oblivious_trees" in model:
        scale_and_bias = model.get("scale_and_bias")
        scale, bias = 1.0, 0.0
        if isinstance(scale_and_bias, dict):
            scale = float(scale_and_bias.get("scale", 1.0))
            bias = float(scale_and_bias.get("bias", 0.0))
        elif isinstance(scale_and_bias, (list, tuple)) and scale_and_bias:
            scale = float(scale_and_bias[0])
            bias_raw = scale_and_bias[1] if len(scale_and_bias) > 1 else 0.0
            bias = float(bias_raw[0]) if isinstance(bias_raw, (list, tuple)) else float(bias_raw)
        total = 0.0
        for tree in model["oblivious_trees"]:
            index = 0
            for level, split in enumerate(tree.get("splits", [])):
                feature = int(split.get("float_feature_index", 0))
                value = features[feature] if feature < len(features) else 0.0
                if value > float(split.get("border", 0.0)):
                    index |= 1 << level
            leaf_values = tree.get("leaf_values", [])
            total += float(leaf_values[index]) if index < len(leaf_values) else 0.0
        return total * scale + bias
    score = float(model.get("average_output", 0.0) or 0.0)
    for tree in model.get("tree_info", []):
        score += float(tree.get("shrinkage", 1.0)) * _tree_value(tree["tree_structure"], features)
    return score


def advantage_feature_vector(obs, candidate_index: int, anchor_index: int) -> list[float]:
    candidate = feature_vector(obs, obs.select.option[candidate_index])
    anchor = feature_vector(obs, obs.select.option[anchor_index])
    return candidate + anchor + [left - right for left, right in zip(candidate, anchor)]


def advantage_override(
    obs,
    anchor: list[int],
    scores: list[float],
    searched: list[int] | None,
    model: dict | None,
) -> list[int] | None:
    ensemble = (model or {}).get("advantage_ensemble") or []
    calibration = (model or {}).get("advantage_calibration") or {}
    if not ensemble or len(anchor) != 1 or obs.select.maxCount != 1 or len(obs.select.option) < 2:
        return None
    if _ADVANTAGE_MEMORY["overrides"] >= int(calibration.get("max_overrides_per_game", 8)):
        return None
    anchor_index = int(anchor[0])
    if not 0 <= anchor_index < len(obs.select.option):
        return None
    candidate_limit = max(2, int(calibration.get("candidate_limit", 4)))
    candidates = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:candidate_limit]
    if anchor_index not in candidates:
        candidates.append(anchor_index)
    context = SelectContext(obs.select.context).name
    thresholds = calibration.get("context_thresholds") or {}
    threshold = float(thresholds.get(context, calibration.get("default_threshold", 0.0)))
    terminal_threshold = float(calibration.get("terminal_threshold", threshold))
    uncertainty_weight = float(calibration.get("uncertainty_weight", 1.5))
    max_std = float(calibration.get("max_std", float("inf")))
    allow_terminal = bool(calibration.get("allow_terminal", False))
    terminal_types = {
        int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)),
        int(getattr(OptionType.END, "value", OptionType.END)),
    }
    best_index = anchor_index
    best_lcb = threshold
    for candidate_index in candidates:
        if candidate_index == anchor_index:
            continue
        option_type = int(getattr(obs.select.option[candidate_index].type, "value", obs.select.option[candidate_index].type))
        is_terminal = option_type in terminal_types
        if is_terminal and (not allow_terminal or searched != [candidate_index]):
            continue
        features = advantage_feature_vector(obs, candidate_index, anchor_index)
        predictions = [model_score(member, features) for member in ensemble if _model_has_trees(member)]
        if len(predictions) < int(calibration.get("minimum_models", 3)):
            continue
        mean = sum(predictions) / len(predictions)
        variance = sum((value - mean) ** 2 for value in predictions) / len(predictions)
        std = math.sqrt(variance)
        if not math.isfinite(mean) or not math.isfinite(std) or std > max_std:
            continue
        lcb = mean - uncertainty_weight * std
        required = terminal_threshold if is_terminal else threshold
        if lcb > required and lcb > best_lcb:
            best_lcb = lcb
            best_index = candidate_index
    if best_index == anchor_index:
        return None
    _ADVANTAGE_MEMORY["overrides"] += 1
    return [best_index]


def aggregate_option_type_features(
    feature_rows: list[list[float]],
    option_types: list[int],
    base_scores: list[float] | None = None,
    feature_mode: str = "full",
) -> tuple[list[list[float]], list[int]]:
    grouped: dict[int, list[tuple[list[float], float]]] = {}
    scores = base_scores if base_scores is not None else [0.0] * len(feature_rows)
    for features, option_type, base_score in zip(feature_rows, option_types, scores):
        grouped.setdefault(int(option_type), []).append((features, float(base_score)))
    aggregated = []
    types = []
    for option_type in sorted(grouped):
        entries = grouped[option_type]
        rows = [entry[0] for entry in entries]
        if not rows or len(rows[0]) < len(FEATURE_NAMES):
            aggregated.append([sum(values) / len(rows) for values in zip(*rows)])
            types.append(option_type)
            continue
        type_scores = [entry[1] for entry in entries]
        ordered_scores = sorted(type_scores, reverse=True)
        first = rows[0]
        state_features = [first[index] for index in TYPE_STATE_FEATURE_INDICES]
        attack_damage = [row[_FEATURE_INDEX["effective_attack_damage"]] for row in rows]
        attack_ko = [row[_FEATURE_INDEX["attack_would_ko"]] for row in rows]
        target_hp = [row[_FEATURE_INDEX["target_hp"]] for row in rows]
        target_energy = [row[_FEATURE_INDEX["target_energy"]] for row in rows]
        option_energy = [row[_FEATURE_INDEX["option_card_energy"]] for row in rows]
        option_hp = [row[_FEATURE_INDEX["option_card_hp"]] for row in rows]
        option_damage = [row[_FEATURE_INDEX["option_card_damage"]] for row in rows]
        option_appeared = [row[_FEATURE_INDEX["option_card_appeared"]] for row in rows]
        card_ids = [int(row[_FEATURE_INDEX["card_id"]] or 0) for row in rows]
        candidate_card_hash = [0.0] * POLICY_CARD_HASH_BUCKETS
        for card_id in card_ids:
            if card_id:
                candidate_card_hash[card_id % POLICY_CARD_HASH_BUCKETS] += 1.0
        score_mean = sum(type_scores) / len(type_scores)
        score_variance = sum((score - score_mean) ** 2 for score in type_scores) / len(type_scores)
        row = state_features + [
            float(option_type),
            *_one_hot(option_type, OPTION_TYPE_COUNT),
            float(len(rows)),
            float(len(feature_rows)),
            max(type_scores),
            score_mean,
            score_variance ** 0.5,
            ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0],
            ordered_scores[2] if len(ordered_scores) > 2 else ordered_scores[-1],
            max(attack_damage),
            max(attack_ko),
            max(target_hp),
            max(target_energy),
            max(option_energy),
            sum(option_energy) / len(option_energy),
            max(option_hp),
            max(option_damage),
            max(option_appeared),
            *candidate_card_hash,
            float(any(card_ids)),
            float(option_type == int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK))),
            max(attack_ko),
        ]
        if feature_mode == "compact_v1":
            row = [row[index] for index in ROUTER_FEATURE_INDICES]
        aggregated.append(row)
        types.append(option_type)
    return aggregated, types


def stop_decision_features(type_features: dict[int, list[float]]) -> list[float] | None:
    if not type_features:
        return None
    feature_names = ROUTER_FEATURE_NAMES if len(next(iter(type_features.values()))) == len(ROUTER_FEATURE_NAMES) else TYPE_FEATURE_NAMES
    score_index = feature_names.index("base_score_max")
    continuing = [value for value in type_features if value not in {
        int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)),
        int(getattr(OptionType.END, "value", OptionType.END)),
    }]
    terminal = [value for value in type_features if value not in continuing]
    if not continuing or not terminal:
        return None
    continue_type = max(continuing, key=lambda value: type_features[value][score_index])
    terminal_type = max(terminal, key=lambda value: type_features[value][score_index])
    continue_row = type_features[continue_type]
    terminal_row = type_features[terminal_type]

    def value(row: list[float], name: str) -> float:
        return row[feature_names.index(name)] if name in feature_names else 0.0

    candidate_index = feature_names.index("candidate_count")
    continue_candidates = sum(type_features[value][candidate_index] for value in continuing)
    continue_counts = {
        option_type: type_features.get(option_type, [0.0] * len(feature_names))[candidate_index]
        for option_type in (
            int(getattr(OptionType.PLAY, "value", OptionType.PLAY)),
            int(getattr(OptionType.ABILITY, "value", OptionType.ABILITY)),
            int(getattr(OptionType.ATTACH, "value", OptionType.ATTACH)),
            int(getattr(OptionType.EVOLVE, "value", OptionType.EVOLVE)),
            int(getattr(OptionType.RETREAT, "value", OptionType.RETREAT)),
        )
    }
    unseen = sum(
        1
        for option_type in continuing
        if value(continue_row, f"turn_option_count_{option_type}") <= 0
    )
    attack_type = int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK))
    attack_row = type_features.get(attack_type, terminal_row)
    extras = [
        float(len(continuing)),
        float(continue_candidates),
        *[float(continue_counts[option_type]) for option_type in continue_counts],
        float(unseen),
        value(continue_row, "turn_action_count"),
        value(attack_row, "attack_damage_max"),
        value(attack_row, "attack_ko_any"),
        value(terminal_row, "base_score_max") - value(continue_row, "base_score_max"),
        value(continue_row, "turn_energy_delta"),
        value(continue_row, "turn_hand_delta"),
        value(continue_row, "turn_bench_delta"),
    ]
    return [*continue_row, *terminal_row, *(terminal - continuing for continuing, terminal in zip(continue_row, terminal_row)), *extras]


def _standardized(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    scale = variance ** 0.5
    if scale <= 1e-9:
        return {key: 0.0 for key in values}
    return {key: (value - mean) / scale for key, value in values.items()}


def score_main_options_v7(
    model: dict,
    feature_rows: list[list[float]],
    option_types: list[int],
    base_scores: list[float],
    fallback_scores: list[float] | None = None,
    model_weight: float = 1.0,
) -> list[float]:
    heads = model.get("heads") or {}
    calibration = model.get("router_calibration") or {}
    feature_mode = str(calibration.get("feature_mode", "compact_v1"))
    type_rows, present_types = aggregate_option_type_features(
        feature_rows, option_types, base_scores, feature_mode=feature_mode
    )
    type_features = dict(zip(present_types, type_rows))
    stop_feature_mode = str(calibration.get("stop_feature_mode", "full"))
    stop_rows, stop_types = aggregate_option_type_features(
        feature_rows, option_types, base_scores, feature_mode=stop_feature_mode
    )
    stop_type_features = dict(zip(stop_types, stop_rows))

    def head_scores(name: str) -> dict[int, float]:
        head = heads.get(name)
        if not _model_has_trees(head):
            return {option_type: 0.0 for option_type in present_types}
        return {
            option_type: model_score(head, type_features[option_type])
            for option_type in present_types
        }

    type_scores = _standardized(head_scores("main_type"))
    continue_scores = head_scores("main_continue_type")
    terminal_scores = head_scores("main_terminal_type")
    stage_scores = _standardized({
        option_type: (
            terminal_scores[option_type]
            if option_type in {
                int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)),
                int(getattr(OptionType.END, "value", OptionType.END)),
            }
            else continue_scores[option_type]
        )
        for option_type in present_types
    })
    terminal = [value for value in present_types if value in {
        int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)),
        int(getattr(OptionType.END, "value", OptionType.END)),
    }]
    continuing = [value for value in present_types if value not in terminal]
    feature_names = ROUTER_FEATURE_NAMES if stop_feature_mode == "compact_v1" else TYPE_FEATURE_NAMES
    score_index = feature_names.index("base_score_max")
    continue_representative = max(continuing, key=lambda value: stop_type_features[value][score_index]) if continuing else None
    terminal_representative = max(terminal, key=lambda value: stop_type_features[value][score_index]) if terminal else None
    stop_features = stop_decision_features(stop_type_features)
    schema_version = int(calibration.get("schema_version", 1) or 1)
    regime = "no_attack"
    if int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)) in terminal:
        ko_index = feature_names.index("attack_ko_any")
        attack_type = int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK))
        regime = "ko_ready" if stop_type_features[attack_type][ko_index] > 0 else "attack_ready"
    regime_calibration = (calibration.get("regimes") or {}).get(regime, calibration)
    if stop_features is not None:
        stop_features = apply_stop_feature_ablation(
            stop_features,
            str(regime_calibration.get("ablation_mode", "full")),
        )
    stop_head = heads.get(f"main_stop_{regime}") if schema_version >= 2 else heads.get("main_stop")
    if stop_head is None:
        stop_head = heads.get("main_stop")
    member_scores = [
        model_score(head, stop_features)
        for name, head in heads.items()
        if name.startswith(f"main_stop_{regime}_seed") and _model_has_trees(head) and stop_features is not None
    ]
    stop_margin = (
        sum(member_scores) / len(member_scores)
        if member_scores else model_score(stop_head, stop_features)
        if _model_has_trees(stop_head) and stop_features is not None else 0.0
    )
    stop_std = (
        (sum((score - stop_margin) ** 2 for score in member_scores) / len(member_scores)) ** 0.5
        if member_scores else 0.0
    )
    threshold = float(regime_calibration.get("stop_threshold", 0.0) or 0.0)
    confidence_band = float(regime_calibration.get("stop_confidence_band", 0.0) or 0.0)
    terminal_preferred = stop_margin >= threshold
    confidence_multiplier = 1.0
    if abs(stop_margin - threshold) < confidence_band or stop_std > float(regime_calibration.get("uncertainty_threshold", float("inf"))):
        confidence_multiplier = float(calibration.get("low_confidence_multiplier", 0.25) or 0.25)
    fallback_scores = fallback_scores or [0.0] * len(option_types)
    standardized_base = _standardized(dict(enumerate(base_scores)))
    raw_option_scores = {}
    for option, (features, option_type) in enumerate(zip(feature_rows, option_types)):
        option_head = heads.get(MAIN_TYPE_HEADS_RUNTIME.get(option_type, ""))
        raw_option_scores[option] = model_score(option_head, features) if _model_has_trees(option_head) else 0.0
    standardized_option = _standardized(raw_option_scores)
    safety_residual = model.get("residual_q") or {}
    residual_enabled = confidence_multiplier < 1.0 and _model_has_trees(safety_residual)
    residual_weight = min(1.0, max(0.0, float(calibration.get("safety_residual_weight", 0.0) or 0.0)))
    result = []
    for option, features, option_type, base_score, fallback in zip(
        range(len(option_types)), feature_rows, option_types, base_scores, fallback_scores
    ):
        stage_gate = 1.0 if (option_type in {
            int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)),
            int(getattr(OptionType.END, "value", OptionType.END)),
        }) == terminal_preferred else -1.0
        raw = (
            standardized_base.get(option, 0.0)
            + float(calibration.get("option_weight", 1.0)) * standardized_option.get(option, 0.0)
            + float(calibration.get("main_type_weight", 0.5)) * type_scores.get(option_type, 0.0)
            + float(calibration.get("stage_type_weight", 0.25)) * stage_scores.get(option_type, 0.0)
            + min(2.0, float(calibration.get("stop_weight", 0.25))) * confidence_multiplier * stage_gate
        )
        if residual_enabled and residual_weight:
            raw += residual_weight * max(-0.5, min(0.5, model_score(safety_residual, features)))
        result.append(float(fallback) + model_weight * raw)
    return result


def model_head(obs) -> str:
    context = SelectContext(obs.select.context)
    if context == SelectContext.MAIN:
        return "main"
    if context in {SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO}:
        return "attach"
    if context in {
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.SWITCH_ENERGY_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.TO_HAND_ENERGY,
        SelectContext.TO_DECK_ENERGY,
        SelectContext.SWITCH_ENERGY,
    }:
        return "energy"
    if context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
        return "retreat"
    return "card"


def resolve_model(model: dict | None, obs) -> tuple[dict | None, float]:
    if not model:
        return None, 0.0
    if "global" not in model:
        return model, 0.0
    head = model_head(obs)
    selected = (model.get("heads") or {}).get(head) or model.get("global")
    threshold = float((model.get("confidence_thresholds") or {}).get(head, (model.get("confidence_thresholds") or {}).get("global", 0.0)) or 0.0)
    return selected, threshold


def _priority(config: dict, key: str, card_id: int, default: float = 0.0) -> float:
    values = config.get(key, {})
    return float(values.get(str(card_id), values.get(card_id, default)))


def fallback_score(obs, option, config: dict) -> float:
    state = obs.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    context = obs.select.context
    card = option_card(obs, option)
    target = option_target(obs, option)
    card_id = getattr(card, "id", 0) or 0
    target_id = getattr(target, "id", 0) or 0
    score = 0.0

    if context == SelectContext.MAIN:
        if option.type == OptionType.PLAY:
            score = 300.0 + _priority(config, "play_priority", card_id)
        elif option.type == OptionType.EVOLVE:
            score = 850.0 + _priority(config, "evolve_priority", target_id)
        elif option.type == OptionType.ATTACH:
            score = 650.0 + _priority(config, "attach_priority", target_id)
        elif option.type == OptionType.ABILITY:
            score = 720.0 + _priority(config, "ability_priority", card_id)
        elif option.type == OptionType.RETREAT:
            score = _priority(config, "active_priority", target_id, -150.0)
        elif option.type == OptionType.ATTACK:
            attack = ATTACK_TABLE.get(option.attackId)
            damage = float(getattr(attack, "damage", 0) or 0)
            score = 900.0 + damage * 2.0 + _priority(config, "attack_priority", option.attackId or 0)
            opponent_active = active(opponent)
            if opponent_active is not None and damage >= opponent_active.hp:
                score += 900.0 + 350.0 * (2 if getattr(CARD_TABLE.get(opponent_active.id), "ex", False) else 1)
        elif option.type == OptionType.END:
            score = -900.0
    elif option.type == OptionType.CARD:
        if context == SelectContext.SETUP_ACTIVE_POKEMON:
            score = _priority(config, "setup_active_priority", card_id)
        elif context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            score = _priority(config, "setup_bench_priority", card_id)
        elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
            if player_index != state.yourIndex:
                score = _priority(config, "target_priority", card_id)
            else:
                score = _priority(config, "active_priority", card_id)
        elif context == SelectContext.TO_HAND:
            score = _priority(config, "search_priority", card_id)
        elif context in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
            score = -_priority(config, "keep_priority", card_id)
        elif context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
            score = -float(getattr(card, "hp", 0) or 0)
            if option.playerIndex != state.yourIndex:
                score = -score + _priority(config, "target_priority", card_id)
        else:
            score = _priority(config, "card_priority", card_id)
    elif option.type == OptionType.YES:
        score = 100.0
    elif option.type == OptionType.NO:
        score = 0.0
    elif option.type == OptionType.NUMBER:
        score = float(option.number or 0)
    else:
        score = float(option.count or 0)

    own_deck_guard = int(config.get("deck_guard", 0) or 0)
    if own_deck_guard and me.deckCount <= own_deck_guard and card_id in set(config.get("optional_thin_cards", [])):
        score -= 3000.0
    if config.get("family") == "grimmsnarl":
        if context == SelectContext.MAIN:
            score += _grim_main_adjustment(obs, option, card, target)
        elif option.type == OptionType.CARD and context in (
            SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD,
        ):
            score += _grim_search_score(obs, card_id, _effect_card_id(obs))
    elif config.get("family") == "mega_lopunny":
        if context == SelectContext.MAIN and config.get("lopunny_main_adjustment"):
            score += _lopunny_main_adjustment(obs, option, card, target)
        elif config.get("lopunny_search_adjustment") and option.type == OptionType.CARD and context in (
            SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD,
        ):
            score += _lopunny_search_score(obs, card_id, _effect_card_id(obs))
    elif config.get("family") == "teal_ogerpon":
        tactical = bool(config.get("ogerpon_tactical"))
        anti_mill = bool(config.get("ogerpon_anti_mill")) and _opponent_is_mill(obs)
        if config.get("ogerpon_targetfix") and context in (SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM):
            score += max(0.0, 3.0 - energy_count(target)) * 90.0
            if getattr(target, "id", 0) == TEAL_MASK_OGERPON_EX:
                score += 180.0
        if tactical and context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
            if player_index != state.yourIndex:
                score += _ogerpon_target_score(obs, card)
        if anti_mill and context == SelectContext.MAIN:
            ready_ogerpon = any(energy_count(pokemon) >= 3 for pokemon in _pokemon_cards(me) if pokemon.id == TEAL_MASK_OGERPON_EX)
            deck_gap_unsafe = int(getattr(me, "deckCount", 60) or 0) <= int(getattr(opponent, "deckCount", 60) or 0) + 6
            if option.type == OptionType.PLAY and card_id == 1120:
                score += 520000.0
            if ready_ogerpon and deck_gap_unsafe:
                if option.type == OptionType.PLAY and card_id in {
                    BUG_CATCHING_SET, ENERGY_SEARCH, TERA_ORB, POKEGEAR_30,
                    LILLIES_DETERMINATION, JUDGE, HARLEQUIN,
                }:
                    score -= 520000.0
                elif option.type == OptionType.ABILITY and card_id == TEAL_MASK_OGERPON_EX:
                    score -= 440000.0
                elif option.type == OptionType.ATTACK:
                    score += 360000.0
        if config.get("ogerpon_endgame"):
            if option.type == OptionType.ATTACK:
                damage = _ogerpon_attack_damage(obs, int(option.attackId or 0))
                opponent_active = active(opponent)
                if opponent_active is not None and damage >= float(opponent_active.hp or 0):
                    score += 1200.0
            elif option.type == OptionType.END and (me.deckCount or 0) <= 4:
                score -= 900.0
    return score


def score_options(obs, config: dict, model: dict | None) -> list[float]:
    scores = []
    model_weight = float(config.get("model_weight", 500.0))
    selected_model, _ = resolve_model(model, obs)
    value_model = model.get("value") if model and "global" in model else None
    residual_model = model.get("residual_q") if model and "global" in model else None
    value_weight = float(config.get("value_weight", 0.0) or 0.0)
    residual_weight = float(config.get("residual_weight", 0.0) or 0.0)
    calibration = (model or {}).get("calibration") or {}
    type_head_weight = float(calibration.get("main_type_weight", config.get("type_head_weight", 0.0)))
    option_head_weight = float(calibration.get("main_option_weight", config.get("type_head_weight", 0.0)))
    type_head_weights = config.get("type_head_weights") or {}
    main_context = SelectContext(obs.select.context) == SelectContext.MAIN
    option_types = [int(getattr(option.type, "value", option.type) or 0) for option in obs.select.option]
    heads = (model or {}).get("heads") or {}
    has_regime_stop = all(
        _model_has_trees(heads.get(f"main_stop_{regime}"))
        for regime in ("no_attack", "attack_ready", "ko_ready")
    )
    main_type_model = heads.get("main_type") if main_context else None
    needs_features = any((
        _model_has_trees(selected_model),
        _model_has_trees(value_model) and bool(value_weight),
        _model_has_trees(residual_model) and bool(residual_weight),
        _model_has_trees(main_type_model) and bool(type_head_weight),
        main_context and int((model or {}).get("version", 0) or 0) >= 6 and (
            _model_has_trees(heads.get("main_stop")) or has_regime_stop
        ),
        main_context and bool(option_head_weight) and any(_model_has_trees(value) for value in heads.values()),
    ))
    feature_rows = [feature_vector(obs, option) for option in obs.select.option] if needs_features else [None] * len(option_types)
    base_scores = [
        model_score(selected_model, features) if features is not None and _model_has_trees(selected_model) else 0.0
        for features in feature_rows
    ]
    if main_context and int((model or {}).get("version", 0) or 0) >= 7 and (
        _model_has_trees(heads.get("main_stop")) or has_regime_stop
    ):
        return score_main_options_v7(
            model or {},
            feature_rows,
            option_types,
            base_scores,
            [fallback_score(obs, option, config) for option in obs.select.option],
            model_weight,
        )
    if main_context and int((model or {}).get("version", 0) or 0) == 6 and _model_has_trees(heads.get("main_stop")):
        type_rows, present_types = aggregate_option_type_features(feature_rows, option_types, base_scores)
        type_features = dict(zip(present_types, type_rows))
        stop_model = heads["main_stop"]
        stop_scores = {option_type: model_score(stop_model, type_features[option_type]) for option_type in present_types}
        terminal_types = {
            int(getattr(OptionType.ATTACK, "value", OptionType.ATTACK)),
            int(getattr(OptionType.END, "value", OptionType.END)),
        }
        terminal = [option_type for option_type in present_types if option_type in terminal_types]
        continuing = [option_type for option_type in present_types if option_type not in terminal_types]
        choose_terminal = bool(terminal) and (
            not continuing or max(stop_scores[value] for value in terminal) > max(stop_scores[value] for value in continuing)
        )
        stage_types = terminal if choose_terminal else continuing
        stage_head = heads.get("main_terminal_type" if choose_terminal else "main_continue_type")
        if _model_has_trees(stage_head):
            chosen_type = max(stage_types, key=lambda value: model_score(stage_head, type_features[value]))
        else:
            chosen_type = max(stage_types, key=lambda value: stop_scores[value])
        routed_scores = []
        for option, features, option_type, base_score in zip(obs.select.option, feature_rows, option_types, base_scores):
            if option_type != chosen_type:
                routed_scores.append(float("-inf"))
                continue
            score = fallback_score(obs, option, config) + model_weight * base_score
            type_head_name, type_head = _main_type_head(model, option)
            if type_head is not None:
                score += model_weight * model_score(type_head, features)
            routed_scores.append(score)
        return routed_scores
    main_type_scores: dict[int, float] = {}
    if main_context and _model_has_trees(main_type_model) and type_head_weight:
        type_rows, present_types = aggregate_option_type_features(feature_rows, option_types, base_scores)
        main_type_scores = {
            option_type: model_score(main_type_model, features)
            for option_type, features in zip(present_types, type_rows)
        }
    if residual_model and config.get("residual_low_confidence_only", True):
        margin, threshold = model_confidence(obs, model)
        if threshold > 0 and margin >= threshold:
            residual_weight = 0.0
    for option, features, option_type, base_score in zip(obs.select.option, feature_rows, option_types, base_scores):
        score = fallback_score(obs, option, config)
        if features is not None and _model_has_trees(selected_model):
            score += model_weight * base_score
            if main_context and type_head_weight and option_type in main_type_scores:
                score += model_weight * type_head_weight * main_type_scores[option_type]
            if main_context and option_head_weight:
                type_head_name, type_head = _main_type_head(model, option)
                if type_head is not None:
                    multiplier = float(type_head_weights.get(type_head_name, 1.0))
                    score += model_weight * option_head_weight * multiplier * model_score(type_head, features)
            if _model_has_trees(value_model) and value_weight:
                score += value_weight * model_score(value_model, features)
            if _model_has_trees(residual_model) and residual_weight:
                score += residual_weight * model_score(residual_model, features)
        scores.append(score)
    return scores


def model_confidence(obs, model: dict | None) -> tuple[float, float]:
    selected_model, threshold = resolve_model(model, obs)
    if selected_model is None:
        return 0.0, threshold
    scores = [model_score(selected_model, feature_vector(obs, option)) for option in obs.select.option]
    scores.sort(reverse=True)
    margin = scores[0] - scores[1] if len(scores) > 1 else float("inf")
    return margin, threshold


def _main_type_head(model: dict | None, option) -> tuple[str | None, dict | None]:
    heads = (model or {}).get("heads") or {}
    option_type = int(getattr(option.type, "value", option.type) or 0)
    name = MAIN_TYPE_HEADS_RUNTIME.get(option_type)
    head = heads.get(name) if name else None
    if not isinstance(head, dict) or "oblivious_trees" not in head:
        return name, None
    return name, head


def choose_from_scores(select, scores: list[float], margin: float | None = None) -> list[int]:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    if margin is not None and order:
        top = scores[order[0]]
        result = [index for index in order if top - scores[index] <= margin][:select.maxCount]
        if len(result) < select.minCount:
            result = order[:min(select.minCount, select.maxCount)]
        return result
    result = []
    for index in order:
        if len(result) >= select.maxCount:
            break
        if scores[index] >= 0 or len(result) < select.minCount:
            result.append(index)
    return result


def selection_margin(obs, config: dict, model: dict | None) -> float | None:
    if not model or "global" not in model:
        return None
    context = int(getattr(obs.select.context, "value", obs.select.context) or 0)
    key = f"{model_head(obs)}:{context}"
    raw = (model.get("selection_margin_thresholds") or {}).get(key)
    if raw is None:
        return None
    return abs(float(config.get("model_weight", 500.0))) * float(raw)


def state_value(
    obs,
    config: dict,
    value_model: dict | None = None,
    perspective_index: int | None = None,
) -> float:
    state = obs.current
    player_index = state.yourIndex if perspective_index is None else perspective_index
    result = int(getattr(state, "result", -1) or 0)
    if result != -1:
        return 1_000_000_000.0 if result == player_index else -1_000_000_000.0
    me = state.players[player_index]
    opponent = state.players[1 - player_index]
    value = 12000.0 * (len(opponent.prize or []) - len(me.prize or []))
    value += 200.0 * (me.deckCount - opponent.deckCount)
    value += 25.0 * ((me.handCount or 0) - (opponent.handCount or 0))
    for pokemon in list(me.active or []) + list(me.bench or []):
        if pokemon is not None:
            value += float(pokemon.hp or 0) + 180.0 * energy_count(pokemon)
            value += _priority(config, "board_priority", pokemon.id)
    for pokemon in list(opponent.active or []) + list(opponent.bench or []):
        if pokemon is not None:
            value -= 0.8 * float(pokemon.hp or 0)
    if config.get("family") == "grimmsnarl":
        my_active = active(me)
        ready_bench_grim = _ready_grim_on_bench(obs)
        if getattr(my_active, "id", 0) == GRIMMSNARL_EX:
            value += 18000.0 + 7000.0 * int(energy_count(my_active) >= 2)
        elif ready_bench_grim is not None and energy_count(ready_bench_grim) >= 2:
            value -= 15000.0
            if getattr(my_active, "id", 0) == IMPIDIMP and state.turn >= 4:
                value -= 12000.0
        support_ids = [pokemon.id for pokemon in list(me.active or []) + list(me.bench or []) if pokemon is not None]
        value += 3500.0 * min(2, support_ids.count(FROSLASS))
        value += 2500.0 * min(2, support_ids.count(MUNKIDORI))
    elif config.get("family") == "teal_ogerpon":
        my_active = active(me)
        opponent_active = active(opponent)
        if getattr(my_active, "id", 0) == TEAL_MASK_OGERPON_EX:
            value += 4500.0 * min(4, energy_count(my_active))
            damage = _ogerpon_attack_damage(obs, MYRIAD_LEAF_SHOWER, perspective_index=player_index)
            if opponent_active is not None and damage >= float(getattr(opponent_active, "hp", 0) or 0):
                value += 18000.0 + 6000.0 * _prize_value(opponent_active)
        if getattr(opponent_active, "id", 0) == CRUSTLE:
            value -= 24000.0
        if config.get("ogerpon_anti_mill") and _opponent_is_mill(obs):
            deck_margin = int(getattr(me, "deckCount", 0) or 0) - int(getattr(opponent, "deckCount", 0) or 0)
            value += 9000.0 * deck_margin
            if int(getattr(me, "deckCount", 0) or 0) <= 8:
                value -= 30000.0 * (9 - int(getattr(me, "deckCount", 0) or 0))
    if me.deckCount <= 3:
        value -= 8000.0 * (4 - me.deckCount)
    if opponent.deckCount <= 3:
        value += 8000.0 * (4 - opponent.deckCount)
    if _model_has_trees(value_model) and obs.select is not None and obs.select.option:
        predicted = max(model_score(value_model, feature_vector(obs, option)) for option in obs.select.option)
        value += float(config.get("search", {}).get("value_model_weight", 24000.0)) * predicted
    return value


def search_action(obs_dict: dict, obs, heuristic: list[int], config: dict, model: dict | None) -> list[int] | None:
    search = config.get("search", {})
    if not search.get("enabled") or obs.select.maxCount != 1 or not heuristic:
        return None
    contexts = set(search.get("contexts", []))
    if contexts and SelectContext(obs.select.context).name not in contexts:
        return None
    if (
        obs.select.context == SelectContext.MAIN
        and search.get("terminal_main_only", False)
        and not _only_terminal_main_actions(obs)
    ):
        return None
    heuristic_option = obs.select.option[heuristic[0]]
    if (
        config.get("family") == "teal_ogerpon"
        and obs.select.context == SelectContext.MAIN
        and heuristic_option.type == OptionType.ABILITY
        and getattr(option_card(obs, heuristic_option), "id", 0) == TEAL_MASK_OGERPON_EX
        and not (config.get("ogerpon_anti_mill") and _opponent_is_mill(obs))
    ):
        return None
    if search.get("low_confidence_only", True):
        margin, threshold = model_confidence(obs, model)
        if threshold > 0 and margin >= threshold:
            return None
    if not getattr(obs, "search_begin_input", None):
        return None
    try:
        from cg.api import search_begin, search_end, search_release, search_step
    except Exception:
        return None
    scores = score_options(obs, config, model)
    candidates = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: int(search.get("candidates", 3))]
    remaining = float(obs_dict.get("remainingOverageTime", 600.0) or 0.0)
    if remaining < float(search.get("minimum_overage_s", 30.0)):
        return None
    budget = float(search.get("budget_s", 0.10))
    if remaining < 120.0:
        budget *= 0.4
    deadline = time.perf_counter() + budget
    best_index = heuristic[0]
    values: dict[int, list[float]] = {candidate: [] for candidate in candidates}
    _SEARCH_STATS["attempts"] += 1
    player_index = obs.current.yourIndex
    start_turn = obs.current.turn
    worlds = _sample_hidden_worlds(obs, config, int(search.get("belief_worlds", 3)))
    for own_deck, own_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active in worlds:
        if time.perf_counter() >= deadline:
            break
        try:
            root = search_begin(
                obs,
                own_deck,
                own_prize,
                opponent_deck,
                opponent_prize,
                opponent_hand,
                opponent_active,
                False,
            )
        except Exception:
            _SEARCH_STATS["failures"] += 1
            continue
        try:
            for candidate in candidates:
                if time.perf_counter() >= deadline:
                    break
                branch = None
                try:
                    branch = search_step(root.searchId, [candidate])
                    current = branch
                    for _ in range(int(search.get("rollout_steps", 5))):
                        if time.perf_counter() >= deadline:
                            break
                        next_obs = current.observation
                        if next_obs.select is None or next_obs.current.result != -1:
                            break
                        if next_obs.current.yourIndex != player_index or next_obs.current.turn != start_turn:
                            break
                        action = forced_action(next_obs, config)
                        if action is None:
                            next_scores = score_options(next_obs, config, model)
                            action = choose_from_scores(
                                next_obs.select,
                                next_scores,
                                selection_margin(next_obs, config, model),
                            )
                            action = ensure_mandatory_minimums(next_obs, action, score_options(next_obs, config, model))
                        current = search_step(current.searchId, action)
                    values[candidate].append(
                        state_value(
                            current.observation,
                            config,
                            value_model=(model or {}).get("win_value") if model else None,
                            perspective_index=player_index,
                        )
                    )
                except Exception:
                    _SEARCH_STATS["failures"] += 1
                finally:
                    if branch is not None:
                        try:
                            search_release(branch.searchId)
                        except Exception:
                            pass
        finally:
            try:
                search_end()
            except Exception:
                pass

    aggregates: dict[int, float] = {}
    risk_penalty = float(search.get("risk_penalty", 0.20))
    for candidate, samples in values.items():
        if not samples:
            continue
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / len(samples)
        aggregates[candidate] = mean - risk_penalty * math.sqrt(variance)
    if not aggregates:
        return None
    _SEARCH_STATS["successes"] += 1
    best_index = max(aggregates, key=aggregates.get)
    margin = float(search.get("margin", 0.0))
    heuristic_value = aggregates.get(heuristic[0], float("-inf"))
    if aggregates[best_index] < heuristic_value + margin:
        best_index = heuristic[0]
    if best_index != heuristic[0]:
        _SEARCH_STATS["changes"] += 1
    return [best_index]


def fallback_action(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if select is None:
        return read_deck_csv()
    options = select.get("option") or []
    minimum = max(0, int(select.get("minCount", 0) or 0))
    maximum = max(0, int(select.get("maxCount", len(options)) or 0))
    return list(range(min(minimum, maximum, len(options))))


def run_agent(obs_dict: dict, config: dict, model: dict | None = None) -> list[int]:
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            _LOPUNNY_MEMORY.update(turn=None, active_serial=None, bench_serials=set(), moved=False)
            _OGERPON_MEMORY.update(turn=None, primary_serial=None)
            _reset_policy_memory()
            _reset_policy_router()
            _SEARCH_STATS.update(attempts=0, successes=0, changes=0, failures=0)
            _ADVANTAGE_MEMORY["overrides"] = 0
            return read_deck_csv()
        config = _select_policy_config(obs, config)
        _sync_policy_memory(obs)
        if config.get("family") in {"mega_lopunny", "generic_replay"}:
            _update_lopunny_memory(obs)
        if config.get("family") in {"teal_ogerpon", "generic_replay"}:
            _update_ogerpon_memory(obs)
        mandatory = forced_action(obs, config)
        if mandatory is not None:
            _record_policy_action(obs, mandatory)
            return mandatory
        scores = score_options(obs, config, model)
        heuristic = choose_from_scores(obs.select, scores, selection_margin(obs, config, model))
        heuristic = ensure_mandatory_minimums(obs, heuristic, scores)
        searched = search_action(obs_dict, obs, heuristic, config, model)
        anchor = searched if searched is not None else heuristic
        overridden = advantage_override(obs, anchor, scores, searched, model)
        result = overridden if overridden is not None else anchor
        _record_policy_action(obs, result)
        return result
    except Exception:
        if os.environ.get("DEBUG_AGENT") == "1":
            import traceback
            traceback.print_exc()
        return fallback_action(obs_dict)
