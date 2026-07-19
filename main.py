import os
from collections import defaultdict
from pathlib import Path

from cg.api import (
    AreaType, CardType, Observation, OptionType, Pokemon,
    SelectContext, all_attack, all_card_data, to_observation_class,
)

# --- Core plan: aggressive Great Tusk LO + Crustle wall package ---
# Core mill package
GREAT_TUSK = 58
DURANT_EX = 198
CORNERSTONE_OGERPON = 386
TATSUGIRI = 122
FLUTTER_MANE = 56

# Crustle wall package
DWEBBLE = 344
CRUSTLE = 345

# Search / disruption / recovery
FIGHT_GONG = 1142
POKEGEAR_30 = 1122
ROTO_STICK = 1077
BUG_CATCHING_SET = 1094
ULTRA_BALL = 1121
HAND_TRIMMER = 1087
JUMBO_ICE_CREAM = 1147
SWITCH = 1123
BUDDY_BUDDY_POFFIN = 1086
POKE_PAD = 1152
FLUTE = 1091
NIGHT_STRETCHER = 1097
SACRED_ASH = 1129
ENERGY_RECYCLER = 1139
ENHANCED_HAMMER = 1081
ENERGY_LASSO = 1149
HANDY_CIRCULATOR = 1161
GRAVITY_GEM = 1166
HERO_CAPE = 1159
EXPLORER_GUIDANCE = 1185
ERI = 1186
XEROSIC_SCHEME = 1197
COLRESS_TENACITY = 1194
JUDGE = 1213
BOSS_ORDERS = 1182
LISIA_APPEAL = 1204
NEUTRAL_CENTER = 1247

# Energies
BASIC_GRASS_ENERGY = 1
BASIC_FIGHTING_ENERGY = 6
GROW_GRASS_ENERGY = 18
MIST_ENERGY = 11
ROCK_FIGHTING_ENERGY = 20
ENERGY_IDS = {BASIC_GRASS_ENERGY, BASIC_FIGHTING_ENERGY, GROW_GRASS_ENERGY, MIST_ENERGY, ROCK_FIGHTING_ENERGY}
GRASS_ENERGY_IDS = {BASIC_GRASS_ENERGY, GROW_GRASS_ENERGY}
BASIC_ENERGY_IDS = {BASIC_GRASS_ENERGY, BASIC_FIGHTING_ENERGY}

# Attack IDs
LAND_COLLAPSE = 62          # Great Tusk: mill 1, +3 if Ancient Supporter was played.
GIANT_TUSK = 63             # Great Tusk: 160 damage.
DURANT_VENGEFUL_CRUSH = 267
ROCK_KAGURA = 538           # Ogerpon: attach Basic Fighting from deck.
MOUNTAIN_RAMMING = 539      # Ogerpon: 100 + mill 1.
ASCENSION = 478             # Dwebble: evolve from deck.
SUPERB_SCISSORS = 479       # Crustle: 120.

# --- Optional emergency attacker package ---
MEGA_HERACROSS_EX = 781
KORAIDON_EX = 979
TERRAKION = 607
MEGA_HAWLUCHA_EX = 886
JUGGERNAUT_HORN = 1130
HERA_MOUNTAIN_RAMMING = 1131
KORAIDON_TERA = 1408
ORICHALCUM_FANG = 1409
TERRAKION_RETALIATE = 873
TERRAKION_LAND_CRUSH = 874
SOMERSAULT_DIVE = 1277
ATTACKER_PIVOTS = {MEGA_HERACROSS_EX, KORAIDON_EX, TERRAKION, MEGA_HAWLUCHA_EX}

POKEMON_IDS = {GREAT_TUSK, DURANT_EX, CORNERSTONE_OGERPON, TATSUGIRI, FLUTTER_MANE, DWEBBLE, CRUSTLE, MEGA_HERACROSS_EX, KORAIDON_EX, TERRAKION, MEGA_HAWLUCHA_EX}
SEARCH_ITEMS = {FIGHT_GONG, POKEGEAR_30, ROTO_STICK, BUG_CATCHING_SET, ULTRA_BALL, BUDDY_BUDDY_POFFIN, POKE_PAD}
RECOVERY_ITEMS = {NIGHT_STRETCHER, SACRED_ASH, ENERGY_RECYCLER}
SUPPORTERS = {EXPLORER_GUIDANCE, ERI, XEROSIC_SCHEME, COLRESS_TENACITY, JUDGE, BOSS_ORDERS, LISIA_APPEAL}
AIR_BALLOON = 1174
SACRED_CHARM = 1177
TOOLS = {HANDY_CIRCULATOR, GRAVITY_GEM, HERO_CAPE, AIR_BALLOON, SACRED_CHARM}

CARD_TABLE = {card.cardId: card for card in all_card_data()}
ATTACK_TABLE = {attack.attackId: attack for attack in all_attack()}


def _ex_evolution_ancestor_names() -> set[str]:
    ancestors = {card.evolvesFrom for card in CARD_TABLE.values() if (card.ex or getattr(card, "megaEx", False)) and card.evolvesFrom}
    changed = True
    while changed:
        changed = False
        for card in CARD_TABLE.values():
            if card.name in ancestors and card.evolvesFrom and card.evolvesFrom not in ancestors:
                ancestors.add(card.evolvesFrom)
                changed = True
    return ancestors


EX_EVOLUTION_ANCESTORS = _ex_evolution_ancestor_names()


def read_deck_csv() -> list[int]:
    candidates = [
        Path(__file__).resolve().parent / "deck.csv",
        Path("deck.csv"),
        Path("/kaggle_simulations/agent/deck.csv"),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    with path.open("r", encoding="utf-8") as file:
        return [int(line) for line in file.read().splitlines()[:60]]


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    player = obs.current.players[player_index]
    zones = {
        AreaType.HAND: player.hand,
        AreaType.DISCARD: player.discard,
        AreaType.ACTIVE: player.active,
        AreaType.BENCH: player.bench,
        AreaType.PRIZE: player.prize,
        AreaType.STADIUM: obs.current.stadium,
        AreaType.LOOKING: obs.current.looking,
        AreaType.DECK: obs.select.deck,
    }
    zone = zones.get(area)
    if zone is None or index is None or not 0 <= index < len(zone):
        return None
    return zone[index]


def field_pokemon(player):
    return [p for p in player.active + player.bench if p is not None]


def active_pokemon(player):
    return player.active[0] if player.active and player.active[0] is not None else None


def attached_energy_count(pokemon: Pokemon | None) -> int:
    return len(pokemon.energies) if pokemon is not None else 0


def damage_on(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    return max(0, pokemon.maxHp - pokemon.hp)


def is_ex_card(card_id: int) -> bool:
    data = CARD_TABLE.get(card_id)
    return bool(data and (data.ex or getattr(data, "megaEx", False)))


def is_ex_pokemon(pokemon: Pokemon | None) -> bool:
    return pokemon is not None and is_ex_card(pokemon.id)


def has_tool(pokemon: Pokemon | None, tool_id: int) -> bool:
    return pokemon is not None and any(c.id == tool_id for c in pokemon.tools)


def count_in_hand(player, card_id: int) -> int:
    return sum(1 for c in (player.hand or []) if c.id == card_id)


def count_in_field(player, card_id: int) -> int:
    return sum(1 for p in field_pokemon(player) if p.id == card_id)


def has_in_field(player, card_id: int) -> bool:
    return count_in_field(player, card_id) > 0


def count_in_discard(player, card_id: int) -> int:
    return sum(1 for c in (player.discard or []) if c.id == card_id)


def count_energy_in_discard(player) -> int:
    return sum(1 for c in (player.discard or []) if c.id in ENERGY_IDS)


def count_pokemon_in_discard(player) -> int:
    n = 0
    for c in (player.discard or []):
        data = CARD_TABLE.get(c.id)
        if data is not None and data.cardType == CardType.POKEMON:
            n += 1
    return n


def can_pay_attack(pokemon: Pokemon | None, attack_id: int) -> bool:
    if pokemon is None:
        return False
    attack = ATTACK_TABLE.get(attack_id)
    if attack is None:
        return False
    # The simulator offers only legal attacks. This approximate check is for planning.
    return len(pokemon.energies) >= len(attack.energies)


def has_ready_tusk(player) -> bool:
    return any(p.id == GREAT_TUSK and can_pay_attack(p, LAND_COLLAPSE) for p in field_pokemon(player))


def active_tusk_ready(player) -> bool:
    a = active_pokemon(player)
    return a is not None and a.id == GREAT_TUSK and can_pay_attack(a, LAND_COLLAPSE)


def ready_tusk_on_bench(player) -> bool:
    return any(p.id == GREAT_TUSK and can_pay_attack(p, LAND_COLLAPSE) for p in player.bench)


def ready_crustle(player) -> bool:
    return any(p.id == CRUSTLE for p in field_pokemon(player))


def active_is_ready_crustle(player) -> bool:
    a = active_pokemon(player)
    return a is not None and a.id == CRUSTLE


def opponent_has_special_energy(opponent) -> bool:
    for pokemon in field_pokemon(opponent):
        for card in pokemon.energyCards:
            data = CARD_TABLE.get(card.id)
            if data is not None and data.cardType == CardType.SPECIAL_ENERGY:
                return True
    return False


def attack_energy_minimum(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 99
    data = CARD_TABLE.get(pokemon.id)
    if data is None or not data.attacks:
        return 99
    costs = [len(ATTACK_TABLE[a].energies) for a in data.attacks if a in ATTACK_TABLE]
    return min(costs, default=99)


def opponent_can_attack_soon(opponent) -> bool:
    active = active_pokemon(opponent)
    if active is None:
        return False
    return attached_energy_count(active) + 1 >= attack_energy_minimum(active)


def opponent_ex_pressure(opponent) -> bool:
    active = active_pokemon(opponent)
    if active is not None and is_ex_pokemon(active) and opponent_can_attack_soon(opponent):
        return True
    for pokemon in opponent.bench:
        if is_ex_pokemon(pokemon) and attached_energy_count(pokemon) + 1 >= attack_energy_minimum(pokemon):
            return True
    return False


def opponent_shows_ex_evolution_line(opponent) -> bool:
    for pokemon in field_pokemon(opponent):
        data = CARD_TABLE.get(pokemon.id)
        if data is not None and (is_ex_card(data.cardId) or data.name in EX_EVOLUTION_ANCESTORS):
            return True
    return False


def retreat_cost(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    data = CARD_TABLE.get(pokemon.id)
    return getattr(data, "retreatCost", 0) if data is not None else 0


def opponent_has_ex_or_ex_line_pressure(opponent) -> bool:
    return opponent_ex_pressure(opponent) or opponent_shows_ex_evolution_line(opponent)


def opponent_has_trappable_bench(opponent) -> bool:
    for p in opponent.bench:
        if p is None:
            continue
        if attached_energy_count(p) == 0 and retreat_cost(p) >= 1:
            return True
    return False


def opponent_has_trappable_basic_bench(opponent) -> bool:
    for p in opponent.bench:
        if p is None:
            continue
        data = CARD_TABLE.get(p.id)
        if data is not None and getattr(data, "basic", False) and attached_energy_count(p) == 0 and retreat_cost(p) >= 1:
            return True
    return False



# --- Opponent package recognition + generic feature layer ---
LUCARIO_STRONG_IDS = {673, 674, 675, 676, 677, 678, 1141, 1252}
DRAGAPULT_SAMPLE_IDS = {119, 120, 121, 1256, 1080}
ABOMASNOW_SAMPLE_IDS = {721, 722, 723, 1262}
ALAKAZAM_IDS = {741, 742, 743, 1264, 19}

def opponent_visible_ids(opponent) -> set[int]:
    ids = set()
    for p in field_pokemon(opponent):
        if p is not None:
            ids.add(p.id)
            for e in getattr(p, 'energyCards', []) or []:
                ids.add(e.id)
            for t in getattr(p, 'tools', []) or []:
                ids.add(t.id)
    for c in (opponent.discard or []):
        ids.add(c.id)
    return ids

def facing_lucario_strong(opponent) -> bool:
    return bool(opponent_visible_ids(opponent) & LUCARIO_STRONG_IDS)

def facing_dragapult_sample(opponent) -> bool:
    return bool(opponent_visible_ids(opponent) & DRAGAPULT_SAMPLE_IDS)

def facing_abomasnow_sample(opponent) -> bool:
    return bool(opponent_visible_ids(opponent) & ABOMASNOW_SAMPLE_IDS)

def facing_alakazam(opponent) -> bool:
    return bool(opponent_visible_ids(opponent) & ALAKAZAM_IDS)

def opponent_bench_counter_pressure(opponent) -> bool:
    if facing_dragapult_sample(opponent):
        return True
    for p in field_pokemon(opponent):
        data = CARD_TABLE.get(p.id)
        if data is None:
            continue
        for aid in getattr(data, 'attacks', []) or []:
            atk = ATTACK_TABLE.get(aid)
            txt = (getattr(atk, 'text', '') or '').lower() if atk is not None else ''
            if 'damage counter' in txt and 'bench' in txt:
                return True
    return False

def opponent_self_deck_pressure(opponent) -> bool:
    if facing_abomasnow_sample(opponent):
        return True
    for p in field_pokemon(opponent):
        data = CARD_TABLE.get(p.id)
        if data is None:
            continue
        for aid in getattr(data, 'attacks', []) or []:
            atk = ATTACK_TABLE.get(aid)
            text = (getattr(atk, 'text', '') or '').lower() if atk is not None else ''
            if ('discard the top' in text and 'your deck' in text) or ('discard' in text and 'your deck' in text and 'damage' in text):
                return True
    return False

def own_deck_safety_guard(me, opponent) -> bool:
    # Only stop optional self-thinning when the opponent itself is already burning deck fast.
    if not opponent_self_deck_pressure(opponent):
        return False
    return me.deckCount <= max(8, opponent.deckCount + 4) and opponent.deckCount > 4

def desired_field_floor(me, opponent, state) -> int:
    if facing_lucario_strong(opponent):
        return 5
    if opponent_bench_counter_pressure(opponent):
        return 3
    if opponent_can_attack_soon(opponent):
        return 3
    return 2

def urgent_field_rebuild(me, opponent, state) -> bool:
    return len(field_pokemon(me)) < desired_field_floor(me, opponent, state)

def generic_active_nonex_race_threat(opponent) -> bool:
    a = active_pokemon(opponent)
    if a is None or is_ex_pokemon(a):
        return False
    return attached_energy_count(a) >= 2 and a.hp <= 170


def should_wall_mode(me, opponent, state) -> bool:
    # A live boosted mill turn is worth more than moving into the wall.
    if active_tusk_ready(me) and count_in_hand(me, EXPLORER_GUIDANCE) > 0 and not state.supporterPlayed:
        return False
    if opponent.deckCount <= 20:
        return False
    active = active_pokemon(me)
    stadium_id = state.stadium[0].id if state.stadium else None
    if stadium_id == NEUTRAL_CENTER and active is not None and not is_ex_pokemon(active):
        # Neutralization Zone already turns Great Tusk into the preferred wall,
        # while keeping the primary mill attack online.
        return False
    # Generic wall mode: prefer a non-ex wall against visible ex/evolution-line pressure.
    if not opponent_has_ex_or_ex_line_pressure(opponent):
        return False
    if has_in_field(me, CRUSTLE) or has_in_field(me, DWEBBLE) or count_in_hand(me, DWEBBLE) or count_in_hand(me, CRUSTLE):
        return True
    return False


def should_ko_mode(me, opponent, state) -> bool:
    active = active_pokemon(me)
    opp_active = active_pokemon(opponent)
    if active is None or opp_active is None or opponent.deckCount <= 8:
        return False

    # Estimate both actual win routes instead of treating damage as a last-resort
    # action. Do not assume an Explorer that is not actually available.
    tusks = [p for p in field_pokemon(me) if p.id == GREAT_TUSK]
    if tusks:
        best_tusk = max(tusks, key=attached_energy_count)
        mill_setup = max(0, 2 - attached_energy_count(best_tusk))
        if best_tusk.serial != active.serial:
            mill_setup += 1
        mill_per_turn = 4 if count_in_hand(me, EXPLORER_GUIDANCE) > 0 and not state.supporterPlayed else 1
        mill_turns = mill_setup + (opponent.deckCount + mill_per_turn - 1) // mill_per_turn
    else:
        # Search, two attachments and a promotion are still required.
        mill_turns = 4 + opponent.deckCount
    prizes_taken_by_opponent = max(0, 6 - len(opponent.prize))
    durant_damage = 30 + 30 * prizes_taken_by_opponent

    plans = []
    attack_profiles = {
        GREAT_TUSK: (GIANT_TUSK, 160),
        DURANT_EX: (DURANT_VENGEFUL_CRUSH, durant_damage),
        CRUSTLE: (SUPERB_SCISSORS, 120),
        CORNERSTONE_OGERPON: (MOUNTAIN_RAMMING, 100),
        MEGA_HERACROSS_EX: (HERA_MOUNTAIN_RAMMING, 170),
        KORAIDON_EX: (ORICHALCUM_FANG, 200),
        TERRAKION: (TERRAKION_RETALIATE, 130),
        MEGA_HAWLUCHA_EX: (SOMERSAULT_DIVE, 260),
    }
    for pokemon in field_pokemon(me):
        profile = attack_profiles.get(pokemon.id)
        if profile is None:
            continue
        attack_id, damage = profile
        attack = ATTACK_TABLE.get(attack_id)
        if attack is None or damage <= 0:
            continue
        setup_turns = max(0, len(attack.energies) - attached_energy_count(pokemon))
        switch_turns = 0 if pokemon.serial == active.serial else 1
        attack_turns = (opp_active.hp + damage - 1) // damage
        plans.append((setup_turns + switch_turns + attack_turns, damage, pokemon.id))
    if not plans:
        return False

    stadium_id = state.stadium[0].id if state.stadium else None
    zone_bypass_attacker = (
        stadium_id == NEUTRAL_CENTER
        and not is_ex_pokemon(opp_active)
        and any(is_ex_pokemon(pokemon) for pokemon in field_pokemon(opponent))
    )
    if zone_bypass_attacker:
        # Remove the non-ex attacker that bypasses the Zone, then return to the
        # protected mill plan against the opponent's ex board.
        return True

    turns_for_active_ko, damage, attacker_id = min(plans)
    prize_gain = 2 if is_ex_pokemon(opp_active) else 1
    immediate_prize_win = turns_for_active_ko == 1 and prize_gain >= len(me.prize)
    immediate_board_win = turns_for_active_ko == 1 and not opponent.bench
    if immediate_prize_win or immediate_board_win:
        return True

    # Approximate the rest of the prize race with the visible active target.
    knockouts_needed = (len(me.prize) + prize_gain - 1) // prize_gain
    # Setup/switch is paid once; subsequent visible targets are approximated by
    # the current target's attack count.
    attacks_for_target = (opp_active.hp + damage - 1) // damage
    ko_turns = turns_for_active_ko + attacks_for_target * (knockouts_needed - 1)
    board_clear_turns = turns_for_active_ko + attacks_for_target * len(opponent.bench)
    if len(opponent.bench) <= 1 and board_clear_turns <= mill_turns:
        return True
    return ko_turns < mill_turns


def bench_space(player) -> int:
    return player.benchMax - len(player.bench)


def can_bench_more(player) -> bool:
    return bench_space(player) > 0


def initial_active_score(card_id: int, me, opponent) -> int:
    # Lead with a disposable/setup body and preserve the mill attacker.
    if card_id == DWEBBLE:
        return 12500
    if card_id == TATSUGIRI:
        return 9800
    if card_id == GREAT_TUSK:
        return 3500
    if card_id == FLUTTER_MANE:
        return 7000
    if card_id == CORNERSTONE_OGERPON:
        return 6500
    if card_id == DURANT_EX:
        return 3000
    return 1000


def setup_bench_score(card_id: int, me, opponent) -> int:
    if card_id == GREAT_TUSK:
        return 10000 + 1200 * (2 - min(2, count_in_field(me, GREAT_TUSK)))
    if card_id == DWEBBLE:
        return 9600 if count_in_field(me, DWEBBLE) + count_in_field(me, CRUSTLE) < 3 else 4200
    if card_id == DURANT_EX:
        return 7100 if count_in_field(me, DURANT_EX) == 0 else 4300
    if card_id == TATSUGIRI:
        return 6500 if count_in_field(me, TATSUGIRI) == 0 else 2500
    if card_id == CORNERSTONE_OGERPON:
        return 5200
    if card_id == FLUTTER_MANE:
        return 3600
    return 1000


def card_keep_value(card_id: int, me, opponent, state, wall_mode: bool, ko_mode: bool) -> int:
    active = active_pokemon(me)
    attacking_tusk = active is not None and active.id == GREAT_TUSK and can_pay_attack(active, LAND_COLLAPSE)
    if card_id == EXPLORER_GUIDANCE:
        return 9800 if attacking_tusk and not state.supporterPlayed else 5400
    if card_id == BOSS_ORDERS:
        return 5200 if opponent_has_trappable_bench(opponent) else 900
    if card_id == LISIA_APPEAL:
        return 5000 if opponent_has_trappable_basic_bench(opponent) else 900
    if card_id == GREAT_TUSK:
        return 8500
    if card_id in ENERGY_IDS:
        if active is not None and active.id == GREAT_TUSK and attached_energy_count(active) < 2:
            return 7400
        if has_in_field(me, CRUSTLE) and card_id in GRASS_ENERGY_IDS:
            return 4700
        return 3600
    if card_id == FIGHT_GONG:
        return 6800
    if card_id == ULTRA_BALL:
        return 6600
    if card_id == POKEGEAR_30:
        return 6200
    if card_id == ROTO_STICK:
        return 6000
    if card_id == BUDDY_BUDDY_POFFIN:
        return 7600 if count_in_field(me, DWEBBLE) + count_in_field(me, CRUSTLE) < 2 or count_in_field(me, TATSUGIRI) == 0 else 2200
    if card_id == POKE_PAD:
        return 7400 if count_in_field(me, GREAT_TUSK) == 0 or (has_in_field(me, DWEBBLE) and count_in_field(me, CRUSTLE) == 0) else 3000
    if card_id == BUG_CATCHING_SET:
        return 5600 if wall_mode or count_in_field(me, DWEBBLE) == 0 else 4200
    if card_id == DWEBBLE:
        return 5700
    if card_id == CRUSTLE:
        return 6200 if has_in_field(me, DWEBBLE) else 3000
    if card_id == DURANT_EX:
        return 5000 if can_bench_more(me) else 200
    if card_id == TATSUGIRI:
        return 4500 if not state.supporterPlayed else 2000
    if card_id == NEUTRAL_CENTER:
        # ACE SPEC is a one-of and opposing ex attackers often appear only
        # after evolution. Preserve it before the threat becomes visible.
        return 14000
    if card_id == COLRESS_TENACITY:
        current_stadium = state.stadium[0].id if state.stadium else None
        if current_stadium != NEUTRAL_CENTER and count_in_hand(me, NEUTRAL_CENTER) == 0:
            return 7600
        return 1200
    if card_id == AIR_BALLOON:
        return 7600 if any(p.id == GREAT_TUSK and not has_tool(p, AIR_BALLOON) for p in field_pokemon(me)) else 2600
    if card_id == SACRED_CHARM:
        return 7600 if opponent_can_attack_soon(opponent) else 2600
    if card_id in TOOLS:
        return 3600
    if card_id == NIGHT_STRETCHER:
        return 3400 if count_in_discard(me, GREAT_TUSK) or count_energy_in_discard(me) else 1200
    if card_id in (SACRED_ASH, ENERGY_RECYCLER):
        return 3000 if me.deckCount <= 18 else 1000
    if card_id == JUDGE:
        return 3000 if opponent.handCount <= 3 else 500
    if card_id in (ERI, XEROSIC_SCHEME):
        return 2300
    if card_id in (FLUTE, HAND_TRIMMER, ENHANCED_HAMMER, ENERGY_LASSO):
        return 2000
    return 800


def play_score(card_id: int, me, opponent, state, wall_mode: bool, ko_mode: bool) -> int:
    active = active_pokemon(me)
    active_ready_tusk = active is not None and active.id == GREAT_TUSK and can_pay_attack(active, LAND_COLLAPSE)
    has_explorer = count_in_hand(me, EXPLORER_GUIDANCE) > 0
    score = -10000

    if card_id == EXPLORER_GUIDANCE:
        if not state.supporterPlayed and active_ready_tusk and me.deckCount >= 6:
            # Highest priority: this is the deck's main win condition.
            score = 520000 + max(0, 20 - opponent.deckCount) * 3500
        elif not state.supporterPlayed and me.deckCount >= 10 and not has_ready_tusk(me):
            # Allowed, but not preferred: preserve Explorer for a boosted attack when possible.
            score = 14000
    elif card_id == COLRESS_TENACITY:
        current_stadium = state.stadium[0].id if state.stadium else None
        need_zone = current_stadium != NEUTRAL_CENTER and count_in_hand(me, NEUTRAL_CENTER) == 0
        ex_threat = opponent_ex_pressure(opponent) or opponent_shows_ex_evolution_line(opponent)
        if not state.supporterPlayed and ex_threat and need_zone:
            # Search the one-of Zone before an evolved ex starts taking prizes.
            score = 340000
        elif not state.supporterPlayed and me.deckCount >= 8:
            score = 9000
    elif card_id == BUDDY_BUDDY_POFFIN:
        if can_bench_more(me):
            wall_count = count_in_field(me, DWEBBLE) + count_in_field(me, CRUSTLE)
            if len(field_pokemon(me)) <= 1 or wall_count < 2:
                score = 94000
            elif count_in_field(me, TATSUGIRI) == 0 and not has_ready_tusk(me):
                score = 36000
            else:
                score = 11000
    elif card_id == POKE_PAD:
        if count_in_field(me, GREAT_TUSK) == 0 or (wall_mode and count_in_field(me, CRUSTLE) == 0):
            score = 90000
        elif has_in_field(me, DWEBBLE) and count_in_field(me, CRUSTLE) == 0:
            score = 68000
        else:
            score = 18000
    elif card_id == FIGHT_GONG:
        need_tusk = count_in_field(me, GREAT_TUSK) == 0 and count_in_hand(me, GREAT_TUSK) == 0
        need_energy_for_tusk = any(p.id == GREAT_TUSK and attached_energy_count(p) < 2 for p in field_pokemon(me))
        if need_tusk or need_energy_for_tusk:
            score = 90000
        else:
            score = 21000
    elif card_id == ULTRA_BALL:
        # Universal Pokémon search: bridges Great Tusk and Crustle packages.
        if count_in_field(me, GREAT_TUSK) == 0 or (wall_mode and count_in_field(me, CRUSTLE) == 0):
            score = 82000
        elif has_in_field(me, DWEBBLE) and count_in_field(me, CRUSTLE) == 0:
            score = 52000
        else:
            score = 16000
    elif card_id == POKEGEAR_30:
        if not state.supporterPlayed and not has_explorer and active_ready_tusk and me.deckCount >= 7:
            score = 88000
        elif not state.supporterPlayed and not has_explorer and me.deckCount >= 10:
            score = 23000
    elif card_id == ROTO_STICK:
        if not state.supporterPlayed and not has_explorer and active_ready_tusk and me.deckCount >= 4:
            score = 86000
        elif not state.supporterPlayed and not has_explorer and me.deckCount >= 9:
            score = 21000
    elif card_id == BUG_CATCHING_SET:
        if wall_mode or count_in_field(me, DWEBBLE) == 0 or (has_in_field(me, DWEBBLE) and count_in_field(me, CRUSTLE) == 0):
            score = 42000
        else:
            score = 9000
    elif card_id == GREAT_TUSK:
        if can_bench_more(me) and count_in_field(me, GREAT_TUSK) < 2:
            score = 95000
    elif card_id == MEGA_HERACROSS_EX:
        if can_bench_more(me) and facing_lucario_strong(opponent) and count_in_field(me, MEGA_HERACROSS_EX) == 0 and len(field_pokemon(me)) >= 3:
            score = 64000
    elif card_id == KORAIDON_EX:
        if can_bench_more(me) and facing_lucario_strong(opponent) and count_in_field(me, KORAIDON_EX) == 0 and len(field_pokemon(me)) >= 3:
            score = 54000
    elif card_id == TERRAKION:
        if can_bench_more(me) and facing_lucario_strong(opponent) and count_in_field(me, TERRAKION) == 0 and len(field_pokemon(me)) >= 2:
            score = 52000
    elif card_id == MEGA_HAWLUCHA_EX:
        if can_bench_more(me) and facing_lucario_strong(opponent) and count_in_field(me, MEGA_HAWLUCHA_EX) == 0 and len(field_pokemon(me)) >= 3:
            score = 56000
    elif card_id == DWEBBLE:
        if can_bench_more(me) and count_in_field(me, DWEBBLE) + count_in_field(me, CRUSTLE) < 2:
            score = 60000 if wall_mode else 36000
    elif card_id == CRUSTLE:
        # Normally handled by EVOLVE option, but keep playable/evolution choices high where applicable.
        score = 34000 if has_in_field(me, DWEBBLE) else -1000
    elif card_id == DURANT_EX:
        if can_bench_more(me) and opponent.deckCount > 0:
            # Guaranteed 1-card mill on play. Keep high, but lower than boosted Explorer.
            score = 70000 if count_in_field(me, DURANT_EX) == 0 else 42000
            if opponent_can_attack_soon(opponent):
                score -= 6000
    elif card_id == TATSUGIRI:
        if can_bench_more(me) and not state.supporterPlayed and not has_explorer and count_in_field(me, TATSUGIRI) == 0:
            score = 28000
    elif card_id == FLUTTER_MANE:
        if can_bench_more(me) and count_in_field(me, FLUTTER_MANE) == 0:
            score = 12000
    elif card_id == CORNERSTONE_OGERPON:
        if can_bench_more(me) and count_in_field(me, CORNERSTONE_OGERPON) == 0:
            score = 15000
    elif card_id == NEUTRAL_CENTER:
        current_stadium = state.stadium[0].id if state.stadium else None
        if not state.stadiumPlayed and current_stadium != NEUTRAL_CENTER:
            # Stadium must be established before ATTACK ends the turn.
            score = 330000
    elif card_id == FLUTE:
        if opponent.benchMax - len(opponent.bench) > 0 and opponent.deckCount >= 5:
            score = 21000
    elif card_id == HAND_TRIMMER:
        their_loss = max(0, opponent.handCount - 5)
        our_loss = max(0, me.handCount - 5)
        if their_loss > our_loss:
            score = 16000 + 1600 * (their_loss - our_loss)
    elif card_id == SWITCH:
        active_id = active.id if active is not None else None
        if wall_mode and active_id != CRUSTLE and any(p.id == CRUSTLE for p in me.bench):
            score = 400000
        elif active_id != GREAT_TUSK and ready_tusk_on_bench(me):
            score = 210000
    elif card_id == JUMBO_ICE_CREAM:
        if active is not None and damage_on(active) >= 40 and attached_energy_count(active) >= 3:
            # Heal before ATTACK ends the turn; especially valuable after a
            # capped wall survives a high-damage non-ex hit.
            score = 350000 + damage_on(active)
    elif card_id == ENHANCED_HAMMER:
        if opponent_has_special_energy(opponent):
            score = 26000
    elif card_id == ENERGY_LASSO:
        if opponent.handCount >= 6 and opponent_can_attack_soon(opponent):
            score = 11000
    elif card_id == BOSS_ORDERS:
        if not state.supporterPlayed and opponent_has_trappable_bench(opponent):
            if active_ready_tusk and count_in_hand(me, EXPLORER_GUIDANCE) > 0:
                score = 120000
            elif opponent.deckCount <= 10 or len(opponent.prize) <= 1:
                score = 390000
            elif opponent_has_ex_or_ex_line_pressure(opponent) and len(opponent.prize) <= 4:
                score = 120000
            else:
                score = 50000
    elif card_id == LISIA_APPEAL:
        if not state.supporterPlayed and opponent_has_trappable_basic_bench(opponent):
            if active_ready_tusk and count_in_hand(me, EXPLORER_GUIDANCE) > 0:
                score = 115000
            elif opponent.deckCount <= 12 or len(opponent.prize) <= 1:
                score = 390000
            elif opponent_has_ex_or_ex_line_pressure(opponent) and len(opponent.prize) <= 4:
                score = 120000
            else:
                score = 52000
    elif card_id == ERI:
        if not state.supporterPlayed and opponent.handCount >= 6 and not active_ready_tusk:
            score = 11000
    elif card_id == XEROSIC_SCHEME:
        if not state.supporterPlayed and active_ready_tusk and count_in_hand(me, EXPLORER_GUIDANCE) > 0:
            score = 60000
        elif not state.supporterPlayed and opponent.handCount >= 8:
            score = 265000 + 2500 * (opponent.handCount - 8)
        elif not state.supporterPlayed and opponent.handCount >= 5 and not active_ready_tusk:
            score = 12000
    elif card_id == JUDGE:
        net_mill = 4 - opponent.handCount
        if not state.supporterPlayed and net_mill > 0 and not active_ready_tusk:
            score = 11000 + 1500 * net_mill
    elif card_id == NIGHT_STRETCHER:
        if count_in_discard(me, GREAT_TUSK) and count_in_field(me, GREAT_TUSK) == 0:
            score = 32000
        elif count_energy_in_discard(me) and any(p.id == GREAT_TUSK and attached_energy_count(p) < 2 for p in field_pokemon(me)):
            score = 18000
    elif card_id == SACRED_ASH:
        if count_pokemon_in_discard(me) >= 3 and me.deckCount <= 18:
            score = 13500
    elif card_id == ENERGY_RECYCLER:
        if count_energy_in_discard(me) >= 3 and me.deckCount <= 18:
            score = 13000
    return score


def attach_score(card_id: int, target: Pokemon | None, in_play_area, me, opponent, wall_mode: bool, ko_mode: bool) -> int:
    if target is None:
        return -10000
    active = active_pokemon(me)
    if card_id in ENERGY_IDS:
        score = 0
        if target.id == GREAT_TUSK:
            if ko_mode and attached_energy_count(target) < 4:
                score = 90000
            else:
                score = 120000 if attached_energy_count(target) < 2 else 20000
            if card_id in (MIST_ENERGY, ROCK_FIGHTING_ENERGY):
                score += 32000
            if in_play_area == AreaType.ACTIVE:
                score += 12000
            if count_in_hand(me, EXPLORER_GUIDANCE) > 0 and attached_energy_count(target) == 1:
                score += 40000
        elif target.id == DWEBBLE:
            # Ascension costs 1 colorless; attach if Dwebble is active and can evolve.
            if wall_mode and in_play_area == AreaType.ACTIVE and attached_energy_count(target) < 1:
                score = 160000 if card_id in GRASS_ENERGY_IDS else 130000
            else:
                score = 52000 if in_play_area == AreaType.ACTIVE and attached_energy_count(target) < 1 else 9000
        elif target.id == CRUSTLE:
            if in_play_area == AreaType.ACTIVE and facing_lucario_strong(opponent) and attached_energy_count(target) >= 2 and can_pay_attack(target, SUPERB_SCISSORS):
                score = 9000
            elif card_id in GRASS_ENERGY_IDS:
                if wall_mode and attached_energy_count(target) < 3:
                    score = 130000 if in_play_area == AreaType.ACTIVE else 110000
                else:
                    score = 18000
            else:
                score = 12000
        elif target.id == MEGA_HERACROSS_EX:
            if facing_lucario_strong(opponent) and card_id in GRASS_ENERGY_IDS and attached_energy_count(target) < 3 and has_ready_tusk(me):
                score = 104000 if in_play_area == AreaType.ACTIVE else 82000
            else:
                score = 3000
        elif target.id == KORAIDON_EX:
            if facing_lucario_strong(opponent) and card_id in (BASIC_FIGHTING_ENERGY, ROCK_FIGHTING_ENERGY, MIST_ENERGY) and attached_energy_count(target) < 3 and has_ready_tusk(me):
                score = 98000 if in_play_area == AreaType.ACTIVE else 76000
            else:
                score = 3000
        elif target.id == TERRAKION:
            if facing_lucario_strong(opponent) and card_id in (BASIC_FIGHTING_ENERGY, ROCK_FIGHTING_ENERGY, MIST_ENERGY) and attached_energy_count(target) < 2 and has_ready_tusk(me):
                score = 82000 if in_play_area == AreaType.ACTIVE else 60000
            else:
                score = 3000
        elif target.id == MEGA_HAWLUCHA_EX:
            if facing_lucario_strong(opponent) and card_id in (BASIC_FIGHTING_ENERGY, ROCK_FIGHTING_ENERGY, MIST_ENERGY) and attached_energy_count(target) < 3 and has_ready_tusk(me):
                score = 90000 if in_play_area == AreaType.ACTIVE else 70000
            else:
                score = 3000
        elif target.id == DURANT_EX:
            if card_id in GRASS_ENERGY_IDS:
                score = (72000 if ko_mode else 22000) if attached_energy_count(target) < 3 else 6000
            else:
                score = 9000
        elif target.id == CORNERSTONE_OGERPON:
            score = 15000 if card_id == BASIC_FIGHTING_ENERGY else 5000
        else:
            score = 2000
        return score
    if card_id == HERO_CAPE:
        if not has_tool(target, HERO_CAPE):
            if target.id == CRUSTLE and wall_mode:
                return 260000 if in_play_area == AreaType.ACTIVE else 220000
            if target.id == GREAT_TUSK:
                return 210000 if in_play_area == AreaType.ACTIVE else 170000
            if target.id == CRUSTLE:
                return 150000
            return 50000
        return -10000
    if card_id == AIR_BALLOON:
        if not has_tool(target, AIR_BALLOON):
            if target.id == GREAT_TUSK:
                return 140000 if in_play_area == AreaType.ACTIVE else 90000
            if target.id in (DWEBBLE, CRUSTLE):
                return 50000
        return -10000
    if card_id == SACRED_CHARM:
        if not has_tool(target, SACRED_CHARM):
            if target.id in (GREAT_TUSK, CRUSTLE, DWEBBLE) and opponent_can_attack_soon(opponent):
                return 110000
            return 25000
        return -10000
    if card_id == GRAVITY_GEM:
        if in_play_area == AreaType.ACTIVE and not has_tool(target, GRAVITY_GEM):
            if target.id == CRUSTLE and wall_mode:
                return 42000
            if target.id == GREAT_TUSK:
                return 20000
            return 12000
        return -10000
    if card_id == HANDY_CIRCULATOR:
        if in_play_area == AreaType.ACTIVE and not has_tool(target, HANDY_CIRCULATOR):
            if target.id == CRUSTLE and wall_mode:
                return 39000
            if target.id == GREAT_TUSK:
                return 18000
            return 8000
        return -10000
    return -10000


def switch_score(card: Pokemon, player_index: int, me, opponent, state, wall_mode: bool, ko_mode: bool) -> int:
    if player_index != state.yourIndex:
        # Select opponent target: in LO mode, trap high-retreat/low-energy targets.
        data = CARD_TABLE.get(card.id)
        retreat = data.retreatCost if data is not None else 0
        score = 1000 + retreat * 700 - attached_energy_count(card) * 800
        if facing_lucario_strong(opponent):
            if card.id in (674, 673) and attached_energy_count(card) <= 1:
                score += 8500
            elif card.id in (675, 676, 677) and attached_energy_count(card) == 0:
                score += 5200
        if facing_abomasnow_sample(opponent):
            if card.id in (723, 722) and attached_energy_count(card) <= 2:
                score += 6500
            if card.id == 721 and attached_energy_count(card) >= 2:
                score -= 2500
        if facing_dragapult_sample(opponent):
            if card.id == 121:
                score += 4500
            if card.id in (119, 235) and retreat == 0:
                score -= 2000
        if facing_alakazam(opponent):
            if card.id in (741, 742, 305, 858, 343) and attached_energy_count(card) == 0:
                score += 4000
        if ko_mode:
            score = max(score, 3500 - card.hp + attached_energy_count(card) * 300)
        if is_ex_pokemon(card) and not ko_mode and not (facing_abomasnow_sample(opponent) or facing_dragapult_sample(opponent)):
            score -= 300
        return score

    if card.id == GREAT_TUSK and can_pay_attack(card, LAND_COLLAPSE) and opponent.deckCount <= 20:
        return 190000 + attached_energy_count(card) * 500 - damage_on(card)
    if wall_mode and card.id == CRUSTLE:
        return 150000 + attached_energy_count(card) * 600 - damage_on(card)
    if card.id == GREAT_TUSK and can_pay_attack(card, LAND_COLLAPSE):
        return 120000 + attached_energy_count(card) * 500 - damage_on(card)
    if card.id == DWEBBLE and wall_mode:
        return 68000 + attached_energy_count(card) * 900
    if facing_lucario_strong(opponent) and card.id == MEGA_HERACROSS_EX and can_pay_attack(card, HERA_MOUNTAIN_RAMMING):
        return 170000 + attached_energy_count(card) * 800 - damage_on(card)
    if facing_lucario_strong(opponent) and card.id == KORAIDON_EX and can_pay_attack(card, ORICHALCUM_FANG):
        return 165000 + attached_energy_count(card) * 800 - damage_on(card)
    if facing_lucario_strong(opponent) and card.id == TERRAKION and can_pay_attack(card, TERRAKION_RETALIATE):
        return 135000 + attached_energy_count(card) * 800 - damage_on(card)
    if facing_lucario_strong(opponent) and card.id == MEGA_HAWLUCHA_EX and can_pay_attack(card, SOMERSAULT_DIVE):
        # Do not abandon Neutralization Zone too casually; this is for KO mode or enemy stadium races.
        if state.stadium and state.stadium[0].id != NEUTRAL_CENTER:
            return 190000 + attached_energy_count(card) * 900 - damage_on(card)
        if ko_mode and opponent.deckCount <= 26:
            return 165000 + attached_energy_count(card) * 900 - damage_on(card)
    if card.id == TATSUGIRI and not state.supporterPlayed and count_in_hand(me, EXPLORER_GUIDANCE) == 0:
        return 36000
    if card.id == FLUTTER_MANE and opponent_can_attack_soon(opponent):
        return 24000
    if card.id == CORNERSTONE_OGERPON and ko_mode:
        return 30000
    if card.id == DURANT_EX:
        return 105000 + attached_energy_count(card) * 500 if ko_mode and can_pay_attack(card, DURANT_VENGEFUL_CRUSH) else -20000
    return 1000 + attached_energy_count(card) * 300 - damage_on(card) // 2


def attack_score(attack_id: int | None, me, opponent, state, wall_mode: bool, ko_mode: bool) -> int:
    active = active_pokemon(me)
    if active is None:
        return -10000
    if attack_id == LAND_COLLAPSE:
        score = 180000
        if state.supporterPlayed:
            score += 90000
        else:
            # Attack is still better than doing nothing, but Explorer + attack should outrank raw attack.
            score += 10000
        if opponent.deckCount <= (4 if state.supporterPlayed else 1):
            score += 100000
        return score
    if attack_id == ASCENSION:
        if active.id == DWEBBLE and count_in_field(me, CRUSTLE) == 0:
            return 125000 if wall_mode else 80000
        return 1000
    if attack_id == MOUNTAIN_RAMMING:
        score = 32000 if not ko_mode else 300000
        if opponent.deckCount <= 1:
            score += 100000
        return score
    if attack_id == ROCK_KAGURA:
        return 42000 if active.id == CORNERSTONE_OGERPON and attached_energy_count(active) < 3 else 8000
    if attack_id == SUPERB_SCISSORS:
        if ko_mode:
            return 350000
        if facing_lucario_strong(opponent) or generic_active_nonex_race_threat(opponent):
            return 235000
        if wall_mode:
            return 65000
        return 9000
    if attack_id == DURANT_VENGEFUL_CRUSH:
        return 300000 if ko_mode else 10000
    if attack_id == HERA_MOUNTAIN_RAMMING:
        if facing_lucario_strong(opponent):
            return 285000 if ko_mode or opponent.deckCount <= 28 else 90000
        return 12000
    if attack_id == JUGGERNAUT_HORN:
        return 250000 if facing_lucario_strong(opponent) and ko_mode else 16000
    if attack_id == ORICHALCUM_FANG:
        return 285000 if facing_lucario_strong(opponent) and ko_mode else 12000
    if attack_id == KORAIDON_TERA:
        return 16000
    if attack_id == TERRAKION_RETALIATE:
        return 235000 if facing_lucario_strong(opponent) and ko_mode else 20000
    if attack_id == TERRAKION_LAND_CRUSH:
        return 220000 if facing_lucario_strong(opponent) and ko_mode else 12000
    if attack_id == SOMERSAULT_DIVE:
        if facing_lucario_strong(opponent):
            if state.stadium and state.stadium[0].id != NEUTRAL_CENTER:
                return 300000 if ko_mode else 185000
            # Avoid discarding our own Neutralization Zone unless the game is already near terminal.
            if state.stadium and state.stadium[0].id == NEUTRAL_CENTER and opponent.deckCount > 12:
                return 18000
            return 260000 if ko_mode else 28000
        return 12000
    if attack_id == GIANT_TUSK:
        return 300000 if ko_mode else -5000
    attack = ATTACK_TABLE.get(attack_id)
    damage = attack.damage if attack is not None else 0
    return (6000 + damage * 8) if ko_mode else (1200 - damage * 5)


def select_card_score(card, player_index, context, me, opponent, state, wall_mode: bool, ko_mode: bool) -> int:
    if card is None:
        return -10000
    cid = card.id
    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        if isinstance(card, Pokemon):
            return switch_score(card, player_index, me, opponent, state, wall_mode, ko_mode)
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        return initial_active_score(cid, me, opponent)
    if context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        return setup_bench_score(cid, me, opponent)
    if context in (SelectContext.EVOLVES_TO, SelectContext.EVOLVE):
        if cid == CRUSTLE:
            return 90000 if wall_mode or has_in_field(me, DWEBBLE) else 30000
        return 1000
    if context == SelectContext.EVOLVES_FROM:
        if cid == DWEBBLE:
            return 90000
        return 1000
    if context == SelectContext.TO_HAND:
        # Search target priorities. Ultra Ball can search both Great Tusk and Crustle.
        if cid == NEUTRAL_CENTER:
            return 120000
        if cid == COLRESS_TENACITY and (opponent_ex_pressure(opponent) or opponent_shows_ex_evolution_line(opponent)):
            return 110000
        if cid == EXPLORER_GUIDANCE and active_tusk_ready(me) and not state.supporterPlayed:
            return 160000
        if cid == BOSS_ORDERS and opponent_has_trappable_bench(opponent):
            return 75000 if opponent.deckCount <= 10 or len(opponent.prize) <= 1 else 26000
        if cid == LISIA_APPEAL and opponent_has_trappable_basic_bench(opponent):
            return 80000 if opponent.deckCount <= 12 or len(opponent.prize) <= 1 else 24000
        if cid == GREAT_TUSK:
            return 85000 if count_in_field(me, GREAT_TUSK) == 0 else 45000
        if cid == CRUSTLE and has_in_field(me, DWEBBLE):
            return 78000 if wall_mode else 43000
        if cid == DWEBBLE:
            return 64000 if wall_mode or count_in_field(me, DWEBBLE) == 0 else 22000
        if cid in ENERGY_IDS and any(p.id == GREAT_TUSK and attached_energy_count(p) < 2 for p in field_pokemon(me)):
            return 56000
        if cid == MEGA_HERACROSS_EX:
            return 52000 if facing_lucario_strong(opponent) and count_in_field(me, MEGA_HERACROSS_EX) == 0 and len(field_pokemon(me)) >= 3 else 8000
        if cid == KORAIDON_EX:
            return 44000 if facing_lucario_strong(opponent) and count_in_field(me, KORAIDON_EX) == 0 and len(field_pokemon(me)) >= 3 else 7000
        if cid == TERRAKION:
            return 40000 if facing_lucario_strong(opponent) and count_in_field(me, TERRAKION) == 0 else 7000
        if cid == MEGA_HAWLUCHA_EX:
            return 42000 if facing_lucario_strong(opponent) and count_in_field(me, MEGA_HAWLUCHA_EX) == 0 and len(field_pokemon(me)) >= 3 else 7000
        if cid == DURANT_EX:
            return 42000 if count_in_field(me, DURANT_EX) == 0 else 24000
        if cid == BUG_CATCHING_SET:
            return 32000
        return card_keep_value(cid, me, opponent, state, wall_mode, ko_mode)
    if context in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
        # Choose low-value cards to discard. Preserve Explorer for the boosted Great Tusk turn.
        value = card_keep_value(cid, me, opponent, state, wall_mode, ko_mode)
        score = 7000 - value
        if cid == EXPLORER_GUIDANCE:
            score -= 8000
        if cid == GREAT_TUSK:
            score -= 9000
        if cid == CRUSTLE and has_in_field(me, DWEBBLE):
            score -= 5000
        if cid in ENERGY_IDS and any(p.id == GREAT_TUSK and attached_energy_count(p) < 2 for p in field_pokemon(me)):
            score -= 4000
        return score
    if context in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
        if cid == GREAT_TUSK:
            return 85000
        if cid in ENERGY_IDS:
            return 70000
        if cid == EXPLORER_GUIDANCE:
            return 55000
        if cid in (DWEBBLE, CRUSTLE, TATSUGIRI, DURANT_EX, MEGA_HERACROSS_EX, KORAIDON_EX, TERRAKION, MEGA_HAWLUCHA_EX):
            return 30000
        return 1000
    if context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
        if cid == GREAT_TUSK:
            return 85000
        if cid == CRUSTLE:
            return 50000 if wall_mode else 18000
        if cid == DURANT_EX:
            return 20000
        if cid == MEGA_HERACROSS_EX:
            return 52000 if facing_lucario_strong(opponent) else 1000
        if cid == KORAIDON_EX:
            return 48000 if facing_lucario_strong(opponent) else 1000
        if cid == TERRAKION:
            return 42000 if facing_lucario_strong(opponent) else 1000
        if cid == MEGA_HAWLUCHA_EX:
            return 46000 if facing_lucario_strong(opponent) else 1000
        return 1000
    if context in (SelectContext.DETACH_FROM, SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_ENERGY):
        if isinstance(card, Pokemon):
            return -attached_energy_count(card) * 1000
        return 100
    if context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
        if isinstance(card, Pokemon):
            if player_index == state.yourIndex:
                return -5000
            return 3500 if ko_mode else -1000
    if context in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
        if isinstance(card, Pokemon) and player_index == state.yourIndex:
            return damage_on(card) + (3000 if card.id in (GREAT_TUSK, CRUSTLE) else 0)
    if isinstance(card, Pokemon):
        return switch_score(card, player_index, me, opponent, state, wall_mode, ko_mode)
    return card_keep_value(cid, me, opponent, state, wall_mode, ko_mode)


def _agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    state = obs.current
    select = obs.select
    context = select.context
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    wall_mode = should_wall_mode(me, opponent, state)
    ko_mode = should_ko_mode(me, opponent, state)
    active = active_pokemon(me)

    scores = []
    for option in select.option:
        score = 0
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
                    # Use Tatsugiri to find Explorer if Tusk is not yet ready or Explorer is missing.
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
                    # Do not leave a useless Great Tusk active if legal retreat exists.
                    score = 70000
                elif active is not None and active.id == TATSUGIRI and (state.supporterPlayed or count_in_hand(me, EXPLORER_GUIDANCE) > 0):
                    score = 36000
                else:
                    score = -10000
            elif option.type == OptionType.ATTACK:
                score = attack_score(option.attackId, me, opponent, state, wall_mode, ko_mode)
                if option.attackId is None:
                    # Generic ATTACK button. Specific attack will usually be selected next.
                    if active_tusk_ready(me):
                        # If Explorer is available, play it first unless already played.
                        if count_in_hand(me, EXPLORER_GUIDANCE) > 0 and not state.supporterPlayed:
                            score = 100000
                        else:
                            score = 200000 + (70000 if state.supporterPlayed else 0)
                    elif active is not None and active.id == DWEBBLE:
                        score = 90000
                    elif active is not None and active.id == CRUSTLE and can_pay_attack(active, SUPERB_SCISSORS):
                        if ko_mode:
                            score = 325000
                        elif facing_lucario_strong(opponent) or generic_active_nonex_race_threat(opponent):
                            score = 230000
                        elif wall_mode:
                            score = 65000
                        else:
                            score = 9000
                    elif active is not None and active.id == CRUSTLE and wall_mode:
                        score = 35000
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
                # Do not over-protect deck if drawing/searching unlocks Tusk mill.
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
        scores.append(score)

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    result = []
    for index in order:
        if len(result) >= select.maxCount:
            break
        if scores[index] >= 0 or len(result) < select.minCount:
            result.append(index)
    return result


def _fallback_action(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if select is None:
        return read_deck_csv()
    options = select.get("option") or []
    min_count = max(0, int(select.get("minCount", 0) or 0))
    max_count = max(0, int(select.get("maxCount", len(options)) or 0))
    return list(range(min(min_count, max_count, len(options))))


def agent(obs_dict: dict, configuration=None) -> list[int]:
    try:
        return _agent(obs_dict)
    except Exception:
        if os.environ.get("DEBUG_AGENT") == "1":
            import traceback
            traceback.print_exc()
        return _fallback_action(obs_dict)


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

GT_SEARCH_ENABLED = True
GT_SEARCH_CANDIDATES = 2
GT_SEARCH_BUDGET_S = 0.03
GT_SEARCH_MARGIN = 2400.0
GT_SEARCH_ROLLOUT_STEPS = 4
GT_BELIEF_WORLDS = 4
GT_RISK_PENALTY = 0.20
GT_OPPONENT_DECKS = {"unknown": read_deck_csv()}


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
    value += (len(getattr(opp, "prize", []) or []) - len(getattr(me, "prize", []) or [])) * 3500.0
    value += (60 - opp_deck) * 950.0
    value -= (60 - my_deck) * 280.0
    if opp_deck <= 4:
        value += (5 - opp_deck) * 18000.0
    if my_deck <= 5:
        value -= (6 - my_deck) * 12000.0
    value += (getattr(me, "handCount", 0) - getattr(opp, "handCount", 0)) * 45.0
    value += (my_deck - opp_deck) * 30.0

    hand = list(getattr(me, "hand", []) or [])
    hand_ids = [getattr(card, "id", None) for card in hand]
    active = active_pokemon(me)
    if active is not None and active.id == GREAT_TUSK:
        value += 1800.0
        if _gt_energy_count(active) >= 2:
            value += 6500.0
            if not getattr(st, "supporterPlayed", False) and EXPLORER_GUIDANCE in hand_ids:
                value += 14000.0
            if getattr(st, "supporterPlayed", False):
                value += 4500.0
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
    return value


def _gt_option_tactical_bonus(obs, option, score):
    state = obs.current
    select = obs.select
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    active = active_pokemon(me)
    bonus = 0
    try:
        if select.context == SelectContext.MAIN:
            if option.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, option.index, state.yourIndex)
                cid = getattr(card, "id", None)
                if cid == EXPLORER_GUIDANCE and active is not None and active.id == GREAT_TUSK and can_pay_attack(active, LAND_COLLAPSE):
                    bonus += 900000
                elif cid in (POKEGEAR_30, POKE_PAD, ULTRA_BALL, FIGHT_GONG, BUDDY_BUDDY_POFFIN):
                    bonus += 220000
                elif cid in (ERI, XEROSIC_SCHEME, HAND_TRIMMER, ENHANCED_HAMMER, ENERGY_LASSO, FLUTE):
                    bonus += 160000
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
            elif option.type == OptionType.ABILITY:
                bonus += 150000
            elif option.type == OptionType.RETREAT:
                bonus += 100000
            elif option.type == OptionType.ATTACK:
                bonus += 260000
                if active is not None and active.id == GREAT_TUSK and can_pay_attack(active, LAND_COLLAPSE):
                    bonus += 220000
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
        key=lambda i: (_gt_option_tactical_bonus(obs, obs.select.option[i], scores[i]), scores[i]),
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
    for own_deck, own_prize, opp_deck, opp_prize, opp_hand in _gt_sample_hidden_worlds(obs, me, opp, deck, fallback_basic):
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
        aggregates[idx] = mean - GT_RISK_PENALTY * variance ** 0.5
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
        return _fallback_action(obs_dict)


# Kaggle executes main.py and picks the last callable in insertion order.
# Keep the search helper functions from becoming the submitted callable.
kaggle_agent = agent
