from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PublicBeliefState:
    visible_ids: tuple[int, ...] = ()
    opponent_hand_count: int = 0
    opponent_deck_count: int = 0
    opponent_prize_count: int = 0
    turn: int = 0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class HiddenWorld:
    archetype: str
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    active_fallback: tuple[int, ...]


def infer_archetype(
    visible_ids: Iterable[int],
    archetype_decks: dict[str, Iterable[int]],
) -> str:
    visible = set(visible_ids)
    if not visible or not archetype_decks:
        return "unknown"
    scores = {
        name: len(visible.intersection(set(deck)))
        for name, deck in archetype_decks.items()
    }
    best_name, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    return best_name if best_score > 0 else "unknown"


def _candidate_cards(
    visible_ids: Iterable[int],
    fallback_deck: Iterable[int],
    archetype_decks: dict[str, Iterable[int]],
) -> dict[str, list[int]]:
    visible = set(visible_ids)
    candidates: dict[str, list[int]] = {}
    for name, deck in archetype_decks.items():
        cards = list(deck)
        if len(set(cards).intersection(visible)):
            candidates[name] = cards
    if not candidates:
        candidates["unknown"] = list(fallback_deck)
    return candidates


def sample_hidden_worlds(
    state: PublicBeliefState,
    archetype_decks: dict[str, Iterable[int]],
    fallback_deck: Iterable[int],
    count: int = 8,
) -> list[HiddenWorld]:
    if count <= 0:
        return []
    candidates = _candidate_cards(state.visible_ids, fallback_deck, archetype_decks)
    rng = random.Random(state.seed)
    worlds: list[HiddenWorld] = []
    for _ in range(count):
        archetype = rng.choice(sorted(candidates))
        cards = list(candidates[archetype])
        available = Counter(cards)
        for card_id in state.visible_ids:
            if available[card_id] > 0:
                available[card_id] -= 1
        pool = [card_id for card_id, quantity in available.items() for _ in range(max(0, quantity))]
        rng.shuffle(pool)

        requested = state.opponent_deck_count + state.opponent_prize_count + state.opponent_hand_count
        if len(pool) < requested:
            fallback = list(fallback_deck)
            rng.shuffle(fallback)
            pool.extend(fallback[: max(0, requested - len(pool))])
        cursor = 0
        deck = tuple(pool[cursor : cursor + state.opponent_deck_count])
        cursor += state.opponent_deck_count
        prize = tuple(pool[cursor : cursor + state.opponent_prize_count])
        cursor += state.opponent_prize_count
        hand = tuple(pool[cursor : cursor + state.opponent_hand_count])
        worlds.append(HiddenWorld(archetype, deck, prize, hand, ()))
    return worlds
