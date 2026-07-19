from __future__ import annotations

from collections import Counter


BASIC_ENERGY_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
ACE_SPEC_IDS = {
    10, 12, 13, 1080, 1082, 1085, 1088, 1089, 1092, 1093,
    1095, 1096, 1100, 1104, 1107, 1109, 1110, 1111, 1125,
    1126, 1128, 1155, 1158, 1159, 1165, 1167, 1169, 1247, 1249,
}


def validate_deck_ids(deck: list[int]) -> None:
    if len(deck) != 60:
        raise ValueError(f"Deck must contain 60 cards, got {len(deck)}")
    if any(card_id <= 0 for card_id in deck):
        raise ValueError("Deck card ids must be positive ints")
    counts = Counter(deck)
    over_limit = [
        card_id
        for card_id, count in counts.items()
        if card_id not in BASIC_ENERGY_IDS and count > 4
    ]
    if over_limit:
        raise ValueError(f"Non-basic cards exceed four-copy limit: {sorted(over_limit)}")
    ace_specs = [card_id for card_id in deck if card_id in ACE_SPEC_IDS]
    if len(ace_specs) > 1:
        raise ValueError(f"Deck cannot contain more than one ACE SPEC card: {sorted(ace_specs)}")


def deck_to_text(deck: list[int]) -> str:
    validate_deck_ids(deck)
    return "\n".join(str(card_id) for card_id in deck) + "\n"


def apply_deck_swaps(
    deck_text: str,
    swaps: list[tuple[int, int]],
    override: list[int] | None = None,
) -> str:
    if override is not None:
        return deck_to_text(list(override))

    deck = [int(line) for line in deck_text.splitlines() if line.strip()]
    counts = Counter(deck)
    for add_id, remove_id in swaps:
        if counts[remove_id] <= 0:
            continue
        if add_id not in BASIC_ENERGY_IDS and counts[add_id] >= 4:
            continue
        counts[remove_id] -= 1
        counts[add_id] += 1

    out: list[int] = []
    for card_id, count in counts.items():
        out.extend([card_id] * count)
    return deck_to_text(out)
