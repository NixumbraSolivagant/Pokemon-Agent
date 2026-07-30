from tools.build_meta_proxies import PROXY_DECKS
from tools.deck_rules import validate_deck_ids


def test_meta_proxy_decks_are_legal():
    assert set(PROXY_DECKS) == {"alakazam", "grimmsnarl", "garchomp", "crustle"}
    for deck in PROXY_DECKS.values():
        validate_deck_ids(deck)
