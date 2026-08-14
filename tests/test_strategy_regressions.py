from types import SimpleNamespace

import main


def card(card_id):
    return SimpleNamespace(id=card_id)


def player(deck=30, prizes=6, hand=(), active=(), bench=()):
    return SimpleNamespace(
        deckCount=deck,
        prize=[card(1) for _ in range(prizes)],
        hand=[card(value) for value in hand],
        active=list(active),
        bench=list(bench),
        discard=[],
        handCount=len(hand),
        benchMax=5,
    )


def state(me, opponent):
    return SimpleNamespace(
        players=[me, opponent],
        yourIndex=0,
        stadium=[],
        supporterPlayed=False,
        stadiumPlayed=False,
    )


def test_late_guard_is_not_limited_to_self_mill_matchups():
    me = player(deck=7, hand=[main.POKEGEAR_30])
    opponent = player(deck=9)
    assert main.own_deck_safety_guard(me, opponent) is True
    assert main.play_score(main.POKEGEAR_30, me, opponent, state(me, opponent), False, False) < 0


def test_ready_boosted_explorer_remains_allowed():
    tusk = SimpleNamespace(id=main.GREAT_TUSK, energies=[card(1), card(6)], hp=160, maxHp=160, tools=[])
    me = player(deck=7, hand=[main.EXPLORER_GUIDANCE], active=[tusk])
    opponent = player(deck=9)
    assert main.own_deck_safety_guard(me, opponent) is True
    assert main.play_score(main.EXPLORER_GUIDANCE, me, opponent, state(me, opponent), False, False) > 0


def test_endgame_wall_survives_low_opponent_deck():
    me = player(deck=20, hand=[main.DWEBBLE])
    opponent = player(deck=10, prizes=2, active=[SimpleNamespace(id=main.KORAIDON_EX, energies=[], hp=200, maxHp=200, tools=[])])
    assert main.should_wall_mode(me, opponent, state(me, opponent)) is True


def test_ready_tusk_does_not_wall_at_terminal_deck():
    tusk = SimpleNamespace(id=main.GREAT_TUSK, energies=[card(1), card(6)], hp=160, maxHp=160, tools=[])
    me = player(deck=20, active=[tusk], hand=[main.EXPLORER_GUIDANCE])
    opponent = player(deck=4, prizes=2)
    assert main.should_wall_mode(me, opponent, state(me, opponent)) is False


def test_opening_tusk_requires_setup_package():
    weak = player(hand=[main.DWEBBLE, main.GREAT_TUSK])
    strong = player(hand=[main.GREAT_TUSK, main.BASIC_FIGHTING_ENERGY])
    assert main.initial_active_score(main.DWEBBLE, weak, player()) > main.initial_active_score(main.GREAT_TUSK, weak, player())
    assert main.initial_active_score(main.GREAT_TUSK, strong, player()) > main.initial_active_score(main.DWEBBLE, strong, player())
