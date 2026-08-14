from pathlib import Path
from types import SimpleNamespace

from local_eval.models import EvalConfig, GameResult, MatchReport, SubmissionInfo
from tools.robust_gold_search import rank_report


def game(p0, p1, winner, loser, reason):
    outcome = "P0_WIN" if winner == p0 else "P1_WIN"
    return GameResult("g", p0, p1, 1, 0, outcome, winner, loser, reason, 1, 0.01)


def test_import_error_opponent_is_excluded_from_candidate_rate(tmp_path: Path):
    candidate = tmp_path / "candidate.tar.gz"
    opponent = tmp_path / "opponent.tar.gz"
    report = MatchReport(
        EvalConfig(),
        [SubmissionInfo("candidate", str(candidate), "a"), SubmissionInfo("opponent", str(opponent), "b")],
        [
            game("candidate", "opponent", "candidate", "opponent", "RESULT"),
            game("candidate", "opponent", "candidate", "opponent", "IMPORT_ERROR"),
        ],
        [],
    )
    rows = rank_report(report, [candidate], [opponent])
    assert rows[0].games == 1
    assert rows[0].wins == 1
    assert rows[0].excluded == 1


def test_common_random_seed_schedule_reuses_pair_seeds(tmp_path: Path, monkeypatch):
    from local_eval import evaluator

    paths = []
    for name in ("c1", "c2", "opponent"):
        path = tmp_path / f"{name}.tar.gz"
        path.write_bytes(name.encode())
        paths.append(path)
    seen = []

    def fake_play_game(p0_tarball, p1_tarball, p0_name, p1_name, seed, game_id, config, project_root):
        seen.append((frozenset((p0_name, p1_name)), seed))
        return GameResult(game_id, p0_name, p1_name, seed, 0, "P0_WIN", p0_name, p1_name, "RESULT", 1, 0.01)

    monkeypatch.setattr(evaluator, "play_game", fake_play_game)
    evaluator.run_candidate_pool(
        paths[:2], paths[2:], 2,
        EvalConfig(seed=100, workers=1, common_random_seeds=True),
        tmp_path, peer_span=0,
    )
    seeds_by_pair = {}
    for pair, seed in seen:
        seeds_by_pair.setdefault(pair, []).append(seed)
    assert list(seeds_by_pair.values()) == [[100, 101], [100, 101]]
