from __future__ import annotations

import tarfile
import shutil
import json
from pathlib import Path

from local_eval.archive import SubmissionArchiveError, safe_extract_tar_gz
from local_eval.evaluator import run_match, save_report
from local_eval.models import EvalConfig, GameResult
from local_eval.rating import KaggleStyleRating
from local_eval.referee import InvalidAction, validate_action, validate_deck


def test_validate_deck_requires_sixty_ints():
    assert validate_deck([1] * 60) == [1] * 60
    try:
        validate_deck([1] * 59)
        raise AssertionError("expected InvalidAction")
    except InvalidAction:
        pass


def test_validate_deck_rejects_invalid_counts_and_ids():
    for deck in ([999] * 5 + [1] * 55, [0] + [1] * 59, [True] + [1] * 59, [1159, 1247] + [1] * 58):
        try:
            validate_deck(deck)
            raise AssertionError(f"expected InvalidAction for {deck[:6]!r}")
        except InvalidAction:
            pass


def test_validate_action_bounds_and_duplicates():
    obs = {"select": {"minCount": 1, "maxCount": 2, "option": [{}, {}, {}]}}
    assert validate_action(obs, [0, 2]) == [0, 2]
    for action in ([0, 0], [3], [], ["0"]):
        try:
            validate_action(obs, action)
            raise AssertionError(f"expected InvalidAction for {action!r}")
        except InvalidAction:
            pass


def test_safe_extract_rejects_path_traversal(tmp_path: Path):
    tar_path = tmp_path / "bad.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(payload, arcname="../evil.txt")
    try:
        safe_extract_tar_gz(tar_path, tmp_path / "out")
        raise AssertionError("expected SubmissionArchiveError")
    except SubmissionArchiveError:
        pass


def test_trueskill_winner_moves_up():
    cfg = EvalConfig()
    system = KaggleStyleRating(cfg)
    p0 = system.new_rating()
    p1 = system.new_rating()
    result = GameResult("g0", "a", "b", 1, 0, "P0_WIN", "a", "b", "RESULT", 10, 0.1)
    new_p0, new_p1 = system.update_game(p0, p1, result)
    assert new_p0.mu > p0.mu
    assert new_p1.mu < p1.mu
    assert new_p0.sigma < p0.sigma


def _make_submission(tmp_path: Path, name: str, main_code: str) -> Path:
    src = tmp_path / name
    baseline_extract = tmp_path / f"{name}_baseline"
    with tarfile.open("基准/submission_sorce_700.tar.gz", "r:gz") as tar:
        tar.extractall(baseline_extract)
    shutil.copytree(baseline_extract / "cg", src / "cg")
    (src / "main.py").write_text(main_code)
    shutil.copy2(baseline_extract / "deck.csv", src / "deck.csv")
    tar_path = tmp_path / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in src.rglob("*"):
            tar.add(path, arcname=str(path.relative_to(src)))
    return tar_path


def test_invalid_action_loses_against_baseline(tmp_path: Path):
    bad = _make_submission(
        tmp_path,
        "bad",
        "def agent(obs):\n"
        "    if obs.get('select') is None:\n"
        "        return [int(x) for x in open('deck.csv').read().split()]\n"
        "    return [999]\n",
    )
    baseline = Path("基准/submission_sorce_700.tar.gz")
    report = run_match(bad, baseline, games=1, config=EvalConfig(max_actions=20, run_timeout_s=60), project_root=Path.cwd())
    bad_stats = next(s for s in report.standings if s.name == "bad")
    assert bad_stats.losses == 1
    assert bad_stats.invalids == 1


def test_timeout_loses_against_baseline(tmp_path: Path):
    slow = _make_submission(
        tmp_path,
        "slow",
        "import time\n"
        "def agent(obs):\n"
        "    if obs.get('select') is None:\n"
        "        return [int(x) for x in open('deck.csv').read().split()]\n"
        "    time.sleep(0.2)\n"
        "    return [0]\n",
    )
    baseline = Path("基准/submission_sorce_700.tar.gz")
    cfg = EvalConfig(act_timeout_s=0.05, overage_time_s=0.0, max_actions=20, run_timeout_s=60)
    report = run_match(slow, baseline, games=1, config=cfg, project_root=Path.cwd())
    slow_stats = next(s for s in report.standings if s.name == "slow")
    assert slow_stats.losses == 1
    assert slow_stats.timeouts == 1


def test_stdout_noise_does_not_break_protocol(tmp_path: Path):
    noisy = _make_submission(
        tmp_path,
        "noisy",
        "print('import noise')\n"
        "def agent(obs):\n"
        "    print('agent noise')\n"
        "    if obs.get('select') is None:\n"
        "        return [int(x) for x in open('deck.csv').read().split()]\n"
        "    return [0]\n",
    )
    baseline = Path("基准/submission_sorce_700.tar.gz")
    report = run_match(noisy, baseline, games=1, config=EvalConfig(max_actions=20, run_timeout_s=60), project_root=Path.cwd())
    noisy_stats = next(s for s in report.standings if s.name == "noisy")
    assert noisy_stats.crashes == 0


def test_import_timeout_loses_against_baseline(tmp_path: Path):
    slow_import = _make_submission(
        tmp_path,
        "slow_import",
        "import time\n"
        "time.sleep(0.2)\n"
        "def agent(obs):\n"
        "    if obs.get('select') is None:\n"
        "        return [int(x) for x in open('deck.csv').read().split()]\n"
        "    return [0]\n",
    )
    baseline = Path("基准/submission_sorce_700.tar.gz")
    cfg = EvalConfig(import_timeout_s=0.05, max_actions=20, run_timeout_s=60)
    report = run_match(slow_import, baseline, games=1, config=cfg, project_root=Path.cwd())
    slow_stats = next(s for s in report.standings if s.name == "slow_import")
    assert slow_stats.losses == 1
    assert slow_stats.crashes == 1
    assert report.games[0].reason == "IMPORT_ERROR"


def test_deck_timeout_loses_against_baseline(tmp_path: Path):
    slow_deck = _make_submission(
        tmp_path,
        "slow_deck",
        "import time\n"
        "def agent(obs):\n"
        "    if obs.get('select') is None:\n"
        "        time.sleep(0.2)\n"
        "        return [int(x) for x in open('deck.csv').read().split()]\n"
        "    return [0]\n",
    )
    baseline = Path("基准/submission_sorce_700.tar.gz")
    cfg = EvalConfig(deck_timeout_s=0.05, max_actions=20, run_timeout_s=60)
    report = run_match(slow_deck, baseline, games=1, config=cfg, project_root=Path.cwd())
    slow_stats = next(s for s in report.standings if s.name == "slow_deck")
    assert slow_stats.losses == 1
    assert slow_stats.crashes == 1
    assert report.games[0].reason == "DECK_ERROR"


def test_save_report_writes_debug_game_records(tmp_path: Path):
    bad = _make_submission(
        tmp_path,
        "bad_record",
        "def agent(obs):\n"
        "    if obs.get('select') is None:\n"
        "        return [int(x) for x in open('deck.csv').read().split()]\n"
        "    return [999]\n",
    )
    baseline = Path("基准/submission_sorce_700.tar.gz")
    report = run_match(bad, baseline, games=1, config=EvalConfig(max_actions=20, run_timeout_s=60), project_root=Path.cwd())
    out = tmp_path / "report"
    save_report(report, out)
    record = out / "game_records" / "g000000.json"
    assert record.exists()
    text = record.read_text()
    assert '"trace"' in text
    assert '"event": "deck"' in text
    assert '"event": "action"' in text
    assert (out / "report.json").exists()
    assert (out / "summary.csv").exists()
    games = (out / "games.jsonl").read_text().splitlines()
    assert len(games) == len(report.games)
    assert json.loads(games[0])["game_id"] == report.games[0].game_id
