from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from tools.kaggle_gold_loop import (
    CandidateRecord,
    Registry,
    assert_no_secret_leak,
    execute_cycle,
    ingest_gate,
    plan_dual_slot,
    remaining_submissions,
)


def write_package(path: Path, main_text: str = "def agent(obs, configuration=None):\n    return []\n") -> None:
    members = {
        "main.py": main_text.encode(),
        "deck.csv": ("1\n" * 56 + "1121\n" * 4).encode(),
        "cg/game.py": b"pass\n",
        "cg/api.py": b"pass\n",
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_registry_upsert_is_idempotent(tmp_path: Path):
    registry = Registry(tmp_path / "state.json")
    record = CandidateRecord("anchor:abc", "anchor", "anchor.tar.gz", "abc", "baseline")

    registry.upsert(record)
    registry.upsert(record)

    reloaded = Registry(tmp_path / "state.json")
    assert list(reloaded.data["candidates"]) == ["anchor:abc"]
    assert reloaded.data["candidates"]["anchor:abc"]["hypothesis"] == "baseline"


def test_dual_slot_always_preserves_champion_first():
    assert plan_dual_slot("champion", "challenger", 2) == ["champion", "challenger"]
    assert plan_dual_slot(
        "champion", "challenger", 1,
        active_submissions=[{"id": 10}], champion_submission_ids={10},
    ) == ["challenger"]
    with pytest.raises(RuntimeError):
        plan_dual_slot(
            "champion", "challenger", 1,
            active_submissions=[{"id": 10}], champion_submission_ids={11},
        )
    with pytest.raises(RuntimeError):
        plan_dual_slot("champion", "challenger", 1)
    with pytest.raises(ValueError):
        plan_dual_slot("champion", "champion", 5)


def test_current_kaggle_limit_payload_is_supported():
    assert remaining_submissions({"numTotal": 47, "numAllowedNow": 3}) == 3


def test_submission_cycle_dry_run_never_calls_submit(tmp_path: Path):
    champion_package = tmp_path / "champion.tar.gz"
    challenger_package = tmp_path / "challenger.tar.gz"
    write_package(champion_package)
    write_package(challenger_package)
    registry = Registry(tmp_path / "state.json")
    registry.upsert(CandidateRecord("champion", "champion", str(champion_package), "a", "anchor"))
    registry.upsert(CandidateRecord("challenger", "challenger", str(challenger_package), "b", "test"))
    registry.data["champion_id"] = "champion"
    registry.data["submission_limits"] = {"remaining": 5}
    registry.save()

    class NeverSubmit:
        def submit(self, *args, **kwargs):
            raise AssertionError("dry run attempted a submission")

        def submission_limits(self, competition):
            raise AssertionError("cached limits should be used")

    actions = execute_cycle(registry, NeverSubmit(), "challenger", execute=False)

    assert [action["role"] for action in actions] == ["champion-preserve", "challenger"]
    assert all("submission_id" not in action for action in actions)


def test_secret_scanner_checks_tar_members(tmp_path: Path):
    safe = tmp_path / "safe.tar.gz"
    unsafe = tmp_path / "unsafe.tar.gz"
    write_package(safe)
    write_package(unsafe, 'ACCESS_TOKEN = "secret-value"\n')

    assert_no_secret_leak(safe)
    with pytest.raises(ValueError):
        assert_no_secret_leak(unsafe)


def test_state_file_contains_no_package_payload(tmp_path: Path):
    registry = Registry(tmp_path / "state.json")
    registry.upsert(CandidateRecord("x", "x", "/tmp/x.tar.gz", "deadbeef", "hypothesis"))
    payload = json.loads((tmp_path / "state.json").read_text())
    assert payload["candidates"]["x"]["package_sha256"] == "deadbeef"
    assert "access_token" not in json.dumps(payload).lower()


def test_ingest_gate_records_pass_and_blocks_catastrophic_flag(tmp_path: Path):
    registry = Registry(tmp_path / "state.json")
    registry.upsert(CandidateRecord("candidate", "candidate", "candidate.tar.gz", "abc", "test"))
    ranking = tmp_path / "ranking.json"
    ranking.write_text(
        json.dumps(
            [
                {
                    "name": "candidate",
                    "games": 240,
                    "wins": 150,
                    "losses": 89,
                    "no_results": 1,
                    "mean_rate": 0.628,
                    "worst_rate": 0.45,
                    "worst_lower": 0.35,
                    "robust_score": 0.43,
                }
            ]
        )
    )

    gate = ingest_gate(registry, "candidate", "holdout", ranking)

    assert gate["passed"] is True
    assert registry.candidate("candidate")["gate_results"]["catastrophic_regression"] is False
