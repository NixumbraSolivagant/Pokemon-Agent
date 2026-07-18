from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from local_eval.archive import safe_extract_tar_gz
from local_eval.models import EvalConfig
from local_eval.player import PlayerProcess
from local_eval.referee import validate_action
from tools.loss_mining import player_value


@dataclass(slots=True)
class ScenarioEvalRow:
    scenario_id: str
    candidate: str
    reached: bool
    actions_replayed: int
    takeover_actions: int
    start_value: float
    end_value: float
    delta: float
    result: int
    reason: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    return [row for row in data if isinstance(row, dict)]


def _fallback_action(obs: dict[str, Any]) -> list[int]:
    select = obs.get("select") or {}
    min_count = max(0, int(select.get("minCount") or 0))
    option_count = len(select.get("option") or [])
    return list(range(min(min_count, option_count)))


def eval_scenario(
    scenario: dict[str, Any],
    candidate_tarball: Path,
    opponent_tarball: Path,
    candidate_name: str,
    opponent_name: str,
    config: EvalConfig,
    project_root: Path,
    takeover_actions: int = 80,
) -> ScenarioEvalRow:
    scenario_id = str(scenario.get("scenario_id", "scenario"))
    focus_seat = int(scenario.get("focus_seat", 0))
    seed = int(scenario.get("seed") or 0)
    random.seed(seed)
    p0_tarball = candidate_tarball if focus_seat == 0 else opponent_tarball
    p1_tarball = opponent_tarball if focus_seat == 0 else candidate_tarball
    p0_name = candidate_name if focus_seat == 0 else opponent_name
    p1_name = opponent_name if focus_seat == 0 else candidate_name
    candidate_seat = focus_seat
    actions_replayed = 0
    takeover_count = 0
    start_value = 0.0
    end_value = 0.0
    result = -1
    p0: PlayerProcess | None = None
    p1: PlayerProcess | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="pokemon_scenario_eval_") as tmp:
            tmp_path = Path(tmp)
            p0_dir = safe_extract_tar_gz(str(p0_tarball), tmp_path / "p0")
            p1_dir = safe_extract_tar_gz(str(p1_tarball), tmp_path / "p1")
            sys.path.insert(0, str(p0_dir))
            from cg.game import battle_finish, battle_select, battle_start

            p0 = PlayerProcess(p0_name, p0_dir, project_root, config.act_timeout_s, config.import_timeout_s)
            p1 = PlayerProcess(p1_name, p1_dir, project_root, config.act_timeout_s, config.import_timeout_s)
            p0.start()
            p1.start()
            p0_deck = scenario.get("p0_deck") or []
            p1_deck = scenario.get("p1_deck") or []
            obs, _ = battle_start([int(x) for x in p0_deck], [int(x) for x in p1_deck])
            if obs is None:
                raise RuntimeError("battle_start returned None")
            for step in scenario.get("prefix_actions", [])[: config.max_actions]:
                cur = obs.get("current") or {}
                result = int(cur.get("result", -1))
                if result in (0, 1, 2):
                    break
                action = [int(x) for x in step.get("action", [])]
                validate_action(obs, action)
                obs = battle_select(action)
                actions_replayed += 1
            reached = actions_replayed == len(scenario.get("prefix_actions", []))
            start_value = player_value(obs, candidate_seat)
            while takeover_count < takeover_actions and actions_replayed + takeover_count < config.max_actions:
                cur = obs.get("current") or {}
                result = int(cur.get("result", -1))
                if result in (0, 1, 2):
                    break
                actor_seat = int(cur.get("yourIndex", 0))
                actor = p0 if actor_seat == 0 else p1
                assert actor is not None
                try:
                    raw_action, _ = actor.request({"kind": "act", "obs": obs}, config.act_timeout_s + config.overage_time_s)
                    action = validate_action(obs, raw_action)
                except Exception:
                    action = _fallback_action(obs)
                obs = battle_select(action)
                takeover_count += 1
            end_value = player_value(obs, candidate_seat)
            result = int((obs.get("current") or {}).get("result", -1))
            try:
                battle_finish()
            except Exception:
                pass
            return ScenarioEvalRow(
                scenario_id=scenario_id,
                candidate=candidate_name,
                reached=reached,
                actions_replayed=actions_replayed,
                takeover_actions=takeover_count,
                start_value=start_value,
                end_value=end_value,
                delta=end_value - start_value,
                result=result,
                reason="ok",
            )
    except Exception as exc:
        return ScenarioEvalRow(
            scenario_id=scenario_id,
            candidate=candidate_name,
            reached=False,
            actions_replayed=actions_replayed,
            takeover_actions=takeover_count,
            start_value=start_value,
            end_value=end_value,
            delta=end_value - start_value,
            result=result,
            reason="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if p0 is not None:
            p0.close()
        if p1 is not None:
            p1.close()


def run_scenarios(
    scenarios_path: Path,
    candidates: list[Path],
    opponent: Path,
    out: Path,
    limit: int = 24,
    takeover_actions: int = 80,
    seed: int = 20260717,
) -> list[ScenarioEvalRow]:
    scenarios = load_scenarios(scenarios_path)[: max(0, limit)]
    rows: list[ScenarioEvalRow] = []
    cfg = EvalConfig(seed=seed, max_actions=220, run_timeout_s=120.0)
    for candidate in candidates:
        candidate_name = candidate.name.removesuffix(".tar.gz")
        opponent_name = opponent.name.removesuffix(".tar.gz")
        for scenario in scenarios:
            rows.append(eval_scenario(scenario, candidate, opponent, candidate_name, opponent_name, cfg, Path.cwd(), takeover_actions))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([row.to_dict() for row in rows], indent=2), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay mined prefixes and let candidates take over for a short scenario check.")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/discovery_gold/scenario_eval.json"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--takeover-actions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args(argv)
    rows = run_scenarios(args.scenarios, args.candidate, args.opponent, args.out, args.limit, args.takeover_actions, args.seed)
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
