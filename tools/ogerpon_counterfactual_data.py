from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.meta_runtime import (
    _sample_hidden_worlds,
    choose_from_scores,
    ensure_mandatory_minimums,
    feature_vector,
    forced_action,
    score_options,
    state_value,
)
from cg.api import SelectContext, search_begin, search_end, search_release, search_step, to_observation_class
from tools.build_meta_submission import family_config


CRITICAL_CONTEXTS = {"MAIN", "SWITCH", "TO_ACTIVE", "ATTACH_FROM", "ATTACH_TO", "SWITCH_ENERGY"}
_WORKER_SETTINGS: dict[str, Any] = {}


def model_bundle(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": int(artifact.get("version", 2)),
        "global": artifact.get("model", {}),
        "heads": artifact.get("head_models", {}),
        "value": artifact.get("value_model", {}),
        "confidence_thresholds": artifact.get("confidence_thresholds", {}),
        "residual_q": artifact.get("residual_q_model", {}),
        "win_value": artifact.get("win_value_model", {}),
    }


def record_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.json"))
    yield from sorted(root.rglob("*.json.gz"))


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def shaped_return(value: float, baseline: float) -> float:
    if value >= 500_000_000.0:
        return 1.0
    if value <= -500_000_000.0:
        return -1.0
    return math.tanh((value - baseline) / 50_000.0)


def rollout_branch(branch, config: dict[str, Any], model: dict[str, Any], player_index: int, start_turn: int, steps: int, deadline: float):
    current = branch
    for _ in range(steps):
        if time.perf_counter() >= deadline:
            break
        obs = current.observation
        if obs.select is None or obs.current.result != -1:
            break
        if obs.current.yourIndex != player_index or obs.current.turn != start_turn:
            break
        scores = score_options(obs, config, model)
        action = forced_action(obs, config)
        if action is None:
            action = choose_from_scores(obs.select, scores)
            action = ensure_mandatory_minimums(obs, action, scores)
        current = search_step(current.searchId, action)
    return current.observation


def collect_state(
    obs_dict: dict[str, Any],
    actual_action: list[int],
    config: dict[str, Any],
    model: dict[str, Any],
    max_candidates: int,
    belief_worlds: int,
    rollout_steps: int,
    risk_penalty: float,
    budget_s: float,
) -> dict[str, Any] | None:
    obs = to_observation_class(obs_dict)
    if obs.select is None or not getattr(obs, "search_begin_input", None):
        return None
    context = SelectContext(obs.select.context).name
    if context not in CRITICAL_CONTEXTS or obs.select.maxCount != 1 or len(obs.select.option) < 2:
        return None
    scores = score_options(obs, config, model)
    ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    anchor_index = ordered[0]
    candidates = ordered[:max_candidates]
    if len(actual_action) == 1 and actual_action[0] not in candidates and 0 <= actual_action[0] < len(scores):
        candidates[-1:] = actual_action
    player_index = obs.current.yourIndex
    start_turn = obs.current.turn
    baseline = state_value(obs, config, perspective_index=player_index)
    values: dict[int, list[float]] = {candidate: [] for candidate in candidates}
    deadline = time.perf_counter() + budget_s
    for world in _sample_hidden_worlds(obs, config, belief_worlds):
        if time.perf_counter() >= deadline:
            break
        try:
            root = search_begin(obs, *world, False)
        except Exception:
            continue
        try:
            for candidate in candidates:
                if time.perf_counter() >= deadline:
                    break
                branch = None
                try:
                    branch = search_step(root.searchId, [candidate])
                    final_obs = rollout_branch(
                        branch,
                        config,
                        model,
                        player_index,
                        start_turn,
                        rollout_steps,
                        deadline,
                    )
                    values[candidate].append(
                        shaped_return(state_value(final_obs, config, perspective_index=player_index), baseline)
                    )
                except Exception:
                    continue
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
    actions = []
    anchor_samples = values.get(anchor_index) or []
    if not anchor_samples:
        return None
    anchor_mean = sum(anchor_samples) / len(anchor_samples)
    anchor_variance = sum((value - anchor_mean) ** 2 for value in anchor_samples) / len(anchor_samples)
    anchor_robust_return = anchor_mean - risk_penalty * math.sqrt(anchor_variance)
    anchor_features = feature_vector(obs, obs.select.option[anchor_index])
    for candidate in candidates:
        samples = values[candidate]
        if not samples:
            continue
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / len(samples)
        std = math.sqrt(variance)
        candidate_features = feature_vector(obs, obs.select.option[candidate])
        robust_return = mean - risk_penalty * std
        actions.append(
            {
                "index": candidate,
                "features": candidate_features,
                "advantage_features": candidate_features + anchor_features + [
                    left - right for left, right in zip(candidate_features, anchor_features)
                ],
                "teacher_score": scores[candidate],
                "samples": samples,
                "mean_return": mean,
                "std_return": std,
                "robust_return": robust_return,
                "advantage_return": robust_return - anchor_robust_return,
            }
        )
    if len(actions) < 2:
        return None
    return {
        "context": context,
        "turn": int(obs.current.turn or 0),
        "player_index": player_index,
        "actual_action": actual_action,
        "anchor_index": anchor_index,
        "option_count": len(obs.select.option),
        "actions": actions,
    }


def process_record(
    path: Path,
    focus_name: str,
    config: dict[str, Any],
    model: dict[str, Any],
    max_states_per_episode: int,
    max_candidates: int,
    belief_worlds: int,
    rollout_steps: int,
    risk_penalty: float,
    budget_s: float,
) -> tuple[int, list[dict[str, Any]]]:
    record = read_json(path)
    game = record.get("game", record)
    game_id = str(game.get("game_id", path.stem))
    winner = game.get("winner")
    reward = 1 if winner == focus_name else -1 if winner else 0
    attempted = 0
    rows = []
    for event in game.get("trace", []):
        if len(rows) >= max_states_per_episode:
            break
        if event.get("event") != "action" or not event.get("training_state") or event.get("actor") != focus_name:
            continue
        observation = event.get("observation")
        action = event.get("action") or []
        if not isinstance(observation, dict) or not isinstance(action, list):
            continue
        attempted += 1
        row = collect_state(
            observation,
            action,
            config,
            model,
            max_candidates,
            belief_worlds,
            rollout_steps,
            risk_penalty,
            budget_s,
        )
        if row is None:
            continue
        row.update(
            {
                "episode_id": game_id,
                "state_id": f"{game_id}:{event.get('step', 0)}",
                "opponent": game.get("p1") if game.get("p0") == focus_name else game.get("p0"),
                "group_id": game.get("p1") if game.get("p0") == focus_name else game.get("p0"),
                "game_reward": reward,
            }
        )
        rows.append(row)
    return attempted, rows


def initialize_worker(
    artifact: dict[str, Any],
    variant: str,
    config_overrides: dict[str, Any],
    focus_name: str,
    max_states_per_episode: int,
    max_candidates: int,
    belief_worlds: int,
    rollout_steps: int,
    risk_penalty: float,
    budget_s: float,
) -> None:
    config = family_config(str(artifact["family"]), variant)
    if config_overrides.get("__replace_config__"):
        config = dict(config_overrides)
        config.pop("__replace_config__", None)
    config["own_deck"] = list(map(int, artifact["deck"]))
    config["opponent_decks"] = artifact.get("opponent_decks", [])
    for key, value in config_overrides.items():
        if key == "search" and isinstance(value, dict):
            config["search"] = {**config.get("search", {}), **value}
        else:
            config[key] = value
    _WORKER_SETTINGS.update(
        {
            "focus_name": focus_name,
            "config": config,
            "model": model_bundle(artifact),
            "max_states_per_episode": max_states_per_episode,
            "max_candidates": max_candidates,
            "belief_worlds": belief_worlds,
            "rollout_steps": rollout_steps,
            "risk_penalty": risk_penalty,
            "budget_s": budget_s,
        }
    )


def process_record_worker(path: str) -> tuple[int, list[dict[str, Any]]]:
    return process_record(Path(path), **_WORKER_SETTINGS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Ogerpon counterfactual state-action returns from training traces.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--focus-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="clone_combat_search")
    parser.add_argument("--config-overrides", type=Path)
    parser.add_argument("--max-states", type=int, default=20_000)
    parser.add_argument("--max-states-per-episode", type=int, default=250)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--belief-worlds", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--risk-penalty", type=float, default=0.20)
    parser.add_argument("--state-budget-s", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    config_overrides = json.loads(args.config_overrides.read_text(encoding="utf-8")) if args.config_overrides else {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = attempted = 0
    paths = list(record_paths(args.records))
    initializer_args = (
        artifact,
        args.variant,
        config_overrides,
        args.focus_name,
        args.max_states_per_episode,
        args.max_candidates,
        args.belief_worlds,
        args.rollout_steps,
        args.risk_penalty,
        args.state_budget_s,
    )
    with args.out.open("w", encoding="utf-8") as output:
        if args.workers <= 1:
            initialize_worker(*initializer_args)
            result_batches = ([process_record_worker(str(path)) for path in paths],)
        else:
            executor = ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_worker,
                initargs=initializer_args,
            )
            batch_size = args.workers * 2
            result_batches = (
                executor.map(process_record_worker, map(str, paths[start:start + batch_size]), chunksize=1)
                for start in range(0, len(paths), batch_size)
            )
        try:
            for results in result_batches:
                for record_attempted, rows in results:
                    attempted += record_attempted
                    for row in rows:
                        if written >= args.max_states:
                            break
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                        written += 1
                    if written >= args.max_states:
                        break
                if written >= args.max_states:
                    break
        finally:
            if args.workers > 1:
                executor.shutdown(wait=True, cancel_futures=True)
    summary = {"attempted_states": attempted, "written_states": written, "output": str(args.out)}
    args.out.with_suffix(args.out.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
