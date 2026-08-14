from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file
from local_eval.evaluator import run_candidate_pool, save_report
from local_eval.models import EvalConfig
from tools.behavior_clone_audit import audit
from tools.build_meta_submission import build
from tools.replay_imitation import _load_cache_file, read_replays, replay_fingerprint, train
from tools.robust_gold_search import rank_report, write_ranking


MODEL_PROFILES = {
    "balanced": {"n_estimators": 1400, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20, "head_n_estimators": 600, "value_n_estimators": 600, "win_weight": 1.0, "loss_weight": 1.0},
    "compact": {"n_estimators": 420, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 18, "win_weight": 1.0, "loss_weight": 1.0},
    "deep": {"n_estimators": 2000, "learning_rate": 0.02, "num_leaves": 255, "min_child_samples": 12, "head_n_estimators": 800, "value_n_estimators": 800, "win_weight": 1.0, "loss_weight": 1.0},
    "winsoft": {"n_estimators": 1600, "learning_rate": 0.025, "num_leaves": 127, "min_child_samples": 16, "head_n_estimators": 700, "value_n_estimators": 700, "win_weight": 1.0, "loss_weight": 1.0},
    "regularized": {"n_estimators": 1800, "learning_rate": 0.022, "num_leaves": 63, "min_child_samples": 28, "head_n_estimators": 750, "value_n_estimators": 700, "reg_lambda": 4.0, "win_weight": 1.0, "loss_weight": 1.0},
    "softmax": {"n_estimators": 1600, "learning_rate": 0.025, "num_leaves": 127, "min_child_samples": 18, "head_n_estimators": 700, "value_n_estimators": 700, "reg_lambda": 2.0, "loss_function": "QuerySoftMax", "win_weight": 1.0, "loss_weight": 1.0},
    "conservative": {"n_estimators": 360, "learning_rate": 0.028, "num_leaves": 15, "min_child_samples": 24, "win_weight": 1.8, "loss_weight": 0.1},
}
VARIANTS = (
    "clone_fidelity",
    "clone_targetfix",
    "clone_endgame",
    "clone_value",
    "clone_search",
    "clone_combat",
    "clone_combat_search",
    "clone_anti_mill",
    "clone_anti_mill_search",
)


def make_manifest(args: argparse.Namespace, profile: str, out: Path, repeat: int = 0) -> Path:
    gpu_devices = str(getattr(args, "gpu_devices", "") or "")
    if not gpu_devices and int(getattr(args, "gpu_device", -1)) >= 0:
        gpu_devices = str(args.gpu_device)
    majkel = str(args.target_name).casefold() == "majkel1337"
    data = {
        "family": "teal_ogerpon",
        "seed": args.seed + 100 * list(MODEL_PROFILES).index(profile) + repeat,
        **MODEL_PROFILES[profile],
        "n_jobs": args.model_jobs,
        "parse_workers": args.parse_workers,
        "task_type": "GPU" if gpu_devices else "CPU",
        "devices": gpu_devices,
        "enable_main_type_heads": True,
        "enable_main_router": True,
        "router_mode": "regime_soft_v2" if majkel else "soft_confidence",
        "router_schema_version": 2 if majkel else 1,
        "router_feature_mode": "compact_v1",
        "stop_no_attack_continue_weight": 1.0,
        "stop_attack_ready_continue_weight": 1.5,
        "stop_ko_ready_continue_weight": 2.0,
        "stop_weight_cap": 2.70,
        "stop_ensemble_size": 3,
        "stop_oof_folds": 5 if majkel else 0,
        "stop_oof_false_stop_multiplier": 1.35,
        "stop_run_ablations": majkel,
        "stop_gate_continue": 0.92 if majkel else 0.90,
        "stop_gate_terminal": 0.75 if majkel else 0.70,
        "stop_gate_balanced": 0.87 if majkel else 0.85,
        "stop_gate_attack_continue": 0.89 if majkel else 0.0,
        "stop_gate_ko_continue": 0.85 if majkel else 0.0,
        "stop_gate_turn_8_continue": 0.85 if majkel else 0.0,
        "stop_require_test_gate": majkel,
        "split_seed": int(getattr(args, "split_seed", 20260813)) if majkel else args.seed + 100 * list(MODEL_PROFILES).index(profile),
        "frozen_dataset": str(getattr(args, "frozen_dataset", "") or "") if majkel else "",
        "cache_dir": str(args.out / "cache"),
        "minimum_head_decisions": args.minimum_head_decisions,
        "confidence_precision": args.confidence_precision,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "sources": [{"name": args.target_name, "path": str(args.replays), "target_name": args.target_name, "split": "train"}],
    }
    label = profile if int(getattr(args, "seed_repeats", 1)) == 1 else f"{profile}_s{repeat}"
    path = out / "manifests" / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def train_profile(profile: str, manifest: Path, out: Path) -> Path:
    artifact_path = out / "artifacts" / f"{profile}.json"
    train(manifest, artifact_path)
    return artifact_path


_AUDIT_REPLAYS: list[dict[str, Any]] | None = None


def _run_audit(
    package: Path,
    replay_dir: Path,
    target_name: str,
    episode_ids: set[str] | None,
) -> tuple[Path, dict[str, Any]]:
    return package, audit(package, replay_dir, target_name, episode_ids, _AUDIT_REPLAYS)


def build_candidates(artifacts: list[Path], args: argparse.Namespace, out: Path) -> list[Path]:
    jobs: list[tuple[Path, Path, Path, Path, str]] = []
    for artifact in artifacts:
        profile = artifact.stem
        for variant in args.variants:
            package = out / "candidates" / f"ogerpon_{profile}_{variant}.tar.gz"
            jobs.append((artifact, package, Path(args.runtime), Path(args.cg_dir), variant))
    with ThreadPoolExecutor(max_workers=min(8, len(jobs) or 1)) as executor:
        return list(executor.map(lambda job: build(*job, linux_only=True), jobs))


def audit_candidates(candidates: list[Path], artifact_by_profile: dict[str, Path], args: argparse.Namespace, out: Path) -> list[Path]:
    global _AUDIT_REPLAYS
    selected: list[Path] = []
    rows: list[dict[str, Any]] = []
    audit_dir = out / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        (out / "audit_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return selected
    cache_dir = args.out / "cache"
    replay_cache = cache_dir / f"replays-{replay_fingerprint(args.replays)[:16]}.pickle.gz"
    if replay_cache.exists():
        _AUDIT_REPLAYS = _load_cache_file(replay_cache)
        print(f"[audit] replay cache hit {replay_cache.name}", flush=True)
    else:
        _AUDIT_REPLAYS = read_replays(args.replays, workers=args.parse_workers)
    tasks: list[tuple[Path, Path, str, set[str] | None]] = []
    for package in candidates:
        profile = package.name.removeprefix("ogerpon_").split("_clone", 1)[0]
        artifact = json.loads(artifact_by_profile[profile].read_text(encoding="utf-8"))
        test_episode_ids = {str(value).split(":")[-1] for value in artifact.get("split_stats", {}).get("test_episodes", [])}
        tasks.append((package, args.replays, args.target_name, test_episode_ids))
    results: dict[str, dict[str, Any]] = {}
    if len(tasks) > 1 and args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(len(tasks), args.workers)) as executor:
            futures = {executor.submit(_run_audit, *task): index for index, task in enumerate(tasks)}
            for number, future in enumerate(as_completed(futures), 1):
                package, result = future.result()
                results[str(package)] = result
                if number % 4 == 0 or number == len(futures):
                    print(f"[audit] {number}/{len(tasks)}", flush=True)
    else:
        for number, task in enumerate(tasks, 1):
            package, result = _run_audit(*task)
            results[str(package)] = result
            print(f"[audit] {number}/{len(tasks)}", flush=True)
    for package in candidates:
        result = results[str(package)]
        (audit_dir / f"{package.stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        ability_rate = float(result.get("by_group", {}).get("ability", {}).get("semantic_rate", 0.0))
        passed = float(result.get("semantic_rate", 0.0)) >= args.min_semantic_rate and ability_rate >= args.min_ability_rate
        rows.append({"package": str(package), "semantic_rate": result.get("semantic_rate", 0.0), "ability_rate": ability_rate, "passed": passed})
        if passed:
            selected.append(package)
    (out / "audit_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return selected


def tournament(
    label: str,
    candidates: list[Path],
    opponents: list[Path],
    games: int,
    profile: str,
    args: argparse.Namespace,
    out: Path,
) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    if not candidates or not opponents:
        return []
    config = EvalConfig(
        profile=profile,
        workers=args.workers,
        seed=args.seed + args.seed_offsets[label],
        record_mode="none",
        progress=True,
        progress_label=label,
        progress_mode="line",
        archive_cache_dir=str(out / "cache"),
        max_in_flight=args.workers * 2,
    )
    report = run_candidate_pool(candidates, opponents, games, config, Path.cwd(), peer_span=0)
    save_report(report, out)
    rows = rank_report(report, candidates, opponents)
    write_ranking(out / "ranking.json", rows)
    return [asdict(row) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe keidroid Ogerpon front-100 local search.")
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--target-name", default="keidroid")
    parser.add_argument("--train-opponents", nargs="+", type=Path, required=True)
    parser.add_argument("--holdout-opponents", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("agents/meta_runtime.py"))
    parser.add_argument("--cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--model-jobs", type=int, default=12)
    parser.add_argument("--parse-workers", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=-1)
    parser.add_argument("--gpu-devices", default="")
    parser.add_argument("--train-parallel", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--split-seed", type=int, default=20260813)
    parser.add_argument("--frozen-dataset", type=Path)
    parser.add_argument("--stage1-games", type=int, default=4)
    parser.add_argument("--stage2-games", type=int, default=16)
    parser.add_argument("--holdout-games", type=int, default=48)
    parser.add_argument("--stage2-count", type=int, default=8)
    parser.add_argument("--holdout-count", type=int, default=3)
    parser.add_argument("--min-semantic-rate", type=float, default=0.85)
    parser.add_argument("--min-ability-rate", type=float, default=0.90)
    parser.add_argument("--min-validation-top1", type=float, default=0.70)
    parser.add_argument("--min-validation-top3", type=float, default=0.92)
    parser.add_argument("--min-main-top1", type=float, default=0.62)
    parser.add_argument("--min-card-top1", type=float, default=0.66)
    parser.add_argument("--max-train-validation-gap", type=float, default=0.08)
    parser.add_argument("--minimum-head-decisions", type=int, default=30)
    parser.add_argument("--confidence-precision", type=float, default=0.80)
    parser.add_argument("--online-budget", type=int, default=8)
    parser.add_argument("--profiles", nargs="+", choices=tuple(MODEL_PROFILES), default=list(MODEL_PROFILES))
    parser.add_argument("--seed-repeats", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args(argv)
    args.seed_offsets = {"stage1": 0, "stage2": 100_000, "holdout": 200_000}
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (profile if args.seed_repeats == 1 else f"{profile}_s{repeat}", make_manifest(args, profile, args.out, repeat))
        for profile in args.profiles
        for repeat in range(max(1, args.seed_repeats))
    ]
    artifacts_by_profile = {}
    training_failures = {}
    with ThreadPoolExecutor(max_workers=min(len(jobs), args.train_parallel)) as executor:
        futures = {
            executor.submit(train_profile, profile, manifest, args.out): profile
            for profile, manifest in jobs
        }
        for future in as_completed(futures):
            profile = futures[future]
            try:
                artifacts_by_profile[profile] = future.result()
            except Exception as exc:
                training_failures[profile] = f"{type(exc).__name__}: {exc}"
                print(f"[train] profile failed: {profile}: {training_failures[profile]}", flush=True)
                traceback.print_exc()
    profiles = [label for label, _ in jobs if label in artifacts_by_profile]
    artifacts = [artifacts_by_profile[profile] for profile in profiles]
    artifact_data = [json.loads(path.read_text(encoding="utf-8")) for path in artifacts]
    rejected = {profile: [reason] for profile, reason in training_failures.items()}
    for profile, row in zip(profiles, artifact_data):
        train_top1 = float(row.get("train_metrics", {}).get("top1", 0.0))
        validation = row.get("runtime_policy_metrics", {}).get("validation", row.get("validation_metrics", {}))
        validation_top1 = float(validation.get("top1", 0.0))
        validation_top3 = float(validation.get("top3", 0.0))
        main_top1 = float(row.get("hierarchical_main_metrics", {}).get("validation", {}).get("top1", 0.0))
        card_top1 = float(row.get("head_metrics", {}).get("card", {}).get("validation", {}).get("top1", 0.0))
        reasons = []
        if validation_top1 < args.min_validation_top1:
            reasons.append(f"validation_top1={validation_top1:.3f}")
        if validation_top3 < args.min_validation_top3:
            reasons.append(f"validation_top3={validation_top3:.3f}")
        if main_top1 < args.min_main_top1:
            reasons.append(f"main_top1={main_top1:.3f}")
        if card_top1 < args.min_card_top1:
            reasons.append(f"card_top1={card_top1:.3f}")
        if train_top1 - validation_top1 > args.max_train_validation_gap:
            reasons.append(f"train_validation_gap={train_top1 - validation_top1:.3f}")
        if reasons:
            rejected[profile] = reasons
    qualified = [
        (profile, artifact, row)
        for profile, artifact, row in zip(profiles, artifacts, artifact_data)
        if profile not in rejected
    ]
    (args.out / "imitation_gate.json").write_text(json.dumps({
        "qualified": [profile for profile, _, _ in qualified],
        "rejected": rejected,
    }, indent=2), encoding="utf-8")
    if not qualified:
        raise RuntimeError(f"all CatBoost profiles failed imitation gates: {rejected}")
    profiles = [profile for profile, _, _ in qualified]
    artifacts = [artifact for _, artifact, _ in qualified]
    artifact_data = [row for _, _, row in qualified]
    deck_hashes = {row["deck_sha256"] for row in artifact_data}
    if len(deck_hashes) != 1:
        raise RuntimeError(f"keidroid deck drift detected across artifacts: {sorted(deck_hashes)}")
    candidates = build_candidates(artifacts, args, args.out)
    selected = audit_candidates(candidates, dict(zip(profiles, artifacts)), args, args.out)
    if not selected:
        raise SystemExit("No candidate passed the independent behavior audit")
    stage1 = tournament("stage1", selected, args.train_opponents, args.stage1_games, "legacy", args, args.out / "stage1")
    stage1_paths = [Path(row["tarball"]) for row in stage1[: args.stage2_count]]
    stage2 = tournament("stage2", stage1_paths, args.train_opponents, args.stage2_games, "legacy", args, args.out / "stage2")
    stage2_paths = [Path(row["tarball"]) for row in stage2[: args.holdout_count]]
    holdout = tournament("holdout", stage2_paths, args.holdout_opponents, args.holdout_games, "kaggle", args, args.out / "holdout")
    submission_queue = [
        {
            "priority": index + 1,
            "name": row["name"],
            "package": row["tarball"],
            "package_sha256": sha256_file(row["tarball"]),
            "hypothesis": "fixed keidroid deck; leakage-safe multi-head/value/search challenger",
            "requires_explicit_kaggle_execute": True,
        }
        for index, row in enumerate(holdout[: max(0, args.online_budget)])
    ]
    (args.out / "submission_queue.json").write_text(json.dumps(submission_queue, indent=2), encoding="utf-8")
    result = {
        "candidate_count": len(candidates),
        "audit_pass_count": len(selected),
        "stage1": stage1,
        "stage2": stage2,
        "holdout": holdout,
        "online_submission_required": True,
        "online_submission_performed": False,
        "deck_is_fixed": True,
        "deck_sha256": next(iter(deck_hashes)),
        "submission_queue": submission_queue,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
