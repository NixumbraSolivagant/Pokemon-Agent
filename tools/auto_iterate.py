from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file
from local_eval.evaluator import run_ladder, save_report
from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.build_models import DEFAULT_BASE, BuildConfig
from tools.build_submission import build_submission
from tools.export_kaggle_submission import export_kaggle_submission
from tools.prior_deck_space import great_tusk_prior_configs, metal_prior_configs, portfolio_seed_configs
from tools.reference_pool import DEFAULT_REFERENCE_PATHS


DEFAULT_POOL = DEFAULT_REFERENCE_PATHS

def config_to_dict(cfg: BuildConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["base"] = str(cfg.base)
    data["out"] = str(cfg.out)
    return data


def config_from_dict(data: dict[str, Any]) -> BuildConfig:
    deck_override = data.get("deck_override")
    if deck_override is not None:
        deck_override = [int(card_id) for card_id in deck_override]
    strategy_weights = {
        str(key): float(value)
        for key, value in dict(data.get("strategy_weights", {})).items()
    }
    return BuildConfig(
        name=str(data.get("name", "champion")),
        family=str(data.get("family", "great_tusk")),
        base=Path(data.get("base", DEFAULT_BASE)),
        out=Path(data.get("out", "outputs/submissions/champion_latest.tar.gz")),
        enable_search=bool(data.get("enable_search", True)),
        injection=str(data.get("injection", "great_tusk")),
        search_candidates=int(data.get("search_candidates", 8)),
        search_budget_s=float(data.get("search_budget_s", 0.25)),
        search_margin=float(data.get("search_margin", 1200.0)),
        search_rollout_steps=int(data.get("search_rollout_steps", 16)),
        deck_swaps=[tuple(map(int, pair)) for pair in data.get("deck_swaps", [])],
        deck_override=deck_override,
        deck_files=tuple(data.get("deck_files", ("deck.csv",))),
        strategy_weights=strategy_weights,
        policy_variant=str(data.get("policy_variant", "default")),
        opponent_model=str(data.get("opponent_model", "perfect")),
        origin=str(data.get("origin", "")),
        notes=str(data.get("notes", "")),
    )


def initial_champion_config(base: Path) -> BuildConfig:
    return BuildConfig(
        name="incumbent_ultraball",
        family="great_tusk",
        base=base,
        out=Path("outputs/submissions/champion_latest.tar.gz"),
        search_candidates=8,
        search_budget_s=0.25,
        search_margin=1200.0,
        deck_swaps=[(1121, 1123)],
        notes="Initial champion from previous top-pool screen: +1 Ultra Ball, -1 Switch.",
    )


def load_state(path: Path, base: Path) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state["incumbent"] = config_to_dict(config_from_dict(state["incumbent"]))
        return state
    cfg = initial_champion_config(base)
    return {
        "generation": 0,
        "incumbent": config_to_dict(cfg),
        "history": [],
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def variant_configs(incumbent: BuildConfig, generation: int, out_dir: Path, limit: int, include_portfolio: bool) -> list[BuildConfig]:
    gt_limit = max(1, limit - (12 if include_portfolio else 0))
    variants = great_tusk_prior_configs(incumbent, generation, out_dir, gt_limit)
    if include_portfolio:
        remaining = max(0, limit - len(variants))
        metal_limit = max(0, remaining - 4)
        variants.extend(metal_prior_configs(generation, out_dir, metal_limit))
        variants.extend(portfolio_seed_configs(generation, out_dir)[: max(0, limit - len(variants))])
    return variants[:limit]


def build_candidates(configs: list[BuildConfig]) -> list[Path]:
    built: list[Path] = []
    for cfg in configs:
        try:
            built.append(build_submission(cfg))
        except Exception as exc:
            print(json.dumps({"skipped": cfg.name, "reason": f"{type(exc).__name__}: {exc}"}))
    return built


def stats_by_name(report: MatchReport) -> dict[str, AgentStats]:
    return {s.name: s for s in report.standings}


def score_delta(report: MatchReport, candidate: str, incumbent: str) -> tuple[float, int, int]:
    stats = stats_by_name(report)
    cand = stats[candidate]
    inc = stats[incumbent]
    rec = cand.opponents.get(incumbent, {"wins": 0, "losses": 0, "draws": 0})
    return cand.kaggle_score_estimate - inc.kaggle_score_estimate, rec["wins"], rec["losses"]


def choose_best_candidate(report: MatchReport, candidate_names: set[str], incumbent_name: str) -> AgentStats | None:
    candidates = [s for s in report.standings if s.name in candidate_names and s.name != incumbent_name]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (s.kaggle_score_estimate, s.wins - s.losses, s.wins))


def choose_top_candidates(report: MatchReport, candidate_names: set[str], incumbent_name: str, limit: int) -> list[AgentStats]:
    candidates = [s for s in report.standings if s.name in candidate_names and s.name != incumbent_name]
    candidates.sort(key=lambda s: (s.kaggle_score_estimate, s.wins - s.losses, s.wins), reverse=True)
    return candidates[: max(1, limit)]


def write_generation_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["generation", "stage", "candidate", "promoted", "score_delta", "h2h_wins", "h2h_losses", "tarball"]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def hall_of_fame_paths(state: dict[str, Any], limit: int) -> list[Path]:
    paths: list[Path] = []
    for row in reversed(state.get("history", [])):
        if not row.get("promoted"):
            continue
        tarball = Path(str(row.get("tarball", "")))
        if tarball.exists() and tarball not in paths:
            paths.append(tarball)
        if len(paths) >= limit:
            break
    return list(reversed(paths))


def unique_tarballs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen_sha: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            digest = sha256_file(path)
        except Exception:
            continue
        if digest in seen_sha:
            continue
        seen_sha.add(digest)
        out.append(path)
    return out


def evaluation_tarball_map(paths: list[Path], out_dir: Path, kaggle_safe: bool) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not kaggle_safe:
        return {path.name.removesuffix(".tar.gz"): path for path in paths if path.exists()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            continue
        target = out_dir / path.name
        export_kaggle_submission(
            path,
            target,
            strip_search_wrapper=False,
            exclude_internal_files=False,
        )
        out[path.name.removesuffix(".tar.gz")] = target
    return out


def run_generation(
    state: dict[str, Any],
    args: argparse.Namespace,
    generation: int,
) -> dict[str, Any]:
    run_dir = args.out / f"generation_{generation:03d}"
    variants_dir = run_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    incumbent_cfg = config_from_dict(state["incumbent"])
    incumbent_cfg = replace(
        incumbent_cfg,
        name=f"g{generation:03d}_incumbent",
        out=variants_dir / f"g{generation:03d}_incumbent.tar.gz",
    )
    configs = variant_configs(incumbent_cfg, generation, variants_dir, args.population, args.include_portfolio_seeds)
    built = build_candidates(configs)
    internal_by_name = {path.name.removesuffix(".tar.gz"): path for path in built}
    eval_built_by_name = evaluation_tarball_map(built, run_dir / "eval_variants", args.evaluate_kaggle_safe)
    name_to_config = {cfg.name: cfg for cfg in configs}
    built_names = {path.name.removesuffix(".tar.gz") for path in built}
    incumbent_name = incumbent_cfg.name
    promotable_names = {
        cfg.name
        for cfg in configs
        if cfg.name in built_names and cfg.name != incumbent_name and (args.allow_cross_archetype_promotion or cfg.family == "great_tusk")
    }

    pool = unique_tarballs([*args.pool, *hall_of_fame_paths(state, args.hof_size)])
    eval_pool_by_name = evaluation_tarball_map(pool, run_dir / "eval_pool", args.evaluate_kaggle_safe)
    stage1_cfg = EvalConfig(
        seed=args.seed + generation * 1000,
        workers=args.workers,
        max_actions=args.max_actions,
        run_timeout_s=args.run_timeout,
        record_mode="none",
        record_gzip=True,
    )
    stage1_report = run_ladder(
        unique_tarballs([*eval_built_by_name.values(), *eval_pool_by_name.values()]),
        args.stage1_games,
        stage1_cfg,
        Path.cwd(),
    )
    save_report(stage1_report, run_dir / "stage1")
    finalists = choose_top_candidates(stage1_report, promotable_names, incumbent_name, args.finalists)
    best_stage1 = finalists[0] if finalists else None

    promoted = False
    row: dict[str, Any] = {
        "generation": generation,
        "stage": "stage1",
        "candidate": best_stage1.name if best_stage1 else "",
        "promoted": False,
        "score_delta": "",
        "h2h_wins": "",
        "h2h_losses": "",
        "tarball": "",
    }
    if best_stage1 is None:
        write_generation_summary(args.out / "generations.csv", [row])
        return state

    finalist_names = [s.name for s in finalists]
    finalist_paths = [internal_by_name[name] for name in finalist_names if name in internal_by_name]
    finalist_eval_paths = [eval_built_by_name[name] for name in finalist_names if name in eval_built_by_name]
    incumbent_eval_path = eval_built_by_name[incumbent_name]
    final_pool = unique_tarballs([incumbent_eval_path, *finalist_eval_paths, *eval_pool_by_name.values()])
    stage2_cfg = EvalConfig(
        seed=args.seed + generation * 1000 + 501,
        workers=args.workers,
        max_actions=args.max_actions,
        run_timeout_s=args.run_timeout,
        record_mode=args.record_mode,
        record_focus=best_stage1.name,
        record_sample_rate=args.record_sample_rate,
        record_gzip=True,
    )
    stage2_report = run_ladder(final_pool, args.stage2_games, stage2_cfg, Path.cwd())
    save_report(stage2_report, run_dir / "stage2")
    best_stage1 = choose_best_candidate(stage2_report, {p.name.removesuffix(".tar.gz") for p in finalist_eval_paths}, incumbent_name)
    if best_stage1 is None:
        write_generation_summary(args.out / "generations.csv", [row])
        return state
    delta, h2h_wins, h2h_losses = score_delta(stage2_report, best_stage1.name, incumbent_name)
    best_stats = stats_by_name(stage2_report)[best_stage1.name]

    no_result_rate = best_stats.no_results / max(1, best_stats.games)
    clean = (
        best_stats.crashes == 0
        and best_stats.timeouts == 0
        and best_stats.invalids == 0
        and no_result_rate <= args.max_no_result_rate
    )
    if args.promotion_policy == "strict":
        promoted = clean and delta >= args.min_score_delta and h2h_wins > h2h_losses
    elif args.promotion_policy == "score":
        promoted = clean and delta >= args.min_score_delta
    else:
        promoted = clean and delta >= args.min_score_delta and (
            h2h_wins >= h2h_losses or delta >= args.large_score_delta
        )
    if promoted:
        args.promote.parent.mkdir(parents=True, exist_ok=True)
        promoted_internal = internal_by_name.get(best_stage1.name, Path(best_stage1.tarball))
        shutil.copyfile(promoted_internal, args.promote)
        export_kaggle_submission(
            args.promote,
            args.submission_out,
            strip_search_wrapper=args.strip_search_wrapper_for_submission,
        )
        state["incumbent"] = config_to_dict(name_to_config[best_stage1.name])
        state["incumbent"]["name"] = "incumbent"
        state["incumbent"]["out"] = str(args.promote)

    row = {
        "generation": generation,
        "stage": "stage2",
        "candidate": best_stage1.name,
        "promoted": promoted,
        "score_delta": f"{delta:.6f}",
        "h2h_wins": h2h_wins,
        "h2h_losses": h2h_losses,
        "tarball": str(internal_by_name.get(best_stage1.name, Path(best_stage1.tarball))),
    }
    state["generation"] = generation
    state.setdefault("history", []).append(row)
    write_generation_summary(args.out / "generations.csv", [row])
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automatic low-resource champion iteration loop.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=Path("outputs/auto_iterate"))
    parser.add_argument("--promote", type=Path, default=Path("outputs/submissions/champion_latest.tar.gz"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submissions/submission.tar.gz"))
    parser.add_argument("--strip-search-wrapper-for-submission", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--evaluate-kaggle-safe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--stage1-games", type=int, default=2)
    parser.add_argument("--stage2-games", type=int, default=6)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--min-score-delta", type=float, default=8.0)
    parser.add_argument("--large-score-delta", type=float, default=40.0)
    parser.add_argument("--max-no-result-rate", type=float, default=0.01)
    parser.add_argument("--promotion-policy", choices=["balanced", "strict", "score"], default="balanced")
    parser.add_argument("--include-portfolio-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cross-archetype-promotion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-mode", choices=["none", "losses", "sample", "all"], default="losses")
    parser.add_argument("--record-sample-rate", type=float, default=0.03)
    parser.add_argument("--pool", nargs="*", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--hof-size", type=int, default=4)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "state.json"
    state = load_state(state_path, args.base)
    if not args.promote.exists():
        build_submission(replace(config_from_dict(state["incumbent"]), out=args.promote, name="incumbent"))
    export_kaggle_submission(
        args.promote,
        args.submission_out,
        strip_search_wrapper=args.strip_search_wrapper_for_submission,
    )

    start_generation = int(state.get("generation", 0)) + 1
    for generation in range(start_generation, start_generation + args.generations):
        state = run_generation(state, args, generation)
        save_state(state_path, state)
        print(json.dumps({"generation": generation, "incumbent": state["incumbent"], "history_tail": state["history"][-1:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
