from __future__ import annotations

import csv
import gzip
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .archive import sha256_file
from .models import AgentStats, EvalConfig, GameResult, MatchReport, SubmissionInfo
from .rating import KaggleStyleRating
from .referee import play_game


def submission_name(path: str | Path) -> str:
    p = Path(path)
    return p.name.removesuffix(".tar.gz")


def build_submission_info(paths: list[str | Path]) -> dict[str, SubmissionInfo]:
    infos: dict[str, SubmissionInfo] = {}
    for path in paths:
        p = Path(path).resolve()
        name = submission_name(p)
        base = name
        suffix = 2
        while name in infos:
            name = f"{base}_{suffix}"
            suffix += 1
        infos[name] = SubmissionInfo(name=name, tarball=str(p), sha256=sha256_file(p))
    return infos


def run_match(
    tarball_a: str | Path,
    tarball_b: str | Path,
    games: int,
    config: EvalConfig,
    project_root: str | Path,
) -> MatchReport:
    return run_ladder([tarball_a, tarball_b], games_per_pair=games, config=config, project_root=project_root)


def run_ladder(
    tarballs: list[str | Path],
    games_per_pair: int,
    config: EvalConfig,
    project_root: str | Path,
) -> MatchReport:
    infos = build_submission_info(tarballs)
    names = list(infos)
    rating = KaggleStyleRating(config)
    ratings = {name: rating.new_rating() for name in names}
    stats = {
        name: AgentStats(
            name=name,
            tarball=infos[name].tarball,
            sha256=infos[name].sha256,
            mu=config.trueskill_mu,
            sigma=config.trueskill_sigma,
            kaggle_score_estimate=config.trueskill_mu - 3.0 * config.trueskill_sigma,
        )
        for name in names
    }

    game_results: list[GameResult] = []
    jobs: list[tuple[int, str, str, int, str]] = []
    game_no = 0
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            for local_idx in range(games_per_pair):
                swap = local_idx % 2 == 1
                p0_name, p1_name = (name_b, name_a) if swap else (name_a, name_b)
                seed = config.seed + game_no
                game_id = f"g{game_no:06d}"
                jobs.append((game_no, p0_name, p1_name, seed, game_id))
                game_no += 1
    _progress(
        config,
        "start",
        done=0,
        total=len(jobs),
        extra=f"agents={len(names)} games_per_pair={games_per_pair} workers={config.workers}",
        force=True,
    )

    def _run_job(job: tuple[int, str, str, int, str]) -> tuple[int, GameResult]:
        idx, p0_name, p1_name, seed, game_id = job
        result = play_game(
            infos[p0_name].tarball,
            infos[p1_name].tarball,
            p0_name,
            p1_name,
            seed,
            game_id,
            config,
            project_root,
        )
        return idx, result

    if config.workers > 1 and len(jobs) > 1:
        results_by_idx: dict[int, GameResult] = {}
        completed = 0
        last_progress = 0.0
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                idx, result = future.result()
                results_by_idx[idx] = result
                completed += 1
                now = time.monotonic()
                if completed == len(jobs) or now - last_progress >= max(0.25, config.progress_interval_s):
                    last_progress = now
                    _progress(
                        config,
                        "games",
                        done=completed,
                        total=len(jobs),
                        extra=_progress_result_summary(result),
                        force=completed == len(jobs),
                    )
        ordered_results = [results_by_idx[idx] for idx in range(len(jobs))]
    else:
        ordered_results = []
        last_progress = 0.0
        for job in jobs:
            result = _run_job(job)[1]
            ordered_results.append(result)
            now = time.monotonic()
            if len(ordered_results) == len(jobs) or now - last_progress >= max(0.25, config.progress_interval_s):
                last_progress = now
                _progress(
                    config,
                    "games",
                    done=len(ordered_results),
                    total=len(jobs),
                    extra=_progress_result_summary(result),
                    force=len(ordered_results) == len(jobs),
                )

    for result in ordered_results:
        game_results.append(result)
        p0_name, p1_name = result.p0, result.p1
        ratings[p0_name], ratings[p1_name] = rating.update_game(ratings[p0_name], ratings[p1_name], result)
        _apply_stats(stats, result)
        for name in (p0_name, p1_name):
            stats[name].mu = ratings[name].mu
            stats[name].sigma = ratings[name].sigma
            stats[name].kaggle_score_estimate = ratings[name].kaggle_score_estimate

    standings = sorted(
        stats.values(),
        key=lambda s: (s.kaggle_score_estimate, s.mu, s.wins - s.losses, -s.losses),
        reverse=True,
    )
    return MatchReport(
        config=config,
        submissions=list(infos.values()),
        games=game_results,
        standings=standings,
        metadata=_metadata(project_root),
    )


def _progress(config: EvalConfig, phase: str, done: int, total: int, extra: str = "", force: bool = False) -> None:
    if not config.progress:
        return
    total = max(0, total)
    ratio = (done / total) if total else 1.0
    width = 28
    filled = min(width, max(0, int(ratio * width)))
    bar = "#" * filled + "-" * (width - filled)
    label = config.progress_label or "ladder"
    text = f"[{label}] {phase} [{bar}] {done}/{total} {ratio * 100:5.1f}%"
    if extra:
        text += f" | {extra}"
    mode = (config.progress_mode or "auto").lower()
    if mode == "file":
        _write_progress_file(config.progress_file, text)
        return
    overwrite = mode == "overwrite" or (mode == "auto" and sys.stdout.isatty())
    if overwrite:
        end = "\n" if force else ""
        print("\r\033[K" + text, end=end, flush=True)
    else:
        print(text, flush=True)


def _write_progress_file(path: str, text: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(target)


def _progress_result_summary(result: GameResult) -> str:
    return (
        f"last={result.game_id} {result.p0} vs {result.p1} "
        f"{result.outcome}/{result.reason} actions={result.actions}"
    )


def _metadata(project_root: str | Path) -> dict[str, object]:
    return {
        "local_eval_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(Path(project_root).resolve()),
        "score_formula": "mu - 3*sigma",
        "scoring_note": (
            "This evaluator runs official cg battles locally and applies a "
            "TrueSkill-style Kaggle score estimate. Kaggle private backend "
            "matchmaking and rating parameters are not guaranteed to match."
        ),
    }


def _apply_stats(stats: dict[str, AgentStats], result: GameResult) -> None:
    p0 = stats[result.p0]
    p1 = stats[result.p1]
    p0.games += 1
    p1.games += 1
    _ensure_opp(p0, result.p1)
    _ensure_opp(p1, result.p0)

    if result.outcome == "DRAW":
        p0.draws += 1
        p1.draws += 1
        p0.opponents[result.p1]["draws"] += 1
        p1.opponents[result.p0]["draws"] += 1
    elif result.winner and result.loser:
        winner = stats[result.winner]
        loser = stats[result.loser]
        winner.wins += 1
        loser.losses += 1
        winner.opponents[result.loser]["wins"] += 1
        loser.opponents[result.winner]["losses"] += 1
        if result.reason == "TIMEOUT":
            loser.timeouts += 1
        elif result.reason == "INVALID_ACTION" or result.reason == "ENGINE_REJECTED_ACTION":
            loser.invalids += 1
        elif result.reason != "RESULT":
            loser.crashes += 1
    else:
        p0.no_results += 1
        p1.no_results += 1


def _ensure_opp(stats: AgentStats, opponent: str) -> None:
    stats.opponents.setdefault(opponent, {"wins": 0, "losses": 0, "draws": 0})


def save_report(report: MatchReport, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_dir = out / "game_records"
    tmp_records_dir = out / f".game_records.{os.getpid()}.tmp"
    if tmp_records_dir.exists():
        shutil.rmtree(tmp_records_dir)
    tmp_records_dir.mkdir(parents=True)
    report_tmp = _tmp_path(out / "report.json")
    games_tmp = _tmp_path(out / "games.jsonl")
    summary_tmp = _tmp_path(out / "summary.csv")
    try:
        report_tmp.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        with games_tmp.open("w", encoding="utf-8") as f:
            for game in report.games:
                f.write(json.dumps(game.to_dict(), ensure_ascii=False) + "\n")
                record = {
                    "metadata": report.metadata,
                    "config": asdict(report.config),
                    "game": game.to_dict(include_trace=True),
                }
                if game.trace:
                    payload = json.dumps(record, ensure_ascii=False, indent=2)
                    if report.config.record_gzip:
                        with gzip.open(tmp_records_dir / f"{game.game_id}.json.gz", "wt", encoding="utf-8") as rf:
                            rf.write(payload)
                    else:
                        (tmp_records_dir / f"{game.game_id}.json").write_text(payload, encoding="utf-8")

        with summary_tmp.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "name",
                "kaggle_score_estimate",
                "mu",
                "sigma",
                "games",
                "wins",
                "losses",
                "draws",
                "crashes",
                "timeouts",
                "invalids",
                "no_results",
                "tarball",
                "sha256",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in report.standings:
                data = asdict(row)
                writer.writerow({field: data[field] for field in fields})

        games_tmp.replace(out / "games.jsonl")
        summary_tmp.replace(out / "summary.csv")
        if records_dir.exists():
            shutil.rmtree(records_dir)
        tmp_records_dir.replace(records_dir)
        report_tmp.replace(out / "report.json")
    except Exception:
        for tmp in (report_tmp, games_tmp, summary_tmp):
            tmp.unlink(missing_ok=True)
        shutil.rmtree(tmp_records_dir, ignore_errors=True)
        raise
    _write_matchup_matrix(report, out / "matchup_matrix.csv")
    _write_agent_intervals(report, out / "agent_intervals.csv")


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _write_matchup_matrix(report: MatchReport, path: Path) -> None:
    names = [s.name for s in report.standings]
    by_name = {s.name: s for s in report.standings}
    tmp = _tmp_path(path)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", *names])
        for name in names:
            row = [name]
            for opp in names:
                if opp == name:
                    row.append("")
                    continue
                rec = by_name[name].opponents.get(opp, {"wins": 0, "losses": 0, "draws": 0})
                total = rec["wins"] + rec["losses"] + rec["draws"]
                score = (rec["wins"] + 0.5 * rec["draws"]) / total if total else 0.0
                row.append(f"{score:.3f} ({rec['wins']}-{rec['losses']}-{rec['draws']})")
            writer.writerow(row)
    tmp.replace(path)


def _write_agent_intervals(report: MatchReport, path: Path) -> None:
    tmp = _tmp_path(path)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "games", "score_rate", "wilson_low", "wilson_high"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for stats in report.standings:
            decisive = stats.wins + stats.losses + stats.draws
            score = stats.wins + 0.5 * stats.draws
            rate, low, high = _wilson(score, decisive)
            writer.writerow(
                {
                    "name": stats.name,
                    "games": decisive,
                    "score_rate": f"{rate:.6f}",
                    "wilson_low": f"{low:.6f}",
                    "wilson_high": f"{high:.6f}",
                }
            )
    tmp.replace(path)


def _wilson(score: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    phat = score / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * ((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) ** 0.5
    return phat, max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)
