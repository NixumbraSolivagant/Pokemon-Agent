from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_eval.evaluator import run_candidate_pool, save_report
from local_eval.models import EvalConfig
from tools.kaggle_gold_loop import KaggleCli


CALIBRATION_VERSION = "kaggle-eval-calibration-v1"
DEFAULT_CONFIG = Path("configs/kaggle_eval_calibration_private.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_config(data, path.parent)
    return data


def validate_config(data: dict[str, Any], base_dir: Path = Path.cwd()) -> None:
    excluded = {int(value) for value in data.get("excluded_submission_ids", [])}
    seen: set[int] = set()
    for row in data.get("submissions", []):
        submission_id = int(row["submission_id"])
        if submission_id in seen and submission_id not in excluded:
            raise ValueError(f"duplicate submission_id: {submission_id}")
        seen.add(submission_id)
        package = _resolve(base_dir, row["package"])
        if package.is_file():
            actual = sha256_file(package)
            if actual != row["package_sha256"]:
                raise ValueError(f"package SHA-256 mismatch for submission {submission_id}: {actual}")
    opponent_names = [str(row["name"]) for row in data.get("opponents", [])]
    if len(opponent_names) != len(set(opponent_names)):
        raise ValueError("opponent names must be unique")


def pava_fit(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    ordered = sorted(zip(xs, ys), key=lambda pair: pair[0])
    blocks: list[dict[str, float]] = []
    for x_value, y_value in ordered:
        blocks.append({"x0": x_value, "x1": x_value, "sum": y_value, "count": 1.0})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left["x1"] != right["x0"] and left["sum"] / left["count"] <= right["sum"] / right["count"]:
                break
            merged = {
                "x0": left["x0"],
                "x1": right["x1"],
                "sum": left["sum"] + right["sum"],
                "count": left["count"] + right["count"],
            }
            blocks[-2:] = [merged]
    return [(block["x1"], block["sum"] / block["count"]) for block in blocks]


def isotonic_predict(model: list[tuple[float, float]], value: float) -> float:
    if not model:
        raise ValueError("empty isotonic model")
    for upper_x, prediction in model:
        if value <= upper_x:
            return prediction
    return model[-1][1]


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    x_ranks = _average_ranks(xs)
    y_ranks = _average_ranks(ys)
    return _pearson(x_ranks, y_ranks)


def pairwise_order_accuracy(xs: list[float], ys: list[float]) -> float | None:
    correct = total = 0
    for first in range(len(xs)):
        for second in range(first + 1, len(xs)):
            if xs[first] == xs[second] or ys[first] == ys[second]:
                continue
            total += 1
            correct += int((xs[first] < xs[second]) == (ys[first] < ys[second]))
    return correct / total if total else None


def report_rank_score(
    report: dict[str, Any],
    package_sha256: str,
    opponent_weights: dict[str, float],
    seat_weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    names_by_sha = {row["sha256"]: row["name"] for row in report.get("submissions", [])}
    candidate = names_by_sha.get(package_sha256)
    if candidate is None:
        raise ValueError(f"candidate SHA-256 {package_sha256} not found in local report")
    weighted_points = weighted_games = unknown_games = all_games = 0.0
    for game in report.get("games", []):
        if candidate not in (game.get("p0"), game.get("p1")) or not game.get("ranking_eligible", True):
            continue
        opponent = game["p1"] if game["p0"] == candidate else game["p0"]
        all_games += 1
        weight = opponent_weights.get(opponent)
        if weight is None:
            unknown_games += 1
            continue
        candidate_seat = "0" if game["p0"] == candidate else "1"
        weight *= (seat_weights or {}).get(candidate_seat, 1.0)
        if game.get("winner") == candidate:
            points = 1.0
        elif game.get("outcome") == "DRAW":
            points = 0.5
        elif game.get("loser") == candidate:
            points = 0.0
        else:
            continue
        weighted_points += weight * points
        weighted_games += weight
    if weighted_games == 0:
        raise ValueError("local report has no known weighted opponent games")
    return weighted_points / weighted_games, unknown_games / all_games if all_games else 0.0


def calibrate(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    excluded = {int(value) for value in data.get("excluded_submission_ids", [])}
    configured_weights = {str(row["name"]): float(row["weight"]) for row in data.get("opponents", [])}
    weights = {str(key): float(value) for key, value in data.get("derived_opponent_weights", configured_weights).items()}
    seat_weights = {str(key): float(value) for key, value in data.get("derived_seat_weights", data.get("seat_weights", {})).items()}
    rows: list[dict[str, Any]] = []
    for submission in data.get("submissions", []):
        submission_id = int(submission["submission_id"])
        if submission_id in excluded or not submission.get("confirmed"):
            continue
        score = submission.get("observed_public_score")
        report_path = submission.get("local_report")
        if score is None or not report_path:
            continue
        report = json.loads(_resolve(base_dir, report_path).read_text(encoding="utf-8"))
        rank_score, unknown_coverage = report_rank_score(report, submission["package_sha256"], weights, seat_weights)
        rows.append({
            "submission_id": submission_id,
            "package_sha256": submission["package_sha256"],
            "kaggle_rank_score": rank_score,
            "observed_public_score": float(score),
            "unknown_opponent_coverage": unknown_coverage,
        })
    xs = [row["kaggle_rank_score"] for row in rows]
    ys = [row["observed_public_score"] for row in rows]
    correlation = spearman_correlation(xs, ys)
    order_accuracy = pairwise_order_accuracy(xs, ys)
    enough = len(rows) >= 5 and len(set(ys)) >= 5
    model = pava_fit(xs, ys) if enough else []
    residuals = [abs(y - isotonic_predict(model, x)) for x, y in zip(xs, ys)] if model else []
    interval_radius = _quantile(residuals, 0.80) if residuals else None
    for row in rows:
        prediction = isotonic_predict(model, row["kaggle_rank_score"]) if model else None
        row["predicted_public_score"] = prediction
        row["prediction_interval"] = (
            [prediction - interval_radius, prediction + interval_radius]
            if prediction is not None and interval_radius is not None
            else None
        )
    calibrated = bool(enough and correlation is not None and correlation >= 0.80 and order_accuracy is not None and order_accuracy >= 0.80)
    return {
        "calibration_version": CALIBRATION_VERSION,
        "verified_submission_count": len(rows),
        "distinct_public_score_count": len(set(ys)),
        "spearman": correlation,
        "pairwise_order_accuracy": order_accuracy,
        "calibrated": calibrated,
        "auto_submit_allowed": calibrated,
        "prediction_status": "available" if enough else "insufficient_data",
        "isotonic_model": model,
        "rows": rows,
    }


def sync(data: dict[str, Any], kaggle_executable: Path) -> dict[str, Any]:
    cli = KaggleCli(kaggle_executable)
    active = cli.team_submissions(int(data["team_id"]))
    by_id = {int(row.get("id", row.get("ref", -1))): row for row in active}
    now = datetime.now(timezone.utc).isoformat()
    episodes_by_submission: dict[int, list[dict[str, Any]]] = {}
    for row in data.get("submissions", []):
        remote = by_id.get(int(row["submission_id"]))
        if remote is None:
            continue
        public_score = remote.get("publicScore")
        row["observed_public_score"] = float(public_score) if public_score not in (None, "") else None
        row["observed_at"] = now
        episodes = cli.episodes(int(row["submission_id"]))
        episodes_by_submission[int(row["submission_id"])] = episodes
        row["episode_count"] = len(episodes)
    data.update(derive_episode_weights(episodes_by_submission, data.get("opponents", [])))
    return data


def derive_episode_weights(
    episodes_by_submission: dict[int, list[dict[str, Any]]],
    opponents: list[dict[str, Any]],
) -> dict[str, Any]:
    opponent_by_submission = {
        int(submission_id): str(row["name"])
        for row in opponents
        for submission_id in row.get("submission_ids", [])
    }
    seat_counts = {"0": 0, "1": 0}
    opponent_counts: dict[str, int] = {}
    unknown = total = 0
    for target_submission, episodes in episodes_by_submission.items():
        for episode in episodes:
            agents = episode.get("agents") or []
            target_seat = next(
                (index for index, agent in enumerate(agents) if int(agent.get("submissionId", -1)) == target_submission),
                None,
            )
            if target_seat not in (0, 1):
                continue
            total += 1
            seat_counts[str(target_seat)] += 1
            opponent_id = int((agents[1 - target_seat] or {}).get("submissionId", -1))
            opponent_name = opponent_by_submission.get(opponent_id)
            if opponent_name is None:
                unknown += 1
                continue
            opponent_counts[opponent_name] = opponent_counts.get(opponent_name, 0) + 1
    known_total = sum(opponent_counts.values())
    seat_total = sum(seat_counts.values())
    return {
        "derived_opponent_weights": {
            name: count / known_total for name, count in sorted(opponent_counts.items())
        } if known_total else {},
        "derived_seat_weights": {
            seat: count / seat_total for seat, count in seat_counts.items()
        } if seat_total else {},
        "episode_weight_coverage": {
            "episodes": total,
            "unknown_opponents": unknown,
            "unknown_opponent_coverage": unknown / total if total else 0.0,
        },
    }


def run_evaluations(data: dict[str, Any], base_dir: Path, out: Path, workers: int) -> None:
    opponents = [_resolve(base_dir, row["package"]) for row in data.get("opponents", [])]
    games = int(data.get("games_per_pair", 20))
    for row in data.get("submissions", []):
        package = _resolve(base_dir, row["package"])
        target = out / str(row["submission_id"])
        report = run_candidate_pool(
            [package],
            opponents,
            games,
            EvalConfig(profile="kaggle", workers=workers, record_mode="none", progress=True, progress_label=str(row["submission_id"])),
            Path.cwd(),
            peer_span=0,
        )
        save_report(report, target)
        row["local_report"] = str((target / "report.json").resolve())


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate local Kaggle-profile ordering against verified public scores.")
    parser.add_argument("command", choices=("validate", "sync", "evaluate", "calibrate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("outputs/kaggle_eval_calibration"))
    parser.add_argument("--kaggle", type=Path, default=Path("kaggle"))
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args(argv)
    data = load_config(args.config)
    if args.command == "sync":
        sync(data, args.kaggle)
        args.config.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    elif args.command == "evaluate":
        run_evaluations(data, args.config.parent, args.out, args.workers)
        args.config.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    elif args.command == "calibrate":
        result = calibrate(data, args.config.parent)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
