from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Iterable

from local_eval.models import AgentStats, MatchReport


def wilson_lower_bound(wins: int, losses: int, draws: int = 0, z: float = 1.96) -> float:
    total = wins + losses + draws
    if total <= 0:
        return 0.0
    successes = wins + 0.5 * draws
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def score_rate(stats: AgentStats) -> float:
    return (stats.wins + 0.5 * stats.draws) / max(1, stats.games - stats.no_results)


def robust_stats(stats: AgentStats) -> dict[str, float]:
    rates = [
        (row["wins"] + 0.5 * row["draws"]) / max(1, row["wins"] + row["losses"] + row["draws"])
        for row in stats.opponents.values()
        if row["wins"] + row["losses"] + row["draws"] > 0
    ]
    rates.sort()
    mean = score_rate(stats)
    worst = rates[0] if rates else 0.0
    quartile = rates[max(0, int(len(rates) * 0.25) - 1)] if rates else 0.0
    variance = sum((value - mean) ** 2 for value in rates) / max(1, len(rates))
    concentration = max(rates) - min(rates) if rates else 1.0
    lower = wilson_lower_bound(stats.wins, stats.losses, stats.draws)
    robust = mean - 0.8 * math.sqrt(variance) - 0.5 * (1.0 - quartile) - 0.2 * concentration
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "worst": worst,
        "quartile": quartile,
        "concentration": concentration,
        "wilson_lower": lower,
        "robust_score": robust,
    }


def rank_for_racing(
    report: MatchReport,
    names: Iterable[str],
    limit: int,
    no_result_rate: float = 0.01,
) -> list[str]:
    allowed = set(names)
    scored: list[tuple[float, AgentStats]] = []
    for stats in report.standings:
        if stats.name not in allowed:
            continue
        if stats.no_results / max(1, stats.games) > no_result_rate:
            continue
        scored.append((robust_stats(stats)["robust_score"], stats))
    scored.sort(key=lambda row: (row[0], row[1].kaggle_score_estimate), reverse=True)
    return [stats.name for _, stats in scored[: max(1, limit)]]


def classify_roles(report: MatchReport, names: Iterable[str], limit: int = 3) -> dict[str, str]:
    rows = {stats.name: stats for stats in report.standings if stats.name in set(names)}
    if not rows:
        return {}
    def family_rate(stats: AgentStats, tokens: tuple[str, ...]) -> float:
        selected = [
            row for name, row in stats.opponents.items()
            if any(token in name.lower() for token in tokens)
        ]
        games = sum(row["wins"] + row["losses"] + row["draws"] for row in selected)
        score = sum(row["wins"] + 0.5 * row["draws"] for row in selected)
        return score / max(1, games)

    robust = sorted(rows, key=lambda name: robust_stats(rows[name])["robust_score"], reverse=True)
    roles: dict[str, str] = {}
    if robust:
        roles[robust[0]] = "generalist"
    fast_ranked = sorted(rows, key=lambda name: family_rate(rows[name], ("fast", "ko", "prize", "aggro")), reverse=True)
    control_ranked = sorted(rows, key=lambda name: family_rate(rows[name], ("mill", "wall", "denial", "control")), reverse=True)
    fast = next((name for name in fast_ranked if name not in roles), fast_ranked[0])
    roles.setdefault(fast, "anti_fast_ko")
    control = next((name for name in control_ranked if name not in roles), control_ranked[0])
    roles.setdefault(control, "anti_control")
    for name in robust[:limit]:
        roles.setdefault(name, "diversity_backup")
    return roles


def report_racing(report: MatchReport, names: Iterable[str]) -> dict[str, Any]:
    selected = set(names)
    rows = {
        stats.name: {
            "stats": asdict(stats),
            "robustness": robust_stats(stats),
        }
        for stats in report.standings
        if stats.name in selected
    }
    return {
        "candidates": rows,
        "roles": classify_roles(report, selected),
        "ranked": [name for name in sorted(rows, key=lambda item: rows[item]["robustness"]["robust_score"], reverse=True)],
    }
