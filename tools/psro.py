from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from local_eval.models import MatchReport


@dataclass(slots=True)
class PsroResult:
    names: list[str]
    payoff: list[list[float]]
    weights: dict[str, float]
    exploitability_proxy: float
    niches: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "names": self.names,
            "payoff": self.payoff,
            "weights": self.weights,
            "exploitability_proxy": self.exploitability_proxy,
            "niches": self.niches,
        }


def payoff_matrix(report: MatchReport, names: list[str] | None = None) -> tuple[list[str], list[list[float]]]:
    by_name = {stats.name: stats for stats in report.standings}
    if names is None:
        names = [stats.name for stats in report.standings]
    names = [name for name in names if name in by_name]
    matrix: list[list[float]] = []
    for row_name in names:
        stats = by_name[row_name]
        row: list[float] = []
        for col_name in names:
            if row_name == col_name:
                row.append(0.0)
                continue
            rec = stats.opponents.get(col_name, {"wins": 0, "losses": 0, "draws": 0})
            total = rec["wins"] + rec["losses"] + rec["draws"]
            if total <= 0:
                row.append(0.0)
            else:
                score_rate = (rec["wins"] + 0.5 * rec["draws"]) / total
                row.append(2.0 * score_rate - 1.0)
        matrix.append(row)
    return names, matrix


def replicator_dynamics(
    matrix: list[list[float]],
    iterations: int = 2000,
    lr: float = 0.12,
    floor: float = 1.0e-9,
) -> list[float]:
    n = len(matrix)
    if n == 0:
        return []
    weights = [1.0 / n] * n
    shifted = [[value + 1.0 for value in row] for row in matrix]
    for _ in range(max(1, iterations)):
        payoffs = [sum(shifted[i][j] * weights[j] for j in range(n)) for i in range(n)]
        avg = sum(weights[i] * payoffs[i] for i in range(n))
        if avg <= 0:
            break
        next_weights = [max(floor, weights[i] * (1.0 + lr * (payoffs[i] - avg) / avg)) for i in range(n)]
        total = sum(next_weights)
        weights = [w / total for w in next_weights]
    return weights


def solve_psro(report: MatchReport, names: list[str] | None = None, iterations: int = 2000) -> PsroResult:
    names, matrix = payoff_matrix(report, names)
    weights = replicator_dynamics(matrix, iterations=iterations)
    if not names:
        return PsroResult([], [], {}, 0.0, [])
    weighted_payoffs = [
        sum(matrix[i][j] * weights[j] for j in range(len(names)))
        for i in range(len(names))
    ]
    value = sum(weights[i] * weighted_payoffs[i] for i in range(len(names)))
    exploitability = max(weighted_payoffs) - value if weighted_payoffs else 0.0
    weights_by_name = {name: weights[i] for i, name in enumerate(names) if weights[i] >= 0.001}
    niches: list[dict[str, object]] = []
    for i, name in enumerate(names):
        weight = weights_by_name.get(name, 0.0)
        if weight < 0.001:
            continue
        beats = sorted(
            ((matrix[i][j], names[j]) for j in range(len(names)) if i != j and matrix[i][j] > 0.15),
            reverse=True,
        )[:5]
        loses_to = sorted(
            ((matrix[i][j], names[j]) for j in range(len(names)) if i != j and matrix[i][j] < -0.15),
        )[:5]
        role = "generalist"
        if weight >= 0.20 and beats and loses_to:
            role = "anti_meta_specialist"
        elif weight >= 0.20:
            role = "meta_anchor"
        elif beats:
            role = "counterpick"
        niches.append(
            {
                "candidate": name,
                "weight": weight,
                "role": role,
                "beats": [opponent for _, opponent in beats],
                "loses_to": [opponent for _, opponent in loses_to],
            }
        )
    return PsroResult(
        names=names,
        payoff=matrix,
        weights=weights_by_name,
        exploitability_proxy=exploitability,
        niches=niches,
    )


def psro_bonus_names(report: MatchReport, candidate_names: set[str], limit: int, min_weight: float = 0.02) -> list[str]:
    names = sorted(candidate_names)
    result = solve_psro(report, names)
    ranked = [
        (weight, name)
        for name, weight in result.weights.items()
        if name in candidate_names and weight >= min_weight
    ]
    ranked.sort(reverse=True)
    return [name for _, name in ranked[: max(0, limit)]]


def write_psro(report: MatchReport, path: Path, names: list[str] | None = None) -> PsroResult:
    result = solve_psro(report, names)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
