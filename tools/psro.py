from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from local_eval.models import MatchReport


@dataclass(slots=True)
class PsroResult:
    names: list[str]
    payoff: list[list[float | None]]
    weights: dict[str, float]
    exploitability_proxy: float
    niches: list[dict[str, object]]
    coverage: float = 0.0
    best_response_targets: list[str] = field(default_factory=list)
    counter_opponent_targets: list[str] = field(default_factory=list)
    uncertain_matchups: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "names": self.names,
            "payoff": self.payoff,
            "weights": self.weights,
            "exploitability_proxy": self.exploitability_proxy,
            "niches": self.niches,
            "coverage": self.coverage,
            "best_response_targets": self.best_response_targets,
            "counter_opponent_targets": self.counter_opponent_targets,
            "uncertain_matchups": self.uncertain_matchups,
        }


def payoff_matrix(report: MatchReport, names: list[str] | None = None) -> tuple[list[str], list[list[float | None]]]:
    by_name = {stats.name: stats for stats in report.standings}
    if names is None:
        names = [stats.name for stats in report.standings]
    names = [name for name in names if name in by_name]
    matrix: list[list[float | None]] = []
    for row_name in names:
        stats = by_name[row_name]
        row: list[float | None] = []
        for col_name in names:
            if row_name == col_name:
                row.append(0.0)
                continue
            rec = stats.opponents.get(col_name, {"wins": 0, "losses": 0, "draws": 0})
            total = rec["wins"] + rec["losses"] + rec["draws"]
            if total <= 0:
                row.append(None)
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
    if not names:
        return PsroResult([], [], {}, 0.0, [])
    possible_pairs = len(names) * (len(names) - 1) // 2
    covered_pairs = sum(1 for i in range(len(names)) for j in range(i + 1, len(names)) if matrix[i][j] is not None)
    coverage = covered_pairs / possible_pairs if possible_pairs else 1.0
    if coverage < 1.0:
        return PsroResult(
            names=names,
            payoff=matrix,
            weights={},
            exploitability_proxy=0.0,
            niches=[],
            coverage=coverage,
            uncertain_matchups=[
                {"first": names[i], "second": names[j], "games": 0, "score_rate": None, "uncertainty": None}
                for i in range(len(names))
                for j in range(i + 1, len(names))
                if matrix[i][j] is None
            ][:12],
        )
    complete_matrix = [[float(value) for value in row] for row in matrix]
    weights = replicator_dynamics(complete_matrix, iterations=iterations)
    weighted_payoffs = [
        sum(complete_matrix[i][j] * weights[j] for j in range(len(names)))
        for i in range(len(names))
    ]
    value = sum(weights[i] * weighted_payoffs[i] for i in range(len(names)))
    exploitability = max(weighted_payoffs) - value if weighted_payoffs else 0.0
    weights_by_name = {name: weights[i] for i, name in enumerate(names) if weights[i] >= 0.001}
    niches: list[dict[str, object]] = []
    uncertain: list[dict[str, object]] = []
    stats_by_name = {stats.name: stats for stats in report.standings}
    for i, name in enumerate(names):
        weight = weights_by_name.get(name, 0.0)
        if weight < 0.001:
            continue
        beats = sorted(
            ((complete_matrix[i][j], names[j]) for j in range(len(names)) if i != j and complete_matrix[i][j] > 0.15),
            reverse=True,
        )[:5]
        loses_to = sorted(
            ((complete_matrix[i][j], names[j]) for j in range(len(names)) if i != j and complete_matrix[i][j] < -0.15),
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
        for j in range(i + 1, len(names)):
            rec = stats_by_name[name].opponents.get(names[j], {"wins": 0, "losses": 0, "draws": 0})
            games = rec["wins"] + rec["losses"] + rec["draws"]
            score_rate = (rec["wins"] + 0.5 * rec["draws"]) / max(1, games)
            uncertainty = (score_rate * (1.0 - score_rate) / max(1, games)) ** 0.5
            uncertain.append(
                {
                    "first": name,
                    "second": names[j],
                    "games": games,
                    "score_rate": score_rate,
                    "uncertainty": uncertainty,
                }
            )
    uncertain.sort(key=lambda row: (row["uncertainty"], -row["games"]), reverse=True)
    meta_ranked = sorted(weights_by_name, key=weights_by_name.get, reverse=True)
    vulnerable = sorted(
        names,
        key=lambda candidate: min(complete_matrix[names.index(candidate)]) if len(names) > 1 else 0.0,
    )
    return PsroResult(
        names=names,
        payoff=matrix,
        weights=weights_by_name,
        exploitability_proxy=exploitability,
        niches=niches,
        coverage=coverage,
        best_response_targets=meta_ranked[: max(1, min(4, len(meta_ranked)))],
        counter_opponent_targets=vulnerable[: max(1, min(4, len(vulnerable)))],
        uncertain_matchups=uncertain[:12],
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
