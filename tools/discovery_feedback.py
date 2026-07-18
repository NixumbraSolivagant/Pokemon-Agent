from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PressureItem:
    pressure_id: str
    kind: str
    severity: float
    confidence: float
    target_opponent: str = ""
    target_family: str = ""
    target_candidate: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_operators: list[str] = field(default_factory=list)
    recommended_motifs: list[str] = field(default_factory=list)
    avoid_mutations: list[str] = field(default_factory=list)
    protected_cards: list[int] = field(default_factory=list)
    strategy_shifts: dict[str, float] = field(default_factory=dict)
    budget_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TAG_TO_PRESSURE = {
    "focus_hand_drop": ("loss_recovery", ["motif_insert", "protect_core"], ["draw_recovery", "hand_disruption"]),
    "opponent_hand_surge": ("anti_matchup", ["anti_matchup_package", "strategy_shift"], ["hand_disruption"]),
    "opponent_took_prize": ("anti_prize_race", ["motif_insert", "strategy_shift"], ["defensive_tools", "switch_pivot"]),
    "focus_prize_regressed": ("anti_prize_race", ["motif_insert", "strategy_shift"], ["defensive_tools"]),
    "focus_deck_drop": ("resource_repair", ["motif_insert", "avoid_cut"], ["draw_recovery", "resource_safety"]),
    "opponent_deck_recovered": ("resource_denial", ["anti_matchup_package"], ["resource_denial"]),
    "focus_active_changed": ("pivot_repair", ["motif_insert", "protect_core"], ["switch_pivot", "defensive_tools"]),
}

MOTIF_STRATEGY_SHIFTS = {
    "draw_recovery": {"hand_delta": 90.0, "self_mill_penalty": 420.0},
    "hand_disruption": {"hand_delta": 80.0, "opp_mill": 1050.0},
    "defensive_tools": {"prize_delta": 5200.0, "great_tusk_active": 2200.0},
    "switch_pivot": {"great_tusk_ready": 7600.0, "supporter_played_attack": 5600.0},
    "resource_denial": {"opp_mill": 1180.0, "opp_deckout_bonus": 23000.0},
    "resource_safety": {"self_deckout_penalty": 15500.0, "deck_delta": 90.0},
}

DEFAULT_BUDGET_SPLIT = {
    "repair": 0.30,
    "niche": 0.22,
    "loss": 0.20,
    "diversity": 0.18,
    "explore": 0.10,
}


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_scenarios(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("scenarios", [])
    return [row for row in data if isinstance(row, dict)]


def load_report(path: Path | None) -> dict[str, Any]:
    return read_json(path)


def family_from_name(name: str) -> str:
    lower = name.lower()
    for family in ("great_tusk", "lucario", "metal", "charizard", "ancient", "stall"):
        if family in lower:
            return family
    return "unknown"


def candidate_family_hint(name: str) -> str:
    family = family_from_name(name)
    return "great_tusk" if family == "unknown" else family


def report_stats(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name")): row
        for row in report.get("standings", [])
        if isinstance(row, dict) and row.get("name")
    }


def candidate_score_rate(row: dict[str, int]) -> float:
    total = int(row.get("wins", 0)) + int(row.get("losses", 0)) + int(row.get("draws", 0))
    if total <= 0:
        return 0.5
    return (int(row.get("wins", 0)) + 0.5 * int(row.get("draws", 0))) / total


def stability_pressure(report: dict[str, Any], stage: str) -> list[PressureItem]:
    out: list[PressureItem] = []
    for row in report.get("standings", []):
        if not isinstance(row, dict):
            continue
        games = max(1, int(row.get("games") or 0))
        failures = int(row.get("timeouts") or 0) + int(row.get("invalids") or 0) + int(row.get("crashes") or 0)
        no_results = int(row.get("no_results") or 0)
        rate = (failures + no_results) / games
        if rate < 0.01 and failures == 0:
            continue
        name = str(row.get("name") or "")
        out.append(
            PressureItem(
                pressure_id=f"{stage}:stability:{name}",
                kind="stability_repair",
                severity=min(1.0, rate * 20.0 + failures * 0.08),
                confidence=min(1.0, games / 60.0),
                target_candidate=name,
                target_family=family_from_name(name),
                evidence={
                    "stage": stage,
                    "games": games,
                    "timeouts": row.get("timeouts", 0),
                    "invalids": row.get("invalids", 0),
                    "crashes": row.get("crashes", 0),
                    "no_results": no_results,
                },
                recommended_operators=["search_shape_shift", "strategy_shift"],
                strategy_shifts={"self_deckout_penalty": 16000.0, "self_mill_penalty": 460.0},
                budget_weight=max(0.25, min(2.0, rate * 50.0)),
            )
        )
    return out


def matchup_pressures(report: dict[str, Any], stage: str, incumbent_name: str, holdout_feedback_mode: str = "audit_only") -> list[PressureItem]:
    if holdout_feedback_mode == "audit_only" and "holdout" in stage:
        return []
    stats = report_stats(report)
    out: list[PressureItem] = []
    inc = stats.get(incumbent_name)
    inc_score = float(inc.get("kaggle_score_estimate") or 0.0) if inc else 0.0
    for name, row in stats.items():
        if name == incumbent_name:
            continue
        score = float(row.get("kaggle_score_estimate") or 0.0)
        candidate_family = candidate_family_hint(name)
        for opponent, rec in dict(row.get("opponents") or {}).items():
            opponent_family = family_from_name(str(opponent))
            rate = candidate_score_rate(rec)
            total = int(rec.get("wins", 0)) + int(rec.get("losses", 0)) + int(rec.get("draws", 0))
            if total >= 2 and rate <= 0.38 and score >= inc_score - 80.0:
                out.append(
                    PressureItem(
                        pressure_id=f"{stage}:repair:{name}:vs:{opponent}",
                        kind="repair_matchup",
                        severity=min(1.0, (0.50 - rate) * 2.4),
                        confidence=min(1.0, total / 24.0),
                        target_candidate=name,
                        target_opponent=str(opponent),
                        target_family=candidate_family,
                        evidence={"stage": stage, "score_rate": rate, "record": rec, "candidate_score": score, "opponent_family": opponent_family},
                        recommended_operators=["anti_matchup_package", "motif_insert", "strategy_shift"],
                        recommended_motifs=["defensive_tools", "switch_pivot", "resource_safety"],
                        strategy_shifts={"prize_delta": 5200.0, "great_tusk_ready": 7600.0},
                        budget_weight=max(0.4, (0.50 - rate) * 4.0),
                    )
                )
            if total >= 2 and rate >= 0.64 and score >= inc_score - 160.0:
                out.append(
                    PressureItem(
                        pressure_id=f"{stage}:exploit:{name}:vs:{opponent}",
                        kind="exploit_niche",
                        severity=min(1.0, (rate - 0.50) * 2.2),
                        confidence=min(1.0, total / 24.0),
                        target_candidate=name,
                        target_opponent=str(opponent),
                        target_family=candidate_family,
                        evidence={"stage": stage, "score_rate": rate, "record": rec, "candidate_score": score, "opponent_family": opponent_family},
                        recommended_operators=["crossover", "anti_matchup_package", "strategy_shift"],
                        recommended_motifs=["hand_disruption", "resource_denial"],
                        strategy_shifts={"opp_mill": 1120.0, "opp_deckout_bonus": 22000.0},
                        budget_weight=max(0.35, (rate - 0.50) * 3.0),
                    )
                )
    return out


def scenario_pressures(scenarios: list[dict[str, Any]], stage: str) -> list[PressureItem]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        tags = [str(tag) for tag in scenario.get("tags") or []]
        for tag in tags:
            kind, operators, motifs = TAG_TO_PRESSURE.get(tag, ("loss_recovery", ["motif_insert"], ["resource_safety"]))
            grouped[(kind, str(scenario.get("focus") or ""), tag)].append(scenario)
    out: list[PressureItem] = []
    for (kind, focus, tag), rows in grouped.items():
        drops = [float(row.get("drop") or 0.0) for row in rows]
        opponents = Counter(str(row.get("opponent") or "") for row in rows if row.get("opponent"))
        _, operators, motifs = TAG_TO_PRESSURE.get(tag, (kind, ["motif_insert"], ["resource_safety"]))
        shifts: dict[str, float] = {}
        for motif in motifs:
            shifts.update(MOTIF_STRATEGY_SHIFTS.get(motif, {}))
        out.append(
            PressureItem(
                pressure_id=f"{stage}:scenario:{focus}:{tag}",
                kind=kind,
                severity=min(1.0, (sum(drops) / max(1, len(drops))) / 900.0),
                confidence=min(1.0, len(rows) / 8.0),
                target_candidate=focus,
                target_opponent=opponents.most_common(1)[0][0] if opponents else "",
                target_family=candidate_family_hint(focus),
                evidence={"stage": stage, "tag": tag, "count": len(rows), "avg_drop": sum(drops) / max(1, len(drops))},
                recommended_operators=operators,
                recommended_motifs=motifs,
                strategy_shifts=shifts,
                budget_weight=max(0.5, min(2.5, len(rows) / 3.0)),
            )
        )
    return out


def psro_pressures(psro_path: Path | None, stage: str) -> list[PressureItem]:
    data = read_json(psro_path)
    out: list[PressureItem] = []
    for niche in data.get("niches", []):
        if not isinstance(niche, dict):
            continue
        candidate = str(niche.get("candidate") or "")
        weight = float(niche.get("weight") or 0.0)
        if not candidate or weight < 0.02:
            continue
        beats = [str(x) for x in niche.get("beats", [])]
        loses_to = [str(x) for x in niche.get("loses_to", [])]
        target = beats[0] if beats else (loses_to[0] if loses_to else "")
        out.append(
            PressureItem(
                pressure_id=f"{stage}:psro:{candidate}",
                kind="psro_niche",
                severity=min(1.0, weight * 3.0),
                confidence=min(1.0, 0.40 + weight * 2.0),
                target_candidate=candidate,
                target_opponent=target,
                target_family=candidate_family_hint(candidate),
                evidence={"stage": stage, "role": niche.get("role"), "weight": weight, "beats": beats, "loses_to": loses_to},
                recommended_operators=["crossover", "anti_matchup_package", "strategy_shift"],
                recommended_motifs=["hand_disruption", "resource_denial"],
                strategy_shifts={"opp_mill": 1100.0, "hand_delta": 85.0},
                budget_weight=max(0.5, weight * 6.0),
            )
        )
    return out


def build_failure_pressures(path: Path | None, stage: str = "build") -> list[PressureItem]:
    data = []
    if path and path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, list) else []
        except Exception:
            data = []
    if not data:
        return []
    return [
        PressureItem(
            pressure_id=f"{stage}:build_failures",
            kind="stability_repair",
            severity=min(1.0, len(data) / 20.0),
            confidence=0.75,
            evidence={"failures": data[:12], "count": len(data)},
            recommended_operators=["avoid_cut", "search_shape_shift"],
            budget_weight=min(2.0, 0.5 + len(data) / 16.0),
        )
    ]


def dedupe_pressures(items: list[PressureItem], limit: int = 64) -> list[PressureItem]:
    best: dict[tuple[str, str, str, str], PressureItem] = {}
    for item in items:
        key = (item.kind, item.target_candidate, item.target_opponent, ",".join(item.recommended_motifs))
        score = item.severity * item.confidence * item.budget_weight
        prev = best.get(key)
        if prev is None or score > prev.severity * prev.confidence * prev.budget_weight:
            best[key] = item
    ranked = sorted(best.values(), key=lambda p: (p.severity * p.confidence * p.budget_weight, p.confidence), reverse=True)
    return ranked[:limit]


def population_plan(items: list[PressureItem], population: int, microburst: bool = False) -> dict[str, Any]:
    base = dict(DEFAULT_BUDGET_SPLIT)
    if microburst:
        base = {"repair": 0.35, "niche": 0.15, "loss": 0.35, "diversity": 0.05, "explore": 0.10}
    kind_bucket = {
        "repair_matchup": "repair",
        "stability_repair": "repair",
        "psro_niche": "niche",
        "exploit_niche": "niche",
        "loss_recovery": "loss",
        "anti_prize_race": "loss",
        "resource_repair": "loss",
        "resource_denial": "loss",
        "pivot_repair": "loss",
        "archetype_diversity": "diversity",
    }
    pressure_by_bucket: Counter[str] = Counter()
    for item in items:
        pressure_by_bucket[kind_bucket.get(item.kind, "explore")] += item.budget_weight
    slots = {name: int(round(population * frac)) for name, frac in base.items()}
    diff = population - sum(slots.values())
    slots["explore"] = max(0, slots.get("explore", 0) + diff)
    return {
        "population": population,
        "mode": "microburst" if microburst else "next_generation",
        "slots": slots,
        "pressure_by_bucket": dict(pressure_by_bucket),
    }


def aggregate_feedback(
    *,
    out: Path,
    generation: int,
    stage_reports: dict[str, Path] | None = None,
    scenario_paths: list[Path] | None = None,
    psro_paths: list[Path] | None = None,
    build_failures: Path | None = None,
    incumbent_name: str = "",
    population: int = 0,
    holdout_feedback_mode: str = "audit_only",
    microburst: bool = False,
    previous_feedback: Path | None = None,
) -> dict[str, Any]:
    items: list[PressureItem] = []
    previous = read_json(previous_feedback)
    for raw in previous.get("pressure_items", [])[:16]:
        if not isinstance(raw, dict):
            continue
        raw = dict(raw)
        raw["severity"] = float(raw.get("severity") or 0.0) * 0.65
        raw["confidence"] = float(raw.get("confidence") or 0.0) * 0.85
        raw["pressure_id"] = f"carry:{raw.get('pressure_id', 'unknown')}"
        for key in ("recommended_operators", "recommended_motifs", "avoid_mutations", "protected_cards"):
            if raw.get(key) is None:
                raw[key] = []
        if raw.get("evidence") is None:
            raw["evidence"] = {}
        if raw.get("strategy_shifts") is None:
            raw["strategy_shifts"] = {}
        if raw.get("budget_weight") is None:
            raw["budget_weight"] = 1.0
        if raw.get("severity") is None:
            raw["severity"] = 0.0
        if raw.get("confidence") is None:
            raw["confidence"] = 0.0
        for key in ("target_opponent", "target_family", "target_candidate"):
            if raw.get(key) is None:
                raw[key] = ""
        try:
            items.append(PressureItem(**{key: raw.get(key) for key in PressureItem.__dataclass_fields__}))
        except Exception:
            continue
    for stage, path in (stage_reports or {}).items():
        report = load_report(path)
        if not report:
            continue
        items.extend(stability_pressure(report, stage))
        items.extend(matchup_pressures(report, stage, incumbent_name, holdout_feedback_mode))
    for path in scenario_paths or []:
        stage_label = path.stem if path.stem != "scenarios" else (path.parent.name if path.parent.name else "scenarios")
        items.extend(scenario_pressures(load_scenarios(path), stage_label))
    for path in psro_paths or []:
        items.extend(psro_pressures(path, path.parent.name if path.parent.name else "psro"))
    items.extend(build_failure_pressures(build_failures))
    ranked = dedupe_pressures(items)
    protected = sorted({card for item in ranked for card in item.protected_cards})
    avoid = sorted({name for item in ranked for name in item.avoid_mutations})
    motifs = Counter(motif for item in ranked for motif in item.recommended_motifs)
    feedback = {
        "version": 2,
        "generation": generation,
        "mode": "microburst" if microburst else "next_generation",
        "summary": {
            "pressure_count": len(ranked),
            "top_kinds": dict(Counter(item.kind for item in ranked).most_common(12)),
            "top_motifs": dict(motifs.most_common(12)),
        },
        "population_plan": population_plan(ranked, population, microburst=microburst),
        "protected_cards": protected,
        "avoid_mutations": avoid,
        "recommended_motifs": [name for name, _ in motifs.most_common(12)],
        "pressure_items": [item.to_dict() for item in ranked],
    }
    write_json(out, feedback)
    return feedback


def write_pressure_artifacts(out_dir: Path, feedback: dict[str, Any]) -> None:
    write_json(out_dir / "pressure_map.json", {"pressure_items": feedback.get("pressure_items", [])})
    write_json(out_dir / "mutation_budget.json", feedback.get("population_plan", {}))
    lineage = [
        {
            "pressure_id": item.get("pressure_id"),
            "kind": item.get("kind"),
            "target_candidate": item.get("target_candidate"),
            "target_opponent": item.get("target_opponent"),
            "operators": item.get("recommended_operators", []),
        }
        for item in feedback.get("pressure_items", [])
        if isinstance(item, dict)
    ]
    write_json(out_dir / "lineage_graph.json", {"nodes": lineage})
