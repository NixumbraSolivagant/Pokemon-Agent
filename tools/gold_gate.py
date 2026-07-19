from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_eval.archive import sha256_file
from local_eval.evaluator import run_ladder, save_report, submission_name
from local_eval.models import AgentStats, EvalConfig, MatchReport
from tools.reference_pool import ANCHOR_TABLE, default_pool, load_anchor_table, load_anchors, pool_manifest


TARGET_LB_SCORE = 1200.0
KEY_ANCHORS = {
    "i-have-one-rear-card": 0.70,
    "submission_820": 0.60,
    "pokemon-ai-battle-best-ptcg-advanced": 0.50,
    "pokemon-steel": 0.50,
    "pokemon-tcg-rahul-jiwane": 0.50,
    "ptcg-mega-lucario-ex-v63": 0.50,
    "multiply-agent-best-940-lb": 0.55,
    "submission_sorce_700": 0.70,
}
REQUIRED_GOLD_ANCHORS = (
    "i-have-one-rear-card",
    "submission_820",
    "pokemon-ai-battle-best-ptcg-advanced",
    "multiply-agent-best-940-lb",
)


@dataclass(slots=True)
class MatchupGate:
    opponent: str
    wins: int
    losses: int
    draws: int
    games: int
    win_rate: float
    wilson_low: float
    target: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def wilson_low(wins: int, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    denom = 1.0 + z * z / games
    centre = p + z * z / (2.0 * games)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games)
    return max(0.0, (centre - margin) / denom)


def score_rate(row: AgentStats | dict[str, Any], opponent: str) -> MatchupGate | None:
    opponents = row.opponents if isinstance(row, AgentStats) else row.get("opponents", {})
    rec = opponents.get(opponent)
    if not rec:
        return None
    wins = int(rec.get("wins") or 0)
    losses = int(rec.get("losses") or 0)
    draws = int(rec.get("draws") or 0)
    games = wins + losses + draws
    target = KEY_ANCHORS.get(opponent, 0.48)
    rate = (wins + 0.5 * draws) / games if games else 0.0
    return MatchupGate(
        opponent=opponent,
        wins=wins,
        losses=losses,
        draws=draws,
        games=games,
        win_rate=rate,
        wilson_low=wilson_low(wins, games),
        target=target,
        passed=rate >= target,
    )


def stats_by_name(report: MatchReport | dict[str, Any]) -> dict[str, AgentStats | dict[str, Any]]:
    standings = report.standings if isinstance(report, MatchReport) else report.get("standings", [])
    return {row.name if isinstance(row, AgentStats) else str(row.get("name")): row for row in standings}


def row_clean(row: AgentStats | dict[str, Any], max_no_result_rate: float) -> tuple[bool, list[str]]:
    games = int(row.games if isinstance(row, AgentStats) else row.get("games") or 0)
    no_results = int(row.no_results if isinstance(row, AgentStats) else row.get("no_results") or 0)
    crashes = int(row.crashes if isinstance(row, AgentStats) else row.get("crashes") or 0)
    timeouts = int(row.timeouts if isinstance(row, AgentStats) else row.get("timeouts") or 0)
    invalids = int(row.invalids if isinstance(row, AgentStats) else row.get("invalids") or 0)
    reasons: list[str] = []
    if invalids:
        reasons.append(f"invalids={invalids}")
    if timeouts:
        reasons.append(f"timeouts={timeouts}")
    if crashes:
        reasons.append(f"crashes={crashes}")
    nr_rate = no_results / max(1, games)
    if nr_rate >= max_no_result_rate:
        reasons.append(f"no_result_rate={nr_rate:.4f} >= {max_no_result_rate:.4f}")
    return not reasons, reasons


def lb_score_for_sha(sha256: str, anchor_table: Path = ANCHOR_TABLE) -> float | None:
    for anchor in load_anchors(anchor_table):
        if anchor.sha256 and anchor.sha256 == sha256 and anchor.lb_score is not None:
            return anchor.lb_score
    return None


def classify_candidate(
    report: MatchReport | dict[str, Any],
    candidate_name: str,
    incumbent_name: str = "incumbent",
    submission_sha256: str = "",
    max_no_result_rate: float = 0.01,
    target_lb_score: float | None = None,
    anchor_table: Path = ANCHOR_TABLE,
    min_matchup_games: int = 0,
) -> dict[str, Any]:
    table = load_anchor_table(anchor_table)
    target_lb = float(target_lb_score if target_lb_score is not None else table.get("target_lb_score", TARGET_LB_SCORE))
    stats = stats_by_name(report)
    row = stats.get(candidate_name)
    if row is None:
        return {
            "decision": "gold_gate_failed",
            "submit_ready": False,
            "gold_gate_passed": False,
            "reasons": [f"candidate {candidate_name!r} not found in report"],
        }
    clean, reasons = row_clean(row, max_no_result_rate)
    gates: list[MatchupGate] = []
    for opponent, target in KEY_ANCHORS.items():
        gate = score_rate(row, opponent)
        if gate is None:
            continue
        gate.target = target
        wilson_target = max(0.0, target - 0.05)
        gate.passed = gate.games >= min_matchup_games and gate.win_rate >= target and (
            min_matchup_games <= 0 or gate.wilson_low >= wilson_target
        )
        gates.append(gate)
        if gate.games < min_matchup_games:
            reasons.append(f"vs {opponent} only {gate.games} games; requires {min_matchup_games}")
        if not gate.passed:
            reasons.append(f"vs {opponent} win_rate {gate.win_rate:.3f} below {target:.3f}")
        if min_matchup_games > 0 and gate.wilson_low < wilson_target:
            reasons.append(f"vs {opponent} Wilson lower bound {gate.wilson_low:.3f} below {wilson_target:.3f}")
    covered = {gate.opponent for gate in gates}
    for required in REQUIRED_GOLD_ANCHORS:
        if required not in covered:
            reasons.append(f"missing required anchor matchup: {required}")
    if incumbent_name == candidate_name:
        reasons.append("candidate cannot be its own incumbent gate")
    elif incumbent_name not in stats:
        reasons.append(f"missing incumbent submission: {incumbent_name}")
    else:
        gate = score_rate(row, incumbent_name)
        if gate is None:
            reasons.append(f"missing incumbent matchup: {incumbent_name}")
        else:
            if gate.games < min_matchup_games:
                reasons.append(f"vs incumbent only {gate.games} games; requires {min_matchup_games}")
            if gate.win_rate < 0.55:
                reasons.append(f"vs incumbent win_rate {gate.win_rate:.3f} below 0.550")
            if gate.wilson_low < 0.50:
                reasons.append(f"vs incumbent Wilson lower bound {gate.wilson_low:.3f} below 0.500")
    known_lb = lb_score_for_sha(submission_sha256, anchor_table) if submission_sha256 else None
    if known_lb is not None and known_lb >= target_lb:
        decision = "gold_confirmed"
        submit_ready = True
        gold_gate_passed = True
    elif reasons or not clean:
        decision = "gold_gate_failed"
        submit_ready = False
        gold_gate_passed = False
    else:
        decision = "kaggle_probe_ready"
        submit_ready = False
        gold_gate_passed = True
        reasons = ["gold gate passed locally; needs Kaggle probe for 1200+ confirmation"]
    return {
        "decision": decision,
        "submit_ready": submit_ready,
        "gold_gate_passed": gold_gate_passed,
        "candidate": candidate_name,
        "target_lb_score": target_lb,
        "known_lb_score": known_lb,
        "submission_sha256": submission_sha256,
        "matchup_gates": [gate.to_dict() for gate in gates],
        "reasons": reasons,
        "score_warning": "local_trueskill_score is not Kaggle leaderboard score",
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    pool = args.pool or default_pool()
    tarballs = [args.candidate, *pool]
    cfg = EvalConfig(
        seed=args.seed,
        workers=max(1, args.workers),
        max_actions=args.max_actions,
        run_timeout_s=args.run_timeout,
        record_mode=args.record_mode,
        record_sample_rate=args.record_sample_rate,
        record_gzip=args.record_gzip,
    )
    report = run_ladder(tarballs, args.games_per_pair, cfg, Path.cwd())
    save_report(report, args.out)
    candidate_name = submission_name(args.candidate)
    sha = sha256_file(args.candidate) if args.candidate.exists() else ""
    decision = classify_candidate(
        report,
        candidate_name,
        args.incumbent_name,
        sha,
        args.max_no_result_rate,
        min_matchup_games=args.min_matchup_games,
    )
    result = {
        "out": str(args.out),
        "candidate": str(args.candidate),
        "candidate_name": candidate_name,
        "candidate_sha256": sha,
        "pool": pool_manifest(pool),
        "decision": decision,
        "standings": [row.to_dict() for row in report.standings],
    }
    write_json(args.out / "gold_gate.json", result)
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    final = read_json(args.final) if args.final else read_json(args.out / "final_report.json")
    report_path = args.report or Path(((final.get("reports") or {}).get("stage_d") or ""))
    if not report_path.exists():
        raise SystemExit(f"Missing report: {report_path}")
    report = read_json(report_path)
    candidate = args.candidate or str(final.get("name") or final.get("best") or "")
    if not candidate:
        raise SystemExit("Missing candidate name; pass --candidate or provide final_report.json with name/best")
    submission = args.submission or Path(str(final.get("submission") or ""))
    sha = sha256_file(submission) if submission.exists() else str(final.get("submission_sha256") or "")
    decision = classify_candidate(
        report,
        candidate,
        args.incumbent_name,
        sha,
        args.max_no_result_rate,
        min_matchup_games=args.min_matchup_games,
    )
    result = {
        "out": str(args.out),
        "candidate": candidate,
        "submission": str(submission),
        "decision": decision,
        "final_status": final.get("status"),
    }
    write_json(args.out / "gold_gate.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gold-medal oriented gate and diagnostics.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_diag = sub.add_parser("diagnose", help="Run candidate vs strong reference pool and classify gold readiness.")
    p_diag.add_argument("--candidate", type=Path, required=True)
    p_diag.add_argument("--out", type=Path, default=Path("outputs/gold_gate"))
    p_diag.add_argument("--pool", nargs="*", type=Path)
    p_diag.add_argument("--games-per-pair", type=int, default=20)
    p_diag.add_argument("--workers", type=int, default=4)
    p_diag.add_argument("--max-actions", type=int, default=1000)
    p_diag.add_argument("--run-timeout", type=float, default=180.0)
    p_diag.add_argument("--seed", type=int, default=20260719)
    p_diag.add_argument("--record-mode", choices=["none", "losses", "sample", "all"], default="losses")
    p_diag.add_argument("--record-sample-rate", type=float, default=0.02)
    p_diag.add_argument("--record-gzip", action=argparse.BooleanOptionalAction, default=True)
    p_diag.add_argument("--incumbent-name", default="incumbent")
    p_diag.add_argument("--max-no-result-rate", type=float, default=0.01)
    p_diag.add_argument("--min-matchup-games", type=int, default=500)

    p_audit = sub.add_parser("audit", help="Classify an existing discovery/gold final report.")
    p_audit.add_argument("--out", type=Path, default=Path("outputs/gold_gate"))
    p_audit.add_argument("--final", type=Path)
    p_audit.add_argument("--report", type=Path)
    p_audit.add_argument("--candidate")
    p_audit.add_argument("--submission", type=Path)
    p_audit.add_argument("--incumbent-name", default="d001_incumbent")
    p_audit.add_argument("--max-no-result-rate", type=float, default=0.01)
    p_audit.add_argument("--min-matchup-games", type=int, default=500)

    args = parser.parse_args(argv)
    result = diagnose(args) if args.cmd == "diagnose" else audit(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
