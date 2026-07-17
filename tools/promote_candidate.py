from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

from tools.export_kaggle_submission import export_kaggle_submission


def load_metadata(tarball: Path) -> dict:
    with tarfile.open(tarball, "r:gz") as tar:
        member = tar.extractfile("build_metadata.json")
        if member is None:
            raise FileNotFoundError(f"{tarball} does not contain build_metadata.json")
        return json.loads(member.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a validated candidate tarball to champion_latest.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--state", type=Path, default=Path("outputs/auto_iterate_server_gold/state.json"))
    parser.add_argument("--promote", type=Path, default=Path("outputs/submissions/champion_latest.tar.gz"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/submissions/submission.tar.gz"))
    parser.add_argument("--strip-search-wrapper-for-submission", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generation", default="manual")
    parser.add_argument("--reason", default="manual promotion after validation")
    parser.add_argument("--score-delta", default="")
    parser.add_argument("--h2h-wins", default="")
    parser.add_argument("--h2h-losses", default="")
    args = parser.parse_args(argv)

    meta = load_metadata(args.candidate)
    args.promote.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.candidate, args.promote)
    export_kaggle_submission(
        args.promote,
        args.submission_out,
        strip_search_wrapper=args.strip_search_wrapper_for_submission,
    )
    state = json.loads(args.state.read_text(encoding="utf-8")) if args.state.exists() else {"generation": 0, "history": []}
    incumbent = {
        "name": "incumbent",
        "family": meta.get("family", "great_tusk"),
        "base": meta.get("base", "outputs/reference_submissions/i-have-one-rear-card.tar.gz"),
        "out": str(args.promote),
        "enable_search": meta.get("enable_search", True),
        "injection": meta.get("injection", "great_tusk"),
        "search_candidates": meta.get("search_candidates", 8),
        "search_budget_s": meta.get("search_budget_s", 0.25),
        "search_margin": meta.get("search_margin", 1200.0),
        "search_rollout_steps": meta.get("search_rollout_steps", 16),
        "deck_swaps": meta.get("deck_swaps", []),
        "notes": meta.get("notes", args.reason),
    }
    state["incumbent"] = incumbent
    state.setdefault("history", []).append(
        {
            "generation": args.generation,
            "stage": "manual_promote",
            "candidate": meta.get("name", args.candidate.name.removesuffix(".tar.gz")),
            "promoted": True,
            "score_delta": args.score_delta,
            "h2h_wins": args.h2h_wins,
            "h2h_losses": args.h2h_losses,
            "tarball": str(args.candidate.resolve()),
            "reason": args.reason,
        }
    )
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "promoted": str(args.promote),
                "submission": str(args.submission_out),
                "candidate": str(args.candidate),
                "incumbent": incumbent,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
