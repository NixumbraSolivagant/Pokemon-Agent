from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.build_models import BuildConfig
from tools.build_submission import build_submission
from tools.deck_rules import validate_deck_ids
from tools.lucario_meta_search import SEARCH_PROFILES, opponent_models, render_runtime


COMMON_STAGE2 = [
    *([1102] * 4),
    *([1121] * 4),
    *([1079] * 4),
    *([1086] * 4),
    *([1192] * 4),
    *([1227] * 4),
    *([1182] * 3),
    *([1123] * 2),
    1159,
    1097,
    1264,
]

PROXY_DECKS = {
    "alakazam": [*([741] * 4), *([742] * 4), *([743] * 4), *([5] * 16), *COMMON_STAGE2],
    "grimmsnarl": [*([646] * 4), *([647] * 4), *([648] * 4), *([7] * 16), *COMMON_STAGE2],
    "garchomp": [*([379] * 4), *([380] * 4), *([381] * 4), *([6] * 16), *COMMON_STAGE2],
    "crustle": [
        *([344] * 4), *([345] * 4), *([1] * 18), *([1102] * 4), *([1121] * 4),
        *([1086] * 4), *([1152] * 4), *([1192] * 4), *([1227] * 4), *([1182] * 3),
        *([1123] * 2), 1159, 1097, 1139, 1213, 1264,
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build current-meta pressure proxy submissions.")
    parser.add_argument("--out", type=Path, default=Path("outputs/meta_proxies"))
    parser.add_argument("--runtime-source", type=Path, default=Path("agents/lucario_meta.py"))
    parser.add_argument("--runtime-cg-dir", type=Path, default=Path("cg"))
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    models = opponent_models(args.models)
    source = render_runtime(args.runtime_source.read_text(encoding="utf-8"), SEARCH_PROFILES[2], models)
    runtime = args.out / "generic_meta_runtime.py"
    runtime.write_text(source, encoding="utf-8")
    built = {}
    for name, deck in PROXY_DECKS.items():
        validate_deck_ids(deck)
        path = build_submission(
            BuildConfig(
                name=f"meta_proxy_{name}",
                family=f"meta_proxy_{name}",
                base=Path("/nonexistent/reference.tar.gz"),
                out=args.out / f"meta_proxy_{name}.tar.gz",
                runtime_source=runtime,
                runtime_cg_dir=args.runtime_cg_dir,
                injection="none",
                enable_search=True,
                deck_override=deck,
                origin="current_meta_proxy",
                notes=f"Pressure proxy for {name}; not a production candidate.",
            )
        )
        built[name] = str(path)
    print(json.dumps(built, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
