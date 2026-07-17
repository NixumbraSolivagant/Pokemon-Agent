from __future__ import annotations

import argparse
import copy
import io
import os
import time
import tarfile
from pathlib import Path


EXCLUDED_ROOT_FILES = {"build_metadata.json"}
EXCLUDED_SUFFIXES = (".pyc",)
EXCLUDED_PARTS = {"__pycache__"}
SEARCH_WRAPPER_MARKER = "# --- Champion Great Tusk search wrapper injected by tools.build_submission ---"


def keep_member(name: str, exclude_internal_files: bool = True) -> bool:
    path = Path(name)
    if name.startswith("/") or ".." in path.parts:
        return False
    if exclude_internal_files and path.name in EXCLUDED_ROOT_FILES and len(path.parts) == 1:
        return False
    if path.name.startswith(".local_eval_"):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    return True


def clean_main(data: bytes, strip_search_wrapper: bool) -> bytes:
    if not strip_search_wrapper:
        return data
    text = data.decode("utf-8")
    index = text.find(SEARCH_WRAPPER_MARKER)
    if index < 0:
        return data
    return text[:index].rstrip().encode("utf-8") + b"\n"


def export_kaggle_submission(
    source: Path,
    out: Path,
    strip_search_wrapper: bool = False,
    exclude_internal_files: bool = True,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(f".{out.name}.tmp")
    names: list[str] = []
    uid = os.getuid()
    gid = os.getgid()
    mtime = int(time.time())
    with tarfile.open(source, "r:gz") as src, tarfile.open(tmp_out, "w:gz") as dst:
        for member in src.getmembers():
            if not member.isfile() or not keep_member(member.name, exclude_internal_files=exclude_internal_files):
                continue
            payload = src.extractfile(member)
            if payload is None:
                continue
            data = payload.read()
            if member.name == "main.py":
                data = clean_main(data, strip_search_wrapper)
            info = copy.copy(member)
            info.size = len(data)
            info.uid = uid
            info.gid = gid
            info.uname = ""
            info.gname = ""
            info.mtime = mtime
            dst.addfile(info, io.BytesIO(data))
            names.append(member.name)

    required = {"main.py", "deck.csv", "cg/game.py", "cg/api.py"}
    missing = sorted(required - set(names))
    if missing:
        tmp_out.unlink(missing_ok=True)
        raise ValueError(f"{source} is missing required Kaggle files: {missing}")
    tmp_out.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an internal candidate tarball as a Kaggle-clean submission.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/submissions/submission.tar.gz"))
    parser.add_argument(
        "--strip-search-wrapper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove the injected cg.search wrapper for heuristic-only ablations.",
    )
    parser.add_argument(
        "--keep-search-wrapper",
        action="store_false",
        dest="strip_search_wrapper",
        help="Keep the injected search wrapper. This is the production default.",
    )
    parser.add_argument("--exclude-internal-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    out = export_kaggle_submission(
        args.source,
        args.out,
        strip_search_wrapper=args.strip_search_wrapper,
        exclude_internal_files=args.exclude_internal_files,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
