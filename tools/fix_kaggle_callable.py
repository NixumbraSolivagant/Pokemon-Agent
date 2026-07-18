from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path


ALIAS = b"""

# Kaggle executes main.py and picks the last callable in insertion order.
# Keep helper functions from becoming the submitted callable.
kaggle_agent = agent
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expand_targets(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.exists() and path.suffixes[-2:] == [".tar", ".gz"]:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    out.append(path)
    return out


def patch_main_payload(data: bytes) -> tuple[bytes, bool]:
    if b"kaggle_agent = agent" in data:
        return data, False
    if b"def agent(" not in data:
        raise ValueError("main.py does not define agent(...)")
    return data.rstrip() + ALIAS + b"\n", True


def patch_tarball(path: Path) -> dict[str, object]:
    before_sha = sha256_file(path)
    tmp = path.with_name(f".{path.name}.kaggle-fix.tmp")
    patched = False
    names: list[str] = []
    uid = os.getuid()
    gid = os.getgid()
    mtime = int(time.time())

    with tarfile.open(path, "r:gz") as src, tarfile.open(tmp, "w:gz") as dst:
        for member in src.getmembers():
            names.append(member.name)
            info = copy.copy(member)
            info.uid = uid
            info.gid = gid
            info.uname = ""
            info.gname = ""
            info.mtime = mtime

            if member.isfile():
                payload = src.extractfile(member)
                if payload is None:
                    continue
                data = payload.read()
                if member.name == "main.py":
                    data, changed = patch_main_payload(data)
                    patched = patched or changed
                info.size = len(data)
                dst.addfile(info, io.BytesIO(data))
            else:
                dst.addfile(info)

    required = {"main.py", "deck.csv", "cg/game.py", "cg/api.py"}
    missing = sorted(required - set(names))
    if missing:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{path} is missing required Kaggle files: {missing}")

    if patched:
        tmp.replace(path)
        after_sha = sha256_file(path)
    else:
        tmp.unlink(missing_ok=True)
        after_sha = before_sha

    return {
        "path": str(path),
        "patched": patched,
        "sha256_before": before_sha,
        "sha256_after": after_sha,
    }


def patch_text_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "patched": False}
    data = path.read_bytes()
    fixed, changed = patch_main_payload(data)
    if changed:
        path.write_bytes(fixed)
    return {"path": str(path), "exists": True, "patched": changed}


def patch_build_submission(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "patched": False}
    text = path.read_text(encoding="utf-8")
    if "kaggle_agent = agent" in text:
        return {"path": str(path), "exists": True, "patched": False}
    marker = "        return list(range(min(min_count, len(options))))\n '''"
    insert = (
        "        return list(range(min(min_count, len(options))))\n\n\n"
        "# Kaggle executes the file and picks the last callable in insertion order.\n"
        "# Rebinding the existing agent under a new final name keeps helper\n"
        "# functions from becoming the submitted callable.\n"
        "kaggle_agent = agent\n '''"
    )
    if marker not in text:
        raise ValueError(f"could not find SEARCH_INJECTION patch point in {path}")
    path.write_text(text.replace(marker, insert, 1), encoding="utf-8")
    return {"path": str(path), "exists": True, "patched": True}


def run_once(args: argparse.Namespace) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if args.patch_sources:
        results.append(patch_text_file(Path("main.py")))
        results.append(patch_build_submission(Path("tools/build_submission.py")))
    for target in expand_targets(args.targets):
        try:
            results.append(patch_tarball(target))
        except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
            results.append(
                {
                    "path": str(target),
                    "patched": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fix Kaggle's last-callable loader issue in Pokemon TCG submission tarballs."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=["outputs/submissions/submission*.tar.gz", "submission*.tar.gz"],
        help="Submission tarballs or glob patterns to patch.",
    )
    parser.add_argument(
        "--patch-sources",
        action="store_true",
        help="Also patch root main.py and tools/build_submission.py for future builds.",
    )
    parser.add_argument("--watch", action="store_true", help="Keep patching matching tarballs as they are rewritten.")
    parser.add_argument("--interval", type=float, default=15.0, help="Watch polling interval in seconds.")
    args = parser.parse_args(argv)

    seen_state: dict[str, tuple[int, int]] = {}
    while True:
        results = run_once(args)
        print(json.dumps({"time": int(time.time()), "results": results}, indent=2), flush=True)
        if not args.watch:
            return 0 if all("error" not in row for row in results) else 1

        current: dict[str, tuple[int, int]] = {}
        for target in expand_targets(args.targets):
            try:
                stat = target.stat()
                current[str(target.resolve())] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                pass
        seen_state = current or seen_state
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
