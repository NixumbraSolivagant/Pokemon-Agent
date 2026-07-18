from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any


LUCARIO_REFERENCE_DECK = [
    673, 673, 674, 674, 675, 675, 676, 676,
    676, 677, 677, 677, 678, 678, 678, 678,
    1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141,
    1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152,
    6, 1159, 1182, 1182, 1192, 1192, 1192, 1192,
    1227, 1227, 1227, 1227, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 1182, 677, 1252,
]

DRAGAPULT_REFERENCE_DECK = [
    119, 119, 119, 119,
    120, 120, 120, 120,
    121, 121, 121,
    140,
    184,
    235, 235,
    1071,
    1079, 1079,
    1080,
    1086, 1086, 1086, 1086,
    1097, 1097,
    1120, 1120, 1120, 1120,
    1121, 1121, 1121, 1121,
    1152, 1152, 1152,
    1156,
    1182, 1182, 1182,
    1198, 1198, 1198, 1198,
    1210, 1210,
    1227, 1227, 1227, 1227,
    1256, 1256,
    2, 2, 2, 2,
    5, 5, 5, 5,
]


def _slug(path: Path) -> str:
    return path.stem.replace(" ", "-").replace("_", "-")


def _write_clean_tar(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for rel in ["main.py", "deck.csv"]:
            tar.add(staging / rel, arcname=rel)
        extra = staging / "lucario_deck.csv"
        if extra.exists():
            tar.add(extra, arcname=extra.name)
        for item in sorted((staging / "cg").rglob("*")):
            rel = item.relative_to(staging / "cg")
            if not item.is_file():
                continue
            if "__pycache__" in rel.parts or item.suffix in {".pyc", ".pyo"}:
                continue
            tar.add(item, arcname=str(Path("cg") / rel))


def _copy_cg(cg_source: Path, staging: Path) -> None:
    target = staging / "cg"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(cg_source, target)


def _execute_code(code: str, staging: Path, globals_dict: dict[str, Any]) -> None:
    old_cwd = Path.cwd()
    os.chdir(staging)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(code, "<notebook-cell>", "exec"), globals_dict)
    finally:
        os.chdir(old_cwd)


def _run_notebook_cells(notebook: Path, staging: Path) -> None:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {
        "__name__": "__notebook_build__",
        "display": lambda *args, **kwargs: None,
    }
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        stripped = source.lstrip()
        if not stripped:
            continue
        if stripped.startswith("%%writefile"):
            first, _, body = stripped.partition("\n")
            parts = first.split()
            if len(parts) < 2:
                continue
            target = staging / parts[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            continue
        if "AGENT_PAYLOADS" in globals_dict and "payload = AGENT_PAYLOADS" in source:
            prefix = source.split("def find_cg_source", 1)[0]
            _execute_code(prefix, staging, globals_dict)
            continue
        if "tarfile.open" in source and "submission.tar.gz" in source:
            continue
        if "importlib.util.spec_from_file_location" in source:
            continue
        _execute_code(source, staging, globals_dict)


def build_reference_submissions(notebooks: list[Path], cg_source: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []
    for notebook in notebooks:
        name = _slug(notebook)
        staging = output_dir / "_build" / name
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _run_notebook_cells(notebook, staging)
        if not (staging / "deck.csv").exists() and "mega-pokemon-reinforcement" in name:
            (staging / "deck.csv").write_text(
                "\n".join(str(card_id) for card_id in LUCARIO_REFERENCE_DECK) + "\n",
                encoding="utf-8",
            )
        if not (staging / "deck.csv").exists() and "pokemon-ai-battle-best-ptcg-advanced" in name:
            (staging / "deck.csv").write_text(
                "\n".join(str(card_id) for card_id in DRAGAPULT_REFERENCE_DECK) + "\n",
                encoding="utf-8",
            )
        selected_build = staging / "selected_agent_build"
        if not (staging / "main.py").exists() and (selected_build / "main.py").exists():
            shutil.copy2(selected_build / "main.py", staging / "main.py")
        if not (staging / "deck.csv").exists() and (selected_build / "deck.csv").exists():
            shutil.copy2(selected_build / "deck.csv", staging / "deck.csv")
        _copy_cg(cg_source, staging)
        missing = [rel for rel in ("main.py", "deck.csv") if not (staging / rel).exists()]
        if missing:
            raise FileNotFoundError(f"{notebook} did not generate: {', '.join(missing)}")
        output = output_dir / f"{name}.tar.gz"
        _write_clean_tar(staging, output)
        built.append(output)
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build reference notebook submissions for local evaluation.")
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--cg-source", type=Path, default=Path("pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"))
    parser.add_argument("--out", type=Path, default=Path("outputs/reference_submissions"))
    args = parser.parse_args(argv)
    built = build_reference_submissions(args.notebooks, args.cg_source, args.out)
    for path in built:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
