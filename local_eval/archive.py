from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


class SubmissionArchiveError(ValueError):
    """Raised when a submission archive is missing required Kaggle files."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract_tar_gz(tarball: str | Path, dest: str | Path) -> Path:
    tarball = Path(tarball).resolve()
    dest = Path(dest).resolve()
    extract_dir = dest / tarball.name.removesuffix(".tar.gz").replace("/", "_")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = extract_dir / member.name
            resolved = member_path.resolve()
            if not str(resolved).startswith(str(extract_dir) + "/") and resolved != extract_dir:
                raise SubmissionArchiveError(f"Unsafe archive member path: {member.name}")
            if member.issym() or member.islnk():
                raise SubmissionArchiveError(f"Archive links are not allowed: {member.name}")
        tar.extractall(extract_dir)

    validate_submission_dir(extract_dir)
    return extract_dir


def validate_submission_dir(path: str | Path) -> None:
    path = Path(path)
    required = ["main.py", "deck.csv", "cg", "cg/game.py", "cg/api.py"]
    missing = [p for p in required if not (path / p).exists()]
    if missing:
        raise SubmissionArchiveError(f"Submission missing required files: {', '.join(missing)}")
