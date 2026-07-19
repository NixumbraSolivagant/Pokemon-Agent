from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import time
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


def cached_extract_tar_gz(tarball: str | Path, cache_root: str | Path) -> Path:
    tarball = Path(tarball).resolve()
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(tarball)
    extract_dir = cache_root / digest
    ready = extract_dir / ".ready"
    if ready.exists():
        validate_submission_dir(extract_dir)
        return extract_dir
    lock = cache_root / f".{digest}.lock"
    acquired = False
    for _ in range(600):
        try:
            lock.mkdir()
            acquired = True
            break
        except FileExistsError:
            if ready.exists():
                validate_submission_dir(extract_dir)
                return extract_dir
            time.sleep(0.05)
    if not acquired:
        raise TimeoutError(f"Timed out waiting for archive cache lock: {tarball}")
    try:
        if ready.exists():
            return extract_dir
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=cache_root))
        try:
            extracted = safe_extract_tar_gz(tarball, temp_dir)
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extracted.replace(extract_dir)
            ready.write_text(digest, encoding="ascii")
            for path in sorted(extract_dir.rglob("*"), reverse=True):
                try:
                    path.chmod(0o555 if path.is_dir() else 0o444)
                except OSError:
                    pass
            extract_dir.chmod(0o555)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass
    validate_submission_dir(extract_dir)
    return extract_dir


def validate_submission_dir(path: str | Path) -> None:
    path = Path(path)
    required = ["main.py", "deck.csv", "cg", "cg/game.py", "cg/api.py"]
    missing = [p for p in required if not (path / p).exists()]
    if missing:
        raise SubmissionArchiveError(f"Submission missing required files: {', '.join(missing)}")
