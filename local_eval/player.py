from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


class PlayerProtocolError(RuntimeError):
    pass


def _load_main(submission_dir: Path) -> ModuleType:
    main_path = submission_dir / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing main.py in {submission_dir}")
    os.chdir(submission_dir)
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location("submission_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError("main.py does not define agent(obs)")
    return module


def _worker_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        module = _load_main(Path(args.submission_dir).resolve())
    print(json.dumps({"ok": True, "kind": "ready"}), flush=True)

    for line in sys.stdin:
        try:
            req = json.loads(line)
            kind = req.get("kind")
            if kind == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                return 0
            if kind == "deck":
                obs = {"select": None, "logs": [], "current": None}
            elif kind == "act":
                obs = req["obs"]
            else:
                raise PlayerProtocolError(f"Unknown request kind: {kind!r}")
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                action = module.agent(obs)
            print(json.dumps({"ok": True, "action": action, "duration_s": time.perf_counter() - started}), flush=True)
        except Exception as exc:
            print(
                json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                flush=True,
            )
    return 0


@dataclass(slots=True)
class PlayerProcess:
    name: str
    submission_dir: Path
    project_root: Path
    timeout_s: float
    import_timeout_s: float = 6.0
    log_dir: Path | None = None
    process: subprocess.Popen[str] | None = None
    stderr_path: Path | None = None
    stderr_handle: Any | None = None

    def start(self) -> None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(self.project_root) if not existing else f"{self.project_root}{os.pathsep}{existing}"
        stderr_dir = self.log_dir or self.submission_dir
        stderr_dir.mkdir(parents=True, exist_ok=True)
        self.stderr_path = stderr_dir / f".local_eval_{self.name}_stderr.log"
        self.stderr_handle = self.stderr_path.open("w+", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "local_eval.player", "--submission-dir", str(self.submission_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            cwd=str(self.submission_dir),
            env=env,
            bufsize=1,
        )
        self._await_ready(self.import_timeout_s)

    def request(self, payload: dict[str, Any], timeout_s: float | None = None) -> tuple[Any, float]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise PlayerProtocolError("Player process is not started")
        if self.process.poll() is not None:
            stderr = self._read_stderr()
            raise PlayerProtocolError(f"Player process exited early: {stderr}")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        deadline = time.perf_counter() + (timeout_s if timeout_s is not None else self.timeout_s)
        while True:
            if time.perf_counter() > deadline:
                self.kill()
                raise TimeoutError(f"{self.name} exceeded action timeout")
            if self.process.poll() is not None:
                stderr = self._read_stderr()
                raise PlayerProtocolError(f"Player process exited: {stderr}")
            remaining = max(0.0, min(0.05, deadline - time.perf_counter()))
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if ready:
                line = self.process.stdout.readline()
                if line:
                    break
        resp = json.loads(line)
        if not resp.get("ok"):
            raise PlayerProtocolError(resp.get("error", "agent error"))
        return resp.get("action"), float(resp.get("duration_s", 0.0))

    def _await_ready(self, timeout_s: float) -> None:
        if self.process is None or self.process.stdout is None:
            raise PlayerProtocolError("Player process is not started")
        deadline = time.perf_counter() + timeout_s
        while True:
            if time.perf_counter() > deadline:
                self.kill()
                raise TimeoutError(f"{self.name} exceeded import timeout")
            if self.process.poll() is not None:
                stderr = self._read_stderr()
                raise PlayerProtocolError(f"Player process exited during import: {stderr}")
            remaining = max(0.0, min(0.05, deadline - time.perf_counter()))
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            line = self.process.stdout.readline()
            if not line:
                continue
            resp = json.loads(line)
            if resp.get("ok") and resp.get("kind") == "ready":
                return
            raise PlayerProtocolError(f"Unexpected startup response: {resp}")

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"kind": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=1.0)
        except Exception:
            self.kill()
        finally:
            if self.stderr_handle is not None:
                try:
                    self.stderr_handle.close()
                except Exception:
                    pass
                self.stderr_handle = None
            self.process = None

    def kill(self) -> None:
        if self.process is None:
            return
        try:
            self.process.kill()
            self.process.wait(timeout=1.0)
        except Exception:
            pass

    def _read_stderr(self) -> str:
        if self.stderr_path is None or not self.stderr_path.exists():
            return ""
        try:
            return self.stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            return ""


if __name__ == "__main__":
    raise SystemExit(_worker_main())
