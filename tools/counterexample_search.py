from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_counterexamples(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    result: list[dict[str, Any]] = []
    for item in data.get("pressure_items", []) or []:
        if isinstance(item, dict):
            result.append(item)
    for item in data.get("failure_fingerprints", []) or []:
        if isinstance(item, dict):
            result.append(item)
    return result


def failure_bias(counterexamples: list[dict[str, Any]]) -> dict[str, float]:
    bias: dict[str, float] = {}
    for item in counterexamples:
        text = json.dumps(item, ensure_ascii=False).lower()
        if "deck" in text or "mill" in text:
            bias["resource_safety"] = bias.get("resource_safety", 0.0) + 1.0
        if "prize" in text or "ko" in text or "active" in text:
            bias["prize_delta"] = bias.get("prize_delta", 0.0) + 1.0
        if "switch" in text or "pivot" in text:
            bias["switch_priority"] = bias.get("switch_priority", 0.0) + 1.0
        if "tool" in text or "wall" in text:
            bias["wall_priority"] = bias.get("wall_priority", 0.0) + 1.0
    return bias
