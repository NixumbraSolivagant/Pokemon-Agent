#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:=outputs/opponent_league_20260804}"
: "${GPU_DEVICES:=1,2}"
: "${PYTHON_BIN:=/home/disk/HMZ/.venv/bin/python}"
: "${POLL_INTERVAL:=60}"
: "${PILOT1_PID_FILE:=$ROOT/pilot_v4_gpu12.pid}"
: "${PID_FILE:=$ROOT/v4_campaign.pid}"
: "${LOG_DIR:=$ROOT/logs}"

mkdir -p "$ROOT" "$LOG_DIR"
echo "$$" > "$PID_FILE"

wait_for_pid_file() {
  local pid_file=$1
  local label=$2
  local pid

  if [[ ! -s "$pid_file" ]]; then
    echo "[campaign] missing $label pid file: $pid_file" >&2
    exit 2
  fi
  pid=$(cat "$pid_file")
  echo "[campaign] waiting for $label pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_INTERVAL"
  done
}

validate_pilot() {
  local expected_selected=$1
  "$PYTHON_BIN" - "$ROOT/pilot_report.json" "$expected_selected" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
expected_selected = int(sys.argv[2])
if not report_path.is_file():
    raise SystemExit(f"missing pilot report: {report_path}")
report = json.loads(report_path.read_text(encoding="utf-8"))
checks = {
    "clone_algorithm_version": int(report.get("clone_algorithm_version", 0)) == 4,
    "selected": int(report.get("selected", 0)) == expected_selected,
    "passed": bool(report.get("passed")),
    "required_team_qualified": bool(report.get("required_team_qualified")),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"pilot gate failed checks={failed} report={report}")
print(
    "[campaign] pilot passed "
    f"selected={report['selected']} qualified={report['qualified']} "
    f"turn_macro={report['turn_macro_median']:.4f} attack={report['attack_median']:.4f}"
)
PY
}

wait_for_pid_file "$PILOT1_PID_FILE" "Majkel pilot"
validate_pilot 1
touch "$ROOT/pilot_v4_majkel.passed"

echo "[campaign] starting diverse pilot on physical GPUs $GPU_DEVICES"
env \
  ROOT="$ROOT" \
  GPU_DEVICES="$GPU_DEVICES" \
  PYTHON_BIN="$PYTHON_BIN" \
  PILOT_SIZE=8 \
  PID_FILE="$ROOT/pilot_v4_diverse.pid" \
  bash tools/run_opponent_pilot_v4.sh \
  > "$LOG_DIR/pilot_v4_diverse.log" 2>&1
validate_pilot 8
touch "$ROOT/pilot_v4_diverse.passed"

echo "[campaign] starting incremental full-pool build"
env \
  ROOT="$ROOT" \
  GPU_DEVICES="$GPU_DEVICES" \
  PYTHON_BIN="$PYTHON_BIN" \
  BUILD_WORKERS=1 \
  MODEL_JOBS=12 \
  bash tools/run_opponent_league_server.sh \
  > "$LOG_DIR/full_pool_v4.log" 2>&1

if [[ ! -s "$ROOT/pool/summary.json" ]]; then
  echo "[campaign] full-pool build ended without pool summary" >&2
  exit 3
fi
touch "$ROOT/v4_campaign.complete"
echo "[campaign] complete pool_summary=$ROOT/pool/summary.json"
