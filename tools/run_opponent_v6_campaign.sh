#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:=outputs/opponent_league_20260804}"
: "${GPU_DEVICES:=0}"
: "${PYTHON_BIN:=/home/disk/HMZ/.venv/bin/python}"
: "${POLL_INTERVAL:=60}"
: "${PILOT1_PID_FILE:=$ROOT/pilot_v6_gpu0.pid}"
: "${PID_FILE:=$ROOT/v6_campaign.pid}"
: "${LOG_DIR:=$ROOT/logs}"
: "${V2_OUT:=outputs/ogerpon_front100_v2_20260803_101303}"
: "${OGERPON_REPLAYS:=outputs/kaggle_logs/top_leaders_20260802/keidroid_55153405}"
: "${OGERPON_OUT:=$ROOT/ogerpon_optimization_v6}"

mkdir -p "$ROOT" "$LOG_DIR"
echo "$$" > "$PID_FILE"

pid=$(cat "$PILOT1_PID_FILE")
echo "[campaign-v6] waiting for Majkel pilot pid=$pid"
while kill -0 "$pid" 2>/dev/null; do sleep "$POLL_INTERVAL"; done

"$PYTHON_BIN" - "$ROOT/pilot_report.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
report = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
if int(report.get("clone_algorithm_version", 0)) != 6 or int(report.get("selected", 0)) != 1:
    raise SystemExit(f"invalid v6 Majkel pilot report: {report}")
if not report.get("passed") or not report.get("required_team_qualified"):
    raise SystemExit(f"v6 Majkel pilot failed: {report}")
print(f"[campaign-v6] Majkel passed macro={report['turn_macro_median']:.4f} attack={report['attack_median']:.4f}")
PY
touch "$ROOT/pilot_v6_majkel.passed"

env ROOT="$ROOT" GPU_DEVICES="$GPU_DEVICES" PYTHON_BIN="$PYTHON_BIN" PILOT_SIZE=8 \
  PID_FILE="$ROOT/pilot_v6_diverse.pid" \
  bash tools/run_opponent_pilot_v6.sh > "$LOG_DIR/pilot_v6_diverse.log" 2>&1

"$PYTHON_BIN" - "$ROOT/pilot_report.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(report.get("clone_algorithm_version", 0)) != 6 or int(report.get("selected", 0)) != 8 or not report.get("passed"):
    raise SystemExit(f"v6 diversity pilot failed: {report}")
PY
touch "$ROOT/pilot_v6_diverse.passed"

env ROOT="$ROOT" GPU_DEVICES="$GPU_DEVICES" PYTHON_BIN="$PYTHON_BIN" BUILD_WORKERS=1 MODEL_JOBS=12 \
  bash tools/run_opponent_league_server.sh > "$LOG_DIR/full_pool_v6.log" 2>&1

[[ -s "$ROOT/pool/summary.json" ]]
touch "$ROOT/v6_campaign.complete"

env ROOT="$ROOT" V2_OUT="$V2_OUT" REPLAYS="$OGERPON_REPLAYS" OUT="$OGERPON_OUT" \
  WORKERS=48 MODEL_JOBS=12 GPU_DEVICE=0 \
  bash tools/run_opponent_league_ogerpon.sh > "$LOG_DIR/ogerpon_optimization_v6.log" 2>&1

[[ -s "$OGERPON_OUT/final/best_submission.tar.gz" ]]
touch "$ROOT/v6_submission.complete"
