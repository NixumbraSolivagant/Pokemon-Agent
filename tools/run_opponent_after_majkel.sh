#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:=outputs/opponent_league_20260804}"
: "${GPU_DEVICES:=1,2}"
: "${BUILD_WORKERS:=4}"
: "${MODEL_JOBS:=12}"
: "${POLL_INTERVAL:=60}"
: "${ACTIVE_OUT_FILE:=outputs/kaggle_logs/top_leaders_20260804/front100_active_out.txt}"
: "${WATCHER_PID_FILE:=$ROOT/after_majkel.pid}"
: "${MAJKEL_PROFILES:=balanced,deep,winsoft}"

mkdir -p "$ROOT"
echo "$$" > "$WATCHER_PID_FILE"

if [ -z "${MAJKEL_OUT:-}" ]; then
  MAJKEL_OUT=$(cat "$ACTIVE_OUT_FILE" 2>/dev/null || true)
fi
if [ -z "$MAJKEL_OUT" ]; then
  echo "[after-majkel] no Majkel output directory resolved" >&2
  exit 2
fi

majkel_process_running() {
  local command_line

  if [ -n "${MAJKEL_PID:-}" ]; then
    command_line=$(ps -p "$MAJKEL_PID" -o args= 2>/dev/null || true)
    [[ "$command_line" == *"tools.ogerpon_front100_search"* && "$command_line" == *"--out $MAJKEL_OUT"* ]]
    return
  fi

  while IFS= read -r command_line; do
    if [[ "$command_line" == *"tools.ogerpon_front100_search"* && "$command_line" == *"--out $MAJKEL_OUT"* ]]; then
      return 0
    fi
  done < <(ps -eo args=)
  return 1
}

majkel_artifacts_complete() {
  local profile
  local profiles

  IFS=',' read -r -a profiles <<< "$MAJKEL_PROFILES"
  for profile in "${profiles[@]}"; do
    if [ ! -s "$MAJKEL_OUT/artifacts/$profile.json" ]; then
      return 1
    fi
  done
  return 0
}

echo "[after-majkel] waiting for $MAJKEL_OUT at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while majkel_process_running; do
  sleep "$POLL_INTERVAL"
done

if [ -f "$MAJKEL_OUT/result.json" ]; then
  echo "[after-majkel] Majkel search completed successfully"
elif majkel_artifacts_complete; then
  echo "[after-majkel] Majkel training artifacts complete; search ended before result.json"
else
  echo "[after-majkel] Majkel stopped before all training artifacts completed; opponent builder not started" >&2
  exit 3
fi

if pgrep -f "tools.opponent_league_pipeline build.*--root ${ROOT}" >/dev/null 2>&1; then
  echo "[after-majkel] opponent builder already running for $ROOT"
  exit 0
fi

echo "[after-majkel] Majkel completed; starting opponent builder on GPU $GPU_DEVICES at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec env \
  ROOT="$ROOT" \
  GPU_DEVICES="$GPU_DEVICES" \
  BUILD_WORKERS="$BUILD_WORKERS" \
  MODEL_JOBS="$MODEL_JOBS" \
  bash tools/run_opponent_league_server.sh
