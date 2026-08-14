#!/usr/bin/env bash
set -euo pipefail

: "${POOL_PID_FILE:=outputs/opponent_league_20260804/after_majkel.pid}"
: "${POLL_INTERVAL:=60}"
: "${GPU_DEVICES:=1,2}"
: "${PYTHON_BIN:=/home/disk/HMZ/.venv/bin/python}"
: "${PID_FILE:=outputs/kaggle_logs/top_leaders_20260804/majkel_v6_after_pool.pid}"

mkdir -p "$(dirname "$PID_FILE")"
echo "$$" > "$PID_FILE"

POOL_PID="${POOL_PID:-}"
if [ -z "$POOL_PID" ] && [ -f "$POOL_PID_FILE" ]; then
  POOL_PID=$(cat "$POOL_PID_FILE")
fi

if [ -n "$POOL_PID" ]; then
  echo "[majkel-v6] waiting for opponent pool pid=$POOL_PID at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while ps -p "$POOL_PID" -o args= 2>/dev/null | grep -qE 'run_opponent_after_majkel|run_opponent_league_server'; do
    sleep "$POLL_INTERVAL"
  done
fi

if pgrep -f '[t]ools.ogerpon_front100_search.*ogerpon_front100_majkel_v6_gpu12' >/dev/null 2>&1; then
  echo "[majkel-v6] optimized Majkel training already running"
  exit 0
fi

: "${OUT:=outputs/ogerpon_front100_majkel_v6_gpu12_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
PROFILES="balanced deep winsoft regularized softmax"
if ! "$PYTHON_BIN" -m tools.catboost_ranker_gpu_smoke --devices 0:1 --loss QuerySoftMax; then
  PROFILES="balanced deep winsoft regularized"
  echo "[majkel-v6] QuerySoftMax GPU unsupported; continuing with YetiRank profiles"
fi
echo "[majkel-v6] starting out=$OUT gpu=$GPU_DEVICES at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec env \
  OUT="$OUT" \
  GPU_DEVICES="$GPU_DEVICES" \
  PROFILES="$PROFILES" \
  RESUME_INCOMPLETE=0 \
  bash tools/run_ogerpon_front100_majkel_server.sh
