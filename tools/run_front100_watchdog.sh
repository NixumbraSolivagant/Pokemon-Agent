#!/usr/bin/env bash
set -u

WS=/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace
OUT=${1:-}
LOG=${2:-}
MAX_RESTARTS=${3:-3}

if [ -z "$OUT" ]; then
  OUT_FILE="$WS/outputs/kaggle_logs/top_leaders_20260804/front100_active_out.txt"
  for _ in $(seq 1 60); do
    OUT=$(cat "$OUT_FILE" 2>/dev/null || true)
    if [ -n "$OUT" ]; then
      break
    fi
    sleep 10
  done
fi
if [ -z "$LOG" ]; then
  LOG="$WS/outputs/kaggle_logs/top_leaders_20260804/front100_watchdog.log"
fi
if [ -z "$OUT" ]; then
  echo "[watchdog] no out dir resolved, exiting" >&2
  exit 1
fi

RESTARTS=0
PATTERN="ogerpon_front100_searc[h].*--out ${OUT}"

while :; do
  if [ -f "$OUT/result.json" ]; then
    echo "[watchdog] $(date -u +%H:%M:%S) result.json found, exiting" >> "$LOG"
    exit 0
  fi
  if [ -f "$OUT/audit_summary.json" ] && ! grep -q '"passed": true' "$OUT/audit_summary.json" 2>/dev/null; then
    echo "[watchdog] $(date -u +%H:%M:%S) audit failed (0 passed), not restarting" >> "$LOG"
    exit 0
  fi
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    if [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
      echo "[watchdog] $(date -u +%H:%M:%S) restart limit reached, out=$OUT" >> "$LOG"
      exit 1
    fi
    RESTARTS=$((RESTARTS + 1))
    echo "[watchdog] $(date -u +%H:%M:%S) restart #$RESTARTS out=$OUT" >> "$LOG"
    cd "$WS" || exit 1
    OUT="$OUT" nohup bash tools/run_ogerpon_front100_majkel_server.sh >> "$LOG" 2>&1 < /dev/null &
  fi
  sleep 60
done
