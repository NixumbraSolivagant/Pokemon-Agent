#!/usr/bin/env bash
set -u

WORKSPACE=/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace
DL_LOG="$WORKSPACE/outputs/kaggle_logs/top_leaders_20260804/majkel_download.log"
PID_FILE=/tmp/opponent_collector_pid.txt

while :; do
  if grep -q '^\[done\]' "$DL_LOG" 2>/dev/null; then
    if [[ -f "$PID_FILE" ]]; then
      PID=$(cat "$PID_FILE")
      if kill -0 "$PID" 2>/dev/null; then
        kill -CONT "$PID"
        echo "[resume-opponent] $(date -u +%H:%M:%S) resumed collector pid=$PID after majkel download done"
      else
        echo "[resume-opponent] $(date -u +%H:%M:%S) collector pid=$PID is dead; manual restart needed"
      fi
    else
      echo "[resume-opponent] $(date -u +%H:%M:%S) pid file missing; manual restart needed"
    fi
    break
  fi
  sleep 60
done
