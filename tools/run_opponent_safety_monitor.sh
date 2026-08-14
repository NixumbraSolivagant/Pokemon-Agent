#!/usr/bin/env bash
set -u

WS=/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace
STATE="$WS/outputs/opponent_league_20260804/collect_state.json"
PID_FILE=/tmp/opponent_collector_pid.txt
LOG=/tmp/opponent_safety.log
PREV=0

while :; do
  sleep 120
  N=$(python3 -c "import json;print(len(json.load(open('$STATE')).get('failures',{})))" 2>/dev/null || echo 0)
  if [ "$N" -gt $((PREV + 3)) ]; then
    echo "[opp-safety] $(date -u +%H:%M:%S) failure jump prev=$PREV now=$N -> pause 30m" >> "$LOG"
    kill -STOP "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null
    sleep 1800
    kill -CONT "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null
    echo "[opp-safety] $(date -u +%H:%M:%S) resumed" >> "$LOG"
  fi
  PREV=$N
done
