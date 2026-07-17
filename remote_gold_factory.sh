#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/disk/HMZ/Notebook/GW/1/Pokémon}"
cd "$PROJECT_DIR"
mkdir -p outputs/remote_logs outputs/submissions

cmd="${1:-start}"
pid_file="outputs/remote_logs/gold_factory.pid"

case "$cmd" in
  start)
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="outputs/remote_logs/gold_factory_${stamp}.log"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup .venv/bin/python -m tools.gold_factory run \
      --profile "${PROFILE:-a800_gold}" \
      --out "${OUT:-outputs/gold_factory}" \
      --promote "${PROMOTE:-outputs/submissions/gold_factory_champion.tar.gz}" \
      --submission-out "${SUBMISSION_OUT:-outputs/submissions/submission.tar.gz}" \
      --population "${POPULATION:-96}" \
      --workers "${WORKERS:-36}" \
      --max-actions "${MAX_ACTIONS:-1000}" \
      --run-timeout "${RUN_TIMEOUT:-180}" \
      --finalists "${FINALISTS:-5}" \
      --manual-slots "${MANUAL_SLOTS:-5}" \
      --record-sample-rate "${RECORD_SAMPLE_RATE:-0.01}" \
      --strip-search-wrapper \
      --resume \
      > "$log" 2>&1 &
    pid=$!
    echo "$pid" > "$pid_file"
    echo "started pid=$pid log=$log"
    ;;
  status)
    pwd
    if test -f "$pid_file"; then
      pid="$(cat "$pid_file")"
      ps -p "$pid" -o pid,etime,cmd || true
    else
      echo "no pid file"
    fi
    .venv/bin/python -m tools.gold_factory status --out "${OUT:-outputs/gold_factory}" || true
    ls -lh "${PROMOTE:-outputs/submissions/gold_factory_champion.tar.gz}" 2>/dev/null || true
    ls -lh "${SUBMISSION_OUT:-outputs/submissions/submission.tar.gz}" 2>/dev/null || true
    ;;
  logs)
    ls -t outputs/remote_logs/gold_factory_*.log 2>/dev/null | head -1 | xargs -r tail -120
    ;;
  stop)
    if test -f "$pid_file"; then
      kill "$(cat "$pid_file")" 2>/dev/null || true
      rm -f "$pid_file"
      echo stopped
    else
      echo no pid file
    fi
    ;;
  *)
    echo "allowed: start | status | logs | stop" >&2
    exit 2
    ;;
esac
