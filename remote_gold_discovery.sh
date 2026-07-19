#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/disk/HMZ/Notebook/GW/1/Pokémon}"
cd "$PROJECT_DIR"
mkdir -p outputs/remote_logs outputs/submissions

cmd="${1:-start}"
pid_file="outputs/remote_logs/gold_discovery.pid"

case "$cmd" in
  start)
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="outputs/remote_logs/gold_discovery_${stamp}.log"
    nohup .venv/bin/python -m tools.discovery_engine run \
      --profile "${PROFILE:-a800_turbo_discovery}" \
      --out "${OUT:-outputs/gold_discovery}" \
      --incumbent "${INCUMBENT:-outputs/submissions/champion_latest.tar.gz}" \
      --promote "${PROMOTE:-outputs/submissions/gold_probe_candidate.tar.gz}" \
      --submission-out "${SUBMISSION_OUT:-outputs/submissions/submission_gold_probe.tar.gz}" \
      --workers "${WORKERS:-30}" \
      --build-workers "${BUILD_WORKERS:-30}" \
      --cpu-headroom "${CPU_HEADROOM:-0}" \
      --max-actions "${MAX_ACTIONS:-1000}" \
      --run-timeout "${RUN_TIMEOUT:-180}" \
      --progress-mode file \
      --progress-interval "${PROGRESS_INTERVAL:-5}" \
      --gold-gate \
      --include-portfolio \
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
      ps -p "$pid" -o pid,etime,pcpu,pmem,cmd || true
    else
      echo "no pid file"
    fi
    out="${OUT:-outputs/gold_discovery}"
    .venv/bin/python -m tools.discovery_engine status --out "$out" || true
    test -f "$out/decision_brief.md" && sed -n '1,120p' "$out/decision_brief.md" || true
    ls -lh "${PROMOTE:-outputs/submissions/gold_probe_candidate.tar.gz}" 2>/dev/null || true
    ls -lh "${SUBMISSION_OUT:-outputs/submissions/submission_gold_probe.tar.gz}" 2>/dev/null || true
    ;;
  logs)
    ls -t outputs/remote_logs/gold_discovery_*.log 2>/dev/null | head -1 | xargs -r tail -120
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
