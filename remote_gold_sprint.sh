#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/disk/HMZ/Notebook/GW/1/Pokémon}"
cd "$PROJECT_DIR"
mkdir -p outputs/remote_logs outputs/submissions

cmd="${1:-start}"
pid_file="outputs/remote_logs/gold_sprint.pid"

case "$cmd" in
  start)
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="outputs/remote_logs/gold_sprint_${stamp}.log"
    cross_arg=()
    if [[ "${ALLOW_CROSS_ARCHETYPE:-0}" == "1" ]]; then
      cross_arg=(--allow-cross-archetype-promotion)
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup .venv/bin/python -m tools.gold_sprint \
      --out "${OUT:-outputs/gold_sprint}" \
      --state "${STATE:-outputs/gold_sprint/state.json}" \
      --source-state "${SOURCE_STATE:-outputs/auto_iterate_server_gold/state.json}" \
      --promote "${PROMOTE:-outputs/submissions/champion_latest.tar.gz}" \
      --submission-out "${SUBMISSION_OUT:-outputs/submissions/submission.tar.gz}" \
      --strip-search-wrapper-for-submission \
      --evaluate-kaggle-safe \
      --cycles "${CYCLES:-8}" \
      --generations-per-cycle "${GENERATIONS_PER_CYCLE:-2}" \
      --population "${POPULATION:-56}" \
      --finalists "${FINALISTS:-10}" \
      --stage1-games "${STAGE1_GAMES:-3}" \
      --stage2-games "${STAGE2_GAMES:-48}" \
      --verify-games "${VERIFY_GAMES:-96}" \
      --final-games "${FINAL_GAMES:-192}" \
      --verify-candidates "${VERIFY_CANDIDATES:-12}" \
      --workers "${WORKERS:-36}" \
      --run-timeout "${RUN_TIMEOUT:-180}" \
      --max-actions "${MAX_ACTIONS:-1000}" \
      --max-no-result-rate "${MAX_NO_RESULT_RATE:-0.02}" \
      --record-mode losses \
      --record-sample-rate "${RECORD_SAMPLE_RATE:-0.01}" \
      "${cross_arg[@]}" \
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
    out="${OUT:-outputs/gold_sprint}"
    promote="${PROMOTE:-outputs/submissions/champion_latest.tar.gz}"
    submission_out="${SUBMISSION_OUT:-outputs/submissions/submission.tar.gz}"
    test -f "$out/sprint_summary.csv" && tail -20 "$out/sprint_summary.csv" || true
    test -f "$out/final_report.json" && .venv/bin/python -m json.tool "$out/final_report.json" | tail -80 || true
    ls -lh "$promote" 2>/dev/null || true
    ls -lh "$submission_out" 2>/dev/null || true
    ;;
  logs)
    ls -t outputs/remote_logs/gold_sprint_*.log 2>/dev/null | head -1 | xargs -r tail -120
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
