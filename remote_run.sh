#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/home/disk/HMZ/Notebook/GW/1/Pokémon"
cd "$PROJECT_DIR"
mkdir -p outputs/remote_logs outputs/submissions
cmd="${1:-status}"
case "$cmd" in
  status)
    pwd
    df -h .
    nvidia-smi || true
    test -f outputs/auto_iterate_server/generations.csv && tail -20 outputs/auto_iterate_server/generations.csv || true
    ls -lh outputs/submissions/champion_latest.tar.gz 2>/dev/null || true
    ls -lh outputs/submissions/submission.tar.gz 2>/dev/null || true
    ;;
  test)
    .venv/bin/python -m pytest tests/local_eval -q
    .venv/bin/python -m py_compile local_eval/*.py tools/*.py
    ;;
  iterate)
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="outputs/remote_logs/auto_iterate_${stamp}.log"
    nohup .venv/bin/python -m tools.auto_iterate \
      --out outputs/auto_iterate_server \
      --generations "${GENERATIONS:-3}" \
      --population "${POPULATION:-18}" \
      --stage1-games "${STAGE1_GAMES:-3}" \
      --stage2-games "${STAGE2_GAMES:-12}" \
      --workers "${WORKERS:-8}" \
      --record-mode losses \
      --record-sample-rate "${RECORD_SAMPLE_RATE:-0.02}" \
      --promotion-policy balanced \
      --promote outputs/submissions/champion_latest.tar.gz \
      --submission-out outputs/submissions/submission.tar.gz \
      --strip-search-wrapper-for-submission \
      --evaluate-kaggle-safe \
      > "$log" 2>&1 &
    echo $! > outputs/remote_logs/auto_iterate.pid
    echo "started pid=$(cat outputs/remote_logs/auto_iterate.pid) log=$log"
    ;;
  stop)
    if test -f outputs/remote_logs/auto_iterate.pid; then
      kill "$(cat outputs/remote_logs/auto_iterate.pid)" 2>/dev/null || true
      rm -f outputs/remote_logs/auto_iterate.pid
      echo stopped
    else
      echo no pid file
    fi
    ;;
  logs)
    ls -t outputs/remote_logs/auto_iterate_*.log 2>/dev/null | head -1 | xargs -r tail -80
    ;;
  *)
    echo "allowed: status | test | iterate | stop | logs" >&2
    exit 2
    ;;
esac
