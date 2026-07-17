#!/usr/bin/env bash
set -euo pipefail
cd "/home/disk/HMZ/Notebook/GW/1/Pokémon"
mkdir -p outputs/remote_logs outputs/submissions
stamp="$(date +%Y%m%d_%H%M%S)"
log="outputs/remote_logs/gold_auto_iterate_${stamp}.log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup .venv/bin/python -m tools.auto_iterate \
  --out outputs/auto_iterate_server_gold \
  --generations "${GENERATIONS:-5}" \
  --population "${POPULATION:-48}" \
  --finalists "${FINALISTS:-8}" \
  --stage1-games "${STAGE1_GAMES:-2}" \
  --stage2-games "${STAGE2_GAMES:-40}" \
  --workers "${WORKERS:-36}" \
  --record-mode losses \
  --record-sample-rate "${RECORD_SAMPLE_RATE:-0.01}" \
  --promotion-policy balanced \
  --allow-cross-archetype-promotion \
  --hof-size "${HOF_SIZE:-8}" \
  --promote outputs/submissions/champion_latest.tar.gz \
  --submission-out outputs/submissions/submission.tar.gz \
  --strip-search-wrapper-for-submission \
  --evaluate-kaggle-safe \
  > "$log" 2>&1 &
pid=$!
echo "$pid" > outputs/remote_logs/gold_auto_iterate.pid
echo "started pid=$pid log=$log"
