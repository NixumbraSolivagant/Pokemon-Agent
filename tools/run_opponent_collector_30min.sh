#!/usr/bin/env bash
set -u

WS=/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace
cd "$WS" || exit 1
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN is required}"

setsid nohup nice -n 19 ionice -c 3 python3 -m tools.opponent_league_pipeline collect \
  --root outputs/opponent_league_20260804 \
  --kaggle "$WS/.tools/uv/uvx --python 3.12 --from kaggle==2.2.4 kaggle" \
  --workers 2 \
  --checkpoint-every 50 \
  --request-interval 1.55 \
  --request-jitter 0.0 \
  --max-retries 5 \
  --max-backoff 300 \
  >> "$WS/outputs/opponent_league_20260804/collector_server.log" 2>&1 < /dev/null &

echo "collector pid=$!"
