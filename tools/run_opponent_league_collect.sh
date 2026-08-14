#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?ROOT is required}"
: "${KAGGLE_CMD:?KAGGLE_CMD is required}"
: "${REMOTE:?REMOTE is required}"
: "${REMOTE_ROOT:?REMOTE_ROOT is required}"
: "${SSH_PORT:=22}"
: "${COMMAND:=collect}"
: "${WORKERS:=2}"
: "${REQUEST_INTERVAL:=1.55}"
: "${REQUEST_JITTER:=0.0}"
: "${MAX_RETRIES:=5}"
: "${MAX_BACKOFF:=300}"

exec nice -n 19 ionice -c 3 python3 -m tools.opponent_league_pipeline "$COMMAND" \
  --root "$ROOT" \
  --kaggle "$KAGGLE_CMD" \
  --workers "$WORKERS" \
  --checkpoint-every 50 \
  --request-interval "$REQUEST_INTERVAL" \
  --request-jitter "$REQUEST_JITTER" \
  --max-retries "$MAX_RETRIES" \
  --max-backoff "$MAX_BACKOFF" \
  --remote "$REMOTE" \
  --remote-root "$REMOTE_ROOT" \
  --ssh-port "$SSH_PORT"
