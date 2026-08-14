#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?ROOT is required}"
: "${V2_OUT:?V2_OUT is required}"
: "${REPLAYS:?REPLAYS is required}"
: "${OUT:=$ROOT/ogerpon_optimization}"
: "${WORKERS:=48}"
: "${MODEL_JOBS:=12}"
: "${TARGET_NAME:=keidroid}"
: "${MINIMUM_POOL_SIZE:=80}"
: "${GPU_DEVICE:=2}"
: "${TRAINING_GAMES:=32}"
: "${HOLDOUT_GAMES:=128}"
: "${EXPECTED_DECK_SHA256:=050d87c524f3258cd84ad8d4e4ebdbacf89b935b9ab0673f62f9a2605a4405a3}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

until [[ -f "$ROOT/pool/summary.json" ]]; do
  sleep 60
done

if [[ -f "$OUT/result.json" && -f "$OUT/submission_queue.json" ]]; then
  exit 0
fi

exec nice -n 19 ionice -c 3 python3 -m tools.ogerpon_residual_pipeline \
  --v2-out "$V2_OUT" \
  --replays "$REPLAYS" \
  --target-name "$TARGET_NAME" \
  --train-manifest "$ROOT/pool/train.json" \
  --dev-manifest "$ROOT/pool/dev.json" \
  --holdout-manifest "$ROOT/pool/holdout.json" \
  --pool-summary "$ROOT/pool/summary.json" \
  --minimum-pool-size "$MINIMUM_POOL_SIZE" \
  --expected-deck-sha256 "$EXPECTED_DECK_SHA256" \
  --out "$OUT" \
  --training-games "$TRAINING_GAMES" \
  --holdout-games "$HOLDOUT_GAMES" \
  --workers "$WORKERS" \
  --model-jobs "$MODEL_JOBS" \
  --gpu-device "$GPU_DEVICE"
