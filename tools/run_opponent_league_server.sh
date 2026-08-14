#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?ROOT is required}"
: "${BUILD_WORKERS:=4}"
: "${MODEL_JOBS:=12}"
: "${GPU_DEVICES:=1,2}"
: "${PYTHON_BIN:=/home/disk/HMZ/.venv/bin/python}"
: "${BUILD_INTERVAL:=300}"
: "${REQUIRED_HISTORY_REPLAYS:=}"

if [ "$GPU_DEVICES" = "auto" ]; then
  GPU_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, - || true)
fi
CATBOOST_DEVICES=""
TASK_TYPE="CPU"
if [ -n "$GPU_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  GPU_COUNT=$(awk -F, '{print NF}' <<< "$GPU_DEVICES")
  CATBOOST_DEVICES=$(seq 0 $((GPU_COUNT - 1)) | paste -sd: -)
  TASK_TYPE="GPU"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

while :; do
  HISTORY_ARGS=()
  if [[ -n "$REQUIRED_HISTORY_REPLAYS" && -d "$REQUIRED_HISTORY_REPLAYS" ]]; then
    HISTORY_ARGS+=(--required-history-replays "$REQUIRED_HISTORY_REPLAYS" --required-history-weight 0.5)
  fi
  nice -n 19 ionice -c 3 "$PYTHON_BIN" -m tools.opponent_league_pipeline build \
    --root "$ROOT" \
    --min-episodes 20 \
    --build-workers "$BUILD_WORKERS" \
    --model-jobs "$MODEL_JOBS" \
    --task-type "$TASK_TYPE" \
    --gpu-devices "$CATBOOST_DEVICES" \
    "${HISTORY_ARGS[@]}"
  if [[ -f "$ROOT/collection_complete.json" ]]; then
    break
  fi
  sleep "$BUILD_INTERVAL"
done

FREEZE_ARGS=(
  --root "$ROOT"
  --target-size 150
  --minimum-size 120
  --minimum-unique-decks 60
  --minimum-unique-behaviors 100
  --minimum-top100 50
  --existing
  outputs/kaggle_manual_submit_20260803/ogerpon_balanced_v2_kaggle.tar.gz
  outputs/kaggle_submissions/20260803_2315/final/ogerpon_balanced_clone_anti_mill_search.tar.gz
  outputs/kaggle_submissions/20260803_2315/final/ogerpon_winsoft_clone_fidelity.tar.gz
)
if [ -n "${GENERATED_PACKAGES:-}" ]; then
  read -r -a GENERATED_ARRAY <<< "$GENERATED_PACKAGES"
  FREEZE_ARGS+=(--generated "${GENERATED_ARRAY[@]}")
fi

nice -n 19 ionice -c 3 "$PYTHON_BIN" -m tools.opponent_league_pipeline freeze \
  "${FREEZE_ARGS[@]}"
