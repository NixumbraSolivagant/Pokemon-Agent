#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:=outputs/opponent_league_20260804}"
: "${GPU_DEVICES:=0}"
: "${PYTHON_BIN:=/home/disk/HMZ/.venv/bin/python}"
: "${PILOT_SIZE:=1}"
: "${PID_FILE:=$ROOT/pilot_v6_gpu0.pid}"
: "${STRENGTH_BASELINE:=$ROOT/baselines/majkel_v3.tar.gz}"

mkdir -p "$ROOT"
echo "$$" > "$PID_FILE"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

strength_args=()
if [[ -f "$STRENGTH_BASELINE" ]]; then
  strength_args+=(--strength-baseline "$STRENGTH_BASELINE" --strength-games 512 --strength-workers 16)
fi

exec nice -n 19 ionice -c 3 "$PYTHON_BIN" -m tools.opponent_league_pipeline build \
  --root "$ROOT" \
  --min-episodes 20 \
  --min-decisions 1000 \
  --build-workers 1 \
  --model-jobs 12 \
  --task-type GPU \
  --gpu-devices 0 \
  --force-rebuild \
  --disable-shared-pretrain \
  --pilot-size "$PILOT_SIZE" \
  --required-team Majkel1337 \
  "${strength_args[@]}"
