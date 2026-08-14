#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:=outputs/opponent_league_20260804}"
: "${GPU_DEVICES:=1,2}"
: "${PYTHON_BIN:=/home/disk/HMZ/.venv/bin/python}"
: "${PILOT_SIZE:=8}"
: "${PID_FILE:=$ROOT/pilot_v3_gpu12.pid}"

mkdir -p "$ROOT"
echo "$$" > "$PID_FILE"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec nice -n 19 ionice -c 3 "$PYTHON_BIN" -m tools.opponent_league_pipeline build \
  --root "$ROOT" \
  --min-episodes 20 \
  --min-decisions 1000 \
  --build-workers 4 \
  --model-jobs 12 \
  --task-type GPU \
  --gpu-devices 0:1 \
  --force-rebuild \
  --pilot-size "$PILOT_SIZE"
