#!/usr/bin/env bash
set -euo pipefail

: "${WORKERS:=48}"
: "${MODEL_JOBS:=12}"
: "${PARSE_WORKERS:=16}"
: "${GPU_DEVICES:=1,2}"
: "${PYTHON_BIN:=/home/disk/HMZ/.venv/bin/python}"
: "${TRAIN_PARALLEL:=3}"
: "${GPU_TRAIN_CONCURRENCY:=1}"
: "${SKIP_FROZEN_CHECK:=0}"
: "${SEED_REPEATS:=3}"
: "${STAGE1_GAMES:=16}"
: "${STAGE2_GAMES:=48}"
: "${HOLDOUT_GAMES:=128}"
: "${PROFILES:=balanced deep winsoft regularized softmax}"
: "${RESUME_INCOMPLETE:=0}"
: "${REPLAYS:?Set REPLAYS to an explicit frozen Majkel replay directory}"
: "${FROZEN_DATASET:?Set FROZEN_DATASET to the matching immutable replay manifest}"
: "${TARGET_NAME:=Majkel1337}"
: "${SPLIT_SEED:=20260813}"

if [ "$SKIP_FROZEN_CHECK" != "1" ]; then
  "$PYTHON_BIN" -m tools.freeze_replay_dataset \
    --replays "$REPLAYS" \
    --target-name "$TARGET_NAME" \
    --out "$FROZEN_DATASET" \
    --seed "$SPLIT_SEED" \
    --parse-workers "$PARSE_WORKERS" \
    --check
else
  echo "[launcher] frozen dataset check skipped after prior successful verification" >&2
fi

if [ "${GPU_DEVICES:-auto}" = "auto" ]; then
  GPU_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, - || true)
fi
GPU_DEVICES="${GPU_DEVICES:-}"
CATBOOST_DEVICES=""
if [ -n "$GPU_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  GPU_COUNT=$(awk -F, '{print NF}' <<< "$GPU_DEVICES")
  CATBOOST_DEVICES=$(seq 0 $((GPU_COUNT - 1)) | paste -sd: -)
fi

if [ -z "${OUT:-}" ]; then
  OUT=""
  if [ "$RESUME_INCOMPLETE" = "1" ]; then
    for candidate in $(ls -dt outputs/ogerpon_front100_v2_majkel_* 2>/dev/null || true); do
      if [ -f "$candidate/result.json" ]; then
        continue
      fi
      if [ -d "$candidate/artifacts" ] && ( ls "$candidate/artifacts/"*_parts >/dev/null 2>&1 || [ -d "$candidate/cache" ] ); then
        OUT="$candidate"
        echo "[launcher] resuming incomplete run: $OUT" >&2
        break
      fi
    done
  fi
  OUT="${OUT:-outputs/ogerpon_front100_v2_majkel_$(date +%Y%m%d_%H%M%S)}"
fi
mkdir -p "$(dirname "$OUT")"
echo "$OUT" > outputs/kaggle_logs/top_leaders_20260804/front100_active_out.txt
echo "[launcher] out=$OUT visible_gpus=${GPU_DEVICES:-cpu} catboost_devices=${CATBOOST_DEVICES:-cpu} at $(date -u +%H:%M:%S)" >&2

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CATBOOST_GPU_CONCURRENCY="$GPU_TRAIN_CONCURRENCY"
read -r -a PROFILE_ARRAY <<< "$PROFILES"

exec nice -n 19 ionice -c 3 "$PYTHON_BIN" -m tools.ogerpon_front100_search \
  --replays "$REPLAYS" \
  --target-name "$TARGET_NAME" \
  --frozen-dataset "$FROZEN_DATASET" \
  --split-seed "$SPLIT_SEED" \
  --train-opponents \
    outputs/meta_imitation/submissions/grim_clone.tar.gz \
    outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz \
    outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz \
    outputs/meta_imitation/submissions/lopunny_clone.tar.gz \
    outputs/meta_imitation/submissions/kang_clone.tar.gz \
    outputs/meta_imitation/submissions/ntuml_clone.tar.gz \
  --holdout-opponents \
    outputs/great_tusk_fixed_confirmation/kaggle/fixed_endgame_wall.tar.gz \
    outputs/reference_submissions/pokemon-steel.tar.gz \
    outputs/reference_submissions/mega-pokemon-reinforcement-ai-battle.tar.gz \
    outputs/reference_submissions/pok-mon-ai-battle-challenge-simulation-solution.tar.gz \
    outputs/reference_submissions/pokemon-ai-battle-best-ptcg-advanced.tar.gz \
  --out "$OUT" \
  --workers "$WORKERS" \
  --model-jobs "$MODEL_JOBS" \
  --parse-workers "$PARSE_WORKERS" \
  --gpu-devices "$CATBOOST_DEVICES" \
  --train-parallel "$TRAIN_PARALLEL" \
  --profiles "${PROFILE_ARRAY[@]}" \
  --seed-repeats "$SEED_REPEATS" \
  --variants \
    clone_fidelity \
    clone_search \
    clone_combat \
    clone_combat_search \
    clone_anti_mill \
    clone_anti_mill_search \
  --stage1-games "$STAGE1_GAMES" \
  --stage2-games "$STAGE2_GAMES" \
  --holdout-games "$HOLDOUT_GAMES" \
  --stage2-count 10 \
  --holdout-count 5
