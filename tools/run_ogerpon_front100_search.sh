#!/usr/bin/env bash
set -euo pipefail

: "${OUT:=outputs/ogerpon_front100_search_$(date +%Y%m%d_%H%M%S)}"
: "${WORKERS:=48}"
: "${MODEL_JOBS:=12}"
: "${TRAIN_PARALLEL:=4}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec nice -n 19 ionice -c 3 python3 -m tools.ogerpon_front100_search \
  --replays outputs/kaggle_logs/top_leaders_20260802/keidroid_55153405 \
  --target-name keidroid \
  --train-opponents \
    outputs/meta_imitation/submissions/grim_clone.tar.gz \
    outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz \
    outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz \
    outputs/meta_imitation/submissions/lopunny_clone.tar.gz \
    outputs/meta_imitation/submissions/kang_clone.tar.gz \
    outputs/reference_submissions/multiply-agent-best-940-lb.tar.gz \
  --holdout-opponents \
    outputs/great_tusk_fixed_confirmation/kaggle/fixed_endgame_wall.tar.gz \
    outputs/reference_submissions/pokemon-steel.tar.gz \
    outputs/reference_submissions/improved-probabilistic-agent.tar.gz \
    outputs/reference_submissions/mega-pokemon-reinforcement-ai-battle.tar.gz \
    outputs/reference_submissions/i-have-one-rear-card.tar.gz \
  --out "$OUT" \
  --workers "$WORKERS" \
  --model-jobs "$MODEL_JOBS" \
  --train-parallel "$TRAIN_PARALLEL"
