#!/usr/bin/env bash
set -euo pipefail

: "${OUT:=outputs/ogerpon_front100_v2_$(date +%Y%m%d_%H%M%S)}"
: "${WORKERS:=48}"
: "${MODEL_JOBS:=12}"
: "${TRAIN_PARALLEL:=3}"

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
  --train-parallel "$TRAIN_PARALLEL" \
  --profiles balanced deep winsoft \
  --variants \
    clone_fidelity \
    clone_search \
    clone_combat \
    clone_combat_search \
    clone_anti_mill \
    clone_anti_mill_search \
  --stage1-games 8 \
  --stage2-games 24 \
  --holdout-games 64 \
  --stage2-count 10 \
  --holdout-count 5
