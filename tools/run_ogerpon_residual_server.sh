#!/usr/bin/env bash
set -euo pipefail

: "${V2_OUT:=outputs/ogerpon_front100_v2_20260803_101303}"
: "${OUT:=outputs/ogerpon_residual_$(date +%Y%m%d_%H%M%S)}"
: "${WORKERS:=48}"
: "${MODEL_JOBS:=12}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

until [[ -f "$V2_OUT/result.json" ]]; do
  sleep 60
done

exec nice -n 19 ionice -c 3 python3 -m tools.ogerpon_residual_pipeline \
  --v2-out "$V2_OUT" \
  --replays outputs/kaggle_logs/top_leaders_20260802/keidroid_55153405 \
  --target-name keidroid \
  --train-opponents \
    outputs/meta_imitation/submissions/grim_clone.tar.gz \
    outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz \
    outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz \
    outputs/meta_imitation/submissions/lopunny_clone.tar.gz \
    outputs/meta_imitation/submissions/kang_clone.tar.gz \
    outputs/meta_imitation/submissions/ntuml_clone.tar.gz \
  --dev-opponents \
    outputs/great_tusk_fixed_confirmation/kaggle/fixed_endgame_wall.tar.gz \
    outputs/reference_submissions/pokemon-steel.tar.gz \
    outputs/reference_submissions/mega-pokemon-reinforcement-ai-battle.tar.gz \
    outputs/reference_submissions/pok-mon-ai-battle-challenge-simulation-solution.tar.gz \
    outputs/reference_submissions/pokemon-ai-battle-best-ptcg-advanced.tar.gz \
  --holdout-opponents \
    outputs/reference_submissions/i-have-one-rear-card.tar.gz \
    outputs/reference_submissions/ptcg-mega-lucario-ex-v62.tar.gz \
    outputs/meta_imitation/submissions/grim_clone_search.tar.gz \
    outputs/meta_imitation/submissions/lopunny_timeline_v3.tar.gz \
    outputs/meta_imitation/submissions/kang_clone_search.tar.gz \
  --out "$OUT" \
  --workers "$WORKERS" \
  --model-jobs "$MODEL_JOBS"
