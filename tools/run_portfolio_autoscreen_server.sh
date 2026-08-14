#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-outputs/portfolio_search/20260802_2340}"
mkdir -p "$OUT"
export MPLCONFIGDIR="/tmp/matplotlib-portfolio-${USER}"
mkdir -p "$MPLCONFIGDIR"

taskset -c 0-47 nice -n 19 ionice -c 3 python3 tools/portfolio_autoscreen_candidates.py \
  --artifacts \
    outputs/meta_imitation/grim.json \
    outputs/meta_imitation/kang.json \
    outputs/meta_imitation/lopunny.json \
    outputs/meta_imitation/lopunny_temporal_v2.json \
    outputs/meta_imitation/ntuml.json \
  --train-manifests \
    configs/meta_imitation_ogerpon_majkel_private.json \
    configs/meta_imitation_ogerpon_keidroid_private.json \
    configs/meta_imitation_ogerpon_combined_private.json \
  --existing \
    outputs/submissions/*.tar.gz \
    outputs/great_tusk_gold/kaggle/*.tar.gz \
    outputs/great_tusk_fixed_confirmation/kaggle/*.tar.gz \
    outputs/meta_imitation/submissions/*.tar.gz \
    outputs/reference_submissions/*.tar.gz \
  --out "$OUT/generated" \
  --n-jobs 12

taskset -c 0-47 nice -n 19 ionice -c 3 python3 tools/server_gold_autoscreen.py \
  --candidates "$OUT/generated/candidates/*.tar.gz" \
  --train \
    outputs/meta_imitation/grim_archaludon_fix_20260802/candidates/grim_clone.tar.gz \
    outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz \
    outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz \
    outputs/meta_imitation/submissions/lopunny_clone.tar.gz \
    outputs/meta_imitation/submissions/kang_clone.tar.gz \
    outputs/reference_submissions/multiply-agent-best-940-lb.tar.gz \
  --holdout \
    outputs/great_tusk_fixed_confirmation/kaggle/fixed_endgame_wall.tar.gz \
    outputs/reference_submissions/pokemon-steel.tar.gz \
    outputs/reference_submissions/improved-probabilistic-agent.tar.gz \
    outputs/reference_submissions/mega-pokemon-reinforcement-ai-battle.tar.gz \
    outputs/reference_submissions/i-have-one-rear-card.tar.gz \
    outputs/reference_submissions/pokemon-ai-battle-best-ptcg-advanced.tar.gz \
  --out "$OUT/screen" \
  --workers 48 \
  --seed 2026080240 \
  --stage1-games 2 \
  --stage2-games 8 \
  --holdout-games 24 \
  --stage2-count 32 \
  --holdout-count 12
