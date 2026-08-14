# Ogerpon Residual Optimization

This pipeline keeps keidroid's 60-card deck fixed while turning the imitation model into a teacher-initialized policy optimizer.

## Models

- The existing multi-head LightGBM rankers remain the teacher policy.
- `residual_q_model` predicts risk-adjusted counterfactual action return with Huber loss.
- `win_value_model` predicts calibrated game outcome probability for search leaf evaluation.
- Runtime scoring blends teacher, residual Q, tactical rules, and real CG belief-world search.

## Search and data loop

1. Finish the V2 baseline search and freeze its best package.
2. Record full observations only for the focus candidate with `record_mode=training`.
3. Branch critical actions through the official `search_begin/search_step` API over sampled hidden worlds.
4. Train artifact v3 on episode-isolated train/validation/test splits.
5. Calibrate the largest residual weight that preserves at least 80% semantic and 85% ability agreement.
6. Repeat the data-generation loop three times.
7. Optimize blend and search parameters with CEM, then run a strict Kaggle-profile holdout.

## Server launch

```bash
OUT=outputs/ogerpon_residual_$(date +%Y%m%d_%H%M%S) \
WORKERS=48 MODEL_JOBS=12 \
nohup tools/run_ogerpon_residual_server.sh > "$OUT/launcher.log" 2>&1 &
```

The launcher waits for the configured V2 `result.json`, uses `nice -n 19` and `ionice -c 3`, and never submits to Kaggle automatically.

## Promotion gates

- Fixed deck SHA must remain unchanged.
- Semantic agreement must be at least 80%; ability agreement must be at least 85%.
- Runtime failures and invalid actions must be zero.
- CEM optimizes robust Wilson-lower-bound performance and penalizes latency.
- Only strict holdout finalists enter `submission_queue.json`.
