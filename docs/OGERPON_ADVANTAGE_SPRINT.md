# Ogerpon Champion-Gated Advantage Sprint

This pipeline keeps `ogerpon_cem_g00_027` as an immutable anchor and only permits an alternative action when an ensemble predicts a positive lower-confidence-bound advantage.

## Safety invariants

- Artifact version 3 behavior remains unchanged.
- Forced actions and multi-select decisions never pass through the advantage router.
- ATTACK and END overrides are disabled by default.
- Missing models, feature errors, excessive ensemble disagreement, or failed calibration return the anchor action.
- Training, validation, and test splits are isolated by opponent group rather than episode.
- If validation cannot identify positive-return overrides at the requested coverage, calibration selects zero overrides.

## Server command

```bash
python -m tools.ogerpon_advantage_sprint \
  --base-artifact outputs/ogerpon_residual_20260803_144127/iteration_02/residual_artifact.json \
  --anchor-package outputs/ogerpon_residual_20260803_144127/cem/finalists/ogerpon_cem_g00_027.tar.gz \
  --records outputs/rl_sprint_20260814/training/iteration_00/training_games/game_records \
  --focus-name anchor_8689 \
  --manifest outputs/rl_sprint_20260814/pool/dev.json \
  --out outputs/advantage_sprint_20260814/round0 \
  --workers 64 --data-workers 48 --gpu-device 0 --games 32 --max-states 20000
```

The command only creates local artifacts and evaluation reports. It does not call the Kaggle submission API.
