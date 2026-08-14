# Opponent League

The opponent league builds a leakage-safe, replay-derived pool for front-100 optimization.

## Design

- Snapshot 500 teams: all top 100, 150 from ranks 101–500, 120 from 501–1500, 80 from 1501–3000, and 50 from the long tail.
- Download only completed public simulation episodes.
- Train arbitrary decks with the `generic_replay` family and teacher-derived fallback priorities.
- Pretrain shared rankers on non-target sources, then fit source-specific GPU residuals from the shared score baseline.
- Admit clones with turn-level fidelity gates: MAIN macro recall at least 80%, ability at least 85%, attack at least 85%, terminal recall at least 70% when covered, median turn action Jaccard at least 75%, and zero audit errors.
- Keep the legacy 88% single-step semantic metric as a diagnostic instead of weakening or using it as the sole admission criterion.
- When `--strength-baseline` is supplied, also require a failure-free paired local evaluation with score at least 55% and Wilson lower bound at least 48%.
- Prefer 150 opponents but never lower quality gates to fill the pool; the minimum frozen pool is 120.
- Require at least 60 unique decks, 100 behavior signatures, and 50 top-100 sources before freezing.
- Freeze train/dev/holdout manifests and never use the permanent holdout for training or CEM tuning.

## Collection

The Kaggle access token remains on the local machine. Collection uses a bounded staging directory, compresses each replay, synchronizes checkpoints to the server, and deletes local replay payloads after a successful remote sync.

```bash
export SSHPASS='...'
ROOT=outputs/opponent_league_20260804 \
KAGGLE_CMD="/tmp/uv-kaggle/uvx --python 3.12 --from kaggle==2.2.4 kaggle" \
REMOTE=runner@124.16.75.152 \
REMOTE_ROOT='/home/disk/HMZ/Notebook/GW/1/Pokémon/outputs/ogerpon_front100_workspace/outputs/opponent_league_20260804' \
SSH_PORT=50030 \
tools/run_opponent_league_collect.sh
```

The `update` command archives the previous snapshot, obtains a fresh leaderboard snapshot, and resumes collection from `collect_state.json`.

Replay collection uses a shared global pacer with a minimum 1.55-second interval, limiting the process to fewer than 40 requests per minute even with two workers. Authentication failures stop collection; rate-limit responses wait and retry indefinitely with shared exponential cooldown.

## Server Build

```bash
ROOT=outputs/opponent_league_20260804 \
BUILD_WORKERS=4 MODEL_JOBS=12 \
tools/run_opponent_league_server.sh
```

Run the Majkel-first v4 gate on physical GPUs 1 and 2 with:

```bash
PILOT_SIZE=1 tools/run_opponent_pilot_v4.sh
```

The v4 launcher preserves the previous Majkel package as the strength baseline, pretrains on 24 non-target sources, fine-tunes Majkel with GPU baseline residuals, and evaluates 256 paired games before qualification.

After the one-source Majkel Pilot is running, start the resumable campaign watcher:

```bash
nohup bash tools/run_opponent_v4_campaign.sh \
  > outputs/opponent_league_20260804/v4_campaign.log 2>&1 &
```

It waits for the Majkel report, requires the v4 teacher and strength gates to pass, runs an eight-source diversity Pilot, and only then starts the incremental full-pool builder. The full build uses physical GPUs 1 and 2 with one CatBoost build at a time to avoid GPU oversubscription, continues while the rate-limited collector runs, and freezes the pool only after collection is complete.

The v5 replacement uses artifact version 6 and clone algorithm version 5. It adds turn-history features, candidate-card set features, and a hard `stop -> type -> option` MAIN router.

> **Known failure — do not retrain or promote this router unchanged.** The 2026-08-12 `majkel_fused_319_gpu0_v6` run proved that type balancing followed by stage balancing inflated terminal weighted mass from a real 15.36% prior to 47.98%. Combined with hard `-inf` routing, the model over-selected ATTACK/END and scored only 24-40 against the previous Majkel v3 baseline. See [Majkel 行为克隆失败复盘](MAJKEL_CLONE_FAILURE_20260812.md). Fix the stop-head weighting, calibrated routing, runtime-equivalent evaluation, and closed-loop audit before running `tools/run_opponent_pilot_v5.sh` or `tools/run_opponent_v5_campaign.sh` again. Do not lower admission gates.

After those fixes, the campaign is intended to require a 512-game Majkel strength gate, run the diversity Pilot and full pool, then launch Ogerpon optimization and write the selected package to `ogerpon_optimization_v5/final/best_submission.tar.gz`. It never submits to Kaggle.

The server launcher runs the collector and GPU CatBoost builder as an incremental producer-consumer loop. A source becomes trainable at 20 episodes, so package construction no longer waits for the entire collection. The current server default is physical GPUs 1 and 2; override with `GPU_DEVICES=...` when needed. It never submits to Kaggle.

Generated PSRO or counterexample packages can be added to the specialist pool with `GENERATED_PACKAGES='path1.tar.gz path2.tar.gz'`. They remain subject to deck and behavior deduplication.

## Automatic Ogerpon Optimization

`tools/run_opponent_league_ogerpon.sh` waits for a successful pool freeze, validates at least 80 unique opponents, rejects train/dev/holdout overlap, verifies the fixed keidroid Ogerpon deck hash, and then runs residual retraining, dev-only CEM, strict holdout, and candidate queue generation. It never submits to Kaggle.

```bash
ROOT=outputs/opponent_league_20260804 \
V2_OUT=outputs/ogerpon_front100_v2_20260803_101303 \
REPLAYS=outputs/kaggle_logs/top_leaders_20260802/keidroid_55153405 \
OUT=outputs/opponent_league_20260804/ogerpon_optimization \
WORKERS=48 MODEL_JOBS=12 \
tools/run_opponent_league_ogerpon.sh
```

## Evaluation

Use `tools.league_eval` with a frozen manifest. Its objective combines weighted Wilson lower bounds, top-100 performance, worst-20% CVaR, mirror performance, and runtime penalties.

```bash
python3 -m tools.league_eval \
  --candidates candidate.tar.gz \
  --manifest outputs/opponent_league_20260804/pool/holdout.json \
  --games 64 --workers 48 \
  --out outputs/opponent_league_20260804/evaluations/candidate_holdout
```

## Outputs

- `snapshot.json`: leaderboard and selected public episodes.
- `collect_state.json`: resumable collection state and failures.
- `registry.json`: source, model, deck, package, audit, and rejection metadata.
- `pilot_report.json`: required-teacher status, turn-level medians, attack median, and overall Pilot result.
- `artifacts/`, `packages/`, `audits/`: trained clone products.
- `pool/train.json`, `pool/dev.json`, `pool/holdout.json`: frozen manifests.
- `pool/summary.json`: final pool coverage and split counts.
- `pool/coverage.json`: unique team, deck, behavior, top-100, and generated-opponent coverage.
