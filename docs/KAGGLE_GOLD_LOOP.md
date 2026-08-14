# Kaggle Gold Loop

This project uses a private champion/challenger loop for simulation competitions. Competitive manifests, reconstructed decks, replay-derived opponent models, logs, and generated submissions stay under ignored `configs/` or `outputs/` paths.

## Principles

1. Change one hypothesis per candidate.
2. Reject candidates with runtime, legality, timeout, or worst-matchup regressions.
3. Use a training pool and a disjoint holdout pool on the remote server.
4. Treat local ratings only as gates, never as a Kaggle score predictor.
5. Preserve the champion in an active slot; re-upload it only when it is no longer active.
6. Promote only after enough non-self Kaggle episodes and a score improvement.

## Build

Create the ignored private manifest at `configs/great_tusk_gold_private.json`, then run:

```bash
python3 -m tools.kaggle_gold_loop build
```

Every candidate receives package, source, and deck hashes in `outputs/kaggle_gold/state.json`.

## Remote Evaluation

Run matches only on the server through the spawn-safe entry point:

```bash
python3 -m tools.server_candidate_eval \
  --candidates 'outputs/great_tusk_gold/kaggle/*.tar.gz' \
  --opponents 'outputs/reference_submissions/*.tar.gz' \
  --games 24 --workers 48 --record-mode losses \
  --out outputs/great_tusk_gold/server_stage
```

Use a separate opponent list for holdout evaluation. Record a result in the registry with:

```bash
python3 -m tools.kaggle_gold_loop ingest-gate CANDIDATE_ID \
  --name holdout --ranking outputs/great_tusk_gold/server_holdout/ranking.json
```

The evaluator uses common per-pair random seeds for candidate comparisons. Ranking excludes referee-wide failures and wins caused by an opponent import/deck failure; a candidate's own runtime failure remains a loss and is penalized.

## Submission

First inspect current slots and limits:

```bash
python3 -m tools.kaggle_gold_loop sync
```

Preview the active-slot-aware upload order without using a submission:

```bash
python3 -m tools.kaggle_gold_loop submit-cycle CHALLENGER_ID
```

Upload only after the preview is correct:

```bash
python3 -m tools.kaggle_gold_loop submit-cycle CHALLENGER_ID --execute
```

The controller synchronizes active slots before a real upload. If the champion already occupies either active slot, it uploads only the challenger and retains one daily submission; otherwise it uploads the champion first and then the challenger, requiring two available submissions.

## Monitoring and Promotion

```bash
python3 -m tools.kaggle_gold_loop monitor CHALLENGER_ID
python3 -m tools.kaggle_gold_loop promote CHALLENGER_ID --min-episodes 40 --min-score SCORE
```

Promotion also refuses candidates marked with a catastrophic server-gate regression.
