from __future__ import annotations

import argparse

import numpy as np
from catboost import CatBoostRanker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0:1")
    parser.add_argument("--loss", default="QuerySoftMax")
    parser.add_argument("--eval-metric", default="NDCG:top=3")
    args = parser.parse_args()
    rng = np.random.default_rng(20260811)
    groups = 256
    group_size = 6
    features = rng.normal(size=(groups * group_size, 32)).astype(np.float32)
    labels = np.zeros(groups * group_size, dtype=np.float32)
    labels[np.arange(groups) * group_size + rng.integers(0, group_size, size=groups)] = 1.0
    group_ids = np.repeat(np.arange(groups), group_size)
    model = CatBoostRanker(
        iterations=8,
        depth=4,
        learning_rate=0.08,
        loss_function=args.loss,
        eval_metric=args.eval_metric,
        task_type="GPU",
        devices=args.devices,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features, labels, group_id=group_ids)
    print(f"[gpu-smoke] loss={args.loss} eval_metric={args.eval_metric} devices={args.devices} trees={model.tree_count_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
