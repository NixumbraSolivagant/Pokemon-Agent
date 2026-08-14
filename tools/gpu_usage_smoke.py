from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.replay_imitation import _train_catboost

rng = np.random.default_rng(11)
rows, cols = 60_000, 260
features = rng.normal(size=(rows, cols)).tolist()
labels = rng.integers(0, 2, size=rows).tolist()
sizes = []
remaining = rows
while remaining > 0:
    size = min(remaining, int(rng.integers(5, 20)))
    sizes.append(size)
    remaining -= size
groups = sizes
weights = [1.0] * rows
manifest = {
    "task_type": "GPU",
    "devices": "1",
    "n_jobs": 12,
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 12,
    "reg_lambda": 1.0,
    "colsample_bytree": 0.9,
    "seed": 7,
    "win_weight": 1.0,
    "loss_weight": 1.0,
}

import subprocess
import os

samples = []
stop = False
own_pid = os.getpid()


def sampler() -> None:
    while not stop:
        util = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", "1"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", "1"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        own_apps = [line for line in apps.splitlines() if str(own_pid) in line]
        samples.append((util, own_apps))
        time.sleep(0.5)


thread = threading.Thread(target=sampler, daemon=True)
thread.start()
start = time.time()
model = _train_catboost(features, labels, groups, weights, manifest)
fit_seconds = time.time() - start
stop = True
thread.join(timeout=5)

utils = [int(line.split(",")[0]) for line, _ in samples if line]
our_pid_seen = any(bool(apps) for _, apps in samples)
own_app_samples = [apps for _, apps in samples if apps]
print("fit_seconds", round(fit_seconds, 2), flush=True)
print("gpu1_samples", len(utils), "max_util", max(utils) if utils else None, "samples_gt0", sum(util > 0 for util in utils), flush=True)
print("our_pid_in_compute_apps", our_pid_seen, flush=True)
print("own_app_samples", own_app_samples[:5], flush=True)
print("trees", len(model.get("oblivious_trees", [])), flush=True)
