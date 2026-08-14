#!/usr/bin/env bash
set -euo pipefail

if [[ "${USE_LEGACY_SINGLE_TEACHER_PIPELINE:-0}" != "1" ]]; then
  exec "$(dirname "$0")/run_ogerpon_front100_search.sh"
fi

OUT="${1:-outputs/ogerpon_single_teacher_20260803}"
REPLAYS="outputs/kaggle_logs/top_leaders_20260802/keidroid_55153405"
TARGET_NAME="keidroid"
MIN_SEMANTIC_RATE="${MIN_SEMANTIC_RATE:-0.9280}"
MIN_ABILITY_RATE="${MIN_ABILITY_RATE:-0.9371}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-4}"
MODEL_JOBS="${MODEL_JOBS:-12}"

MANIFESTS=(
  configs/meta_imitation_ogerpon_keidroid_private.json
  configs/meta_imitation_ogerpon_keidroid_balanced_v2.json
  configs/meta_imitation_ogerpon_keidroid_compact_v2.json
  configs/meta_imitation_ogerpon_keidroid_deep_v2.json
  configs/meta_imitation_ogerpon_keidroid_winsoft_v2.json
)

TRAIN_OPPONENTS=(
  outputs/meta_imitation/grim_archaludon_fix_20260802/candidates/grim_clone.tar.gz
  outputs/reference_submissions/ptcg-mega-lucario-ex-v63.tar.gz
  outputs/reference_submissions/pokemon-tcg-rahul-jiwane.tar.gz
  outputs/meta_imitation/submissions/lopunny_clone.tar.gz
  outputs/meta_imitation/submissions/kang_clone.tar.gz
  outputs/reference_submissions/multiply-agent-best-940-lb.tar.gz
)

HOLDOUT_OPPONENTS=(
  outputs/great_tusk_fixed_confirmation/kaggle/fixed_endgame_wall.tar.gz
  outputs/reference_submissions/pokemon-steel.tar.gz
  outputs/reference_submissions/improved-probabilistic-agent.tar.gz
  outputs/reference_submissions/mega-pokemon-reinforcement-ai-battle.tar.gz
  outputs/reference_submissions/i-have-one-rear-card.tar.gz
  outputs/reference_submissions/pokemon-ai-battle-best-ptcg-advanced.tar.gz
)

mkdir -p "$OUT"/{manifests,artifacts,candidates,audits,selected,logs,screen}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

jobs_file="$OUT/train_jobs.tsv"
: > "$jobs_file"
for manifest in "${MANIFESTS[@]}"; do
  stem="$(basename "${manifest%.json}")"
  generated_manifest="$OUT/manifests/${stem}.json"
  artifact="$OUT/artifacts/${stem}.json"
  python3 - "$manifest" "$generated_manifest" "$MODEL_JOBS" <<'PY'
import json
import sys
from pathlib import Path

source, destination, jobs = sys.argv[1:]
data = json.loads(Path(source).read_text(encoding="utf-8"))
data["n_jobs"] = int(jobs)
Path(destination).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  printf '%s\t%s\t%s\n' "$generated_manifest" "$artifact" "$OUT/logs/train_${stem}.log" >> "$jobs_file"
done

cat "$jobs_file" | xargs -P "$TRAIN_PARALLEL" -n 3 bash -c '
  manifest="$1"
  artifact="$2"
  log="$3"
  python3 tools/replay_imitation.py --manifest "$manifest" --out "$artifact" > "$log" 2>&1
' _

for artifact in "$OUT"/artifacts/*.json; do
  stem="$(basename "${artifact%.json}")"
  package="$OUT/candidates/${stem}_clone.tar.gz"
  python3 tools/build_meta_submission.py \
    --artifact "$artifact" \
    --variant clone \
    --out "$package" \
    > "$OUT/logs/build_${stem}.log" 2>&1
  python3 tools/behavior_clone_audit.py \
    --package "$package" \
    --replays "$REPLAYS" \
    --target-name "$TARGET_NAME" \
    --out "$OUT/audits/${stem}.json" \
    > "$OUT/logs/audit_${stem}.log" 2>&1
done

python3 - "$OUT" "$MIN_SEMANTIC_RATE" "$MIN_ABILITY_RATE" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_semantic = float(sys.argv[2])
min_ability = float(sys.argv[3])
rows = []
for audit_path in sorted((root / "audits").glob("*.json")):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    semantic_rate = float(audit.get("semantic_rate", 0.0))
    ability = audit.get("by_group", {}).get("ability", {})
    ability_rate = float(ability.get("semantic_rate", 0.0))
    passed = semantic_rate >= min_semantic and ability_rate >= min_ability
    package = Path(audit["package"])
    if passed:
        destination = root / "selected" / package.name
        shutil.copy2(package, destination)
    rows.append({
        "candidate": package.name,
        "semantic_rate": semantic_rate,
        "ability_semantic_rate": ability_rate,
        "passed": passed,
    })
(root / "audit_summary.json").write_text(json.dumps({
    "minimum_semantic_rate": min_semantic,
    "minimum_ability_semantic_rate": min_ability,
    "candidates": rows,
    "passed": sum(row["passed"] for row in rows),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

mapfile -t selected < <(find "$OUT/selected" -maxdepth 1 -type f -name '*.tar.gz' -print | sort)
if ((${#selected[@]} == 0)); then
  echo "No candidate passed behavior audit; skipping server battles."
  exit 0
fi

for opponent in "${TRAIN_OPPONENTS[@]}" "${HOLDOUT_OPPONENTS[@]}"; do
  if [[ ! -f "$opponent" ]]; then
    echo "Missing server battle opponent: $opponent" >&2
    exit 1
  fi
done

python3 tools/server_gold_autoscreen.py \
  --candidates "${selected[@]}" \
  --train "${TRAIN_OPPONENTS[@]}" \
  --holdout "${HOLDOUT_OPPONENTS[@]}" \
  --out "$OUT/screen" \
  --workers 48 \
  --seed 20260803 \
  --stage1-games 4 \
  --stage2-games 12 \
  --holdout-games 24 \
  --stage2-count 5 \
  --holdout-count 3
