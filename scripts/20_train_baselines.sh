#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

YOLO_WEIGHTS="${YOLO_WEIGHTS:?Set YOLO_WEIGHTS to the manually uploaded yolo11m.pt}"
CONFIG_DIR="${CONFIG_DIR:-$FHIT_ROOT/configs/generated/scene811_v3}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-120}"
RECIPES="${RECIPES:-official,mix}"
SEEDS="${SEEDS:-42,3407,20260809}"
DATASET_ID="scene811_v3_grouped_clean_r10"
LEDGER_ROOT="${LEDGER_ROOT:-$FHIT_ROOT/reports/scene811_v3/ledger/runs}"
LEDGER_REGISTRY="${LEDGER_REGISTRY:-$FHIT_ROOT/reports/scene811_v3/ledger/experiment_registry.jsonl}"

require_file "$YOLO_WEIGHTS" "YOLO11m initialization checkpoint"
require_file "$CONFIG_DIR/matrix.json" "generated V3 config matrix"
cd "$FHIT_ROOT"

IFS=',' read -r -a recipe_values <<<"$RECIPES"
IFS=',' read -r -a seed_values <<<"$SEEDS"
for recipe in "${recipe_values[@]}"; do
  for seed in "${seed_values[@]}"; do
    config="$CONFIG_DIR/baseline_${recipe}_seed${seed}.yaml"
    name="b_${recipe}_s${seed}"
    run_dir="$FHIT_ROOT/runs/$DATASET_ID/$name"
    last="$run_dir/weights/last.pt"
    ledger_dir="$LEDGER_ROOT/$name"
    require_file "$config" "generated ${recipe}/seed${seed} baseline config"
    rows="$(completed_epochs "$run_dir/results.csv")"
    if (( rows >= BASELINE_EPOCHS )); then
      echo "BASELINE SKIP: $name already has $rows/$BASELINE_EPOCHS result rows."
      continue
    fi
    args=(-m src.train_detector --config "$config" --run-name "$name" --checkpoint "$YOLO_WEIGHTS" --seed "$seed" --epochs "$BASELINE_EPOCHS")
    command_text="$PYTHON ${args[*]}"
    if [[ -f "$last" ]]; then
      echo "BASELINE RESUME: $name from $last ($rows/$BASELINE_EPOCHS rows)."
      args+=(--resume)
      command_text="$PYTHON ${args[*]}"
      if [[ -f "$ledger_dir/evidence/run_manifest.json" && ! -f "$ledger_dir/evidence/completion.json" ]]; then
        "$PYTHON" -m src.experiment_ledger resume --run-dir "$ledger_dir" --checkpoint "$last" --command "$command_text"
      elif [[ ! -f "$ledger_dir/evidence/run_manifest.json" ]]; then
        echo "LEDGER ERROR: resumable model $name has no pre-training ledger. Preserve it; set up a separate retrospective record manually." >&2
        exit 5
      fi
    elif [[ -e "$run_dir" ]]; then
      echo "BASELINE ERROR: $run_dir exists without a resumable last.pt; preserve it and recover manually." >&2
      exit 3
    else
      echo "BASELINE START: $name"
      "$PYTHON" -m src.experiment_ledger init \
        --run-dir "$ledger_dir" \
        --experiment "B-${recipe}" \
        --dataset-report "$DATASET_ROOT/dataset_fingerprint.json" \
        --config "$config" \
        --seed "$seed" \
        --initial-checkpoint "$YOLO_WEIGHTS" \
        --command "$command_text" \
        --registry "$LEDGER_REGISTRY" \
        --repo "$FHIT_ROOT"
    fi
    "$PYTHON" "${args[@]}"
    "$PYTHON" - "$ledger_dir" "$run_dir/results.csv" <<'PY'
import sys
from pathlib import Path

from src.experiment_ledger import evidence_dir, parse_results_csv, read_chain, record_epoch

ledger_dir = Path(sys.argv[1])
results = Path(sys.argv[2])
existing = {
    int(item["event"]["payload"]["epoch"])
    for item in read_chain(evidence_dir(ledger_dir) / "epoch_events.jsonl")
}
for index, metrics in enumerate(parse_results_csv(results), start=1):
    raw_epoch = metrics.get("epoch", index)
    epoch = int(float(raw_epoch))
    if epoch <= 0:
        epoch = index
    if epoch not in existing:
        record_epoch(run_dir=ledger_dir, epoch=epoch, metrics=metrics)
PY
  done
done
