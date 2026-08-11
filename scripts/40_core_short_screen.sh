#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the verified Scene811 V3 directory}"
CONFIG="${CONFIG:-$FHIT_ROOT/configs/generated/scene811_v3/experiment_v3.yaml}"
SHORT_EPOCHS="${SHORT_EPOCHS:-8}"
require_file "$CONFIG" "generated V3 KD config"
cd "$FHIT_ROOT"
namespace="$(v3_namespace "$DATASET_ROOT")"
LEDGER_ROOT="${LEDGER_ROOT:-$FHIT_ROOT/reports/scene811_v3/ledger/runs}"
LEDGER_REGISTRY="${LEDGER_REGISTRY:-$FHIT_ROOT/reports/scene811_v3/ledger/experiment_registry.jsonl}"
baseline_weights="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
from pathlib import Path
print(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))["paths"]["baseline_weights"])
PY
)"
require_file "$baseline_weights" "fixed B-mix baseline checkpoint"

run_or_resume() {
  local config="$1" exp="$2" name="$3"
  local run_dir="$FHIT_ROOT/runs/$namespace/$name"
  local last="$run_dir/weights/last.pt"
  local ledger_dir="$LEDGER_ROOT/$name"
  local rows
  rows="$(completed_epochs "$run_dir/results.csv")"
  if (( rows >= SHORT_EPOCHS )); then
    echo "SHORT SCREEN SKIP: $name already has $rows/$SHORT_EPOCHS result rows."
    return 0
  fi
  local args=(-m src.train_ablation --config "$config" --exp "$exp" --run-name "$name" --epochs "$SHORT_EPOCHS")
  local command_text="$PYTHON ${args[*]}"
  if [[ -f "$last" ]]; then
    args+=(--resume)
    command_text="$PYTHON ${args[*]}"
    if [[ -f "$ledger_dir/evidence/run_manifest.json" && ! -f "$ledger_dir/evidence/completion.json" ]]; then
      "$PYTHON" -m src.experiment_ledger resume --run-dir "$ledger_dir" --checkpoint "$last" --command "$command_text"
    elif [[ ! -f "$ledger_dir/evidence/run_manifest.json" ]]; then
      echo "LEDGER ERROR: resumable $name has no pre-training ledger." >&2
      exit 5
    fi
  elif [[ -e "$run_dir" ]]; then
    echo "SHORT SCREEN ERROR: $run_dir exists without resumable last.pt." >&2
    exit 3
  else
    "$PYTHON" -m src.experiment_ledger init \
      --run-dir "$ledger_dir" \
      --experiment "${exp^^}" \
      --dataset-report "$DATASET_ROOT/dataset_fingerprint.json" \
      --config "$config" \
      --seed 42 \
      --initial-checkpoint "$baseline_weights" \
      --command "$command_text" \
      --registry "$LEDGER_REGISTRY" \
      --repo "$FHIT_ROOT"
  fi
  "$PYTHON" "${args[@]}"
  "$PYTHON" - "$ledger_dir" "$run_dir/results.csv" "$run_dir/kd_health.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from src.experiment_ledger import evidence_dir, parse_results_csv, read_chain, record_epoch

ledger_dir, results, health_path = map(Path, sys.argv[1:4])
existing = {
    int(item["event"]["payload"]["epoch"])
    for item in read_chain(evidence_dir(ledger_dir) / "epoch_events.jsonl")
}
health = {}
if health_path.is_file():
    for line in health_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            health[int(value["epoch"])] = value
for index, metrics in enumerate(parse_results_csv(results), start=1):
    raw_epoch = metrics.get("epoch", index)
    epoch = int(float(raw_epoch))
    if epoch <= 0:
        epoch = index
    if epoch not in existing:
        record_epoch(run_dir=ledger_dir, epoch=epoch, metrics=metrics, kd_health=health.get(epoch))
PY
}

help_text="$("$PYTHON" -m src.train_ablation --help 2>&1)"
if [[ "$help_text" != *"g"* || "$help_text" != *"gp"* ]]; then
  echo "BLOCKED: G/P/GP CLI is unavailable in this checkout; no legacy fallback is allowed." >&2
  exit 4
fi

run_or_resume "$CONFIG" c0 v3_c0_s42
run_or_resume "$CONFIG" g v3_g_s42
run_or_resume "$CONFIG" p v3_p_s42
run_or_resume "$CONFIG" gp v3_gp_s42
