#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the verified Scene811 V3 directory}"
CONFIG="${CONFIG:-$FHIT_ROOT/configs/generated/scene811_v3/experiment_v3.yaml}"
HEALTH_BATCHES="${HEALTH_BATCHES:-10}"
require_file "$CONFIG" "generated V3 KD config"
cd "$FHIT_ROOT"

namespace="$(v3_namespace "$DATASET_ROOT")"
teacher_dir="$FHIT_ROOT/runs/$namespace/teacher"
if [[ -f "$teacher_dir/last.pt" ]]; then
  "$PYTHON" -m src.train_teacher --config "$CONFIG" --resume
elif [[ -e "$teacher_dir" ]]; then
  echo "TEACHER ERROR: $teacher_dir exists without last.pt; preserve it and recover manually." >&2
  exit 3
else
  "$PYTHON" -m src.train_teacher --config "$CONFIG"
fi

cache_dir="$FHIT_ROOT/cache/teacher_signals/$namespace/train"
cache_args=(-m src.cache_teacher_signals --config "$CONFIG" --split train)
[[ -f "$cache_dir/manifest.json" ]] && cache_args+=(--resume)
"$PYTHON" "${cache_args[@]}"
"$PYTHON" -m src.verify_kd_cache --config "$CONFIG" --split train --samples 64

bank="$FHIT_ROOT/cache/prototype_banks/$namespace/leave_one_scene_out.pt"
if [[ -f "$bank" ]]; then
  "$PYTHON" -m src.prototype_bank verify --config "$CONFIG" --cache "$cache_dir" --bank "$bank"
else
  "$PYTHON" -m src.prototype_bank build --config "$CONFIG" --cache "$cache_dir" --bank "$bank" --min-count 4 --verify-samples 64
fi

help_text="$("$PYTHON" -m src.train_ablation --help 2>&1)"
if [[ "$help_text" != *"g"* || "$help_text" != *"gp"* ]]; then
  echo "BLOCKED: this checkout has not exposed the G/P/GP train_ablation interface. Do not fall back silently to legacy F/K/FK." >&2
  exit 4
fi

for exp in g p gp; do
  name="health_${exp}_s42"
  run_dir="$FHIT_ROOT/runs/$namespace/$name"
  if [[ -e "$run_dir" ]]; then
    echo "HEALTH SKIP: preserve existing $run_dir; health runs are write-once."
    continue
  fi
  extra_args=()
  [[ "$exp" == "p" || "$exp" == "gp" ]] && extra_args+=(--prototype-bank "$bank")
  "$PYTHON" -m src.train_ablation --config "$CONFIG" --exp "$exp" --run-name "$name" --epochs 1 --health-batches "$HEALTH_BATCHES" "${extra_args[@]}"
done
