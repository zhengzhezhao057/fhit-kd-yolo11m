#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-/root/miniconda3/envs/fhit-kd/bin/python}"
CONFIG="${CONFIG:-configs/global_kd.yaml}"
cd "$ROOT"

if [[ -f runs/teacher/last.pt ]]; then
  "$PYTHON" -m src.train_teacher --config "$CONFIG" --resume
else
  "$PYTHON" -m src.train_teacher --config "$CONFIG"
fi
cache_args=(-m src.cache_teacher_signals --config "$CONFIG" --split train)
[[ -f cache/teacher_signals/train/manifest.json ]] && cache_args+=(--resume)
"$PYTHON" "${cache_args[@]}"
"$PYTHON" -m src.verify_kd_cache --config "$CONFIG" --split train --samples 64

for exp in f k fk; do
  name="health_${exp}"
  [[ -d "runs/$name" ]] && continue
  "$PYTHON" -m src.train_ablation --config "$CONFIG" --exp "$exp" --run-name "$name" --epochs 1 --health-batches 10
done
