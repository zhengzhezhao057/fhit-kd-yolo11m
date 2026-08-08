#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-/root/miniconda3/envs/fhit-kd/bin/python}"
CHECKPOINT="${CHECKPOINT:-/root/rsdet/weights/yolo11m.pt}"
cd "$ROOT"

for seed in 0 1; do
  name="baseline_seed${seed}"
  last="runs/scene811_v2/$name/weights/last.pt"
  args=(-m src.train_detector --config configs/baseline.yaml --run-name "$name" --checkpoint "$CHECKPOINT" --seed "$seed")
  [[ -f "$last" ]] && args+=(--resume)
  "$PYTHON" "${args[@]}"
done
