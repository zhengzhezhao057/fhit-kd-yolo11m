#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-/root/miniconda3/envs/fhit-kd/bin/python}"
cd "$ROOT"

run_or_resume() {
  local config="$1" exp="$2" name="$3"
  local last="runs/$name/weights/last.pt"
  local args=(-m src.train_ablation --config "$config" --exp "$exp" --run-name "$name")
  [[ -f "$last" ]] && args+=(--resume)
  "$PYTHON" "${args[@]}"
}

run_or_resume configs/global_kd.yaml c0 v2_c0_seed0
run_or_resume configs/global_kd.yaml fk v2_global_kd_seed0

if [[ -f configs/fah_kd.yaml ]]; then
  run_or_resume configs/fah_kd.yaml fk v2_fah_kd_seed0
else
  echo "SKIP FAH-KD: build OOF hard manifest and configs/fah_kd.yaml first."
fi
