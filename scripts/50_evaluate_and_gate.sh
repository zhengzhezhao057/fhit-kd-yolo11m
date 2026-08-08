#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-/root/miniconda3/envs/fhit-kd/bin/python}"
CONFIG="${CONFIG:-configs/global_kd.yaml}"
cd "$ROOT"
mkdir -p reports/scene811_v2/core

models=("C0=runs/v2_c0_seed0/weights/best.pt")
competitions=()
health=()
parity=()

register_kd() {
  local name="$1" run="$2"
  local full="runs/$run/weights/best.pt"
  local deploy="runs/$run/weights/best_deploy.pt"
  [[ -f "$deploy" ]] || return 0
  models+=("$name=$deploy")
  "$PYTHON" -m src.check_deploy_parity --full "$full" --deploy "$deploy" \
    --out "reports/scene811_v2/core/${name}_parity.json"
  health+=("$name=runs/$run/kd_health.jsonl")
  parity+=("$name=reports/scene811_v2/core/${name}_parity.json")
}

register_kd Global_KD v2_global_kd_seed0
register_kd FAH_KD v2_fah_kd_seed0

native_args=()
for model in "${models[@]}"; do native_args+=(--model "$model"); done
"$PYTHON" -m src.validate_models --config "$CONFIG" "${native_args[@]}" \
  --out reports/scene811_v2/core/native.json

for model in "${models[@]}"; do
  name="${model%%=*}"; path="${model#*=}"
  out="reports/scene811_v2/core/${name}_competition.json"
  "$PYTHON" -m src.competition_eval --config "$CONFIG" --model "$path" --split val --class-aware --out "$out"
  competitions+=("$name=$out")
done

standard_args=(-m src.standardize_results --native reports/scene811_v2/core/native.json --baseline C0)
for item in "${competitions[@]}"; do standard_args+=(--competition "$item"); done
for item in "${health[@]}"; do standard_args+=(--health "$item"); done
for item in "${parity[@]}"; do standard_args+=(--parity "$item"); done
standard_args+=(--out reports/scene811_v2/core/standard_metrics.json)
"$PYTHON" "${standard_args[@]}"
"$PYTHON" -m src.result_gate --metrics reports/scene811_v2/core/standard_metrics.json \
  --out reports/scene811_v2/core/gate_decision.json
