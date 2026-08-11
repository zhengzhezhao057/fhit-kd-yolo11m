#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the verified Scene811 V3 directory}"
CONFIG="${CONFIG:-$FHIT_ROOT/configs/generated/scene811_v3/experiment_v3.yaml}"
EVAL_DIR="${EVAL_DIR:-$FHIT_ROOT/reports/scene811_v3/core_seed42}"
require_file "$CONFIG" "generated V3 KD config"
cd "$FHIT_ROOT"
mkdir -p "$EVAL_DIR"
namespace="$(v3_namespace "$DATASET_ROOT")"

models=("C0=$FHIT_ROOT/runs/$namespace/v3_c0_s42/weights/best.pt")
competitions=()
health=()
parity=()

register_kd() {
  local name="$1" run="$2"
  local full="$FHIT_ROOT/runs/$namespace/$run/weights/best.pt"
  local deploy="$FHIT_ROOT/runs/$namespace/$run/weights/best_deploy.pt"
  [[ -f "$deploy" ]] || return 0
  models+=("$name=$deploy")
  "$PYTHON" -m src.check_deploy_parity --full "$full" --deploy "$deploy" \
    --out "$EVAL_DIR/${name}_parity.json"
  health+=("$name=$FHIT_ROOT/runs/$namespace/$run/kd_health.jsonl")
  parity+=("$name=$EVAL_DIR/${name}_parity.json")
}

register_kd G v3_g_s42
register_kd P v3_p_s42
register_kd GP v3_gp_s42

require_file "${models[0]#*=}" "C0 short-screen checkpoint"

native_args=()
for model in "${models[@]}"; do native_args+=(--model "$model"); done
"$PYTHON" -m src.validate_models --config "$CONFIG" "${native_args[@]}" \
  --out "$EVAL_DIR/native.json"

for model in "${models[@]}"; do
  name="${model%%=*}"; path="${model#*=}"
  out="$EVAL_DIR/${name}_competition.json"
  "$PYTHON" -m src.competition_eval --config "$CONFIG" --model "$path" --split val --class-aware \
    --confidence 0.01 --confidence 0.03 --confidence 0.05 --confidence 0.08 --confidence 0.10 \
    --confidence 0.15 --confidence 0.20 --confidence 0.25 --confidence 0.30 --confidence 0.35 \
    --confidence 0.40 --confidence 0.50 --confidence 0.60 --confidence 0.70 --confidence 0.80 --confidence 0.90 \
    --nms-iou 0.50 --out "$out"
  competitions+=("$name=$out")
done

standard_args=(-m src.standardize_results --native "$EVAL_DIR/native.json" --baseline C0)
for item in "${competitions[@]}"; do standard_args+=(--competition "$item"); done
for item in "${health[@]}"; do standard_args+=(--health "$item"); done
for item in "${parity[@]}"; do standard_args+=(--parity "$item"); done
standard_args+=(--out "$EVAL_DIR/standard_metrics.json")
"$PYTHON" "${standard_args[@]}"
"$PYTHON" -m src.result_gate --metrics "$EVAL_DIR/standard_metrics.json" \
  --out "$EVAL_DIR/gate_decision.json"

echo "EVALUATION COMPLETE: $EVAL_DIR"
