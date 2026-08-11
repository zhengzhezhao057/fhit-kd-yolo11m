#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the verified Scene811 V3 directory}"
CONFIG_DIR="${CONFIG_DIR:-$FHIT_ROOT/configs/generated/scene811_v3}"
SEEDS="${SEEDS:-42,3407,20260809}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-120}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
BASELINE_BATCH="${BASELINE_BATCH:-16}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-0}"

args=(
  -m src.prepare_scene811_v3_configs
  --dataset-root "$DATASET_ROOT"
  --out "$CONFIG_DIR"
  --seeds "$SEEDS"
  --epochs "$BASELINE_EPOCHS"
  --image-size "$IMAGE_SIZE"
  --batch "$BASELINE_BATCH"
  --workers "$WORKERS"
  --device "$DEVICE"
)
if [[ -n "${BASELINE_WEIGHTS:-}" || -n "${DINO_WEIGHTS:-}" ]]; then
  : "${BASELINE_WEIGHTS:?Set both BASELINE_WEIGHTS and DINO_WEIGHTS for KD config generation}"
  : "${DINO_WEIGHTS:?Set both BASELINE_WEIGHTS and DINO_WEIGHTS for KD config generation}"
  args+=(--baseline-weights "$BASELINE_WEIGHTS" --dino-weights "$DINO_WEIGHTS")
  [[ -n "${DINO_REPO:-}" ]] && args+=(--dino-repo "$DINO_REPO")
fi

cd "$FHIT_ROOT"
"$PYTHON" "${args[@]}"
echo "CONFIG GENERATION COMPLETE: $CONFIG_DIR/matrix.json"
