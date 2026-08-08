#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-/root/miniconda3/envs/fhit-kd/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/root/rsdet/input/dataset_6699_scene811}"
OFFICIAL_LIST="${OFFICIAL_LIST:-/root/rsdet/input/official_4481_images.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/scene811_v2/split}"

cd "$ROOT"
"$PYTHON" -m src.build_source_manifest \
  --dataset-root "$DATASET_ROOT" \
  --official-list "$OFFICIAL_LIST" \
  --out artifacts/scene811_v2/source_manifest.csv

"$PYTHON" -m src.source_aware_split \
  --dataset-root "$DATASET_ROOT" \
  --source-manifest artifacts/scene811_v2/source_manifest.csv \
  --out "$OUTPUT_ROOT" --seed 42 --link-mode hardlink

echo "DATASET FREEZE COMPLETE: $OUTPUT_ROOT/dataset.yaml"
