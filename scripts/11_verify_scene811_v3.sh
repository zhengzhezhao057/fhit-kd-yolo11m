#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the unpacked scene811_v3_grouped_clean_r10 directory}"
EXPECTED_FINGERPRINT="${EXPECTED_FINGERPRINT:-b4367981f59e0d04cf7925587582acb0d3f25a2e9b145dfef662a8da8f0797b9}"
VERIFY_OUT="${VERIFY_OUT:-$FHIT_ROOT/reports/scene811_v3/preflight/verify_server.json}"

require_dir "$DATASET_ROOT" "Scene811 V3 dataset"
mkdir -p "$(dirname "$VERIFY_OUT")"
cd "$FHIT_ROOT"
"$PYTHON" -m src.verify_scene811_v3_dataset \
  --root "$DATASET_ROOT" \
  --expected-fingerprint "$EXPECTED_FINGERPRINT" \
  --out "$VERIFY_OUT"

echo "SERVER DATASET VERIFY COMPLETE: $VERIFY_OUT"
