#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

LATEST_ZIP="${LATEST_ZIP:?Set LATEST_ZIP to the latest combined Scene811 ZIP}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:?Set OFFICIAL_MANIFEST to the audited 4,481-image identity manifest}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new empty destination directory}"
EXPECTED_ZIP_SHA256="${EXPECTED_ZIP_SHA256:-f66212d1693baa92c6342ddac003775671a9c99e38fb6d26eee2cacd28d63bc5}"
REVIEW_JSON="${REVIEW_JSON:-$FHIT_ROOT/configs/scene811_v3_near_duplicate_review.json}"
SEMANTIC_REVIEW_JSON="${SEMANTIC_REVIEW_JSON:-$FHIT_ROOT/configs/scene811_v3_semantic_scene_unions.json}"

require_file "$LATEST_ZIP" "latest combined dataset ZIP"
require_file "$OFFICIAL_MANIFEST" "official identity manifest"
require_file "$REVIEW_JSON" "immutable near-duplicate review"
require_file "$SEMANTIC_REVIEW_JSON" "immutable semantic same-scene review"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite existing OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

cd "$FHIT_ROOT"
"$PYTHON" -m src.build_scene811_v3_dataset \
  --latest-zip "$LATEST_ZIP" \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --out "$OUTPUT_ROOT" \
  --dataset-id scene811_v3_grouped_clean_r10 \
  --seed 20260810 \
  --expected-zip-sha256 "$EXPECTED_ZIP_SHA256" \
  --near-duplicate-review "$REVIEW_JSON" \
  --semantic-same-scene-review "$SEMANTIC_REVIEW_JSON"

"$PYTHON" -m src.verify_scene811_v3_dataset \
  --root "$OUTPUT_ROOT" \
  --expected-fingerprint "b4367981f59e0d04cf7925587582acb0d3f25a2e9b145dfef662a8da8f0797b9" \
  --out "$OUTPUT_ROOT/reports/verify_after_build.json"

echo "DATASET FREEZE COMPLETE: $OUTPUT_ROOT"
