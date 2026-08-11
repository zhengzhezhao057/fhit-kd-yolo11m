#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"

REGISTRY="${REGISTRY:-$FHIT_ROOT/reports/scene811_v3/ledger/experiment_registry.jsonl}"
OUT="${OUT:-$FHIT_ROOT/reports/scene811_v3/ledger_summary}"
mkdir -p "$OUT"
cd "$FHIT_ROOT"
"$PYTHON" -m src.experiment_ledger summarize --registry "$REGISTRY" --out-dir "$OUT"
echo "LEDGER SUMMARY COMPLETE: $OUT"
