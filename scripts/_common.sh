#!/usr/bin/env bash

# Shared helpers for the Scene811 V3 server workflow. This file is sourced by
# numbered scripts and is not intended to be executed directly.
FHIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-python}"

require_file() {
  local path="$1" description="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${description}: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1" description="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing ${description}: $path" >&2
    exit 2
  fi
}

completed_epochs() {
  local results_csv="$1"
  "$PYTHON" - "$results_csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        print(sum(1 for _ in csv.DictReader(stream)))
PY
}

v3_namespace() {
  local dataset_root="$1"
  "$PYTHON" - "$dataset_root/dataset_fingerprint.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"{value['dataset_id']}__{value['dataset_fingerprint'][:12]}")
PY
}
