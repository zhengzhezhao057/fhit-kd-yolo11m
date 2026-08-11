#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

git submodule update --init --recursive

if [[ "${FHIT_USE_CURRENT_PYTHON:-0}" != "1" ]]; then
  if [[ -z "${CONDA_SH:-}" ]]; then
    for candidate in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "$candidate" ]]; then
        CONDA_SH="$candidate"
        break
      fi
    done
  fi
  if [[ -z "${CONDA_SH:-}" || ! -f "$CONDA_SH" ]]; then
    echo "Miniconda was not found. Set CONDA_SH, or set FHIT_USE_CURRENT_PYTHON=1 for an existing environment." >&2
    exit 1
  fi
  source "$CONDA_SH"
  ENV_NAME="${FHIT_ENV_NAME:-fhit-kd}"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 -y
  fi
  conda activate "$ENV_NAME"
fi

python -m src.server_doctor --install --cuda-index "${CUDA_INDEX:-cu126}"
python -m pip check
python -m pytest -q
python -m src.server_doctor --full --min-free-gb "${MIN_FREE_GB:-80}"
python -m src.experiment_ledger --help >/dev/null
echo "BOOTSTRAP COMPLETE: export PYTHON=$(command -v python) before running later scripts."
