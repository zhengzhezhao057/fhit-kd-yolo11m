#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

git submodule update --init --recursive

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing $CONDA_SH. Install Miniconda or set CONDA_SH to its conda.sh path." >&2
  exit 1
fi
source "$CONDA_SH"
if ! conda env list | awk '{print $1}' | grep -qx fhit-kd; then
  conda create -n fhit-kd python=3.11 -y
fi
conda activate fhit-kd

python -m src.server_doctor --install --cuda-index "${CUDA_INDEX:-cu126}"
python -m pytest -q
python -m src.server_doctor --full --min-free-gb "${MIN_FREE_GB:-80}"
echo "BOOTSTRAP COMPLETE: use /root/miniconda3/envs/fhit-kd/bin/python for every later command."
