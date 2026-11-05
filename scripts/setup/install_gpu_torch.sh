#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${SAFE_FAIR_VENV_DIR:-.venv-sg}"
TORCH_INDEX_URL="${SAFE_FAIR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_PACKAGE_SPEC="${SAFE_FAIR_TORCH_PACKAGE_SPEC:-torch==2.4.1}"

source "${VENV_DIR}/bin/activate"
python -m pip uninstall -y torch || true
python -m pip install "${TORCH_PACKAGE_SPEC}" --index-url "${TORCH_INDEX_URL}"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
