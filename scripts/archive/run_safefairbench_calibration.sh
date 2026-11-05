#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-sg/bin/python"
CALIBRATION_SCRIPT="${ROOT_DIR}/scripts/calibrate_safefairbench.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${CALIBRATION_SCRIPT}" "$@"
