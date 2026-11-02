#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${SAFE_FAIR_VENV_DIR:-.venv-sg}"
source "${VENV_DIR}/bin/activate"
python -m safe_fair_macpo.cli prepare-quick "$@"
