#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="${SAFE_FAIR_PYTHON_VERSION:-3.9}"
VENV_DIR="${SAFE_FAIR_VENV_DIR:-.venv-sg}"
TORCH_INDEX_URL="${SAFE_FAIR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
TORCH_PACKAGE_SPEC="${SAFE_FAIR_TORCH_PACKAGE_SPEC:-torch==2.4.1}"

case "$VENV_DIR" in
  ""|"/"|"."|"..") echo "Refusing unsafe SAFE_FAIR_VENV_DIR: '$VENV_DIR'" >&2; exit 2 ;;
esac

if command -v uv >/dev/null 2>&1; then
  rm -rf -- "$VENV_DIR"
  uv venv --seed --python "$PYTHON_VERSION" "$VENV_DIR"
else
  PYTHON_BIN="${SAFE_FAIR_PYTHON_BIN:-python3}"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "$TORCH_PACKAGE_SPEC" --index-url "$TORCH_INDEX_URL"
python -m pip install -e .

echo "Environment ready in $ROOT_DIR/$VENV_DIR"
