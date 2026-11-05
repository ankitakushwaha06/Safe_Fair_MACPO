#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/scripts/internal/run_dry_run.sh" --cpu
cd "${ROOT_DIR}"
python -m unittest discover -s tests -v
