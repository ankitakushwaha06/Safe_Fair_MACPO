#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAT_ROOT="${ROOT_DIR}/safe_fair_macpo_runs/official_sources/mat"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

if [[ ! -d "${MAT_ROOT}" ]]; then
  echo "MAT repo missing at ${MAT_ROOT}" >&2
  exit 2
fi

python - <<'PY'
import setproctitle  # noqa: F401
import tensorboardX  # noqa: F401
import wandb  # noqa: F401
print("mat_deps_ok")
PY

(
  cd "${MAT_ROOT}/mat/scripts"
  python train/train_mujoco.py --help >/dev/null
)

echo "mat_setup_ok"
