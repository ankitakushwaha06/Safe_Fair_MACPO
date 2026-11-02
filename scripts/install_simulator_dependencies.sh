#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${SAFE_FAIR_VENV_DIR:-.venv-sg}"
SAFETY_GYM_URL="${SAFE_FAIR_SAFETY_GYM_URL:-https://github.com/PKU-Alignment/safety-gymnasium.git}"
SAFETY_GYM_COMMIT="${SAFE_FAIR_SAFETY_GYM_COMMIT:-ae966e511b9927f06b39c727ca5c650136a4e696}"
SAFETY_GYM_ROOT="${ROOT_DIR}/vendor/safety-gymnasium"

source "${VENV_DIR}/bin/activate"

if [[ ! -d "${SAFETY_GYM_ROOT}/.git" ]]; then
  mkdir -p "${ROOT_DIR}/vendor"
  git clone --filter=blob:none --no-checkout "${SAFETY_GYM_URL}" "${SAFETY_GYM_ROOT}"
fi

git -C "${SAFETY_GYM_ROOT}" fetch --depth 1 origin "${SAFETY_GYM_COMMIT}"
git -C "${SAFETY_GYM_ROOT}" checkout --detach "${SAFETY_GYM_COMMIT}"

CHECKED_OUT_COMMIT="$(git -C "${SAFETY_GYM_ROOT}" rev-parse HEAD)"
if [[ "${CHECKED_OUT_COMMIT}" != "${SAFETY_GYM_COMMIT}" ]]; then
  echo "Safety-Gymnasium revision mismatch: expected ${SAFETY_GYM_COMMIT}, got ${CHECKED_OUT_COMMIT}" >&2
  exit 1
fi

python -m pip install \
  'gym==0.17.2' \
  'absl-py>=1.0' \
  'xmltodict>=0.13.0' \
  'pyyaml>=6.0' \
  'imageio>=2.27.0' \
  'gymnasium==0.28.1' \
  'gymnasium-robotics==1.2.2' \
  'mujoco==2.3.3' \
  'mujoco-py==2.1.2.14' \
  'scipy==1.10.1'

python -m pip install -e "${SAFETY_GYM_ROOT}" --no-deps
python scripts/patch_external_sources.py
python -m pip install 'gymnasium==0.28.1'

echo "Installed pinned simulator dependencies into ${ROOT_DIR}/${VENV_DIR}"
