#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APT_ROOT="${ROOT_DIR}/vendor/apt/extracted"

export BASELINE_VENV="${BASELINE_VENV:-${ROOT_DIR}/.venv-sg}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"

_inc_paths=(
  "${APT_ROOT}/libosmesa6-dev/usr/include"
  "${APT_ROOT}/libglew-dev/usr/include"
  "${APT_ROOT}/libglu1-mesa-dev/usr/include"
  "${APT_ROOT}/libgl-dev/usr/include"
)

_lib_paths=(
  "${MUJOCO_PY_MUJOCO_PATH}/bin"
  "${APT_ROOT}/libosmesa6-dev/usr/lib/x86_64-linux-gnu"
  "${APT_ROOT}/libosmesa6/usr/lib/x86_64-linux-gnu"
  "${APT_ROOT}/libglew-dev/usr/lib/x86_64-linux-gnu"
  "${APT_ROOT}/libglew2.2/usr/lib/x86_64-linux-gnu"
  "${APT_ROOT}/libglu1-mesa-dev/usr/lib/x86_64-linux-gnu"
  "${APT_ROOT}/libglu1-mesa/usr/lib/x86_64-linux-gnu"
  "${APT_ROOT}/libgl-dev/usr/lib/x86_64-linux-gnu"
  "${APT_ROOT}/libgl1/usr/lib/x86_64-linux-gnu"
  "/usr/lib/x86_64-linux-gnu"
)

join_by_colon() {
  local IFS=":"
  printf '%s' "$*"
}

export C_INCLUDE_PATH="$(join_by_colon "${_inc_paths[@]}")"
export CPATH="${C_INCLUDE_PATH}"
export LIBRARY_PATH="$(join_by_colon "${_lib_paths[@]}")"
export LD_LIBRARY_PATH="$(join_by_colon "${_lib_paths[@]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPPFLAGS="$(printf ' -I%s' "${_inc_paths[@]}")${CPPFLAGS:+ ${CPPFLAGS}}"
export CFLAGS="${CPPFLAGS}${CFLAGS:+ ${CFLAGS}}"
export LDFLAGS="$(printf ' -L%s' "${_lib_paths[@]}")${LDFLAGS:+ ${LDFLAGS}}"
export PATH="${APT_ROOT}/patchelf/usr/bin:${PATH}"

if [[ -f "${BASELINE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${BASELINE_VENV}/bin/activate"
fi
