#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/setup_mat_local.sh"

"${ROOT_DIR}/scripts/run_mat_native_audit.sh" mat Ant-v2 2x4 180000 0 native_mat_ant_audit
"${ROOT_DIR}/scripts/run_mat_native_audit.sh" mat HalfCheetah-v2 2x3 180000 0 native_mat_halfcheetah_audit
