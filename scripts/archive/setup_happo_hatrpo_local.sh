#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HAPPO_ROOT="${ROOT_DIR}/safe_fair_macpo_runs/official_sources/happo_hatrpo"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/setup/_external_baseline_env.sh"

python - <<'PY'
import importlib
mods = ["setproctitle", "tensorboardX", "mujoco_py"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: ok ({getattr(mod, '__file__', 'builtin')})")
    except Exception as exc:
        print(f"{name}: fail ({type(exc).__name__}: {exc})")
        raise
PY

(
  cd "${HAPPO_ROOT}"
  python - <<'PY'
from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

checks = [
    ("HalfCheetah-v2", "2x3"),
    ("Ant-v2", "2x4"),
]

for scenario, conf in checks:
    env = MujocoMulti(
        env_args={
            "scenario": scenario,
            "agent_conf": conf,
            "agent_obsk": 2,
            "episode_limit": 1000,
        }
    )
    print(f"env_ok: {scenario} {conf} agents={env.n_agents}")
    env.close()
PY
)
