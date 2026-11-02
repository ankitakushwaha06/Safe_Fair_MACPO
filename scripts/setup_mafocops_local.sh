#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAFOCOPS_ROOT="${ROOT_DIR}/safe_fair_macpo_runs/official_sources/mafocops_supplemental"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/_external_baseline_env.sh"

python -m pip install --quiet --upgrade "wandb>=0.18,<0.20"
python -m pip install --quiet -e "${MAFOCOPS_ROOT}"

python - <<'PY'
import importlib
mods = ["wandb", "setproctitle", "tensorboardX", "mujoco_py", "macpo"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: ok ({getattr(mod, '__file__', 'builtin')})")
    except Exception as exc:
        print(f"{name}: fail ({type(exc).__name__}: {exc})")
        raise
PY

(
  cd "${MAFOCOPS_ROOT}"
  python - <<'PY'
from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import MujocoMulti

checks = [
    ("HalfCheetah-v2", "2x3"),
    ("Ant-v2", "2x4"),
]

for scenario, conf in checks:
    env = MujocoMulti(
        env_args={
            "scenario": scenario,
            "agent_conf": conf,
            "agent_obsk": 1,
            "episode_limit": 1000,
        }
    )
    print(f"env_ok: {scenario} {conf} agents={env.n_agents}")
    env.close()
PY
)
