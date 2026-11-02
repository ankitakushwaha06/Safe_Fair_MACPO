from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SCRIPT = REPO_ROOT / "codes" / "safe_fair_macpo_experiments.py"
DEFAULT_RUN_ROOT = REPO_ROOT / "safe_fair_macpo_runs"
MUJOCO_ROOT = Path.home() / ".mujoco" / "mujoco210"
MUJOCO_BIN = MUJOCO_ROOT / "bin"
SYSTEM_LIB = Path("/usr/lib/x86_64-linux-gnu")


def prepend_env_path(existing: str | None, *parts: Path) -> str:
    values = [str(part) for part in parts if part.exists()]
    if existing:
        values.append(existing)
    seen = set()
    ordered: list[str] = []
    for value in values:
        for item in value.split(":"):
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
    return ":".join(ordered)


def backend_env(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    quick: bool,
    run_training: bool,
    clone_official_repos: bool = False,
    run_evaluation: bool = True,
    use_cuda: bool = True,
) -> Dict[str, str]:
    ld_library_path = prepend_env_path(os.environ.get("LD_LIBRARY_PATH"), MUJOCO_BIN, SYSTEM_LIB)
    return {
        "SAFE_FAIR_MACPO_ROOT": str(run_root),
        "SAFE_FAIR_QUICK_TEST": "1" if quick else "0",
        "SAFE_FAIR_RUN_TRAINING": "1" if run_training else "0",
        "SAFE_FAIR_CLONE_OFFICIAL_REPOS": "1" if clone_official_repos else "0",
        "SAFE_FAIR_RUN_EVALUATION": "1" if run_evaluation else "0",
        "SAFE_FAIR_USE_CUDA": "1" if use_cuda else "0",
        "MUJOCO_PY_MUJOCO_PATH": str(MUJOCO_ROOT),
        "LD_LIBRARY_PATH": ld_library_path,
    }


def run_backend(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    quick: bool,
    run_training: bool,
    clone_official_repos: bool = False,
    run_evaluation: bool = True,
    use_cuda: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        backend_env(
            run_root=run_root,
            quick=quick,
            run_training=run_training,
            clone_official_repos=clone_official_repos,
            run_evaluation=run_evaluation,
            use_cuda=use_cuda,
        )
    )
    return subprocess.run(
        [sys.executable, str(BACKEND_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        check=True,
    )
