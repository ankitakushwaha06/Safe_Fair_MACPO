#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-sg" / "bin" / "python"
BACKEND = ROOT / "codes" / "safe_fair_macpo_experiments.py"

TASK_ALIASES = {
    "safety_gym_multigoal_fair": "safetygym_point_multigoal1_fair",
    "safetygym_multigoal_fair": "safetygym_point_multigoal1_fair",
    "safety_gym_chargeshare_fair": "safetygym_chargeshare_fair",
    "vmas_safe_multigoal": "safefairbench_vmas_multigoal",
    "highway_merge_fair": "safefairbench_highway_merge",
}

METHOD_ALIASES = {
    "sfmacpo": "safe_fair_macpo",
    "macpo_recovery": "macpo_recovery",
    "mappo_lag_recovery": "mappo_lagrangian_recovery",
    "recovery_only": "macpo_recovery",
    "sfmacpo_fair_assist": "safe_fair_macpo",
    "safety_slot_control": "macpo_recovery",
    "immediate_recovery": "mappo_lagrangian_recovery",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_task(name: str) -> str:
    return TASK_ALIASES.get(name, name)


def canonical_method(name: str) -> str:
    return METHOD_ALIASES.get(name, name)


def per_step_budget(total_budget: Any, episode_len: int = 250) -> str:
    if str(total_budget).lower() == "inf":
        return "1000000.0"
    val = float(total_budget)
    return f"{val / episode_len:.8f}"


def base_env(run_group: str, task: str, method: str, seed: int, steps: int, tag: str, fair_budget: str = "") -> str:
    include_sfb = "1" if task.startswith("safefairbench_") else "0"
    parts = [
        f"SAFE_FAIR_MACPO_ROOT='{ROOT / 'safe_fair_macpo_runs'}'",
        "SAFE_FAIR_QUICK_TEST=0",
        "SAFE_FAIR_RUN_TRAINING=1",
        "SAFE_FAIR_RUN_EVALUATION=1",
        "SAFE_FAIR_CLONE_OFFICIAL_REPOS=0",
        "SAFE_FAIR_USE_CUDA=1",
        f"SAFE_FAIR_INCLUDE_SAFEFAIRBENCH={include_sfb}",
        f"SAFE_FAIR_RUN_GROUP_FILTER='{run_group}'",
        f"SAFE_FAIR_TASK_FILTER='{task}'",
        f"SAFE_FAIR_METHOD_FILTER='{method}'",
        f"SAFE_FAIR_SEED_FILTER='{seed}'",
        f"SAFE_FAIR_CFG_TOTAL_STEPS='{steps}'",
        "SAFE_FAIR_CFG_ROLLOUT_STEPS='2048'",
        "SAFE_FAIR_CFG_PPO_EPOCHS='8'",
        "SAFE_FAIR_CFG_MINIBATCH_SIZE='512'",
        "SAFE_FAIR_CFG_EVAL_EPISODES='24'",
        f"SAFE_FAIR_CFG_RUN_TAG='{tag}'",
    ]
    if task.startswith("safefairbench_"):
        parts[parts.index("SAFE_FAIR_CFG_ROLLOUT_STEPS='2048'")] = "SAFE_FAIR_CFG_ROLLOUT_STEPS='256'"
        parts[parts.index("SAFE_FAIR_CFG_EVAL_EPISODES='24'")] = "SAFE_FAIR_CFG_EVAL_EPISODES='32'"
    if fair_budget:
        parts.extend([
            f"SAFE_FAIR_TASK_FAIRNESS_BUDGET_PER_STEP='{fair_budget}'",
            "SAFE_FAIR_MAX_RUNS=1",
        ])
    return " ".join(parts) + f" '{PY}' '{BACKEND}'"


def build_jobs(manifest: Dict[str, Any]) -> List[str]:
    jobs: List[str] = []
    for sweep in manifest.get("training_sweeps", []):
        name = sweep["name"]
        task = canonical_task(sweep["env"])
        raw_steps = sweep.get("total_steps", 180000)
        steps = int(raw_steps[0] if isinstance(raw_steps, list) else raw_steps)
        if "fairness_budgets" in sweep:
            methods = [canonical_method(sweep.get("algo", "sfmacpo"))]
            for method in methods:
                for seed in sweep.get("seeds", [0]):
                    for budget in sweep["fairness_budgets"]:
                        fair = per_step_budget(budget)
                        tag = f"reproducibility_{name}_df{str(budget).replace('.', 'p')}_s{seed}"
                        # Use an existing runnable manifest row and override the fairness budget.
                        # The backend's synthetic fairness_budget_sweep group is not defined
                        # for every task in the time-bounded reproducibility manifest, which can silently
                        # select zero runs for MultiGoal/ChargeShare.
                        jobs.append(base_env("main", task, method, int(seed), steps, tag, fair))
        for method_key in sweep.get("algos", []):
            method = canonical_method(method_key)
            step_values = sweep.get("total_steps", [steps])
            if isinstance(step_values, int):
                step_values = [step_values]
            for step_value in step_values:
                for seed in sweep.get("seeds", [0]):
                    tag = f"reproducibility_{name}_{method}_t{step_value}_s{seed}"
                    jobs.append(base_env("safefairbench", task, method, int(seed), int(step_value), tag))
        for variant in sweep.get("variants", []):
            method = canonical_method(variant)
            for seed in sweep.get("seeds", [0]):
                tag = f"reproducibility_{name}_{variant}_s{seed}"
                jobs.append(base_env("safefairbench", task, method, int(seed), steps, tag))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = load_yaml(Path(args.manifest))
    jobs = build_jobs(manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(jobs) + "\n", encoding="utf-8")
    print(f"wrote {out} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
