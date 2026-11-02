#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "codes" / "safe_fair_macpo_experiments.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sfm_exp", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def feature_names_for_task(task_name: str):
    if task_name == "smamujoco_halfcheetah_2x3_fair":
        return ["control_effort", "safety_share", "action_jerk", "overload", "stall_load"]
    return ["f0", "f1", "f2", "f3", "f4"]


def evaluate_components(module, run_dir: Path, episodes: int):
    cfg_data = json.load(open(run_dir / "config.json", "r", encoding="utf-8"))
    cfg = module.RunConfig(
        task=module.TASKS[cfg_data["task"]["name"]],
        method=module.METHODS[cfg_data["method"]["name"]],
        seed=int(cfg_data["seed"]),
        total_steps=1,
        rollout_steps=1,
        run_group=cfg_data.get("run_group", "main"),
        quick=bool(cfg_data.get("quick", False)),
        device=str(module.DEVICE),
    )
    trainer = module.SafeFairTrainer(cfg)
    ckpt = run_dir / "model.pt"
    trainer.ac.load_state_dict(torch.load(ckpt, map_location=module.DEVICE))

    feature_sums = []
    burden_sums = []
    debt_terms = []
    envy_terms = []
    returns = []
    safety_costs = []
    fairness_costs = []

    for ep in range(episodes):
        obs, _ = trainer.env.reset(seed=cfg.seed + 20_000 + ep)
        done = False
        ep_return = 0.0
        ep_safety = 0.0
        ep_fair = 0.0
        steps = 0
        while not done and steps < trainer.task.episode_length:
            out = trainer.ac.act(obs, device=trainer.device, deterministic=True)
            obs, reward, done, info = trainer.env.step(out["actions"])
            ep_return += float(reward)
            ep_safety += float(info.get("safety_cost", 0.0))
            ep_fair += float(info.get("fairness_cost", 0.0))
            steps += 1
        feature_hist = np.asarray(trainer.env.logger.feature_history, dtype=np.float64)
        burden_hist = np.asarray(trainer.env.logger.burden_history, dtype=np.float64)
        feature_sums.append(feature_hist.sum(axis=0))
        burden_sums.append(burden_hist.sum(axis=0))
        debt_terms.append(float(np.mean([abs(np.mean(d)) for d in trainer.env.logger.debt_history])) if trainer.env.logger.debt_history else 0.0)
        envy_terms.append(float(np.mean([module.excess_envy(b / np.maximum(trainer.env.logger.q, 1e-8)) for b in trainer.env.logger.burden_history])) if trainer.env.logger.burden_history else 0.0)
        returns.append(ep_return)
        safety_costs.append(ep_safety)
        fairness_costs.append(ep_fair)

    trainer.close()

    feature_sums = np.asarray(feature_sums, dtype=np.float64)
    burden_sums = np.asarray(burden_sums, dtype=np.float64)
    weights = trainer.env.logger.weights.as_vector()
    weighted_component_sums = feature_sums * weights[None, None, :]
    agent_gap = np.abs(feature_sums[:, 0, :] - feature_sums[:, 1, :]) if feature_sums.shape[1] == 2 else np.zeros((episodes, feature_sums.shape[2]))

    return {
        "task": trainer.task.name,
        "method": trainer.method.name,
        "feature_names": feature_names_for_task(trainer.task.name),
        "return_mean": float(np.mean(returns)),
        "safety_cost_mean": float(np.mean(safety_costs)),
        "fairness_cost_mean": float(np.mean(fairness_costs)),
        "per_agent_burden_mean": burden_sums.mean(axis=0).tolist(),
        "component_mean_agent0": feature_sums.mean(axis=0)[0].tolist(),
        "component_mean_agent1": feature_sums.mean(axis=0)[1].tolist(),
        "weighted_component_mean_agent0": weighted_component_sums.mean(axis=0)[0].tolist(),
        "weighted_component_mean_agent1": weighted_component_sums.mean(axis=0)[1].tolist(),
        "component_gap_mean": agent_gap.mean(axis=0).tolist(),
        "weights": weights.tolist(),
    }


def print_table(title: str, feature_names, left, right, gap):
    headers = [("component", 16), ("agent0", 12), ("agent1", 12), ("gap", 12)]
    line = "+" + "+".join("-" * (w + 2) for _, w in headers) + "+"
    print(title)
    print(line)
    print("| " + " | ".join(name.ljust(w) for name, w in headers) + " |")
    print(line)
    for name, a, b, g in zip(feature_names, left, right, gap):
        print("| " + " | ".join([
            name.ljust(16),
            f"{a:.2f}".rjust(12),
            f"{b:.2f}".rjust(12),
            f"{g:.2f}".rjust(12),
        ]) + " |")
    print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    module = load_module()
    for raw in args.run_dirs:
        run_dir = Path(raw).resolve()
        result = evaluate_components(module, run_dir, episodes=args.episodes)
        print(f"RUN: {run_dir.name}")
        print(
            json.dumps(
                {
                    "return_mean": result["return_mean"],
                    "safety_cost_mean": result["safety_cost_mean"],
                    "fairness_cost_mean": result["fairness_cost_mean"],
                    "per_agent_burden_mean": result["per_agent_burden_mean"],
                },
                indent=2,
            )
        )
        print_table(
            "Raw component totals",
            result["feature_names"],
            result["component_mean_agent0"],
            result["component_mean_agent1"],
            result["component_gap_mean"],
        )
        print_table(
            "Weighted component totals",
            result["feature_names"],
            result["weighted_component_mean_agent0"],
            result["weighted_component_mean_agent1"],
            result["component_gap_mean"],
        )


if __name__ == "__main__":
    main()
