#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs"
FIG_ROOT = ROOT / "safe_fair_macpo_runs" / "figures"
RUN_TAG = "core_benchmark_180k"

TASK_LABEL = {
    "smamujoco_halfcheetah_2x3_fair": "HalfCheetah-2x3-Fair",
    "smamujoco_ant_2x4_fair": "Ant-2x4-Fair",
    "safetygym_point_multigoal1_fair": "MultiGoal-Fair",
    "safetygym_chargeshare_fair": "ChargeShare-Fair",
}

METHOD_LABEL = {
    "ippo": "IPPO",
    "mappo": "MAPPO",
    "macpo": "MACPO",
    "mappo_lagrangian": "MAPPO-Lagrangian",
    "dfrl_soto_ggf": "DFRL / SOTO-GGF",
    "reward_shaped_mappo_fair": "Reward-shaped MAPPO-Fair",
    "reward_shaped_macpo_fair": "Reward-shaped MACPO-Fair",
    "safe_fair_macpo": "Safe-Fair MACPO",
    "safe_fair_no_debt": "Safe-Fair no debt",
    "safe_fair_no_envy": "Safe-Fair no envy",
    "safe_fair_no_recovery": "Safe-Fair no recovery",
}


def latest_histories() -> dict[tuple[str, str], Path]:
    latest: dict[tuple[str, str], tuple[float, Path]] = {}
    for path in LOG_ROOT.rglob("train_history.csv"):
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0]
        run_tag = ""
        cfg_path = path.parent / "config.json"
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    run_tag = json.load(f).get("run_tag", "")
            except Exception:
                run_tag = ""
        if run_tag != RUN_TAG:
            continue
        key = (str(row.get("task", "")), str(row.get("method", "")))
        mtime = path.stat().st_mtime
        if key not in latest or mtime > latest[key][0]:
            latest[key] = (mtime, path)
    return {k: v[1] for k, v in latest.items()}


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    histories = latest_histories()
    by_task: dict[str, list[tuple[str, Path]]] = {}
    for (task, method), path in histories.items():
        by_task.setdefault(task, []).append((method, path))

    for task, items in by_task.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        for method, path in sorted(items):
            df = pd.read_csv(path)
            label = METHOD_LABEL.get(method, method)
            x = df["global_step"].to_numpy()
            axes[0].plot(x, df["rollout_return_mean"], label=label, linewidth=1.8)
            axes[1].plot(x, df["rollout_safety_cost_mean"], label=label, linewidth=1.8)
            axes[2].plot(x, df["rollout_fairness_cost_mean"], label=label, linewidth=1.8)
        axes[0].set_title("Return")
        axes[1].set_title("Safety Cost")
        axes[2].set_title("Fairness Cost")
        for ax in axes:
            ax.set_xlabel("Env steps")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Metric")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(TASK_LABEL.get(task, task))
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        out = FIG_ROOT / f"{task}_{RUN_TAG}_comparison.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
