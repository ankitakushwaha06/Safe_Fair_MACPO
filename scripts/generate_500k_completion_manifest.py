#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "safe_fair_macpo_runs"
LOG_ROOT = RUN_ROOT / "logs"
CONFIG_ROOT = RUN_ROOT / "configs"

TARGET_TASKS: Sequence[str] = (
    "smamujoco_halfcheetah_2x3_fair",
    "safetygym_point_multigoal1_fair",
    "safetygym_chargeshare_fair",
    "smamujoco_ant_2x4_fair",
    "smamujoco_manyagent_ant_fair",
)

TARGET_METHODS: Sequence[str] = (
    "ippo",
    "mappo",
    "macpo",
    "mappo_lagrangian",
    "dfrl_soto_ggf",
    "reward_shaped_mappo_fair",
    "reward_shaped_macpo_fair",
    "safe_fair_macpo",
)

STANDARD_METHODS: Set[str] = {
    "ippo",
    "mappo",
    "macpo",
    "mappo_lagrangian",
    "dfrl_soto_ggf",
    "safe_fair_macpo",
}

FAIR_BUDGETS: Dict[str, str] = {
    "smamujoco_halfcheetah_2x3_fair": "0.04",
    "safetygym_point_multigoal1_fair": "0.05",
    "safetygym_chargeshare_fair": "0.06",
    "smamujoco_ant_2x4_fair": "0.05",
    "smamujoco_manyagent_ant_fair": "0.07",
}


def completed_500k(log_root: Path = LOG_ROOT) -> Set[Tuple[str, str, int]]:
    completed: Set[Tuple[str, str, int]] = set()
    for cfg_path in log_root.rglob("config.json"):
        eval_path = cfg_path.parent / "eval_summary.json"
        if not eval_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(cfg.get("run_group", "")) != "main":
            continue
        try:
            steps = int(cfg.get("total_steps") or 0)
            seed = int(cfg.get("seed"))
        except Exception:
            continue
        if steps < 500_000:
            continue
        task_obj = cfg.get("task", {})
        method_obj = cfg.get("method", {})
        task = task_obj.get("name") if isinstance(task_obj, dict) else str(task_obj)
        method = method_obj.get("name") if isinstance(method_obj, dict) else str(method_obj)
        if task and method:
            completed.add((task, method, seed))
    return completed


def row_settings(task: str) -> Tuple[int, int, int]:
    # Match the completed 500k rows already in the ledger within each task
    # family. HalfCheetah and ManyAgent used the later promotion settings;
    # Safety-Gym and Ant used the default 500k backend settings.
    if task in {"smamujoco_halfcheetah_2x3_fair", "smamujoco_manyagent_ant_fair"}:
        return 2048, 16, 8
    return 4096, 10, 5


def build_rows(seed: int = 0) -> List[Dict[str, object]]:
    done = completed_500k()
    rows: List[Dict[str, object]] = []
    for task in TARGET_TASKS:
        rollout, eval_episodes, ppo_epochs = row_settings(task)
        for method in TARGET_METHODS:
            if (task, method, seed) in done:
                continue
            priority = 10 if method in STANDARD_METHODS else 20
            phase = "standard_500k_missing" if priority == 10 else "reward_shaped_500k_missing"
            rows.append(
                {
                    "priority": priority,
                    "phase": phase,
                    "run_group": "main",
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "total_steps": 500000,
                    "rollout_steps": rollout,
                    "eval_episodes": eval_episodes,
                    "ppo_epochs": ppo_epochs,
                    "minibatch_size": 512 if rollout == 2048 else 1024,
                    "fair_budget": FAIR_BUDGETS[task],
                    "include_sfb": 0,
                }
            )
    return rows


def write_csv(rows: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "priority",
        "phase",
        "run_group",
        "task",
        "method",
        "seed",
        "total_steps",
        "rollout_steps",
        "eval_episodes",
        "ppo_epochs",
        "minibatch_size",
        "fair_budget",
        "include_sfb",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out = CONFIG_ROOT / "missing_500k_matrix_queue.csv"
    rows = build_rows(seed=0)
    write_csv(rows, out)
    standard = sum(1 for row in rows if int(row["priority"]) == 10)
    optional = len(rows) - standard
    print(f"wrote {out}")
    print(f"missing 500k rows: {len(rows)} total = {standard} standard + {optional} reward-shaped appendix")
    for row in rows:
        print(
            f"{row['priority']:>2} {row['phase']:28s} "
            f"{row['task']:34s} {row['method']:28s} seed={row['seed']}"
        )


if __name__ == "__main__":
    main()
