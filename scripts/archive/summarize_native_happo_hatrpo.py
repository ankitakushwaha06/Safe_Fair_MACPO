#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "safe_fair_macpo_runs" / "official_sources" / "happo_hatrpo" / "scripts" / "results" / "mujoco"
TABLES_ROOT = ROOT / "safe_fair_macpo_runs" / "tables"
OUT_CSV = TABLES_ROOT / "external_native_mujoco_results.csv"
OUT_TEX = TABLES_ROOT / "external_native_mujoco_results.tex"


def metric_last(summary: dict, suffix: str) -> float | None:
    for key, value in summary.items():
        if key.endswith(suffix):
            if (
                isinstance(value, list)
                and value
                and isinstance(value[-1], list)
                and len(value[-1]) >= 3
            ):
                return float(value[-1][-1])
    return None


def collect_rows() -> list[dict]:
    rows = []
    for summary_path in sorted(RESULTS_ROOT.glob("*/*/*/*/run*/logs/summary.json")):
        parts = summary_path.parts
        # ... / mujoco / <scenario> / <algo> / <exp_name> / <seed> / <run> / logs / summary.json
        scenario = parts[-7]
        algo = parts[-6]
        exp_name = parts[-5]
        seed = int(parts[-4])
        run_name = parts[-3]

        with summary_path.open() as fh:
            summary = json.load(fh)

        models_dir = summary_path.parent.parent / "models"
        checkpoint_count = len(list(models_dir.glob("*.pt")))

        row = {
            "scenario": scenario,
            "algorithm": algo,
            "experiment": exp_name,
            "seed": seed,
            "run_name": run_name,
            "total_steps": int(metric_last(summary, "/eval_average_episode_rewards/eval_average_episode_rewards")[1] if False else 0),
            "eval_average_episode_rewards": metric_last(summary, "/eval_average_episode_rewards/eval_average_episode_rewards"),
            "eval_max_episode_rewards": metric_last(summary, "/eval_max_episode_rewards/eval_max_episode_rewards"),
            "train_episode_rewards": metric_last(summary, "/train_episode_rewards/aver_rewards"),
            "agent0_average_step_rewards": metric_last(summary, "/agent0/average_step_rewards"),
            "agent1_average_step_rewards": metric_last(summary, "/agent1/average_step_rewards"),
            "agent0_value_loss": metric_last(summary, "/agent0/value_loss"),
            "agent1_value_loss": metric_last(summary, "/agent1/value_loss"),
            "checkpoint_count": checkpoint_count,
            "summary_json": str(summary_path),
            "models_dir": str(models_dir),
        }
        # recover steps from any metric triple
        for value in summary.values():
            if (
                isinstance(value, list)
                and value
                and isinstance(value[-1], list)
                and len(value[-1]) >= 3
            ):
                row["total_steps"] = int(value[-1][1])
                break
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "algorithm",
        "experiment",
        "seed",
        "run_name",
        "total_steps",
        "eval_average_episode_rewards",
        "eval_max_episode_rewards",
        "train_episode_rewards",
        "agent0_average_step_rewards",
        "agent1_average_step_rewards",
        "agent0_value_loss",
        "agent1_value_loss",
        "checkpoint_count",
        "summary_json",
        "models_dir",
    ]
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{x:.2f}"


def write_tex(rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Scenario & Algorithm & Steps & Eval Avg Reward & Train Reward \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['scenario']} & {row['algorithm'].upper()} & {row['total_steps']} & "
            f"{fmt(row['eval_average_episode_rewards'])} & {fmt(row['train_episode_rewards'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    OUT_TEX.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = collect_rows()
    rows.sort(key=lambda r: (r["scenario"], r["algorithm"], r["seed"], r["experiment"], r["run_name"]))
    write_csv(rows)
    write_tex(rows)
    for row in rows:
        print(
            f"{row['scenario']:14s} {row['algorithm']:6s} seed={row['seed']} steps={row['total_steps']:6d} "
            f"eval={fmt(row['eval_average_episode_rewards'])} train={fmt(row['train_episode_rewards'])}"
        )
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
