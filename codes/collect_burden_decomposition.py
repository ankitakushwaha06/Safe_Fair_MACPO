#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs"
TABLE_ROOT = ROOT / "safe_fair_macpo_runs" / "tables"


CANONICAL_COMPONENTS = [
    "burden_waiting",
    "burden_hazard",
    "burden_yield",
    "burden_detour",
    "burden_contested_access",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return np.asarray([])
    return np.asarray(value, dtype=np.float64)


def collect_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for eval_path in LOG_ROOT.rglob("eval_summary.json"):
        try:
            summary = load_json(eval_path)
        except Exception:
            continue
        comps = as_array(summary.get("per_agent_burden_components_mean", []))
        totals = as_array(summary.get("per_agent_burden_mean", []))
        names = summary.get("burden_component_names", ["wait", "hazard", "yield", "detour", "resource"])
        if comps.ndim != 2 or comps.shape[0] == 0:
            continue
        env = summary.get("task", "")
        method = summary.get("method", "")
        run_tag = summary.get("run_tag", "")
        for agent_id in range(comps.shape[0]):
            row: Dict[str, Any] = {
                "env": env,
                "method": method,
                "run_tag": run_tag,
                "eval_seed": summary.get("seed", ""),
                "episode": "mean",
                "agent_id": agent_id,
                "burden_total": float(totals[agent_id]) if totals.size > agent_id else float(np.sum(comps[agent_id])),
                "log_dir": str(eval_path.parent.relative_to(ROOT)),
            }
            for c in CANONICAL_COMPONENTS:
                row[c] = 0.0
            for idx, name in enumerate(names):
                if idx >= comps.shape[1]:
                    continue
                key = {
                    "wait": "burden_waiting",
                    "hazard": "burden_hazard",
                    "yield": "burden_yield",
                    "detour": "burden_detour",
                    "resource": "burden_contested_access",
                }.get(str(name), f"burden_{name}")
                row[key] = float(comps[agent_id, idx])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_bars(df: pd.DataFrame, fig_dir: Path, max_plots: int = 40) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    preferred_envs = [
        "safetygym_point_multigoal1_fair",
        "safetygym_chargeshare_fair",
        "safefairbench_highway_merge",
        "safefairbench_vmas_multigoal",
    ]
    sub = df[df["env"].isin(preferred_envs)].copy()
    if sub.empty:
        sub = df.copy()
    for (env, method, run_tag), g in sub.groupby(["env", "method", "run_tag"], dropna=False):
        if plotted >= max_plots:
            break
        g = g.sort_values("agent_id")
        x = np.arange(len(g))
        fig, ax = plt.subplots(figsize=(6.3, 3.8))
        bottom = np.zeros(len(g))
        for col in CANONICAL_COMPONENTS:
            vals = g[col].to_numpy(dtype=float)
            if np.allclose(vals, 0.0):
                continue
            ax.bar(x, vals, bottom=bottom, label=col.replace("burden_", ""))
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"A{int(a)}" for a in g["agent_id"]])
        ax.set_ylabel("Cumulative burden")
        title_tag = str(run_tag)[:36]
        ax.set_title(f"{env} / {method} / {title_tag}")
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.legend(fontsize=8)
        fig.tight_layout()
        slug = "_".join(str(x) for x in [env, method, run_tag])[:180].replace("/", "_").replace(" ", "_")
        fig.savefig(fig_dir / f"{slug}.pdf")
        plt.close(fig)
        plotted += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-seeds", nargs="*", default=[])
    parser.add_argument("--out", default=str(TABLE_ROOT / "burden_decomposition.csv"))
    parser.add_argument("--fig-dir", default=str(ROOT / "figures" / "burden_decomposition"))
    args = parser.parse_args()
    df = collect_rows()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    plot_bars(df, Path(args.fig_dir))
    print(f"wrote {args.out} rows={len(df)}")
    print(f"wrote burden decomposition plots under {args.fig_dir}")


if __name__ == "__main__":
    main()
