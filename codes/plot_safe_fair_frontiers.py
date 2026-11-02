#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "safe_fair_macpo_runs" / "tables"


def parse_num(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or text == "--":
        return np.nan
    text = text.replace("\\pm", "±").split("±", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def first(row: pd.Series, names: Iterable[str]) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def load_inputs(paths: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            try:
                source = str(path.resolve().relative_to(ROOT))
            except Exception:
                source = str(path)
            rows.append({
                "source": source,
                "env": first(r, ["env", "task", "Environment"]),
                "method": first(r, ["method", "Method"]),
                "run_tag": first(r, ["run_tag", "tag", "Run Tag"]),
                "return": parse_num(first(r, ["return", "return_mean", "Return", "Return ↑"])),
                "safety": parse_num(first(r, ["safety", "safety_cost_mean", "Safety", "Safety cost ↓"])),
                "hard": parse_num(first(r, ["hard", "hard_violation_rate", "Hard", "Hard violation ↓"])),
                "fairness": parse_num(first(r, ["fair", "fairness", "fairness_mean", "fairness_cost_mean", "Fairness"])),
                "jain": parse_num(first(r, ["jain", "jain_burden_mean", "Jain", "Jain burden ↑"])),
                "gini": parse_num(first(r, ["gini", "gini_burden_mean", "Gini", "Gini burden ↓"])),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.dropna(subset=["env", "method"], how="any")
    return out


def nondominated(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    sub = df.dropna(subset=[x_col, y_col]).copy()
    keep = []
    for i, row in sub.iterrows():
        dominated = ((sub[x_col] <= row[x_col]) & (sub[y_col] >= row[y_col]) & ((sub[x_col] < row[x_col]) | (sub[y_col] > row[y_col]))).any()
        if not dominated:
            keep.append(i)
    return sub.loc[keep]


def plot_env(env: str, df: pd.DataFrame, out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    env_slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in env)
    panels = [
        ("fairness", "return", "Fairness cost", "Return", "return_vs_fairness"),
        ("jain", "return", "Jain burden", "Return", "return_vs_jain"),
        ("safety", "fairness", "Safety cost", "Fairness cost", "fairness_vs_safety"),
    ]
    for x_col, y_col, x_label, y_label, suffix in panels:
        sub = df.dropna(subset=[x_col, y_col]).copy()
        if sub.empty:
            continue
        exact = (sub["hard"].fillna(1.0) == 0.0) & (sub["safety"].fillna(1.0) == 0.0)
        hard_safe = sub["hard"].fillna(1.0) == 0.0
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        unsafe = sub[~hard_safe]
        ax.scatter(unsafe[x_col], unsafe[y_col], c="#b8b8b8", s=34, alpha=0.45, label="unsafe")
        hs = sub[hard_safe & ~exact]
        ax.scatter(hs[x_col], hs[y_col], c="#4c78a8", s=42, alpha=0.75, label="hard-safe")
        ex = sub[exact]
        ax.scatter(ex[x_col], ex[y_col], c="#2ca02c", s=58, alpha=0.9, label="exact-safe")
        for _, r in sub.iterrows():
            label = str(r["method"]).replace("safe_fair_", "sf_").replace("reward_shaped_", "rs_")
            ax.annotate(label, (r[x_col], r[y_col]), fontsize=7, alpha=0.75)
        ax.set_title(env)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{env_slug}_{suffix}.pdf"
        fig.savefig(path)
        plt.close(fig)
        frontier = nondominated(sub[exact], x_col, y_col)
        if x_col == "jain":
            # For Jain, higher is better on x; summarize separately by return winner.
            frontier = sub[exact].sort_values([y_col, x_col], ascending=[False, False]).head(5)
        rows.append({
            "env": env,
            "plot": suffix,
            "safety_filter": "exact-safe",
            "non_dominated_methods": "; ".join(sorted(set(frontier["method"].astype(str)))) if len(frontier) else "",
            "best_safe_fair_point": "; ".join(frontier[frontier["method"].astype(str).str.contains("safe_fair|sfmacpo", case=False, regex=True)]["method"].astype(str).head(3)),
            "best_baseline_point": "; ".join(frontier[~frontier["method"].astype(str).str.contains("safe_fair|sfmacpo", case=False, regex=True)]["method"].astype(str).head(3)),
            "path": str(path),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=[
        str(TABLE_ROOT / "main_results.csv"),
        str(TABLE_ROOT / "all_eval_rows_current.csv"),
        str(TABLE_ROOT / "ablation_results.csv"),
        str(TABLE_ROOT / "burden_summary_recomputed.csv"),
    ])
    parser.add_argument("--out-dir", default=str(ROOT / "figures" / "frontiers"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args.inputs).drop_duplicates()
    rows: List[Dict[str, Any]] = []
    for env, sub in df.groupby("env"):
        rows.extend(plot_env(str(env), sub, out_dir))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "frontier_summary.csv", index=False)
    print(f"wrote {len(summary)} frontier plot summaries to {out_dir}")


if __name__ == "__main__":
    main()
