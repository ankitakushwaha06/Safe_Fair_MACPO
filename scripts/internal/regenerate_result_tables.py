#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from codes.release_runs import DEFAULT_MANIFEST, collect_selected_runs, missing_identities
LOG_ROOT = ROOT / "safe_fair_macpo_runs" / "logs"
TABLE_ROOT = ROOT / "safe_fair_macpo_runs" / "tables"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_manifest_eval_rows(
    log_root: Path = LOG_ROOT, manifest: Path = DEFAULT_MANIFEST
) -> pd.DataFrame:
    return pd.DataFrame([summary for _, _, summary in collect_selected_runs(log_root, manifest)])


def mean_std_str(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x.iloc[0]:.3f}"
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def make_summary_table(df: pd.DataFrame, run_group: str = "main") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sub = df[df["run_group"] == run_group].copy()
    metrics = [
        "return_mean",
        "safety_cost_mean",
        "violation_rate",
        "success_rate",
        "jain_burden_mean",
        "envy_gap_mean",
        "debt_auc_mean",
    ]
    rows: List[Dict[str, str]] = []
    for (task, method), g in sub.groupby(["task", "method"]):
        row: Dict[str, str] = {"Environment": task, "Method": method}
        tags = sorted({str(x) for x in g.get("run_tag", pd.Series(dtype=str)).fillna("") if str(x)})
        row["Run Tag"] = tags[-1] if tags else ""
        for metric in metrics:
            row[metric] = mean_std_str(g[metric]) if metric in g else "--"
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["Environment", "Method", "Run Tag", *metrics])
    return pd.DataFrame(rows).sort_values(["Environment", "Method"]).reset_index(drop=True)


def write_tables(name: str, table: pd.DataFrame, table_root: Path = TABLE_ROOT) -> None:
    table_root.mkdir(parents=True, exist_ok=True)
    path_csv = table_root / f"{name}.csv"
    path_tex = table_root / f"{name}.tex"
    table.to_csv(path_csv, index=False)
    try:
        table.to_latex(path_tex, index=False, escape=False)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate aggregate result tables from evaluation logs.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "safe_fair_macpo_runs",
        help="directory containing logs/ and tables/",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    selected = collect_selected_runs(args.run_root / "logs", args.manifest)
    missing = missing_identities(selected, args.manifest)
    if args.require_complete and missing:
        raise SystemExit(f"missing {len(missing)} frozen runs; first missing identity: {missing[0]}")
    eval_df = pd.DataFrame([summary for _, _, summary in selected])
    main_table = make_summary_table(eval_df, "main")
    ablation_table = make_summary_table(eval_df, "ablation")
    safefairbench_ablation_table = make_summary_table(eval_df, "safefairbench_ablation")
    recovery_table = make_summary_table(eval_df, "safefairbench")
    sweep_table = make_summary_table(eval_df, "fairness_budget_sweep")
    write_tables("main_results", main_table, args.run_root / "tables")
    write_tables("ablation_results", ablation_table, args.run_root / "tables")
    write_tables("safefairbench_ablation_results", safefairbench_ablation_table, args.run_root / "tables")
    write_tables("recovery_control_results", recovery_table, args.run_root / "tables")
    write_tables("fairness_sweep", sweep_table, args.run_root / "tables")
    print(f"selected rows={len(eval_df)} missing={len(missing)} main_results={len(main_table)}")


if __name__ == "__main__":
    main()
