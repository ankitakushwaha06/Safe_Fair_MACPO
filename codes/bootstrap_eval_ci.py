#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def bootstrap_delta(a: np.ndarray, b: np.ndarray, n: int = 5000) -> tuple:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(20260505)
    vals = []
    for _ in range(n):
        vals.append(float(np.mean(rng.choice(a, size=a.size, replace=True)) - np.mean(rng.choice(b, size=b.size, replace=True))))
    vals = np.asarray(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="safe_fair_macpo_runs/tables/multiseed_evaluation.csv")
    parser.add_argument("--out", default="safe_fair_macpo_runs/tables/multiseed_evaluation_ci.csv")
    parser.add_argument("--sf-pattern", default="safe_fair|sfmacpo")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df = df[df.get("status", "ok") == "ok"].copy()
    rows: List[dict] = []
    for (env, stress), g in df.groupby(["env", "stress_mode"]):
        sf = g[g["method"].astype(str).str.contains(args.sf_pattern, case=False, regex=True)]
        for baseline, b in g[~g.index.isin(sf.index)].groupby("method"):
            if sf.empty:
                continue
            for metric in ["return_mean", "fairness_mean", "jain_mean", "gini_mean", "overburdened_agent_share", "sacrifice_entropy"]:
                mean, lo, hi = bootstrap_delta(sf[metric].to_numpy(float), b[metric].to_numpy(float))
                rows.append({
                    "env": env,
                    "stress_mode": stress,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_delta_sf_minus_baseline": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "note": "fixed-checkpoint evaluation-seed interval, not training-seed uncertainty",
                })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
