#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_num(v: Any) -> float:
    if v is None or pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    text = str(v).replace("\\pm", "±").split("±", 1)[0].strip()
    try:
        return float(text)
    except Exception:
        return np.nan


def load_all(paths: List[str]) -> pd.DataFrame:
    frames = []
    for pattern in paths:
        for path_str in glob.glob(pattern):
            path = Path(path_str)
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df["source"] = str(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def classify(row: pd.Series) -> str:
    hard = parse_num(row.get("hard_rate", row.get("hard", row.get("hard_violation_rate", np.nan))))
    safety = parse_num(row.get("safety_mean", row.get("safety", row.get("safety_cost_mean", np.nan))))
    ret = parse_num(row.get("return_mean", row.get("return", np.nan)))
    fair = parse_num(row.get("fairness_mean", row.get("fairness", row.get("fair", np.nan))))
    if not np.isfinite(ret) and not np.isfinite(fair):
        return "BUG / RERUN"
    if np.isfinite(hard) and hard > 0:
        return "NEGATIVE BUT INFORMATIVE"
    if np.isfinite(safety) and safety > 0:
        return "APPENDIX SUPPORT"
    if str(row.get("method", "")).find("safe_fair") >= 0:
        return "PROMOTE"
    return "APPENDIX SUPPORT"


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=["safe_fair_macpo_runs/tables/*.csv"])
    parser.add_argument("--out", default="safe_fair_macpo_runs/tables/evaluation_summary.md")
    args = parser.parse_args()
    df = load_all(args.inputs)
    lines = ["# Reproducibility Evaluation Summary", "", "These classifications are triage labels; verify exact paper claims against the source rows before editing text.", ""]
    if df.empty:
        lines.append("No CSV rows found.")
    else:
        normalized: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            env = row.get("env", row.get("task", row.get("Environment", "")))
            method = row.get("method", row.get("Method", ""))
            if not env or not method:
                continue
            item = {
                "env": env,
                "method": method,
                "run_tag": row.get("run_tag", row.get("tag", row.get("Run Tag", ""))),
                "return": parse_num(row.get("return_mean", row.get("return", row.get("Return", np.nan)))),
                "safety": parse_num(row.get("safety_mean", row.get("safety", row.get("safety_cost_mean", np.nan)))),
                "hard": parse_num(row.get("hard_rate", row.get("hard", row.get("hard_violation_rate", np.nan)))),
                "fairness": parse_num(row.get("fairness_mean", row.get("fairness", row.get("fair", np.nan)))),
                "source": row.get("source", ""),
            }
            item["classification"] = classify(pd.Series(item))
            normalized.append(item)
        triage = pd.DataFrame(normalized).drop_duplicates()
        for label in ["PROMOTE", "APPENDIX SUPPORT", "NEGATIVE BUT INFORMATIVE", "BUG / RERUN"]:
            sub = triage[triage["classification"] == label].head(30)
            lines.extend([f"## {label}", ""])
            if sub.empty:
                lines.extend(["None.", ""])
                continue
            lines.append(markdown_table(sub[["env", "method", "run_tag", "return", "safety", "hard", "fairness", "source"]]))
            lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
