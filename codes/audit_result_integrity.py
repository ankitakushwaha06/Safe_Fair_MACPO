#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "safe_fair_macpo_runs"
TABLE_ROOT = RUN_ROOT / "tables"
LOG_ROOT = RUN_ROOT / "logs"

METRIC_COLUMNS = {
    "return": ["return", "return_mean", "Return", "Return ↑"],
    "safety": ["safety", "safety_cost_mean", "Safety", "Safety cost ↓"],
    "hard": ["hard", "hard_violation_rate", "Hard", "Hard violation ↓"],
    "fairness": ["fair", "fairness", "fairness_cost_mean", "Fairness"],
    "jain": ["jain", "jain_burden_mean", "Jain", "Jain burden ↑"],
    "gini": ["gini", "gini_burden_mean", "Gini", "Gini burden ↓"],
}


def stable_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def file_hash(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_num(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"--", "nan", "None"}:
        return math.nan
    text = text.replace("$", "").replace("\\pm", "±")
    text = text.split("±", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return math.nan


def first_present(row: pd.Series, names: Iterable[str]) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def display_path(path: Path, root: Path = ROOT) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def normalize_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out: Dict[str, Any] = {
            "source_table": display_path(path),
            "env": first_present(row, ["env", "task", "Environment"]),
            "method": first_present(row, ["method", "Method"]),
            "run_tag": first_present(row, ["run_tag", "tag", "Run Tag"]),
            "seed": parse_num(first_present(row, ["seed", "Seed"])),
            "budget": parse_num(first_present(row, ["budget", "steps", "total_steps"])),
            "log_dir": first_present(row, ["dir", "_path", "log_dir"]),
        }
        for canonical, names in METRIC_COLUMNS.items():
            out[canonical] = parse_num(first_present(row, names))
        rows.append(out)
    return pd.DataFrame(rows)


def collect_log_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for eval_path in LOG_ROOT.rglob("eval_summary.json"):
        try:
            summary = read_json(eval_path)
        except Exception:
            continue
        run_dir = eval_path.parent
        cfg_path = run_dir / "config.json"
        cfg: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                cfg = read_json(cfg_path)
            except Exception:
                cfg = {}
        task_obj = cfg.get("task") if isinstance(cfg.get("task"), dict) else {}
        method_obj = cfg.get("method") if isinstance(cfg.get("method"), dict) else {}
        checkpoint = run_dir / "model.pt"
        row = {
            "source_table": "logs/eval_summary.json",
            "env": summary.get("task") or task_obj.get("name"),
            "method": summary.get("method") or method_obj.get("name"),
            "run_tag": summary.get("run_tag") or cfg.get("run_tag", ""),
            "seed": int(summary.get("seed", cfg.get("seed", 0))),
            "budget": int(summary.get("global_step", cfg.get("total_steps", 0) or 0)),
            "config_hash": stable_hash(cfg) if cfg else "",
            "checkpoint_path": str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else "",
            "checkpoint_hash": file_hash(checkpoint),
            "return": parse_num(summary.get("return_mean")),
            "safety": parse_num(summary.get("safety_cost_mean")),
            "hard": parse_num(summary.get("hard_violation_rate", summary.get("violation_rate"))),
            "fairness": parse_num(summary.get("fairness_cost_mean")),
            "jain": parse_num(summary.get("jain_burden_mean")),
            "gini": parse_num(summary.get("gini_burden_mean")),
            "log_dir": str(run_dir.relative_to(ROOT)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_table_rows(table_rows: pd.DataFrame, log_rows: pd.DataFrame) -> pd.DataFrame:
    if table_rows.empty or log_rows.empty:
        return table_rows
    enriched = []
    for _, row in table_rows.iterrows():
        out = row.to_dict()
        matches = log_rows.copy()
        if pd.notna(row.get("env")):
            matches = matches[matches["env"].astype(str) == str(row.get("env"))]
        if pd.notna(row.get("method")):
            matches = matches[matches["method"].astype(str) == str(row.get("method"))]
        tag = str(row.get("run_tag") or "").strip()
        if tag and "run_tag" in matches:
            matches = matches[matches["run_tag"].astype(str) == tag]
        seed = row.get("seed")
        if pd.notna(seed):
            matches = matches[matches["seed"] == int(seed)]
        if len(matches) == 1:
            for key in ["config_hash", "checkpoint_path", "checkpoint_hash", "log_dir"]:
                out[key] = matches.iloc[0].get(key, "")
        enriched.append(out)
    return pd.DataFrame(enriched)


def duplicate_audit(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    audit_rows: List[Dict[str, Any]] = []
    numeric = ["return", "safety", "hard", "fairness", "jain", "gini"]
    work = rows.copy()
    for col in numeric:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    for env, g in work.groupby("env", dropna=False):
        records = g.to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                if str(a.get("method")) == str(b.get("method")) and str(a.get("run_tag")) == str(b.get("run_tag")):
                    continue
                vals_a = [a.get(c) for c in numeric]
                vals_b = [b.get(c) for c in numeric]
                comparable = [k for k, (x, y) in enumerate(zip(vals_a, vals_b)) if not (pd.isna(x) or pd.isna(y))]
                same_metrics = bool(comparable) and all(float(vals_a[k]) == float(vals_b[k]) for k in comparable)
                same_config = bool(a.get("config_hash")) and a.get("config_hash") == b.get("config_hash")
                same_checkpoint = bool(a.get("checkpoint_hash")) and a.get("checkpoint_hash") == b.get("checkpoint_hash")
                suspicious_pair = same_metrics or same_config or same_checkpoint
                if suspicious_pair:
                    audit_rows.append(
                        {
                            "env": env,
                            "method_a": a.get("method"),
                            "method_b": b.get("method"),
                            "run_tag_a": a.get("run_tag", ""),
                            "run_tag_b": b.get("run_tag", ""),
                            "seed_a": a.get("seed", ""),
                            "seed_b": b.get("seed", ""),
                            "same_metrics": same_metrics,
                            "same_config": same_config,
                            "same_checkpoint": same_checkpoint,
                            "checkpoint_a": a.get("checkpoint_path", ""),
                            "checkpoint_b": b.get("checkpoint_path", ""),
                            "metrics": json.dumps({c: a.get(c) for c in numeric}, sort_keys=True),
                        }
                    )
    return pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="*", default=[
        str(TABLE_ROOT / "main_results.csv"),
        str(TABLE_ROOT / "all_eval_rows_current.csv"),
        str(TABLE_ROOT / "ablation_results.csv"),
    ])
    parser.add_argument("--out", default=str(TABLE_ROOT / "integrity_audit.csv"))
    parser.add_argument("--duplicates-out", default=str(TABLE_ROOT / "duplicate_row_audit.csv"))
    args = parser.parse_args()

    log_rows = collect_log_rows()
    table_frames = [normalize_table(Path(p)) for p in args.tables]
    table_rows = pd.concat([f for f in table_frames if not f.empty], ignore_index=True) if any(not f.empty for f in table_frames) else pd.DataFrame()
    table_rows = enrich_table_rows(table_rows, log_rows)
    combined = pd.concat([table_rows, log_rows], ignore_index=True, sort=False)
    keep = [
        "env", "method", "run_tag", "seed", "budget", "config_hash",
        "checkpoint_path", "checkpoint_hash", "return", "safety", "hard",
        "fairness", "jain", "gini", "source_table", "log_dir",
    ]
    for col in keep:
        if col not in combined:
            combined[col] = ""
    out_path = Path(args.out)
    duplicates_path = Path(args.duplicates_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    combined[keep].to_csv(out_path, index=False)
    dup = duplicate_audit(combined[keep])
    dup.to_csv(duplicates_path, index=False)
    print(f"wrote {args.out} rows={len(combined)}")
    print(f"wrote {args.duplicates_out} suspicious_pairs={len(dup)}")
    if len(dup):
        print(dup.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
