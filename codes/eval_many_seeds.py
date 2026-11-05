#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "safe_fair_macpo_runs"
LOG_ROOT = RUN_ROOT / "logs"

TASK_ALIASES = {
    "safety_gym_multigoal_fair": "safetygym_point_multigoal1_fair",
    "safetygym_multigoal_fair": "safetygym_point_multigoal1_fair",
    "safety_gym_chargeshare_fair": "safetygym_chargeshare_fair",
    "vmas_safe_multigoal": "safefairbench_vmas_multigoal",
    "highway_merge_fair": "safefairbench_highway_merge",
    "pogema": "safefairbench_pogema_warehouse",
    "pogema_warehouse": "safefairbench_pogema_warehouse",
    "halfcheetah_2x3_fair": "smamujoco_halfcheetah_2x3_fair",
}

METHOD_ALIASES = {
    "sfmacpo": "safe_fair_macpo",
    "sfmacpo_assist": "safe_fair_macpo",
    "mappo_lag": "mappo_lagrangian",
    "mappo_lag_recovery": "mappo_lagrangian_recovery",
    "mappo-lag": "mappo_lagrangian",
}

STRESS_KWARGS = {
    "safetygym_point_multigoal1_fair": {
        "safety_stress": {"hazard_scale": 1.25, "start_near_hazard_prob": 0.25},
        "fairness_stress": {"contested_goal_prob": 0.8, "bottleneck_width_scale": 0.75, "asymmetric_start_bias": True},
    },
    "safetygym_chargeshare_fair": {
        "safety_stress": {"energy_drain_scale": 1.2, "low_energy_threshold_scale": 1.1, "hazard_near_charger": True},
        "fairness_stress": {"charger_count": 1, "contested_charger_start": True, "queue_region_radius_scale": 1.25},
    },
    "safefairbench_vmas_multigoal": {
        "safety_stress": {"hazard_radius_scale": 1.25, "collision_radius_scale": 1.1},
        "fairness_stress": {"bottleneck_width_scale": 0.70, "shared_goal_prob": 1.0},
    },
    "safefairbench_highway_merge": {
        "safety_stress": {"traffic_density_scale": 1.25, "min_gap_scale": 0.80},
        "fairness_stress": {"ramp_vehicle_priority_bias": True, "main_lane_gap_opening_required": True},
    },
}


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for eval_many_seeds manifests") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_task(name: str) -> str:
    return TASK_ALIASES.get(name, name)


def canonical_method(name: str) -> str:
    return METHOD_ALIASES.get(name, name)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def find_run(item: Dict[str, Any]) -> Optional[Path]:
    task = canonical_task(str(item.get("env", "")))
    method = canonical_method(str(item.get("method", "")))
    tag = str(item.get("run_tag", "")).strip()
    candidates: List[Path] = []
    for cfg_path in LOG_ROOT.rglob("config.json"):
        try:
            cfg = read_json(cfg_path)
        except Exception:
            continue
        task_obj = cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}
        method_obj = cfg.get("method", {}) if isinstance(cfg.get("method"), dict) else {}
        if task and task_obj.get("name") != task:
            continue
        if method and method_obj.get("name") != method:
            continue
        cfg_tag = str(cfg.get("run_tag", ""))
        if tag and tag != cfg_tag:
            continue
        if not (cfg_path.parent / "eval_summary.json").exists():
            continue
        candidates.append(cfg_path)
    if not candidates:
        return None
    if len(candidates) > 1:
        rendered = ", ".join(display_path(path.parent) for path in candidates)
        raise RuntimeError(
            f"manifest entry is ambiguous for env={task!r} method={method!r} "
            f"run_tag={tag!r}: {rendered}"
        )
    return candidates[0]


def load_backend():
    os.environ.setdefault("SAFE_FAIR_MACPO_ROOT", str(RUN_ROOT))
    os.environ.setdefault("SAFE_FAIR_RUN_TRAINING", "0")
    os.environ.setdefault("SAFE_FAIR_RUN_EVALUATION", "0")
    os.environ.setdefault("SAFE_FAIR_CLONE_OFFICIAL_REPOS", "0")
    os.environ.setdefault("SAFE_FAIR_QUICK_TEST", "0")
    os.environ.setdefault("SAFE_FAIR_INCLUDE_SAFEFAIRBENCH", "1")
    spec = importlib.util.spec_from_file_location("sfm_backend", ROOT / "codes" / "safe_fair_macpo_experiments.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load backend")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataclass_from_dict(cls, data: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None):
    merged = dict(defaults or {})
    merged.update(data or {})
    allowed = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in merged.items() if k in allowed})


def sem(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def burden_concentration(summary: Dict[str, Any]) -> Dict[str, float]:
    b = np.asarray(summary.get("per_agent_burden_mean", []), dtype=np.float64)
    if b.size == 0 or float(np.sum(b)) <= 1e-8:
        return {"overburdened_agent_share": np.nan, "sacrifice_entropy": np.nan, "mean_total_burden": np.nan}
    p = b / float(np.sum(b))
    entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-12)))) / float(np.log(len(p))) if len(p) > 1 else 1.0
    return {"overburdened_agent_share": float(np.max(p)), "sacrifice_entropy": entropy, "mean_total_burden": float(np.sum(b))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--deterministic-policy", default="true")
    parser.add_argument("--stress-modes", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_obj = load_yaml(Path(args.manifest))
    if isinstance(manifest_obj, dict) and "eval_many_seeds" in manifest_obj:
        eval_section = manifest_obj.get("eval_many_seeds") or {}
        items = eval_section.get("checkpoints", [])
    elif isinstance(manifest_obj, dict):
        items = manifest_obj.get("checkpoints", manifest_obj)
    else:
        items = manifest_obj
    backend = load_backend()
    rows: List[Dict[str, Any]] = []
    deterministic = str(args.deterministic_policy).lower() in {"1", "true", "yes", "on"}
    eval_root = LOG_ROOT / "multiseed_evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)

    for item in items:
        cfg_path = find_run(item)
        if cfg_path is None:
            rows.append({"name": item.get("name", ""), "status": "missing_config", "requested_run_tag": item.get("run_tag", "")})
            continue
        run_dir = cfg_path.parent
        checkpoint = Path(item.get("checkpoint") or (run_dir / "model.pt"))
        if not checkpoint.exists():
            rows.append({"name": item.get("name", ""), "status": "missing_checkpoint", "config_path": str(cfg_path.relative_to(ROOT))})
            continue
        cfg_raw = read_json(cfg_path)
        task_defaults = dataclasses.asdict(backend.TASKS[cfg_raw["task"]["name"]])
        method_defaults = dataclasses.asdict(backend.METHODS[cfg_raw["method"]["name"]])
        base_kwargs = dict(cfg_raw["task"].get("kwargs", {}))
        stress_modes = args.stress_modes or item.get("stress_modes", ["standard"])
        for stress in stress_modes:
            task_data = dict(cfg_raw["task"])
            task_data["kwargs"] = {**base_kwargs, **STRESS_KWARGS.get(task_data["name"], {}).get(stress, {})}
            for eval_seed in args.eval_seeds:
                try:
                    task = dataclass_from_dict(backend.TaskSpec, task_data, task_defaults)
                    method = dataclass_from_dict(backend.MethodSpec, cfg_raw["method"], method_defaults)
                    cfg_data = dict(cfg_raw)
                    cfg_data.update({
                        "task": task,
                        "method": method,
                        "seed": int(eval_seed),
                        "eval_episodes": int(args.episodes),
                        "run_group": "multiseed_evaluation",
                        "run_tag": f"{item.get('name', cfg_raw.get('run_tag', 'eval'))}_{stress}_evalseed{eval_seed}",
                        "init_checkpoint": str(checkpoint),
                    })
                    cfg = dataclass_from_dict(backend.RunConfig, cfg_data)
                    trainer = backend.SafeFairTrainer(cfg)
                    save_path = eval_root / str(item.get("name", cfg.run_tag)) / stress / f"eval_seed_{eval_seed}.json"
                    result = trainer.evaluate(episodes=args.episodes, deterministic=deterministic, save_path=save_path)
                    conc = burden_concentration(result)
                    rows.append({
                        "name": item.get("name", ""),
                        "env": cfg.task.name,
                        "method": cfg.method.name,
                        "source_run_tag": cfg_raw.get("run_tag", ""),
                        "stress_mode": stress,
                        "eval_seed": eval_seed,
                        "episodes": args.episodes,
                        "status": "ok",
                        "return_mean": result.get("return_mean"),
                        "return_sem": sem([result.get("return_mean", np.nan)]),
                        "safety_mean": result.get("safety_cost_mean"),
                        "hard_rate": result.get("hard_violation_rate"),
                        "fairness_mean": result.get("fairness_cost_mean"),
                        "jain_mean": result.get("jain_burden_mean"),
                        "gini_mean": result.get("gini_burden_mean"),
                        "success_rate": result.get("success_rate"),
                        **conc,
                        "config_path": display_path(cfg_path),
                        "checkpoint": display_path(checkpoint),
                    })
                except Exception as exc:
                    rows.append({
                        "name": item.get("name", ""),
                        "stress_mode": stress,
                        "eval_seed": eval_seed,
                        "status": f"error:{type(exc).__name__}",
                        "error": str(exc),
                        "config_path": display_path(cfg_path),
                    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if len(df) and "status" in df:
        ok = df[df["status"] == "ok"].copy()
        if not ok.empty:
            agg_cols = ["return_mean", "safety_mean", "hard_rate", "fairness_mean", "jain_mean", "gini_mean", "success_rate", "mean_total_burden", "overburdened_agent_share", "sacrifice_entropy"]
            summary = ok.groupby(["name", "env", "method", "stress_mode"], as_index=False).agg({c: ["mean", sem] for c in agg_cols if c in ok})
            summary.columns = ["_".join([x for x in col if x]) for col in summary.columns.to_flat_index()]
            summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(f"wrote {out} rows={len(df)}")


if __name__ == "__main__":
    main()
