#!/usr/bin/env python
"""Evaluate a SafeFairBench checkpoint and decompose burden by component."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("SAFE_FAIR_RUN_TRAINING", "0")
os.environ.setdefault("SAFE_FAIR_RUN_EVALUATION", "0")
os.environ.setdefault("SAFE_FAIR_CLONE_OFFICIAL_REPOS", "0")
os.environ.setdefault("SAFE_FAIR_INCLUDE_SAFEFAIRBENCH", "1")
os.environ.setdefault("SAFE_FAIR_QUICK_TEST", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from codes import safe_fair_macpo_experiments as exp  # noqa: E402


def _dataclass_kwargs(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    return {f.name: data[f.name] for f in fields(cls) if f.name in data}


def _load_config(run_dir: Path) -> Any:
    cfg_data = json.loads((run_dir / "config.json").read_text())
    task = exp.TaskSpec(**cfg_data["task"])
    method = exp.MethodSpec(**cfg_data["method"])
    cfg_kwargs = _dataclass_kwargs(exp.RunConfig, cfg_data)
    cfg_kwargs["task"] = task
    cfg_kwargs["method"] = method
    return exp.RunConfig(**cfg_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing config.json and model.pt")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg = _load_config(run_dir)
    cfg.eval_episodes = int(args.episodes)
    trainer = exp.SafeFairTrainer(cfg)
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    trainer.ac.load_state_dict(exp.torch.load(ckpt, map_location=exp.DEVICE))
    save_path = args.output or (run_dir / "burden_decomposition.json")
    result = trainer.evaluate(episodes=args.episodes, deterministic=not args.stochastic, save_path=save_path)
    trainer.close()

    names = result.get("burden_component_names", [])
    components = result.get("per_agent_burden_components_mean", [])
    print(json.dumps({
        "run_dir": str(run_dir),
        "run_tag": result.get("run_tag"),
        "return_mean": result.get("return_mean"),
        "safety_cost_mean": result.get("safety_cost_mean"),
        "hard_violation_rate": result.get("hard_violation_rate"),
        "fairness_cost_mean": result.get("fairness_cost_mean"),
        "jain_burden_mean": result.get("jain_burden_mean"),
        "gini_burden_mean": result.get("gini_burden_mean"),
        "per_agent_burden_mean": result.get("per_agent_burden_mean"),
        "burden_component_names": names,
        "per_agent_burden_components_mean": components,
        "output": str(save_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
