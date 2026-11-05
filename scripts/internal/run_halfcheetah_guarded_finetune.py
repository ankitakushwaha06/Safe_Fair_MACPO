#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUEUE_DIR = ROOT / "safe_fair_macpo_runs" / "logs" / "queue"
STATUS_FILE = QUEUE_DIR / os.environ.get("SAFE_FAIR_HC_GUARD_STATUS_FILE", "halfcheetah_guarded_finetune_status.tsv")
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

APT_ROOT = ROOT / "vendor" / "apt" / "extracted"
MUJOCO_PATH = Path.home() / ".mujoco" / "mujoco210"


def prepend_env_paths(name: str, paths: list[Path]) -> None:
    existing = [p for p in os.environ.get(name, "").split(":") if p]
    prefix = [str(p) for p in paths if p.exists()]
    os.environ[name] = ":".join(prefix + existing)


include_paths = [
    APT_ROOT / "libosmesa6-dev" / "usr" / "include",
    APT_ROOT / "libglew-dev" / "usr" / "include",
    APT_ROOT / "libglu1-mesa-dev" / "usr" / "include",
    APT_ROOT / "libgl-dev" / "usr" / "include",
]
library_paths = [
    MUJOCO_PATH / "bin",
    APT_ROOT / "libosmesa6-dev" / "usr" / "lib" / "x86_64-linux-gnu",
    APT_ROOT / "libosmesa6" / "usr" / "lib" / "x86_64-linux-gnu",
    APT_ROOT / "libglew-dev" / "usr" / "lib" / "x86_64-linux-gnu",
    APT_ROOT / "libglew2.2" / "usr" / "lib" / "x86_64-linux-gnu",
    APT_ROOT / "libglu1-mesa-dev" / "usr" / "lib" / "x86_64-linux-gnu",
    APT_ROOT / "libglu1-mesa" / "usr" / "lib" / "x86_64-linux-gnu",
    APT_ROOT / "libgl-dev" / "usr" / "lib" / "x86_64-linux-gnu",
    APT_ROOT / "libgl1" / "usr" / "lib" / "x86_64-linux-gnu",
    Path("/usr/lib/x86_64-linux-gnu"),
]

prepend_env_paths("C_INCLUDE_PATH", include_paths)
prepend_env_paths("CPATH", include_paths)
prepend_env_paths("LIBRARY_PATH", library_paths)
prepend_env_paths("LD_LIBRARY_PATH", library_paths)
prepend_env_paths("PATH", [APT_ROOT / "patchelf" / "usr" / "bin"])

os.environ.setdefault("SAFE_FAIR_MACPO_ROOT", str(ROOT / "safe_fair_macpo_runs"))
os.environ.setdefault("SAFE_FAIR_QUICK_TEST", "0")
os.environ.setdefault("SAFE_FAIR_RUN_TRAINING", "0")
os.environ.setdefault("SAFE_FAIR_RUN_EVALUATION", "1")
os.environ.setdefault("SAFE_FAIR_CLONE_OFFICIAL_REPOS", "0")
os.environ.setdefault("SAFE_FAIR_USE_CUDA", "1")
os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(MUJOCO_PATH))
os.environ.setdefault("MUJOCO_PY_FORCE_CPU", "1")

spec = importlib.util.spec_from_file_location(
    "safe_fair_macpo_experiments",
    ROOT / "codes" / "safe_fair_macpo_experiments.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load experiment backend")
exp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exp
spec.loader.exec_module(exp)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def find_base_checkpoint(seed: int = 0) -> Path:
    log_root = (
        ROOT
        / "safe_fair_macpo_runs"
        / "logs"
        / "main"
        / "smamujoco_halfcheetah_2x3_fair"
        / "safe_fair_macpo"
    )
    preferred_tags_env = os.environ.get("SAFE_FAIR_HC_GUARD_BASE_TAGS", "").strip()
    if preferred_tags_env:
        preferred_tags = [x.strip() for x in preferred_tags_env.split(",") if x.strip()]
    else:
        preferred_tags = [
            f"extended_benchmark_180k_main_hc_pareto_d_s{seed}",
            f"promoted_multiseed_20260426_180k_base_hc_pareto_d_s{seed}",
            f"hc_pareto_d_s{seed}",
            "hc_sf_pareto_d",
        ]
    for tag in preferred_tags:
        for cfg_path in log_root.glob("seed_*/config.json"):
            cfg = load_json(cfg_path)
            ckpt = cfg_path.parent / "model.pt"
            eval_path = cfg_path.parent / "eval_summary.json"
            if int(cfg.get("seed", -1)) == int(seed) and cfg.get("run_tag") == tag and ckpt.exists() and eval_path.exists():
                print(f"Using base checkpoint {ckpt} from tag {tag}", flush=True)
                return ckpt
    raise FileNotFoundError(f"No safe HalfCheetah hc_pareto_d checkpoint found for seed {seed}")


def state_dict_to_cpu(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def metric(result: dict, key: str, default: float) -> float:
    try:
        return float(result.get(key, default))
    except (TypeError, ValueError):
        return default


def is_safe(result: dict, threshold: float) -> bool:
    hard = metric(result, "hard_violation_rate", metric(result, "violation_rate", 1.0))
    safety = metric(result, "safety_cost_mean", float("inf"))
    return hard <= 0.0 and safety <= threshold


def is_fair_enough(result: dict) -> bool:
    fair = metric(result, "fairness_cost_mean", float("inf"))
    jain = metric(result, "jain_burden_mean", 0.0)
    return fair <= 250.0 and jain >= 0.80


def score(result: dict) -> float:
    ret = metric(result, "return_mean", -1e9)
    fair = metric(result, "fairness_cost_mean", 1e9)
    jain = metric(result, "jain_burden_mean", 0.0)
    return ret - 0.25 * fair + 200.0 * jain


def write_status(row: dict) -> None:
    header = [
        "tag",
        "status",
        "update",
        "return",
        "safety",
        "hard",
        "fair",
        "jain",
        "best_tag",
        "elapsed_s",
        "utc_time",
    ]
    if not STATUS_FILE.exists():
        STATUS_FILE.write_text("\t".join(header) + "\n", encoding="utf-8")
    values = [str(row.get(k, "")) for k in header]
    with STATUS_FILE.open("a", encoding="utf-8") as f:
        f.write("\t".join(values) + "\n")


def summarize_result(result: dict) -> dict:
    return {
        "return": f"{metric(result, 'return_mean', float('nan')):.2f}",
        "safety": f"{metric(result, 'safety_cost_mean', float('nan')):.2f}",
        "hard": f"{metric(result, 'hard_violation_rate', metric(result, 'violation_rate', float('nan'))):.2f}",
        "fair": f"{metric(result, 'fairness_cost_mean', float('nan')):.2f}",
        "jain": f"{metric(result, 'jain_burden_mean', float('nan')):.3f}",
    }


def make_cfg(tag: str, base_checkpoint: Path, params: dict, seed: int):
    cfg = exp.make_run_config(
        "smamujoco_halfcheetah_2x3_fair",
        "safe_fair_macpo",
        seed=seed,
        run_group="main",
        quick=False,
    )
    cfg.run_tag = tag
    cfg.total_steps = 320_000
    cfg.rollout_steps = 4096
    cfg.ppo_epochs = 8
    cfg.minibatch_size = 512
    cfg.eval_episodes = 16
    cfg.init_checkpoint = str(base_checkpoint)
    for key, value in params.items():
        setattr(cfg, key, value)
    return cfg


def run_guarded_variant(tag: str, base_checkpoint: Path, params: dict, seed: int) -> dict:
    start = time.time()
    cfg = make_cfg(tag, base_checkpoint, params, seed)
    trainer = exp.SafeFairTrainer(cfg)
    threshold = float(cfg.task.safety_budget_per_step * cfg.task.episode_length)
    n_updates = max(1, int(math.ceil(cfg.total_steps / cfg.rollout_steps)))
    eval_every = int(params.get("guard_eval_every_updates", 4))
    guard_eval_episodes = int(params.get("guard_eval_episodes", 16))
    patience = int(params.get("guard_unsafe_patience", 1))
    min_updates = int(params.get("guard_min_updates", 8))
    unsafe_streak = 0
    best_state = None
    best_result = None
    best_kind = "none"
    candidates = []

    def consider(update: int, result: dict) -> None:
        nonlocal best_state, best_result, best_kind
        if not is_safe(result, threshold):
            return
        kind = "safe_fair" if is_fair_enough(result) else "safe"
        state = state_dict_to_cpu(trainer.ac)
        candidate = {
            "state": state,
            "result": dict(result),
            "kind": kind,
            "score": score(result),
            "update": update,
        }
        candidates.append(candidate)
        if best_result is None:
            best_state = state
            best_result = dict(result)
            best_kind = kind
            return
        best_is_fair = best_kind == "safe_fair"
        current_is_fair = kind == "safe_fair"
        should_replace = False
        if current_is_fair and not best_is_fair:
            should_replace = True
        elif current_is_fair == best_is_fair and score(result) > score(best_result):
            should_replace = True
        if should_replace:
            best_state = state
            best_result = dict(result)
            best_kind = kind
        if should_replace or best_result.get("guard_update") is None:
            best_result["guard_update"] = update

    print(f"==== GUARDED HC FINETUNE {tag} ====", flush=True)
    initial = trainer.evaluate(
        episodes=guard_eval_episodes,
        deterministic=True,
        save_path=trainer.run_dir / "guard_eval_update_000.json",
    )
    initial["guard_update"] = 0
    consider(0, initial)
    write_status(
        {
            "tag": tag,
            "status": "initial_eval",
            "update": 0,
            **summarize_result(initial),
            "best_tag": best_kind,
            "elapsed_s": int(time.time() - start),
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )

    for update in range(1, n_updates + 1):
        batch = trainer.collect_rollout()
        rec = trainer.update(batch)
        exp.pd.DataFrame(trainer.history).to_csv(trainer.run_dir / "train_history.csv", index=False)
        print(
            f"{tag} update={update:03d}/{n_updates} "
            f"R={rec['rollout_return_mean']:.3f} Cs={rec['rollout_safety_cost_mean']:.4f} "
            f"Cf={rec['rollout_fairness_cost_mean']:.3f} "
            f"ls={trainer.lambda_safety:.3f} lf={trainer.lambda_fairness:.3f}",
            flush=True,
        )
        if update % eval_every != 0 and update != n_updates:
            continue
        result = trainer.evaluate(
            episodes=guard_eval_episodes,
            deterministic=True,
            save_path=trainer.run_dir / f"guard_eval_update_{update:03d}.json",
        )
        result["guard_update"] = update
        safe_now = is_safe(result, threshold)
        consider(update, result)
        if safe_now:
            unsafe_streak = 0
            status = "guard_safe_eval"
        else:
            unsafe_streak += 1
            status = "guard_unsafe_eval"
        write_status(
            {
                "tag": tag,
                "status": status,
                "update": update,
                **summarize_result(result),
                "best_tag": best_kind,
                "elapsed_s": int(time.time() - start),
                "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        if update >= min_updates and unsafe_streak >= patience and best_state is not None:
            print(f"{tag}: stopping after unsafe deterministic guard eval at update {update}", flush=True)
            break

    ordered_candidates = sorted(
        candidates,
        key=lambda c: (1 if c["kind"] == "safe_fair" else 0, c["score"]),
        reverse=True,
    )
    selected = None
    final = None
    for candidate in ordered_candidates:
        trainer.ac.load_state_dict(candidate["state"])
        probe = trainer.evaluate(
            episodes=16,
            deterministic=True,
            save_path=trainer.run_dir / f"final_probe_update_{candidate['update']:03d}.json",
        )
        if is_safe(probe, threshold):
            selected = candidate
            final = probe
            break
    if selected is None:
        if best_state is not None:
            trainer.ac.load_state_dict(best_state)
        final = trainer.evaluate(episodes=16, deterministic=True, save_path=trainer.run_dir / "eval_summary.json")
        selected_update = int(best_result.get("guard_update", -1)) if best_result else -1
        selected_kind = "no_final_safe_candidate"
    else:
        trainer.ac.load_state_dict(selected["state"])
        selected_update = int(selected["update"])
        selected_kind = str(selected["kind"])
    exp.torch.save(trainer.ac.state_dict(), trainer.run_dir / "model.pt")
    final = trainer.evaluate(episodes=16, deterministic=True, save_path=trainer.run_dir / "eval_summary.json")
    final["selected_guard_kind"] = selected_kind
    final["selected_guard_update"] = selected_update
    save_json(final, trainer.run_dir / "eval_summary.json")
    stochastic = trainer.evaluate(episodes=16, deterministic=False, save_path=trainer.run_dir / "eval_summary_stochastic.json")
    save_json({"final": final, "stochastic": stochastic, "best_guard_eval": best_result or {}}, trainer.run_dir / "guard_summary.json")
    write_status(
        {
            "tag": tag,
            "status": f"selected_{selected_kind}",
            "update": final["selected_guard_update"],
            **summarize_result(final),
            "best_tag": selected_kind,
            "elapsed_s": int(time.time() - start),
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    trainer.close()
    print(f"==== DONE GUARDED {tag}: {summarize_result(final)} selected={selected_kind} ====", flush=True)
    return final


def main() -> None:
    seed = int(os.environ.get("SAFE_FAIR_HC_GUARD_SEED", "0"))
    base = find_base_checkpoint(seed)
    tag_prefix = os.environ.get("SAFE_FAIR_HC_GUARD_TAG_PREFIX", "hc_sf_guard").strip() or "hc_sf_guard"
    tag_suffix = os.environ.get("SAFE_FAIR_HC_GUARD_TAG_SUFFIX", "").strip()
    requested = {
        x.strip().lower()
        for x in os.environ.get("SAFE_FAIR_HC_GUARD_VARIANTS", "a,b,c").split(",")
        if x.strip()
    }
    variant_defs = [
        (
            "a",
            {
                "safety_target_scale": 0.60,
                "init_lambda_safety": 0.60,
                "safety_dual_lr": 0.20,
                "max_lambda_safety": 25.0,
                "fairness_dual_lr": 0.008,
                "max_lambda_fairness": 0.80,
                "fairness_warmup_updates": 16,
                "fairness_budget_scale_init": 5.0,
                "fairness_budget_ramp_updates": 60,
                "fairness_safety_margin": 0.003,
                "init_lambda_fairness": 0.0,
                "entropy_coef": 0.0025,
                "guard_eval_every_updates": 4,
                "guard_eval_episodes": 16,
                "guard_min_updates": 8,
            },
        ),
        (
            "b",
            {
                "safety_target_scale": 0.45,
                "init_lambda_safety": 1.20,
                "safety_dual_lr": 0.35,
                "max_lambda_safety": 50.0,
                "fairness_dual_lr": 0.004,
                "max_lambda_fairness": 0.40,
                "fairness_warmup_updates": 32,
                "fairness_budget_scale_init": 10.0,
                "fairness_budget_ramp_updates": 100,
                "fairness_safety_margin": 0.006,
                "init_lambda_fairness": 0.0,
                "entropy_coef": 0.0010,
                "guard_eval_every_updates": 4,
                "guard_eval_episodes": 16,
                "guard_min_updates": 8,
            },
        ),
        (
            "c",
            {
                "safety_target_scale": 0.75,
                "init_lambda_safety": 0.50,
                "safety_dual_lr": 0.18,
                "max_lambda_safety": 30.0,
                "fairness_dual_lr": 0.012,
                "max_lambda_fairness": 1.20,
                "fairness_warmup_updates": 12,
                "fairness_budget_scale_init": 4.0,
                "fairness_budget_ramp_updates": 50,
                "fairness_safety_margin": 0.002,
                "init_lambda_fairness": 0.005,
                "entropy_coef": 0.0030,
                "guard_eval_every_updates": 4,
                "guard_eval_episodes": 16,
                "guard_min_updates": 8,
            },
        ),
    ]
    for name, params in variant_defs:
        if name not in requested:
            continue
        if tag_prefix == "hc_sf_guard" and not tag_suffix and seed == 0:
            tag = f"hc_sf_guard_{name}_320k"
        else:
            suffix = f"_{tag_suffix}" if tag_suffix else ""
            tag = f"{tag_prefix}_{name}_320k_s{seed}{suffix}"
        run_guarded_variant(tag, base, params, seed)
    exp.collect_eval_summaries(exp.LOG_ROOT)
    script = ROOT / "scripts" / "regenerate_latest_tables.py"
    os.system(f"{sys.executable} {script}")


if __name__ == "__main__":
    main()
