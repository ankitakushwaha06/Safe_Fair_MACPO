
# =============================================================================
# 0. Global switches
# =============================================================================
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_optional_flag(name: str):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_csv(name: str):
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_int_tuple(name: str, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(default)
    return tuple(int(item) for item in raw.replace(",", " ").split() if item.strip())


def env_float(name: str, default=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def env_int(name: str, default=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


ROOT = Path(os.environ.get("SAFE_FAIR_MACPO_ROOT", "./safe_fair_macpo_runs")).resolve()
EXTERNAL_ROOT = ROOT / "official_sources"
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor"
LOG_ROOT = ROOT / "logs"
TABLE_ROOT = ROOT / "tables"
FIG_ROOT = ROOT / "figures"
CONFIG_ROOT = ROOT / "configs"

# Safe default: do not install packages or launch a long run automatically.
# Override with environment variables when running from the CLI/wrapper scripts.
INSTALL_EXTERNAL = env_flag("SAFE_FAIR_INSTALL_EXTERNAL", False)
CLONE_OFFICIAL_REPOS = env_flag("SAFE_FAIR_CLONE_OFFICIAL_REPOS", False)
RUN_TRAINING = env_flag("SAFE_FAIR_RUN_TRAINING", False)
RUN_EVALUATION = env_flag("SAFE_FAIR_RUN_EVALUATION", True)
TASK_FILTERS = set(env_csv("SAFE_FAIR_TASK_FILTER"))
METHOD_FILTERS = set(env_csv("SAFE_FAIR_METHOD_FILTER"))
RUN_GROUP_FILTERS = set(env_csv("SAFE_FAIR_RUN_GROUP_FILTER"))
SEED_FILTERS = {int(item) for item in env_csv("SAFE_FAIR_SEED_FILTER")}
FAIR_BUDGET_FILTERS = {float(item) for item in env_csv("SAFE_FAIR_FAIR_BUDGET_FILTER")}
MAX_RUNS = int(os.environ.get("SAFE_FAIR_MAX_RUNS", "0") or "0")
INCLUDE_SAFEFAIRBENCH = env_flag("SAFE_FAIR_INCLUDE_SAFEFAIRBENCH", False)

# QUICK_TEST uses the built-in lightweight bottleneck environment.
# Set QUICK_TEST=False for official MuJoCo / Safety-Gymnasium tasks.
QUICK_TEST = env_flag("SAFE_FAIR_QUICK_TEST", True)
EXPERIMENT_SEEDS = env_int_tuple("SAFE_FAIR_EXPERIMENT_SEEDS", (0, 1, 2))
ABLATION_EXPERIMENT_SEEDS = env_int_tuple("SAFE_FAIR_ABLATION_EXPERIMENT_SEEDS", (0, 1, 2, 3, 4))
SAFEFAIRBENCH_EXPERIMENT_SEEDS = env_int_tuple("SAFE_FAIR_SAFEFAIRBENCH_EXPERIMENT_SEEDS", EXPERIMENT_SEEDS)
SAFEFAIRBENCH_ABLATION_EXPERIMENT_SEEDS = env_int_tuple("SAFE_FAIR_SAFEFAIRBENCH_ABLATION_EXPERIMENT_SEEDS", ABLATION_EXPERIMENT_SEEDS)

# Unified trainer settings.
USE_CUDA_IF_AVAILABLE = env_flag("SAFE_FAIR_USE_CUDA", True)
USE_WANDB = env_flag("SAFE_FAIR_USE_WANDB", False)
PROJECT_NAME = os.environ.get("SAFE_FAIR_PROJECT_NAME", "safe-fair-macpo")

for d in [ROOT, EXTERNAL_ROOT, LOG_ROOT, TABLE_ROOT, FIG_ROOT, CONFIG_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

print("Experiment root:", ROOT)


def preferred_source_path(key: str) -> Path:
    if key == "safety_gymnasium":
        vendor_path = VENDOR_ROOT / "safety-gymnasium"
        if vendor_path.exists():
            return vendor_path
    return EXTERNAL_ROOT / key


# =============================================================================
# 1. Imports, package installation helpers, seeding, and subprocess utilities
# =============================================================================
def run_command(cmd: Union[str, List[str]], cwd: Optional[Path] = None, check: bool = True, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    print(f"$ {display}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=isinstance(cmd, str),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=merged_env,
    )
    if result.stdout:
        print(result.stdout[-4000:])
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {display}")
    return result

def pip_install(packages: Sequence[str]) -> None:
    run_command([sys.executable, "-m", "pip", "install", *packages], check=True)

if INSTALL_EXTERNAL:
    # The official environments use MuJoCo and can require system libraries.
    # Install only in a fresh experiment environment.
    pip_install([
        "numpy", "pandas", "matplotlib", "pyyaml", "tqdm", "tensorboard",
        "gymnasium", "torch", "scipy", "rich"
    ])

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
except Exception as exc:
    raise RuntimeError("PyTorch is required for the standalone trainer. Install torch first.") from exc

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if (USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()) else "cpu")
print("Device:", DEVICE)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@contextmanager
def prepend_sys_path(paths: Sequence[Path]):
    added = []
    for path in paths:
        p = str(path)
        if path.exists() and p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        yield
    finally:
        for p in added:
            while p in sys.path:
                sys.path.remove(p)


# =============================================================================
# 2. Official upstream sources
# =============================================================================
OFFICIAL_SOURCES: Dict[str, Dict[str, str]] = {
    "macpo": {
        "url": "https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation.git",
        "role": "Official MACPO and MAPPO-Lagrangian implementation",
        "paper": "Multi-Agent Constrained Policy Optimisation / Safe Multi-Agent Reinforcement Learning for Multi-Robot Control",
    },
    "safe_mamujoco": {
        "url": "https://github.com/chauncygu/Safe-Multi-Agent-Mujoco.git",
        "role": "Official Safe Multi-Agent MuJoCo benchmark",
        "paper": "Safe Multi-Agent Reinforcement Learning for Multi-Robot Control",
    },
    "safety_gymnasium": {
        "url": "https://github.com/PKU-Alignment/safety-gymnasium.git",
        "role": "Official Safety-Gymnasium environments",
        "paper": "Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark",
    },
    "safe_policy_optimization": {
        "url": "https://github.com/PKU-Alignment/Safe-Policy-Optimization.git",
        "role": "Official SafePO single-agent safe RL baselines, useful for sanity checks",
        "paper": "Safety-Gymnasium / Safe Policy Optimization",
    },
    "mappo": {
        "url": "https://github.com/marlbenchmark/on-policy.git",
        "role": "Official MAPPO implementation",
        "paper": "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games",
    },
    "happo_hatrpo": {
        "url": "https://github.com/morning9393/HAPPO-HATRPO.git",
        "role": "Official HAPPO/HATRPO implementation",
        "paper": "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning",
    },
    "dfrl": {
        "url": "https://github.com/matthieu637/DFRL.git",
        "role": "Official DFRL / SOTO-GGF fair cooperative MARL implementation",
        "paper": "Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning",
    },
    "dec_safe_marl": {
        "url": "https://github.com/zhykoties/Decentralized-Safe-MARL-with-General-Utilities.git",
        "role": "Implementation of scalable decentralized safe MARL with general utilities",
        "paper": "Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent Reinforcement Learning with General Utilities",
    },
    "def_marl": {
        "url": "https://github.com/MIT-REALM/def-marl.git",
        "role": "Official Def-MARL implementation for distributed epigraph-form safe MARL",
        "paper": "Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL",
    },
    "safe_marobosuite": {
        "url": "https://github.com/chauncygu/Safe-Multi-Agent-Robosuite.git",
        "role": "Optional official Safe Multi-Agent Robosuite benchmark",
        "paper": "Safe Multi-Agent Reinforcement Learning for Multi-Robot Control",
    },
}

def clone_or_update_repo(key: str, depth: int = 1) -> Path:
    spec = OFFICIAL_SOURCES[key]
    target = preferred_source_path(key)
    if key == "safety_gymnasium" and target.exists():
        print(f"[source] {key}: using vendored checkout at {target}")
        return target
    if target.exists() and (target / ".git").exists():
        print(f"[source] {key}: already cloned at {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if depth is not None and depth > 0:
        cmd += ["--depth", str(depth)]
    cmd += [spec["url"], str(target)]
    run_command(cmd)
    return target

def repo_commit(path: Path) -> str:
    if not (path / ".git").exists():
        return "not-a-git-repo"
    res = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], check=False)
    return (res.stdout or "").strip().splitlines()[-1] if res.stdout else "unknown"

def prepare_official_sources(keys: Optional[List[str]] = None) -> pd.DataFrame:
    keys = keys or list(OFFICIAL_SOURCES.keys())
    rows = []
    for key in keys:
        spec = OFFICIAL_SOURCES[key]
        path = preferred_source_path(key)
        commit = "not-cloned"
        if CLONE_OFFICIAL_REPOS:
            try:
                path = clone_or_update_repo(key)
                commit = repo_commit(path)
            except Exception as exc:
                print(f"[source] could not clone {key}: {exc}")
                commit = f"clone-failed: {type(exc).__name__}"
        rows.append({"key": key, "role": spec["role"], "url": spec["url"], "commit": commit, "path": str(path)})
    df = pd.DataFrame(rows)
    save_json(df.to_dict(orient="records"), ROOT / "official_source_manifest.json")
    return df

if CLONE_OFFICIAL_REPOS:
    source_manifest = prepare_official_sources()
else:
    source_manifest = pd.DataFrame([
        {"key": k, "role": v["role"], "url": v["url"], "commit": "not-cloned", "path": str(preferred_source_path(k))}
        for k, v in OFFICIAL_SOURCES.items()
    ])
source_manifest


# =============================================================================
# 3. Config dataclasses and experiment registries
# =============================================================================
@dataclass
class TaskSpec:
    name: str
    source: str
    env_id: str
    n_agents: int
    episode_length: int
    safety_budget_per_step: float
    fairness_budget_per_step: float
    official_source_key: str
    description: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MethodSpec:
    name: str
    family: str
    decentralized_critic: bool = False
    use_safety_constraint: bool = False
    use_fairness_constraint: bool = False
    fairness_reward_coeff: float = 0.0
    safety_reward_coeff: float = 0.0
    use_debt: bool = True
    use_envy: bool = True
    safety_first: bool = True
    target_shares: Optional[List[float]] = None
    local_k_hop: Optional[int] = None
    disabled_burden_components: Tuple[str, ...] = ()
    fairness_cost_mode: str = "debt_envy"
    fairness_critic_scope: str = "central"
    use_safety_recovery: bool = False
    safety_recovery_margin: float = 0.08
    safety_recovery_blend: float = 0.85
    safety_recovery_speed: float = 0.95
    use_burden_aware_recovery: bool = False
    burden_recovery_strength: float = 0.45
    notes: str = ""

    @property
    def uses_fairness_objective(self) -> bool:
        return bool(self.use_fairness_constraint or self.fairness_reward_coeff > 0)

@dataclass
class RunConfig:
    task: TaskSpec
    method: MethodSpec
    seed: int = 0
    total_steps: int = 200_000
    rollout_steps: int = 2048
    ppo_epochs: int = 5
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    dual_lr: float = 0.05
    init_lambda_safety: float = 0.0
    init_lambda_fairness: float = 0.0
    max_lambda_safety: float = 10.0
    max_lambda_fairness: float = 5.0
    safety_dual_lr: Optional[float] = None
    fairness_dual_lr: Optional[float] = None
    safety_target_scale: float = 1.0
    fairness_warmup_updates: int = 0
    fairness_safety_margin: float = 0.0
    fairness_budget_scale_init: float = 1.0
    fairness_budget_ramp_updates: int = 0
    eval_episodes: int = 8
    hidden_sizes: Tuple[int, int] = (128, 128)
    device: str = str(DEVICE)
    run_group: str = "main"
    quick: bool = False
    run_tag: str = ""
    init_checkpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

TASKS: Dict[str, TaskSpec] = {
    "smoke_bottleneck": TaskSpec(
        name="smoke_bottleneck",
        source="builtin",
        env_id="BuiltinBottleneckFair-v0",
        n_agents=4,
        episode_length=160,
        safety_budget_per_step=0.02,
        fairness_budget_per_step=0.05,
        official_source_key="none",
        description="Lightweight smoke-test environment; not a paper dataset.",
        kwargs={"n_agents": 4, "n_hazards": 2, "goal_radius": 0.12},
    ),
    "smamujoco_halfcheetah_2x3_fair": TaskSpec(
        name="smamujoco_halfcheetah_2x3_fair",
        source="safe_mamujoco",
        env_id="HalfCheetah-v2",
        n_agents=2,
        episode_length=1000,
        safety_budget_per_step=0.005,
        fairness_budget_per_step=0.04,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo HalfCheetah 2x3 with fair burden wrapper.",
        kwargs={"scenario": "HalfCheetah-v2", "agent_conf": "2x3", "agent_obsk": 1},
    ),
    "smamujoco_halfcheetah_3x2_workload": TaskSpec(
        name="smamujoco_halfcheetah_3x2_workload",
        source="safe_mamujoco",
        env_id="HalfCheetah-v2",
        n_agents=3,
        episode_length=1000,
        safety_budget_per_step=0.005,
        fairness_budget_per_step=0.04,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo HalfCheetah 3x2 actuator-partition workload-imbalance diagnostic.",
        kwargs={"scenario": "HalfCheetah-v2", "agent_conf": "3x2", "agent_obsk": 1},
    ),
    "smamujoco_halfcheetah_6x1_workload": TaskSpec(
        name="smamujoco_halfcheetah_6x1_workload",
        source="safe_mamujoco",
        env_id="HalfCheetah-v2",
        n_agents=6,
        episode_length=1000,
        safety_budget_per_step=0.005,
        fairness_budget_per_step=0.04,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo HalfCheetah 6x1 actuator-level workload-imbalance diagnostic.",
        kwargs={"scenario": "HalfCheetah-v2", "agent_conf": "6x1", "agent_obsk": 1},
    ),
    "smamujoco_ant_2x4_fair": TaskSpec(
        name="smamujoco_ant_2x4_fair",
        source="safe_mamujoco",
        env_id="Ant-v2",
        n_agents=2,
        episode_length=1000,
        safety_budget_per_step=0.010,
        fairness_budget_per_step=0.05,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo Ant 2x4 bottleneck/yield burden wrapper.",
        kwargs={"scenario": "Ant-v2", "agent_conf": "2x4", "agent_obsk": 1},
    ),
    "smamujoco_ant_2x4d_workload": TaskSpec(
        name="smamujoco_ant_2x4d_workload",
        source="safe_mamujoco",
        env_id="Ant-v2",
        n_agents=2,
        episode_length=1000,
        safety_budget_per_step=0.010,
        fairness_budget_per_step=0.05,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo Ant 2x4 diagonal-leg workload-imbalance diagnostic.",
        kwargs={"scenario": "Ant-v2", "agent_conf": "2x4d", "agent_obsk": 1},
    ),
    "smamujoco_ant_4x2_workload": TaskSpec(
        name="smamujoco_ant_4x2_workload",
        source="safe_mamujoco",
        env_id="Ant-v2",
        n_agents=4,
        episode_length=1000,
        safety_budget_per_step=0.010,
        fairness_budget_per_step=0.05,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo Ant 4x2 leg-pair workload-imbalance diagnostic.",
        kwargs={"scenario": "Ant-v2", "agent_conf": "4x2", "agent_obsk": 1},
    ),
    "smamujoco_ant_8x1_workload": TaskSpec(
        name="smamujoco_ant_8x1_workload",
        source="safe_mamujoco",
        env_id="Ant-v2",
        n_agents=8,
        episode_length=1000,
        safety_budget_per_step=0.010,
        fairness_budget_per_step=0.05,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo Ant 8x1 actuator-level workload-imbalance diagnostic.",
        kwargs={"scenario": "Ant-v2", "agent_conf": "8x1", "agent_obsk": 1},
    ),
    "smamujoco_manyagent_ant_fair": TaskSpec(
        name="smamujoco_manyagent_ant_fair",
        source="safe_mamujoco",
        env_id="manyagent_ant",
        n_agents=4,
        episode_length=1000,
        safety_budget_per_step=0.010,
        fairness_budget_per_step=0.07,
        official_source_key="safe_mamujoco",
        description="Safety-Gymnasium multi-agent Ant fairness scalability task (4x2 factorization).",
        kwargs={"scenario": "manyagent_ant", "agent_conf": "4x2", "agent_obsk": 1, "sg_scenario": "Ant"},
    ),
    "smamujoco_walker2d_2x3_workload": TaskSpec(
        name="smamujoco_walker2d_2x3_workload",
        source="safe_mamujoco",
        env_id="Walker2d-v2",
        n_agents=2,
        episode_length=1000,
        safety_budget_per_step=0.010,
        fairness_budget_per_step=0.05,
        official_source_key="safe_mamujoco",
        description="Safe MAMuJoCo Walker2d 2x3 left/right leg workload-imbalance diagnostic.",
        kwargs={"scenario": "Walker2d-v2", "agent_conf": "2x3", "agent_obsk": 1},
    ),
    "safetygym_point_multigoal1_fair": TaskSpec(
        name="safetygym_point_multigoal1_fair",
        source="safety_gymnasium",
        env_id="SafetyPointMultiGoal0-v0",
        n_agents=2,
        episode_length=1000,
        safety_budget_per_step=0.02,
        fairness_budget_per_step=0.05,
        official_source_key="safety_gymnasium",
        description="Paper MultiGoal-Fair built on the easier Safety-Gymnasium MultiGoal0 base with fair queue/exposure wrapper.",
        kwargs={},
    ),
    "safetygym_ant_multigoal1_fair": TaskSpec(
        name="safetygym_ant_multigoal1_fair",
        source="safety_gymnasium",
        env_id="SafetyAntMultiGoal1-v0",
        n_agents=2,
        episode_length=1000,
        safety_budget_per_step=0.02,
        fairness_budget_per_step=0.06,
        official_source_key="safety_gymnasium",
        description="Safety-Gymnasium Ant MultiGoal fair wrapper.",
        kwargs={},
    ),
    "safetygym_chargeshare_fair": TaskSpec(
        name="safetygym_chargeshare_fair",
        source="safety_gymnasium_chargeshare",
        env_id="SafetyPointMultiGoal0-v0",
        n_agents=2,
        episode_length=1000,
        safety_budget_per_step=0.02,
        fairness_budget_per_step=0.06,
        official_source_key="safety_gymnasium",
        description="Paper ChargeShare-Fair built on MultiGoal0 plus wrapper-level scarce charging stations and energy-gated control.",
        kwargs={
            "station_positions": [[0.0, 0.0]],
            "station_radius": 0.30,
            "low_energy_threshold": 0.40,
            "energy_drain_idle": 0.003,
            "energy_drain_action": 0.018,
            "energy_charge_rate": 0.055,
            "min_action_scale": 0.35,
        },
    ),
    "safefairbench_vmas_multigoal": TaskSpec(
        name="safefairbench_vmas_multigoal",
        source="safefairbench_v0",
        env_id="SafeFairBench-v0/VMAS-MultiGoal",
        n_agents=3,
        episode_length=180,
        safety_budget_per_step=0.04,
        fairness_budget_per_step=0.05,
        official_source_key="none",
        description="Dependency-free VMAS-style contested MultiGoal adapter with explicit wait, detour, yield, and hazard burden events.",
        kwargs={"scenario": "vmas_multigoal", "n_agents": 3, "goal_radius": 0.10},
    ),
    "safefairbench_vmas_chargeshare": TaskSpec(
        name="safefairbench_vmas_chargeshare",
        source="safefairbench_v0",
        env_id="SafeFairBench-v0/VMAS-ChargeShare",
        n_agents=3,
        episode_length=180,
        safety_budget_per_step=0.04,
        fairness_budget_per_step=0.06,
        official_source_key="none",
        description="Dependency-free VMAS-style scarce charging station adapter with resource-denial burden events.",
        kwargs={"scenario": "vmas_chargeshare", "n_agents": 3, "goal_radius": 0.10},
    ),
    "safefairbench_pogema_warehouse": TaskSpec(
        name="safefairbench_pogema_warehouse",
        source="safefairbench_v0",
        env_id="SafeFairBench-v0/POGEMA-Warehouse",
        n_agents=4,
        episode_length=220,
        safety_budget_per_step=0.03,
        fairness_budget_per_step=0.06,
        official_source_key="none",
        description="Dependency-free POGEMA-style warehouse adapter with aisle, shelf-blockage, pickup, and delivery burden events.",
        kwargs={"scenario": "pogema_warehouse", "n_agents": 4, "goal_radius": 0.09},
    ),
    "safefairbench_highway_merge": TaskSpec(
        name="safefairbench_highway_merge",
        source="safefairbench_v0",
        env_id="SafeFairBench-v0/Highway-Merge",
        n_agents=3,
        episode_length=180,
        safety_budget_per_step=0.035,
        fairness_budget_per_step=0.05,
        official_source_key="none",
        description="Dependency-free Highway-style merge adapter with close-following, forced-yield, and merge-denial burden events.",
        kwargs={"scenario": "highway_merge", "n_agents": 3, "goal_radius": 0.12},
    ),
    "safe_marobosuite_lift_2x4_fair_optional": TaskSpec(
        name="safe_marobosuite_lift_2x4_fair_optional",
        source="safe_marobosuite",
        env_id="SafeLift-2x4",
        n_agents=2,
        episode_length=500,
        safety_budget_per_step=0.02,
        fairness_budget_per_step=0.06,
        official_source_key="safe_marobosuite",
        description="Optional Safe Multi-Agent Robosuite fairness wrapper.",
        kwargs={"task": "Lift", "agent_conf": "2x4"},
    ),
}

METHODS: Dict[str, MethodSpec] = {
    "ippo": MethodSpec(name="ippo", family="performance_only_independent", decentralized_critic=True, notes="Independent PPO baseline with decentralized critics and no explicit safety/fairness objective."),
    "mappo": MethodSpec(name="mappo", family="performance_only", notes="Performance-only MAPPO-style baseline."),
    "macpo": MethodSpec(name="macpo", family="safety_only", use_safety_constraint=True, notes="Safety-only constrained baseline; exact native version available from official MACPO repository."),
    "mappo_lagrangian": MethodSpec(name="mappo_lagrangian", family="safety_only_lagrangian", use_safety_constraint=True, notes="Safety-only Lagrangian baseline; exact native version available from official MACPO repository."),
    "macpo_recovery": MethodSpec(name="macpo_recovery", family="safety_only_recovery_control", use_safety_constraint=True, use_safety_recovery=True, notes="Control: MACPO-style safety constraint with the same SafeFairBench recovery/projection path as Safe-Fair, but no fairness objective."),
    "mappo_lagrangian_recovery": MethodSpec(name="mappo_lagrangian_recovery", family="safety_only_lagrangian_recovery_control", use_safety_constraint=True, use_safety_recovery=True, notes="Control: MAPPO-Lagrangian safety constraint with the same SafeFairBench recovery/projection path as Safe-Fair, but no fairness objective."),
    "dfrl_soto_ggf": MethodSpec(name="dfrl_soto_ggf", family="fairness_only", fairness_reward_coeff=1.0, use_debt=False, use_envy=True, notes="Fairness-only welfare/reward-shaped baseline inspired by DFRL/SOTO-GGF."),
    "reward_shaped_mappo_fair": MethodSpec(name="reward_shaped_mappo_fair", family="reward_shaping", fairness_reward_coeff=1.0, use_debt=True, use_envy=True, notes="MAPPO with fairness penalty in reward; no explicit safety or fairness constraint."),
    "reward_shaped_macpo_fair": MethodSpec(name="reward_shaped_macpo_fair", family="safety_plus_reward_shaping", use_safety_constraint=True, fairness_reward_coeff=1.0, use_debt=True, use_envy=True, notes="Safety constrained plus fairness reward shaping, without a fairness dual variable."),
    "safe_fair_macpo": MethodSpec(name="safe_fair_macpo", family="safe_fair", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=True, use_safety_recovery=True, notes="Proposed method: safety and fairness constraints with debt and safety-first recovery."),
    "safe_fair_no_fairness": MethodSpec(name="safe_fair_no_fairness", family="ablation", use_safety_constraint=True, use_fairness_constraint=False, use_debt=False, use_envy=False, safety_first=True, use_safety_recovery=True, notes="Ablation: keep the Safe-Fair safety/recovery path but remove the fairness cost and fairness dual."),
    "safe_fair_no_safety": MethodSpec(name="safe_fair_no_safety", family="ablation", use_safety_constraint=False, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=False, use_safety_recovery=False, notes="Ablation: keep the fairness objective but remove the safety constraint and safety-first recovery."),
    "safe_fair_no_debt": MethodSpec(name="safe_fair_no_debt", family="ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=False, use_envy=True, safety_first=True, use_safety_recovery=True, notes="Ablation: remove temporal debt, use instantaneous envy/burden cost."),
    "safe_fair_no_envy": MethodSpec(name="safe_fair_no_envy", family="ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=False, safety_first=True, use_safety_recovery=True, notes="Ablation: remove pairwise excess envy term."),
    "safe_fair_no_recovery": MethodSpec(name="safe_fair_no_recovery", family="ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=False, notes="Ablation: no lexicographic safety-first recovery in dual update."),
    "safe_fair_local_k1": MethodSpec(name="safe_fair_local_k1", family="scalability_ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=True, local_k_hop=1, use_safety_recovery=True, notes="Scalability variant: approximate global fairness by local one-hop neighborhoods."),
    "safe_fair_no_hazard_burden": MethodSpec(name="safe_fair_no_hazard_burden", family="burden_ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=True, disabled_burden_components=("hazard",), use_safety_recovery=True, notes="Ablation: remove hazard exposure from burden fairness while preserving the safety cost channel."),
    "safe_fair_no_contested_access_burden": MethodSpec(name="safe_fair_no_contested_access_burden", family="burden_ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=True, disabled_burden_components=("resource",), use_safety_recovery=True, notes="Ablation: remove contested resource/access burden from fairness cost."),
    "safe_fair_no_yield_burden": MethodSpec(name="safe_fair_no_yield_burden", family="burden_ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=True, disabled_burden_components=("yield",), use_safety_recovery=True, notes="Ablation: remove forced-yield burden from fairness cost."),
    "safe_fair_jain_only": MethodSpec(name="safe_fair_jain_only", family="fairness_metric_ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=False, use_envy=False, safety_first=True, fairness_cost_mode="jain_only", use_safety_recovery=True, notes="Ablation: use instantaneous Jain-burden shortfall as the fairness cost."),
    "safe_fair_envy_only": MethodSpec(name="safe_fair_envy_only", family="fairness_metric_ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=False, use_envy=True, safety_first=True, fairness_cost_mode="envy_only", use_safety_recovery=True, notes="Ablation: use instantaneous pairwise envy without temporal debt."),
    "safe_fair_scalar_critic": MethodSpec(name="safe_fair_scalar_critic", family="critic_ablation", use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=True, fairness_critic_scope="central", use_safety_recovery=True, notes="Ablation: central scalar fairness critic, matching the default implementation."),
    "safe_fair_per_agent_critic": MethodSpec(name="safe_fair_per_agent_critic", family="critic_ablation", decentralized_critic=True, use_safety_constraint=True, use_fairness_constraint=True, use_debt=True, use_envy=True, safety_first=True, fairness_critic_scope="per_agent", use_safety_recovery=True, notes="Ablation: per-agent critic heads averaged in the policy update."),
}
pd.DataFrame([{"task": k, **asdict(v)} for k, v in TASKS.items()]).head()


# =============================================================================
# 4. Fairness metrics and burden logger
# =============================================================================
def as_np(x: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)

def jain_index(x: Sequence[float], eps: float = 1e-8) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    denom = len(arr) * np.sum(arr ** 2) + eps
    return float((np.sum(arr) ** 2) / denom)

def gini_index(x: Sequence[float], eps: float = 1e-8) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    arr = np.sort(np.maximum(arr, 0.0))
    total = float(np.sum(arr))
    if total <= eps:
        return 0.0
    idx = np.arange(1, arr.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * arr)) / (arr.size * total) - (arr.size + 1.0) / arr.size)

def unit_vector(x: Sequence[float], fallback: Optional[Sequence[float]] = None, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > eps:
        return arr / norm
    if fallback is None:
        if arr.size == 0:
            return np.zeros(1, dtype=np.float64)
        out = np.zeros_like(arr, dtype=np.float64)
        out[0] = 1.0
        return out
    return unit_vector(fallback, fallback=None, eps=eps)

def excess_envy(normalized_burdens: Sequence[float], tau: float = 0.0) -> float:
    u = np.asarray(normalized_burdens, dtype=np.float64).reshape(-1)
    n = len(u)
    if n <= 1:
        return 0.0
    diffs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                diffs.append(max(0.0, float(u[i] - u[j] - tau)))
    return float(np.mean(diffs))

def normalize_target_shares(n_agents: int, target_shares: Optional[Sequence[float]]) -> np.ndarray:
    if target_shares is None:
        return np.ones(n_agents, dtype=np.float64) / n_agents
    q = np.asarray(target_shares, dtype=np.float64)
    if len(q) != n_agents:
        raise ValueError(f"target_shares length {len(q)} != n_agents {n_agents}")
    q = np.maximum(q, 1e-8)
    return q / q.sum()

def pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)

@dataclass
class BurdenWeights:
    wait: float = 0.25
    hazard: float = 0.25
    yield_: float = 0.20
    detour: float = 0.20
    resource: float = 0.10

    def as_vector(self) -> np.ndarray:
        v = np.array([self.wait, self.hazard, self.yield_, self.detour, self.resource], dtype=np.float64)
        return v / max(v.sum(), 1e-8)

def apply_disabled_burden_components(weights: BurdenWeights, disabled: Sequence[str]) -> BurdenWeights:
    values = {
        "wait": weights.wait,
        "hazard": weights.hazard,
        "yield": weights.yield_,
        "detour": weights.detour,
        "resource": weights.resource,
    }
    aliases = {"yield_": "yield", "contested_access": "resource", "access": "resource"}
    for name in disabled:
        key = str(name).strip().lower()
        key = aliases.get(key, key)
        if key in values:
            values[key] = 0.0
    return BurdenWeights(
        wait=values["wait"],
        hazard=values["hazard"],
        yield_=values["yield"],
        detour=values["detour"],
        resource=values["resource"],
    )

class FairnessBurdenLogger:
    def __init__(
        self,
        n_agents: int,
        weights: BurdenWeights = BurdenWeights(),
        target_shares: Optional[Sequence[float]] = None,
        rho: float = 0.95,
        z_max: float = 10.0,
        alpha_debt: float = 1.0,
        alpha_envy: float = 0.5,
        envy_tau: float = 0.0,
        use_debt: bool = True,
        use_envy: bool = True,
        fairness_cost_mode: str = "debt_envy",
        local_k_hop: Optional[int] = None,
        wait_action_threshold: float = 0.075,
        collision_yield_distance: float = 0.25,
    ):
        self.n_agents = int(n_agents)
        self.weights = weights
        self.q = normalize_target_shares(n_agents, target_shares)
        self.rho = rho
        self.z_max = z_max
        self.alpha_debt = alpha_debt
        self.alpha_envy = alpha_envy
        self.envy_tau = envy_tau
        self.use_debt = use_debt
        self.use_envy = use_envy
        self.fairness_cost_mode = fairness_cost_mode
        self.local_k_hop = local_k_hop
        self.wait_action_threshold = wait_action_threshold
        self.collision_yield_distance = collision_yield_distance
        self.reset()

    def reset(self) -> None:
        self.debt = np.zeros(self.n_agents, dtype=np.float64)
        self.total_burden = np.zeros(self.n_agents, dtype=np.float64)
        self.debt_history: List[np.ndarray] = []
        self.burden_history: List[np.ndarray] = []
        self.feature_history: List[np.ndarray] = []
        self.last_positions: Optional[np.ndarray] = None
        self.last_dist_to_goal: Optional[np.ndarray] = None
        self.last_x_positions: Optional[np.ndarray] = None
        self.last_action_vectors: Optional[np.ndarray] = None

    def _extract_positions(self, obs: Any, info: Dict[str, Any]) -> Optional[np.ndarray]:
        for key in ["agent_positions", "positions", "xy", "agent_xy", "robot_positions"]:
            if isinstance(info, dict) and key in info:
                arr = as_np(info[key], dtype=np.float64)
                if arr.ndim >= 2 and arr.shape[0] >= self.n_agents:
                    return arr[: self.n_agents, :2]
        obs_list = normalize_obs_container(obs, self.n_agents)
        arrs = []
        for ob in obs_list:
            flat = as_np(ob, dtype=np.float64).reshape(-1)
            if flat.size >= 2:
                arrs.append(flat[:2])
        if len(arrs) == self.n_agents:
            return np.stack(arrs, axis=0)
        return None

    def _extract_goals(self, obs: Any, info: Dict[str, Any]) -> Optional[np.ndarray]:
        for key in ["goal_positions", "goals", "agent_goals", "target_positions"]:
            if isinstance(info, dict) and key in info:
                arr = as_np(info[key], dtype=np.float64)
                if arr.ndim >= 2 and arr.shape[0] >= self.n_agents:
                    return arr[: self.n_agents, :2]
                if arr.ndim == 1 and arr.shape[0] >= 2:
                    return np.repeat(arr[None, :2], self.n_agents, axis=0)
        obs_list = normalize_obs_container(obs, self.n_agents)
        arrs = []
        for ob in obs_list:
            flat = as_np(ob, dtype=np.float64).reshape(-1)
            if flat.size >= 4:
                arrs.append(flat[2:4])
        if len(arrs) == self.n_agents:
            return np.stack(arrs, axis=0)
        return None

    def _extract_vector(self, info: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> np.ndarray:
        if not isinstance(info, dict):
            return np.full(self.n_agents, default, dtype=np.float64)
        for key in keys:
            if key in info:
                arr = as_np(info[key], dtype=np.float64).reshape(-1)
                if arr.size == 1:
                    return np.full(self.n_agents, float(arr[0]), dtype=np.float64)
                if arr.size >= self.n_agents:
                    return arr[: self.n_agents]
        return np.full(self.n_agents, default, dtype=np.float64)

    def _extract_nested_agent_vector(self, info: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> np.ndarray:
        if not isinstance(info, dict):
            return np.full(self.n_agents, default, dtype=np.float64)
        vals = []
        found = False
        for i in range(self.n_agents):
            agent_info = info.get(f"agent_{i}")
            if not isinstance(agent_info, dict):
                vals.append(default)
                continue
            value = default
            for key in keys:
                if key in agent_info:
                    value = float(np.asarray(agent_info[key], dtype=np.float64).reshape(-1)[0])
                    found = True
                    break
            vals.append(value)
        if not found:
            return np.full(self.n_agents, default, dtype=np.float64)
        return np.asarray(vals, dtype=np.float64)

    def _local_centered_burden(self, u: np.ndarray, positions: Optional[np.ndarray]) -> np.ndarray:
        if self.local_k_hop is None or positions is None or self.n_agents <= 2:
            return u - np.mean(u)
        k = max(1, int(self.local_k_hop))
        d = pairwise_distances(positions)
        centered = np.zeros_like(u)
        for i in range(self.n_agents):
            neigh = np.argsort(d[i])[: min(self.n_agents, k + 1)]
            centered[i] = u[i] - np.mean(u[neigh])
        return centered

    def step(self, obs: Any, actions: Sequence[np.ndarray], next_obs: Any, info: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(info, dict) and "agent_burdens" in info:
            burden = as_np(info["agent_burdens"], dtype=np.float64).reshape(-1)[: self.n_agents]
            features = np.zeros((self.n_agents, 5), dtype=np.float64)
            features[:, 0] = burden
        else:
            features = self.compute_features(obs, actions, next_obs, info)
            burden = features @ self.weights.as_vector()

        q = np.maximum(self.q, 1e-8)
        normalized = burden / q
        positions = self._extract_positions(next_obs, info)
        centered = self._local_centered_burden(normalized, positions)

        if self.use_debt:
            self.debt = np.clip(self.rho * self.debt + centered, -self.z_max, self.z_max)
            # Scale the debt back to a per-step magnitude so the fairness budget
            # is comparable to instantaneous burden asymmetry.
            debt_term = float((1.0 - self.rho) * np.mean(np.abs(self.debt)))
        else:
            self.debt = np.zeros(self.n_agents, dtype=np.float64)
            debt_term = float(np.mean(np.abs(centered)))

        envy_term = excess_envy(normalized, self.envy_tau) if self.use_envy else 0.0
        jain_shortfall = max(0.0, 1.0 - jain_index(np.abs(normalized) + 1e-8))
        if self.fairness_cost_mode == "jain_only":
            fair_cost = jain_shortfall
        elif self.fairness_cost_mode == "envy_only":
            fair_cost = envy_term
        elif self.fairness_cost_mode == "debt_only":
            fair_cost = self.alpha_debt * debt_term
        else:
            fair_cost = self.alpha_debt * debt_term + self.alpha_envy * envy_term

        self.total_burden += burden
        self.debt_history.append(self.debt.copy())
        self.burden_history.append(burden.copy())
        self.feature_history.append(features.copy())

        return {
            "burden_features": features,
            "burden": burden,
            "normalized_burden": normalized,
            "fairness_debt": self.debt.copy(),
            "fairness_cost": float(fair_cost),
            "debt_term": float(debt_term),
            "envy_gap": float(envy_term),
            "gini_burden": gini_index(self.total_burden + 1e-8),
            "jain_burden": jain_index(self.total_burden + 1e-8),
        }

    def compute_features(self, obs: Any, actions: Sequence[np.ndarray], next_obs: Any, info: Dict[str, Any]) -> np.ndarray:
        actions_arr = np.stack([as_np(a, dtype=np.float64).reshape(-1) for a in actions], axis=0)
        nested_x_velocity = self._extract_nested_agent_vector(info, ["x_velocity"], default=0.0)
        nested_reward_ctrl = self._extract_nested_agent_vector(info, ["reward_ctrl"], default=0.0)
        nested_x_position = self._extract_nested_agent_vector(info, ["x_position"], default=0.0)
        if np.any(np.abs(nested_x_velocity) > 0.0) or np.any(np.abs(nested_reward_ctrl) > 0.0):
            # In MaMuJoCo locomotion, agents often control disjoint actuator groups
            # on the same body. The environment then emits the same body-level
            # x-velocity and control reward to every agent, which makes any
            # fairness signal built directly from those fields degenerate.
            # In that case, measure burden through actuator workload instead.
            shared_body_metrics = (
                self.n_agents > 1
                and np.allclose(nested_x_velocity, nested_x_velocity[0], atol=1e-9)
                and np.allclose(nested_reward_ctrl, nested_reward_ctrl[0], atol=1e-9)
            )
            hazard_raw = self._extract_vector(
                info,
                ["agent_safety_costs", "agent_costs", "hazard_exposure", "agent_hazard_exposure", "cost_agents"],
                default=0.0,
            )
            if shared_body_metrics:
                control_effort = np.mean(np.square(actions_arr), axis=1)
                effort_total = float(np.sum(control_effort))
                if effort_total > 1e-8:
                    effort_share = control_effort / effort_total
                else:
                    effort_share = np.ones(self.n_agents, dtype=np.float64) / self.n_agents
                safety_total = max(float(np.sum(hazard_raw)), float(scalar_cost_from_info(info)))
                safety_share = safety_total * effort_share
                action_jerk = np.zeros(self.n_agents, dtype=np.float64)
                if self.last_action_vectors is not None and self.last_action_vectors.shape == actions_arr.shape:
                    action_jerk = np.mean(np.square(actions_arr - self.last_action_vectors), axis=1)
                self.last_action_vectors = actions_arr.copy()
                # For shared-body locomotion, keep only burden terms with a clear
                # semantic meaning. Overload and stall_load mostly re-encode
                # control effort and make the fairness penalty harsher without
                # adding an interpretable notion of unfairness.
                zeros = np.zeros(self.n_agents, dtype=np.float64)
                return np.stack([control_effort, safety_share, action_jerk, zeros, zeros], axis=1)

            # If body metrics differ across agents, keep a relative locomotion burden.
            mean_velocity = float(np.mean(nested_x_velocity))
            slowdown = np.maximum(0.0, mean_velocity - nested_x_velocity)
            control_effort = np.maximum(0.0, -nested_reward_ctrl)
            progress_burden = np.zeros(self.n_agents, dtype=np.float64)
            if self.last_x_positions is not None:
                progress = nested_x_position - self.last_x_positions
                mean_progress = float(np.mean(progress))
                progress_burden = np.maximum(0.0, mean_progress - progress)
            self.last_x_positions = nested_x_position.copy()
            zeros = np.zeros(self.n_agents, dtype=np.float64)
            return np.stack([slowdown, hazard_raw, zeros, progress_burden, control_effort], axis=1)

        positions = self._extract_positions(next_obs, info)
        goals = self._extract_goals(next_obs, info)
        action_norm = np.linalg.norm(actions_arr, axis=1)

        if positions is not None and goals is not None:
            dist_to_goal = np.linalg.norm(goals - positions, axis=1)
            wait = ((action_norm < self.wait_action_threshold) & (dist_to_goal > 0.15)).astype(np.float64)
        else:
            wait = (action_norm < self.wait_action_threshold).astype(np.float64)

        hazard = self._extract_vector(
            info,
            ["agent_safety_costs", "agent_costs", "hazard_exposure", "agent_hazard_exposure", "cost_agents"],
            default=0.0,
        )

        if positions is not None and len(positions) == self.n_agents:
            d = pairwise_distances(positions)
            np.fill_diagonal(d, np.inf)
            near = (np.min(d, axis=1) < self.collision_yield_distance).astype(np.float64)
            yield_event = near * (action_norm < 0.2).astype(np.float64)
        else:
            yield_event = self._extract_vector(info, ["yield_events", "agent_yield", "slowdown_events"], default=0.0)

        detour = np.zeros(self.n_agents, dtype=np.float64)
        if positions is not None:
            if self.last_positions is not None:
                step_len = np.linalg.norm(positions - self.last_positions, axis=1)
            else:
                step_len = np.zeros(self.n_agents, dtype=np.float64)
            if goals is not None:
                dist = np.linalg.norm(goals - positions, axis=1)
                if self.last_dist_to_goal is not None:
                    progress = self.last_dist_to_goal - dist
                else:
                    progress = np.zeros(self.n_agents, dtype=np.float64)
                detour = np.maximum(0.0, step_len - np.maximum(progress, 0.0))
                self.last_dist_to_goal = dist
            self.last_positions = positions.copy()
        else:
            detour = self._extract_vector(info, ["detour", "agent_detour", "path_excess"], default=0.0)

        resource = self._extract_vector(
            info,
            ["resource_denial", "agent_resource_denial", "charger_wait", "low_energy_wait"],
            default=0.0,
        )

        features = np.stack([wait, hazard, yield_event, detour, resource], axis=1)
        return np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=0.0)

    def summary(self) -> Dict[str, float]:
        debt_arr = np.asarray(self.debt_history, dtype=np.float64) if self.debt_history else np.zeros((1, self.n_agents))
        burden_arr = np.asarray(self.burden_history, dtype=np.float64) if self.burden_history else np.zeros((1, self.n_agents))
        total_burden = burden_arr.sum(axis=0)
        return {
            "jain_burden": jain_index(total_burden + 1e-8),
            "gini_burden": gini_index(total_burden + 1e-8),
            "envy_gap_final": excess_envy(total_burden / np.maximum(self.q, 1e-8)),
            "debt_auc": float(np.mean(np.abs(debt_arr))),
            "total_burden_mean": float(np.mean(total_burden)),
            "total_burden_std": float(np.std(total_burden)),
        }


# =============================================================================
# 5. Environment normalization, built-in smoke environment, and fair wrappers
# =============================================================================
class BoxLike:
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)

def normalize_obs_container(obs: Any, n_agents: int) -> List[np.ndarray]:
    if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
        obs = obs[0]
    if isinstance(obs, dict):
        keys = sorted(obs.keys(), key=lambda x: str(x))
        vals = [obs[k] for k in keys]
        if len(vals) >= n_agents:
            return [as_np(v).reshape(-1) for v in vals[:n_agents]]
    if isinstance(obs, (list, tuple)):
        if len(obs) >= n_agents:
            return [as_np(v).reshape(-1) for v in list(obs)[:n_agents]]
    arr = as_np(obs)
    if arr.ndim >= 2 and arr.shape[0] >= n_agents:
        return [arr[i].reshape(-1) for i in range(n_agents)]
    if arr.ndim == 1:
        chunks = np.array_split(arr, n_agents)
        return [c.reshape(-1) for c in chunks]
    raise ValueError(f"Cannot normalize observations of type {type(obs)} with shape {getattr(obs, 'shape', None)}")

def normalize_done(done: Any, trunc: Any = None) -> bool:
    vals = []
    for x in [done, trunc]:
        if x is None:
            continue
        if isinstance(x, dict):
            vals.extend([bool(v) for v in x.values()])
        elif isinstance(x, (list, tuple, np.ndarray)):
            vals.extend([bool(v) for v in np.asarray(x).reshape(-1)])
        else:
            vals.append(bool(x))
    return bool(any(vals)) if vals else False

def scalar_reward(reward: Any) -> float:
    if isinstance(reward, dict):
        vals = [float(v) for v in reward.values()]
        return float(np.sum(vals))
    arr = as_np(reward, dtype=np.float64).reshape(-1)
    return float(np.mean(arr))

def scalar_cost_from_info(info: Dict[str, Any]) -> float:
    if not isinstance(info, dict):
        return 0.0
    for key in ["safety_cost", "cost", "cost_sum", "cost_total", "constraint_cost"]:
        if key in info:
            arr = as_np(info[key], dtype=np.float64).reshape(-1)
            return float(np.mean(arr))
    if "costs" in info and isinstance(info["costs"], dict):
        arr = np.asarray([float(v) for v in info["costs"].values()], dtype=np.float64).reshape(-1)
        return float(np.mean(arr))
    for key in ["agent_safety_costs", "agent_costs", "cost_agents"]:
        if key in info:
            arr = as_np(info[key], dtype=np.float64).reshape(-1)
            return float(np.sum(arr))
    return 0.0

def pack_actions_for_env(actions: Sequence[np.ndarray], action_space: Any = None, env: Any = None) -> Any:
    if isinstance(action_space, dict):
        keys = sorted(action_space.keys(), key=lambda x: str(x))
        return {k: actions[i] for i, k in enumerate(keys[:len(actions)])}
    if callable(action_space) and env is not None and hasattr(env, "possible_agents"):
        keys = list(getattr(env, "possible_agents"))
        return {k: np.asarray(actions[i], dtype=np.float32).reshape(-1) for i, k in enumerate(keys[:len(actions)])}
    if isinstance(action_space, (list, tuple)):
        return [np.asarray(a, dtype=np.float32) for a in actions]
    arr = np.stack([np.asarray(a, dtype=np.float32).reshape(-1) for a in actions], axis=0)
    if hasattr(action_space, "shape"):
        shape = tuple(action_space.shape)
        if len(shape) == 1 and int(np.prod(shape)) == arr.size:
            return arr.reshape(shape)
        if len(shape) >= 2 and shape[0] == len(actions):
            return arr.reshape(shape)
    return arr

class BottleneckFairEnv:
    def __init__(self, n_agents: int = 4, n_hazards: int = 2, episode_length: int = 160, goal_radius: float = 0.12, seed: int = 0):
        self.n_agents = n_agents
        self.n_hazards = n_hazards
        self.max_steps = episode_length
        self.goal_radius = goal_radius
        self.dt = 0.06
        self.step_count = 0
        self.rng = np.random.default_rng(seed)
        self.observation_space = [BoxLike(-np.inf, np.inf, (10,)) for _ in range(n_agents)]
        self.action_space = [BoxLike(-1.0, 1.0, (2,)) for _ in range(n_agents)]
        self.hazards = np.array([[0.0, -0.18], [0.0, 0.18], [0.22, 0.0]], dtype=np.float64)[:n_hazards]
        self.reset(seed=seed)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
        y = np.linspace(-0.45, 0.45, self.n_agents)
        self.positions = np.stack([np.full(self.n_agents, -1.0), y], axis=1)
        self.goals = np.stack([np.full(self.n_agents, 1.0), -y], axis=1)
        self.energy = np.ones(self.n_agents, dtype=np.float64)
        self.step_count = 0
        return self._obs(), {"agent_positions": self.positions.copy(), "goal_positions": self.goals.copy()}

    def _obs(self) -> List[np.ndarray]:
        obs = []
        d = pairwise_distances(self.positions)
        for i in range(self.n_agents):
            nearest = np.argsort(d[i])[1] if self.n_agents > 1 else i
            rel_other = self.positions[nearest] - self.positions[i]
            rel_goal = self.goals[i] - self.positions[i]
            nearest_hazard = self.hazards[np.argmin(np.linalg.norm(self.hazards - self.positions[i], axis=1))] - self.positions[i]
            obs_i = np.concatenate([self.positions[i], self.goals[i], rel_goal, rel_other, nearest_hazard]).astype(np.float32)
            obs.append(obs_i)
        return obs

    def step(self, actions: Sequence[np.ndarray]):
        actions = [np.clip(as_np(a, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0) for a in actions]
        old_positions = self.positions.copy()
        proposed = self.positions + self.dt * np.stack(actions, axis=0)
        in_gate = np.abs(proposed[:, 0]) < 0.25
        proposed[in_gate, 1] = np.clip(proposed[in_gate, 1], -0.16, 0.16)
        proposed[:, 0] = np.clip(proposed[:, 0], -1.25, 1.25)
        proposed[:, 1] = np.clip(proposed[:, 1], -0.8, 0.8)
        self.positions = proposed
        self.energy = np.maximum(0.0, self.energy - 0.002 - 0.003 * np.array([np.linalg.norm(a) for a in actions]))

        dist_goal = np.linalg.norm(self.goals - self.positions, axis=1)
        success = dist_goal < self.goal_radius
        progress = np.linalg.norm(self.goals - old_positions, axis=1) - dist_goal
        reward = float(2.0 * np.mean(progress) - 0.05 * np.mean([np.linalg.norm(a) for a in actions]) + 2.0 * np.mean(success))

        hazard_d = np.min(np.linalg.norm(self.positions[:, None, :] - self.hazards[None, :, :], axis=-1), axis=1)
        hazard_cost = (hazard_d < 0.16).astype(np.float64)
        d = pairwise_distances(self.positions)
        np.fill_diagonal(d, np.inf)
        collision_cost = (np.min(d, axis=1) < 0.10).astype(np.float64)
        agent_costs = np.maximum(hazard_cost, collision_cost)
        safety_cost = float(np.sum(agent_costs))

        low_energy_wait = ((self.energy < 0.35) & (np.linalg.norm(self.positions - np.array([0.0, 0.0]), axis=1) > 0.2)).astype(np.float64)
        actions_norm = np.array([np.linalg.norm(a) for a in actions])
        yield_events = ((np.min(d, axis=1) < 0.24) & (actions_norm < 0.20)).astype(np.float64)

        self.step_count += 1
        terminated = bool(np.all(success))
        truncated = bool(self.step_count >= self.max_steps)
        info = {
            "agent_positions": self.positions.copy(),
            "goal_positions": self.goals.copy(),
            "agent_safety_costs": agent_costs.copy(),
            "safety_cost": safety_cost,
            "hazard_exposure": hazard_cost.copy(),
            "yield_events": yield_events.copy(),
            "resource_denial": low_energy_wait.copy(),
            "success": float(np.all(success)),
            "progress": progress.copy(),
        }
        return self._obs(), reward, terminated, truncated, info


class SafeFairBenchEnv:
    """Small SafeFairBench-v0 adapters for smokeable benchmark development.

    These are dependency-free stand-ins for VMAS/POGEMA/Highway-style task
    families. They expose a normalized multi-agent continuous-control API and
    the burden event keys used by the Safe-Fair wrapper, while keeping the
    scenario logic compact enough for CPU smoke tests.
    """

    def __init__(self, task: TaskSpec, seed: int = 0):
        self.task = task
        self.scenario = str(task.kwargs.get("scenario", "vmas_multigoal"))
        self.n_agents = int(task.kwargs.get("n_agents", task.n_agents))
        self.max_steps = int(task.episode_length)
        self.goal_radius = float(task.kwargs.get("goal_radius", 0.10))
        self.dt = 0.065
        default_collision_radius = 0.18 if self.scenario == "highway_merge" else 0.13
        default_hazard_radius = 0.20 if self.scenario == "highway_merge" else 0.15
        self.collision_radius = float(task.kwargs.get("collision_radius", default_collision_radius))
        self.hazard_radius = float(task.kwargs.get("hazard_radius", default_hazard_radius))
        self.yield_distance = float(task.kwargs.get("yield_distance", 0.24))
        self.yield_action_threshold = float(task.kwargs.get("yield_action_threshold", 0.18))
        self.station_radius = float(task.kwargs.get("station_radius", 0.24))
        self.lane_conflict_radius = float(task.kwargs.get("lane_conflict_radius", 0.25))
        self.shelf_block_cost = float(task.kwargs.get("shelf_block_cost", 0.5))
        self.step_count = 0
        self.reset_count = 0
        self.rng = np.random.default_rng(seed)
        self.observation_space = [BoxLike(-np.inf, np.inf, (14,)) for _ in range(self.n_agents)]
        self.action_space = [BoxLike(-1.0, 1.0, (2,)) for _ in range(self.n_agents)]
        self.hazards = np.zeros((0, 2), dtype=np.float64)
        self.station_positions = np.zeros((0, 2), dtype=np.float64)
        self.shelves: List[Tuple[float, float, float, float]] = []
        self.use_recovery_projection = False
        self.reset(seed=seed)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
        self.step_count = 0
        y = np.linspace(-0.45, 0.45, self.n_agents)
        self.velocities = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.energy = np.ones(self.n_agents, dtype=np.float64)
        self.completed = np.zeros(self.n_agents, dtype=np.float64)
        self.station_positions = np.zeros((0, 2), dtype=np.float64)
        self.shelves = []

        if self.scenario == "vmas_chargeshare":
            self.positions = np.stack([np.full(self.n_agents, -1.0), y], axis=1)
            self.goals = np.stack([np.full(self.n_agents, 1.05), -y], axis=1)
            self.hazards = np.asarray([[0.16, 0.18], [0.16, -0.18]], dtype=np.float64)
            self.station_positions = np.asarray([[0.0, 0.0]], dtype=np.float64)
            self.energy = self.rng.uniform(0.62, 0.92, size=self.n_agents)
        elif self.scenario == "pogema_warehouse":
            self.positions = np.stack([np.full(self.n_agents, -1.08), y], axis=1)
            self.goals = np.stack([np.full(self.n_agents, 1.08), -y], axis=1)
            self.hazards = np.asarray([[0.0, 0.0], [0.42, 0.0]], dtype=np.float64)
            self.shelves = [(-0.55, -0.22, -0.58, -0.12), (-0.55, -0.22, 0.12, 0.58), (0.22, 0.55, -0.58, -0.12), (0.22, 0.55, 0.12, 0.58)]
        elif self.scenario == "highway_merge":
            base = np.asarray([[-1.10, 0.18], [-0.58, 0.18], [-1.05, -0.42]], dtype=np.float64)
            if self.n_agents > len(base):
                extra_y = np.linspace(-0.20, 0.20, self.n_agents - len(base))
                extra = np.stack([np.full(len(extra_y), -0.85), extra_y], axis=1)
                base = np.concatenate([base, extra], axis=0)
            self.positions = base[: self.n_agents].copy()
            main_lane_y = float(self.task.kwargs.get("highway_main_lane_y", 0.18))
            goal_base_x = float(self.task.kwargs.get("highway_goal_base_x", 1.20))
            self.goals = np.stack([np.full(self.n_agents, goal_base_x), np.full(self.n_agents, main_lane_y)], axis=1)
            if bool(self.task.kwargs.get("highway_stagger_goals", False)):
                spacing_x = float(self.task.kwargs.get("highway_goal_x_spacing", 0.30))
                spacing_y = float(self.task.kwargs.get("highway_goal_y_spacing", 0.0))
                order = np.argsort(-self.positions[:, 0])
                centered = np.arange(self.n_agents, dtype=np.float64) - 0.5 * (self.n_agents - 1)
                for rank, agent_idx in enumerate(order):
                    self.goals[agent_idx, 0] = goal_base_x - spacing_x * rank
                    self.goals[agent_idx, 1] = main_lane_y + spacing_y * centered[rank]
            self.velocities[:, 0] = np.linspace(0.035, 0.055, self.n_agents)
            self.hazards = np.asarray([[-0.12, -0.08], [0.05, 0.08]], dtype=np.float64)
        else:
            self.positions = np.stack([np.full(self.n_agents, -1.0), y], axis=1)
            shared_goal_y = np.linspace(-0.22, 0.22, self.n_agents)
            goal_y = shared_goal_y[::-1]
            if bool(self.task.kwargs.get("rotate_roles", False)) and self.n_agents > 1:
                role_offset = int(seed) % self.n_agents if seed is not None else self.reset_count % self.n_agents
                goal_y = np.roll(goal_y, role_offset)
            self.reset_count += 1
            self.goals = np.stack([np.full(self.n_agents, 1.0), goal_y], axis=1)
            self.hazards = np.asarray([[-0.04, 0.18], [0.12, -0.18], [0.42, 0.0]], dtype=np.float64)

        return self._obs(), self._info_template(np.zeros(self.n_agents), np.zeros(self.n_agents), np.zeros(self.n_agents), np.zeros(self.n_agents), np.zeros(self.n_agents))

    def _obs(self) -> List[np.ndarray]:
        d = pairwise_distances(self.positions)
        if self.n_agents > 1:
            np.fill_diagonal(d, np.inf)
        obs = []
        for i in range(self.n_agents):
            nearest = int(np.argmin(d[i])) if self.n_agents > 1 else i
            rel_other = self.positions[nearest] - self.positions[i] if self.n_agents > 1 else np.zeros(2, dtype=np.float64)
            rel_goal = self.goals[i] - self.positions[i]
            if self.hazards.size:
                nearest_hazard = self.hazards[np.argmin(np.linalg.norm(self.hazards - self.positions[i], axis=1))] - self.positions[i]
            else:
                nearest_hazard = np.zeros(2, dtype=np.float64)
            obs_i = np.concatenate(
                [
                    self.positions[i],
                    rel_goal,
                    self.velocities[i],
                    rel_other,
                    nearest_hazard,
                    np.asarray(
                        [
                            self.energy[i],
                            self.step_count / max(1, self.max_steps),
                            np.linalg.norm(rel_goal),
                            i / max(1, self.n_agents - 1),
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
            obs.append(obs_i.astype(np.float32))
        return obs

    def _inside_shelf(self, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in self.shelves)

    def _pogema_outer_lane_y(self, agent_idx: int) -> float:
        lanes = np.asarray([-0.68, -0.64, 0.64, 0.68], dtype=np.float64)
        lane_mode = str(self.task.kwargs.get("pogema_lane_mode", "default"))
        if lane_mode == "paired" and self.n_agents == 4:
            lanes = np.asarray([-0.66, -0.66, 0.66, 0.66], dtype=np.float64)
        elif lane_mode == "biased" and self.n_agents == 4:
            lanes = np.asarray([-0.66, 0.66, 0.66, 0.66], dtype=np.float64)
        if self.n_agents != len(lanes):
            half = max(1, self.n_agents // 2)
            lower = np.linspace(-0.68, -0.62, half, dtype=np.float64)
            upper = np.linspace(0.62, 0.68, self.n_agents - half, dtype=np.float64)
            lanes = np.concatenate([lower, upper])
        return float(lanes[agent_idx % len(lanes)])

    def _clip_scenario_bounds(self, points: np.ndarray) -> np.ndarray:
        clipped = points.copy()
        if self.scenario == "highway_merge":
            clipped[:, 0] = np.clip(clipped[:, 0], -1.25, 1.30)
            clipped[:, 1] = np.clip(clipped[:, 1], -0.46, 0.28)
        elif self.scenario == "pogema_warehouse":
            clipped[:, 0] = np.clip(clipped[:, 0], -1.22, 1.22)
            clipped[:, 1] = np.clip(clipped[:, 1], -0.70, 0.70)
        else:
            clipped[:, 0] = np.clip(clipped[:, 0], -1.25, 1.25)
            clipped[:, 1] = np.clip(clipped[:, 1], -0.75, 0.75)
        return clipped

    def _project_recovery_safe_positions(self, points: np.ndarray) -> np.ndarray:
        projected = np.asarray(points, dtype=np.float64).copy()
        clearance = float(self.task.kwargs.get("recovery_projection_clearance", 0.05))
        hazard_limit = self.hazard_radius + clearance
        if self.hazards.size:
            for _ in range(4):
                moved = False
                for i in range(self.n_agents):
                    for hazard in self.hazards:
                        diff = projected[i] - hazard
                        dist = float(np.linalg.norm(diff))
                        if dist < hazard_limit:
                            direction = unit_vector(diff, fallback=np.asarray([0.0, 1.0], dtype=np.float64))
                            projected[i] = hazard + direction * hazard_limit
                            moved = True
                projected = self._clip_scenario_bounds(projected)
                if not moved:
                    break
        collision_limit = self.collision_radius + clearance
        for _ in range(3):
            moved = False
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    diff = projected[i] - projected[j]
                    dist = float(np.linalg.norm(diff))
                    if dist < collision_limit:
                        direction = unit_vector(diff, fallback=np.asarray([1.0, 0.0], dtype=np.float64))
                        push = 0.5 * (collision_limit - dist + 1e-6)
                        projected[i] += push * direction
                        projected[j] -= push * direction
                        moved = True
            projected = self._clip_scenario_bounds(projected)
            if not moved:
                break
        if self.scenario == "highway_merge" and bool(self.task.kwargs.get("highway_project_lane_conflicts", False)):
            lane_y = float(self.task.kwargs.get("highway_main_lane_y", 0.18))
            lane_band = float(self.task.kwargs.get("highway_lane_conflict_band", 0.16))
            lane_limit = float(self.lane_conflict_radius + clearance)
            for _ in range(4):
                moved = False
                for i in range(self.n_agents):
                    for j in range(i + 1, self.n_agents):
                        if abs(float(projected[i, 1]) - lane_y) >= lane_band and abs(float(projected[j, 1]) - lane_y) >= lane_band:
                            continue
                        diff = projected[i] - projected[j]
                        dist = float(np.linalg.norm(diff))
                        if dist < lane_limit:
                            if abs(float(diff[0])) > 1e-6:
                                direction = np.asarray([math.copysign(1.0, float(diff[0])), 0.0], dtype=np.float64)
                            else:
                                direction = np.asarray([1.0 if i > j else -1.0, 0.0], dtype=np.float64)
                            push = 0.5 * (lane_limit - dist + 1e-6)
                            projected[i] += push * direction
                            projected[j] -= push * direction
                            moved = True
                projected = self._clip_scenario_bounds(projected)
                if not moved:
                    break
        return self._clip_scenario_bounds(projected)

    def _apply_domain_constraints(self, proposed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        blocked = np.zeros(self.n_agents, dtype=np.float64)
        clipped = proposed.copy()
        if self.scenario in {"vmas_multigoal", "vmas_chargeshare"}:
            in_gate = np.abs(clipped[:, 0]) < 0.25
            clipped[in_gate, 1] = np.clip(clipped[in_gate, 1], -0.18, 0.18)
            clipped[:, 0] = np.clip(clipped[:, 0], -1.25, 1.25)
            clipped[:, 1] = np.clip(clipped[:, 1], -0.75, 0.75)
        elif self.scenario == "pogema_warehouse":
            clipped[:, 0] = np.clip(clipped[:, 0], -1.22, 1.22)
            clipped[:, 1] = np.clip(clipped[:, 1], -0.70, 0.70)
            for i in range(self.n_agents):
                x = float(clipped[i, 0])
                if -0.62 <= x <= 0.64:
                    lane_y = self._pogema_outer_lane_y(i)
                    clipped[i, 1] = lane_y
                if self._inside_shelf(clipped[i]):
                    clipped[i] = self.positions[i]
                    blocked[i] = 1.0
        elif self.scenario == "highway_merge":
            clipped[:, 0] = np.clip(clipped[:, 0], -1.25, 1.30)
            clipped[:, 1] = np.clip(clipped[:, 1], -0.46, 0.28)
            ramp_mask = clipped[:, 1] < -0.18
            merge_zone = (-0.35 < clipped[:, 0]) & (clipped[:, 0] < 0.28)
            clipped[ramp_mask & merge_zone, 1] = np.minimum(clipped[ramp_mask & merge_zone, 1] + 0.045, 0.18)
        if bool(getattr(self, "use_recovery_projection", False)):
            clipped = self._project_recovery_safe_positions(clipped)
        return clipped, blocked

    def _charge_update(self, actions_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.scenario != "vmas_chargeshare":
            zeros = np.zeros(self.n_agents, dtype=np.float64)
            return zeros, zeros, zeros, np.ones(self.n_agents, dtype=np.float64)
        norms = np.linalg.norm(actions_arr, axis=1)
        charge_received = np.zeros(self.n_agents, dtype=np.float64)
        charger_wait = np.zeros(self.n_agents, dtype=np.float64)
        near_station = np.zeros(self.n_agents, dtype=np.float64)
        dists = np.linalg.norm(self.positions[:, None, :] - self.station_positions[None, :, :], axis=-1)
        near_mask = dists <= self.station_radius
        near_station = near_mask.any(axis=1).astype(np.float64)
        for station_idx in range(self.station_positions.shape[0]):
            contenders = np.where(near_mask[:, station_idx])[0]
            if len(contenders) == 0:
                continue
            winner = int(contenders[np.argmin(self.energy[contenders])])
            charge_received[winner] += 0.075
            for idx in contenders:
                if idx != winner:
                    charger_wait[idx] = 1.0
        self.energy = np.clip(self.energy - 0.004 - 0.022 * norms + charge_received, 0.0, 1.0)
        low_energy = (self.energy < 0.34).astype(np.float64)
        action_scale = np.ones(self.n_agents, dtype=np.float64)
        if np.any(low_energy > 0.0):
            action_scale[low_energy > 0.0] = np.clip(self.energy[low_energy > 0.0] / 0.34, 0.30, 1.0)
        resource_denial = np.maximum(charger_wait, ((low_energy > 0.0) & (near_station <= 0.0)).astype(np.float64))
        return resource_denial, charger_wait, low_energy, action_scale

    def _info_template(
        self,
        agent_costs: np.ndarray,
        hazard_exposure: np.ndarray,
        yield_events: np.ndarray,
        resource_denial: np.ndarray,
        detour: np.ndarray,
    ) -> Dict[str, Any]:
        wait_time = np.zeros(self.n_agents, dtype=np.float64)
        event_dict = {
            "wait_time": wait_time.copy(),
            "yield_count": yield_events.copy(),
            "hazard_dwell": hazard_exposure.copy(),
            "detour_length": detour.copy(),
            "resource_miss": resource_denial.copy(),
            "task_load": self.completed.copy(),
        }
        return {
            "agent_positions": self.positions.copy(),
            "goal_positions": self.goals.copy(),
            "agent_safety_costs": np.asarray(agent_costs, dtype=np.float64).copy(),
            "hazard_exposure": np.asarray(hazard_exposure, dtype=np.float64).copy(),
            "yield_events": np.asarray(yield_events, dtype=np.float64).copy(),
            "resource_denial": np.asarray(resource_denial, dtype=np.float64).copy(),
            "agent_detour": np.asarray(detour, dtype=np.float64).copy(),
            "burden_events": event_dict,
            "safety_cost": float(np.sum(agent_costs)),
            "success": float(np.all(np.linalg.norm(self.goals - self.positions, axis=1) < self.goal_radius)),
        }

    def step(self, actions: Sequence[np.ndarray]):
        actions_arr = np.stack([np.clip(as_np(a, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0) for a in actions], axis=0)
        old_positions = self.positions.copy()
        old_dist = np.linalg.norm(self.goals - old_positions, axis=1)

        resource_denial, charger_wait, low_energy, action_scale = self._charge_update(actions_arr)
        actions_arr = actions_arr * action_scale[:, None]

        if self.scenario == "highway_merge":
            self.velocities[:, 0] = np.clip(self.velocities[:, 0] + 0.020 * actions_arr[:, 0], 0.015, 0.085)
            self.velocities[:, 1] = 0.70 * self.velocities[:, 1] + 0.035 * actions_arr[:, 1]
        else:
            self.velocities = 0.45 * self.velocities + self.dt * actions_arr

        proposed = self.positions + self.velocities
        proposed, blocked = self._apply_domain_constraints(proposed)
        self.positions = proposed
        self.velocities[blocked > 0.0] = 0.0

        dist_goal = np.linalg.norm(self.goals - self.positions, axis=1)
        success = (dist_goal < self.goal_radius).astype(np.float64)
        self.completed = np.maximum(self.completed, success)
        progress = old_dist - dist_goal
        step_len = np.linalg.norm(self.positions - old_positions, axis=1)
        detour = np.maximum(0.0, step_len - np.maximum(progress, 0.0))

        d = pairwise_distances(self.positions)
        if self.n_agents > 1:
            np.fill_diagonal(d, np.inf)
        nearest = np.min(d, axis=1) if self.n_agents > 1 else np.full(self.n_agents, np.inf)
        close_conflict = (nearest < self.collision_radius).astype(np.float64)
        if self.hazards.size:
            hazard_d = np.min(np.linalg.norm(self.positions[:, None, :] - self.hazards[None, :, :], axis=-1), axis=1)
            hazard_exposure = (hazard_d < self.hazard_radius).astype(np.float64)
        else:
            hazard_exposure = np.zeros(self.n_agents, dtype=np.float64)

        action_norm = np.linalg.norm(actions_arr, axis=1)
        yield_events = ((nearest < self.yield_distance) & (action_norm < self.yield_action_threshold)).astype(np.float64)
        if self.scenario == "highway_merge":
            merge_zone = ((-0.35 < self.positions[:, 0]) & (self.positions[:, 0] < 0.30)).astype(np.float64)
            slow = (self.velocities[:, 0] < 0.035).astype(np.float64)
            yield_events = np.maximum(yield_events, merge_zone * slow)
            lane_conflict = ((nearest < self.lane_conflict_radius) & (np.abs(self.positions[:, 1] - 0.18) < 0.16)).astype(np.float64)
            close_conflict = np.maximum(close_conflict, lane_conflict)
            resource_denial = np.maximum(resource_denial, merge_zone * (self.positions[:, 1] < -0.08).astype(np.float64))
        elif self.scenario == "pogema_warehouse":
            narrow_aisle = (np.abs(self.positions[:, 1]) < 0.14).astype(np.float64)
            yield_events = np.maximum(yield_events, narrow_aisle * (nearest < 0.28).astype(np.float64))
            resource_denial = np.maximum(resource_denial, blocked)

        agent_costs = np.maximum(hazard_exposure, close_conflict)
        if self.scenario == "pogema_warehouse":
            agent_costs = np.maximum(agent_costs, self.shelf_block_cost * blocked)

        reward = float(
            2.5 * np.mean(progress)
            + 1.2 * np.mean(success)
            - 0.04 * np.mean(action_norm)
            - 0.35 * np.mean(agent_costs)
            - 0.08 * np.mean(resource_denial)
        )
        if self.scenario == "highway_merge":
            reward += float(0.5 * np.mean(np.maximum(0.0, self.velocities[:, 0])))

        self.step_count += 1
        terminated = bool(np.all(success > 0.0))
        truncated = bool(self.step_count >= self.max_steps)
        info = self._info_template(agent_costs, hazard_exposure, yield_events, resource_denial, detour)
        if self.scenario == "vmas_chargeshare":
            info.update(
                {
                    "energy": self.energy.copy(),
                    "charger_wait": charger_wait.copy(),
                    "low_energy_wait": low_energy.copy(),
                    "charge_received": np.maximum(0.0, self.energy - 0.34),
                }
            )
        return self._obs(), reward, terminated, truncated, info


class SafetyGymTaskAdapter:
    def __init__(self, env: Any, task: TaskSpec):
        self.env = env
        self.task = task
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)
        self.possible_agents = list(getattr(env, "possible_agents", [f"agent_{i}" for i in range(task.n_agents)]))
        station_positions = task.kwargs.get("station_positions", [])
        self.station_positions = np.asarray(station_positions, dtype=np.float64).reshape(-1, 2) if station_positions else np.zeros((0, 2), dtype=np.float64)
        self.station_radius = float(task.kwargs.get("station_radius", 0.30))
        self.low_energy_threshold = float(task.kwargs.get("low_energy_threshold", 0.35))
        self.energy_drain_idle = float(task.kwargs.get("energy_drain_idle", 0.003))
        self.energy_drain_action = float(task.kwargs.get("energy_drain_action", 0.018))
        self.energy_charge_rate = float(task.kwargs.get("energy_charge_rate", 0.055))
        self.min_action_scale = float(task.kwargs.get("min_action_scale", 0.35))
        self.energy = np.ones(task.n_agents, dtype=np.float64)
        self.last_goal_dist = np.zeros(task.n_agents, dtype=np.float64)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def _agent_positions(self) -> np.ndarray:
        task = getattr(self.env, "task", None)
        agent = getattr(task, "agent", None)
        positions = []
        for i in range(self.task.n_agents):
            pos = getattr(agent, f"pos_{i}", np.zeros(3, dtype=np.float64))
            positions.append(np.asarray(pos, dtype=np.float64)[:2])
        return np.stack(positions, axis=0)

    def _goal_positions(self) -> np.ndarray:
        task = getattr(self.env, "task", None)
        goals = []
        for name in ["goal_red", "goal_blue"]:
            goal = getattr(task, name, None)
            if goal is None or not hasattr(goal, "pos"):
                goals.append(np.zeros(2, dtype=np.float64))
            else:
                goals.append(np.asarray(goal.pos, dtype=np.float64)[:2])
        while len(goals) < self.task.n_agents:
            goals.append(goals[-1].copy())
        return np.stack(goals[: self.task.n_agents], axis=0)

    def _hazard_positions(self) -> np.ndarray:
        task = getattr(self.env, "task", None)
        hazards = getattr(task, "hazards", None)
        if hazards is None or not hasattr(hazards, "pos"):
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray([np.asarray(pos, dtype=np.float64)[:2] for pos in hazards.pos], dtype=np.float64)

    def _goal_distances(self) -> np.ndarray:
        task = getattr(self.env, "task", None)
        dists = []
        for fn_name in ["dist_goal_red", "dist_goal_blue"]:
            fn = getattr(task, fn_name, None)
            dists.append(float(fn()) if callable(fn) else 0.0)
        while len(dists) < self.task.n_agents:
            dists.append(dists[-1])
        return np.asarray(dists[: self.task.n_agents], dtype=np.float64)

    def _nested_cost_vector(self, costs: Optional[Dict[str, Any]], info: Dict[str, Any], field: str) -> np.ndarray:
        vals = []
        for i in range(self.task.n_agents):
            agent_key = f"agent_{i}"
            value = 0.0
            if isinstance(costs, dict) and agent_key in costs:
                value = float(np.asarray(costs[agent_key], dtype=np.float64).reshape(-1)[0])
            if isinstance(info, dict):
                probe = info.get(agent_key)
                if isinstance(probe, dict):
                    nested = probe.get(agent_key)
                    if isinstance(nested, dict) and field in nested:
                        value = float(np.asarray(nested[field], dtype=np.float64).reshape(-1)[0])
            vals.append(value)
        return np.asarray(vals, dtype=np.float64)

    def _charge_share_action_transform(
        self,
        actions: Dict[str, np.ndarray],
        positions_before: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        action_arr = np.stack([np.asarray(actions[agent], dtype=np.float64).reshape(-1) for agent in self.possible_agents], axis=0)
        norms = np.linalg.norm(action_arr, axis=1)
        charge_received = np.zeros(self.task.n_agents, dtype=np.float64)
        charger_wait = np.zeros(self.task.n_agents, dtype=np.float64)
        near_station = np.zeros(self.task.n_agents, dtype=np.float64)
        if self.station_positions.size:
            dists = np.linalg.norm(positions_before[:, None, :] - self.station_positions[None, :, :], axis=-1)
            near_mask = dists <= self.station_radius
            near_station = near_mask.any(axis=1).astype(np.float64)
            for station_idx in range(self.station_positions.shape[0]):
                contenders = np.where(near_mask[:, station_idx])[0]
                if len(contenders) == 0:
                    continue
                winner = int(contenders[np.argmin(self.energy[contenders])])
                charge_received[winner] += self.energy_charge_rate
                for idx in contenders:
                    if idx != winner:
                        charger_wait[idx] = 1.0
        drain = self.energy_drain_idle + self.energy_drain_action * norms
        self.energy = np.clip(self.energy - drain + charge_received, 0.0, 1.0)
        low_energy = (self.energy < self.low_energy_threshold).astype(np.float64)
        scale = np.ones(self.task.n_agents, dtype=np.float64)
        low_mask = low_energy > 0.0
        if np.any(low_mask):
            scale[low_mask] = np.clip(self.energy[low_mask] / max(self.low_energy_threshold, 1e-8), self.min_action_scale, 1.0)
        scaled_arr = action_arr * scale[:, None]
        denied_at_station = (
            (low_energy > 0.0)
            & (near_station > 0.0)
            & (charge_received <= 1e-8)
        ).astype(np.float64)
        low_energy_wait = ((low_energy > 0.0) & (near_station <= 0.0)).astype(np.float64)
        resource_denial = np.maximum(charger_wait, denied_at_station)
        scaled_actions = {
            agent: scaled_arr[i].astype(np.float32)
            for i, agent in enumerate(self.possible_agents)
        }
        extra = {
            "energy": self.energy.copy(),
            "charger_wait": charger_wait,
            "low_energy_wait": low_energy_wait,
            "resource_denial": resource_denial,
            "charge_received": charge_received,
            "near_station": near_station,
            "action_scale": scale,
        }
        return scaled_actions, extra

    def _augment_info(
        self,
        info: Optional[Dict[str, Any]],
        costs: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        info = dict(info) if isinstance(info, dict) else {}
        positions = self._agent_positions()
        goals = self._goal_positions()
        goal_dist = self._goal_distances()
        goal_progress = np.maximum(0.0, self.last_goal_dist - goal_dist)
        self.last_goal_dist = goal_dist.copy()
        cost_agents = self._nested_cost_vector(costs, info, "cost_sum")
        hazard_costs = self._nested_cost_vector(costs, info, "cost_hazards")
        hazard_positions = self._hazard_positions()
        if hazard_positions.size:
            min_hazard_dist = np.min(np.linalg.norm(positions[:, None, :] - hazard_positions[None, :, :], axis=-1), axis=1)
        else:
            min_hazard_dist = np.full(self.task.n_agents, np.inf, dtype=np.float64)
        info.update(
            {
                "agent_positions": positions,
                "goal_positions": goals,
                "agent_goal_distances": goal_dist,
                "goal_progress": goal_progress,
                "agent_safety_costs": cost_agents,
                "hazard_exposure": hazard_costs,
                "min_hazard_distance": min_hazard_dist,
                "success": float(bool(info.get("goal_met", False))),
            }
        )
        if extra:
            info.update(extra)
        return info

    def reset(self, seed: Optional[int] = None):
        self.energy = np.ones(self.task.n_agents, dtype=np.float64)
        out = self.env.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self.last_goal_dist = self._goal_distances()
        info = self._augment_info(info)
        return obs, info

    def step(self, actions: Dict[str, np.ndarray]):
        extra: Dict[str, Any] = {}
        positions_before = self._agent_positions()
        env_actions = {
            agent: np.asarray(actions[agent], dtype=np.float32).reshape(-1)
            for agent in self.possible_agents
        }
        if self.task.source == "safety_gymnasium_chargeshare":
            env_actions, extra = self._charge_share_action_transform(env_actions, positions_before)
        out = self.env.step(env_actions)
        if len(out) != 6:
            return out
        next_obs, reward, costs, terminated, truncated, info = out
        info = self._augment_info(info, costs=costs, extra=extra)
        return next_obs, reward, costs, terminated, truncated, info

class FairMultiAgentWrapper:
    def __init__(self, env: Any, task: TaskSpec, method: MethodSpec):
        self.env = env
        self.task = task
        self.method = method
        self.n_agents = task.n_agents
        weights = BurdenWeights()
        if task.source == "safety_gymnasium":
            weights = BurdenWeights(wait=0.30, hazard=0.30, yield_=0.20, detour=0.15, resource=0.05)
        elif task.source == "safety_gymnasium_chargeshare":
            weights = BurdenWeights(wait=0.25, hazard=0.20, yield_=0.15, detour=0.15, resource=0.25)
        elif task.source == "safefairbench_v0":
            scenario = str(task.kwargs.get("scenario", ""))
            if scenario == "vmas_chargeshare":
                weights = BurdenWeights(wait=0.20, hazard=0.15, yield_=0.15, detour=0.15, resource=0.35)
            elif scenario == "pogema_warehouse":
                weights = BurdenWeights(wait=0.25, hazard=0.10, yield_=0.25, detour=0.25, resource=0.15)
            elif scenario == "highway_merge":
                weights = BurdenWeights(wait=0.20, hazard=0.25, yield_=0.35, detour=0.05, resource=0.15)
            else:
                weights = BurdenWeights(wait=0.30, hazard=0.25, yield_=0.20, detour=0.20, resource=0.05)
        weights = apply_disabled_burden_components(weights, method.disabled_burden_components)
        self.logger = FairnessBurdenLogger(
            n_agents=task.n_agents,
            weights=weights,
            target_shares=method.target_shares,
            use_debt=method.use_debt,
            use_envy=method.use_envy,
            fairness_cost_mode=method.fairness_cost_mode,
            local_k_hop=method.local_k_hop,
        )
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)

    def _predict_safefairbench_positions(self, actions_arr: np.ndarray) -> np.ndarray:
        positions = np.asarray(getattr(self.env, "positions", np.zeros((self.n_agents, 2))), dtype=np.float64)
        velocities = np.asarray(getattr(self.env, "velocities", np.zeros_like(positions)), dtype=np.float64)
        dt = float(getattr(self.env, "dt", 0.065))
        scenario = str(self.task.kwargs.get("scenario", ""))
        if scenario == "highway_merge":
            next_vel = velocities.copy()
            next_vel[:, 0] = np.clip(next_vel[:, 0] + 0.020 * actions_arr[:, 0], 0.015, 0.085)
            next_vel[:, 1] = 0.70 * next_vel[:, 1] + 0.035 * actions_arr[:, 1]
        else:
            next_vel = 0.45 * velocities + dt * actions_arr
        proposed = positions + next_vel
        if hasattr(self.env, "_apply_domain_constraints"):
            try:
                proposed, _ = self.env._apply_domain_constraints(proposed)
            except Exception:
                pass
        return np.asarray(proposed, dtype=np.float64)

    def _safefairbench_route_target(self, agent_idx: int, position: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, float]:
        """Scenario-aware safety waypoints for dependency-free SafeFairBench adapters.

        The generic recovery field handles local hazard/collision repulsion. These
        waypoints handle cases where a safe local action requires choosing the
        correct corridor before the agent is already in danger.
        """
        scenario = str(self.task.kwargs.get("scenario", ""))
        pos = np.asarray(position, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        x = float(pos[0])

        if scenario == "pogema_warehouse":
            lane_values = np.asarray([-0.68, -0.64, 0.64, 0.68], dtype=np.float64)
            lane_mode = str(self.task.kwargs.get("pogema_lane_mode", "default"))
            if lane_mode == "paired" and self.n_agents == 4:
                lane_values = np.asarray([-0.66, -0.66, 0.66, 0.66], dtype=np.float64)
            elif lane_mode == "biased" and self.n_agents == 4:
                lane_values = np.asarray([-0.66, 0.66, 0.66, 0.66], dtype=np.float64)
            if self.n_agents != len(lane_values):
                half = max(1, self.n_agents // 2)
                lower = np.linspace(-0.68, -0.62, half, dtype=np.float64)
                upper = np.linspace(0.62, 0.68, self.n_agents - half, dtype=np.float64)
                lane_values = np.concatenate([lower, upper])
            lane_y = float(lane_values[agent_idx % len(lane_values)])
            strength_scale = float(self.task.kwargs.get("pogema_route_strength_scale", 1.0))
            if x < -0.62 and abs(float(pos[1]) - lane_y) > 0.035:
                target = np.asarray([min(x, -0.66), lane_y], dtype=np.float64)
                strength = 1.70 * strength_scale
            elif x < 0.68:
                target = np.asarray([0.72, lane_y], dtype=np.float64)
                strength = 1.45 * strength_scale
            else:
                target = goal
                strength = 0.55 * strength_scale
            return target, strength

        if scenario == "highway_merge":
            main_lane_y = float(self.task.kwargs.get("highway_main_lane_y", 0.18))
            ramp_lane_y = float(self.task.kwargs.get("highway_ramp_lane_y", -0.43))
            shoulder_y = float(self.task.kwargs.get("highway_shoulder_y", 0.275))
            merge_x = float(self.task.kwargs.get("highway_merge_x", 0.34))
            strength = float(self.task.kwargs.get("highway_route_strength", 1.05))
            lane_y = main_lane_y
            if float(pos[1]) < -0.10:
                lane_y = ramp_lane_y if x < merge_x else main_lane_y
            elif -0.28 < x < 0.24:
                # The upper shoulder avoids the merge-zone hazard that lies on
                # the nominal lane centerline in this lightweight adapter.
                lane_y = shoulder_y
            target = np.asarray([min(1.22, x + 0.42), lane_y], dtype=np.float64)
            return target, strength

        if scenario == "vmas_chargeshare" and bool(self.task.kwargs.get("chargeshare_route_recovery", False)):
            lane_span = float(self.task.kwargs.get("chargeshare_lane_span", 0.22))
            lane_values = np.linspace(-lane_span, lane_span, self.n_agents, dtype=np.float64)
            lane_y = float(lane_values[agent_idx])
            strength = float(self.task.kwargs.get("chargeshare_route_strength", 0.60))
            if x < -0.30:
                target = np.asarray([-0.24, lane_y], dtype=np.float64)
            elif x < 0.34:
                target = np.asarray([0.42, lane_y], dtype=np.float64)
            else:
                target = goal
            return target, strength

        if scenario == "vmas_multigoal":
            gate_lanes = np.linspace(-0.16, 0.16, self.n_agents, dtype=np.float64)
            exit_lanes = np.asarray([-0.18, 0.18, -0.18], dtype=np.float64)
            if self.n_agents != len(exit_lanes):
                exit_lanes = np.where(np.arange(self.n_agents) % 2 == 0, -0.18, 0.18).astype(np.float64)
            lane_y = float(gate_lanes[agent_idx])
            exit_y = float(exit_lanes[agent_idx])
            if x < -0.32:
                target = np.asarray([-0.24, lane_y], dtype=np.float64)
                strength = 0.55
            elif x < 0.28:
                target = np.asarray([0.34, lane_y], dtype=np.float64)
                strength = 0.80
            elif x < 0.58:
                target = np.asarray([0.62, exit_y], dtype=np.float64)
                strength = 0.90
            else:
                target = goal
                strength = 0.20
            return target, strength

        return goal, 0.0

    def _highway_fair_merge_assist_target(
        self,
        agent_idx: int,
        positions: np.ndarray,
        route_target: np.ndarray,
        route_strength: float,
        relative_burden: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Bias Highway recovery to pay back repeated ramp merge burden.

        Safety recovery already separates vehicles. This extra route hint only
        runs for the proposed burden-aware recovery and asks under-burdened
        main-lane vehicles to open a shoulder gap while the over-burdened ramp
        vehicle merges earlier. It is disabled by default and controlled by the
        Highway task kwargs used in repair sweeps.
        """
        if not bool(self.task.kwargs.get("highway_fair_merge_assist", False)):
            return route_target, route_strength

        pos = np.asarray(positions[agent_idx], dtype=np.float64)
        main_lane_y = float(self.task.kwargs.get("highway_main_lane_y", 0.18))
        shoulder_y = float(self.task.kwargs.get("highway_shoulder_y", 0.275))
        ramp_y_threshold = float(self.task.kwargs.get("highway_fair_merge_ramp_y_threshold", -0.10))
        assist_start_x = float(self.task.kwargs.get("highway_fair_merge_start_x", -1.15))
        assist_end_x = float(self.task.kwargs.get("highway_fair_merge_end_x", 0.34))
        assist_strength = float(self.task.kwargs.get("highway_fair_merge_strength", 1.45))
        ramp_forward = float(self.task.kwargs.get("highway_fair_merge_ramp_forward", 0.52))
        slot_forward = float(self.task.kwargs.get("highway_fair_merge_slot_forward", 0.48))
        slot_back = float(self.task.kwargs.get("highway_fair_merge_slot_back", 0.18))
        main_window = float(self.task.kwargs.get("highway_fair_merge_main_window", 0.62))

        ramp_mask = (
            (positions[:, 1] < ramp_y_threshold)
            & (positions[:, 0] >= assist_start_x)
            & (positions[:, 0] <= assist_end_x)
        )
        ramp_indices = np.where(ramp_mask)[0]
        if len(ramp_indices) == 0:
            return route_target, route_strength

        # If there are multiple ramp-like agents, prioritize the one with the
        # largest accumulated burden; break ties by proximity to the merge zone.
        scores = relative_burden[ramp_indices] + 0.05 * positions[ramp_indices, 0]
        ramp_idx = int(ramp_indices[int(np.argmax(scores))])
        ramp_pos = np.asarray(positions[ramp_idx], dtype=np.float64)
        ramp_pressure = max(0.0, float(relative_burden[ramp_idx]))
        boosted_strength = assist_strength * (1.0 + 0.20 * ramp_pressure)

        if agent_idx == ramp_idx:
            target_x = min(1.22, max(float(pos[0]) + ramp_forward, -0.22))
            target = np.asarray([target_x, main_lane_y], dtype=np.float64)
            return target, max(route_strength, boosted_strength)

        if abs(float(pos[1]) - main_lane_y) > 0.18:
            return route_target, route_strength
        if abs(float(pos[0]) - float(ramp_pos[0])) > main_window:
            return route_target, route_strength

        # Leading vehicles move forward and up to the shoulder; trailing
        # vehicles slow slightly and also move up. Both actions transfer some
        # merge burden away from the ramp vehicle without creating a collision.
        if float(pos[0]) >= float(ramp_pos[0]):
            target_x = min(1.22, float(pos[0]) + slot_forward)
        else:
            target_x = max(-1.22, float(pos[0]) - slot_back)
        target = np.asarray([target_x, shoulder_y], dtype=np.float64)
        main_strength = boosted_strength * float(self.task.kwargs.get("highway_fair_merge_main_strength_scale", 0.88))
        return target, max(route_strength, main_strength)

    def _safefairbench_recovery_actions(self, actions: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if self.task.source != "safefairbench_v0" or not self.method.use_safety_recovery:
            return [np.asarray(a, dtype=np.float32) for a in actions], {}
        if not hasattr(self.env, "positions"):
            return [np.asarray(a, dtype=np.float32) for a in actions], {}

        actions_arr = np.stack([np.clip(as_np(a, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0) for a in actions], axis=0)
        positions = np.asarray(getattr(self.env, "positions"), dtype=np.float64)
        goals = np.asarray(getattr(self.env, "goals", positions), dtype=np.float64)
        hazards = np.asarray(getattr(self.env, "hazards", np.zeros((0, 2))), dtype=np.float64)
        predicted = self._predict_safefairbench_positions(actions_arr)

        hazard_radius = float(getattr(self.env, "hazard_radius", self.task.kwargs.get("hazard_radius", 0.15)))
        collision_radius = float(getattr(self.env, "collision_radius", self.task.kwargs.get("collision_radius", 0.13)))
        margin = float(self.method.safety_recovery_margin)
        blend = float(np.clip(self.method.safety_recovery_blend, 0.0, 1.0))
        speed = float(np.clip(self.method.safety_recovery_speed, 0.0, 1.0))

        recovered = actions_arr.copy()
        intervention = np.zeros(self.n_agents, dtype=np.float64)
        scores = np.zeros(self.n_agents, dtype=np.float64)
        burden_factor = np.ones(self.n_agents, dtype=np.float64)
        burden_goal_weight = np.full(self.n_agents, 0.25, dtype=np.float64)
        hazard_escape_factor = np.ones(self.n_agents, dtype=np.float64)
        relative_burden = np.zeros(self.n_agents, dtype=np.float64)
        if self.method.use_burden_aware_recovery:
            total_burden = np.asarray(getattr(self.logger, "total_burden", np.zeros(self.n_agents)), dtype=np.float64).reshape(-1)
            if total_burden.size >= self.n_agents and float(np.sum(np.abs(total_burden[: self.n_agents]))) > 1e-8:
                q = np.maximum(np.asarray(getattr(self.logger, "q", np.ones(self.n_agents) / self.n_agents), dtype=np.float64), 1e-8)
                normalized_burden = total_burden[: self.n_agents] / q
                centered_burden = normalized_burden - float(np.mean(normalized_burden))
                scale = float(np.mean(np.abs(centered_burden))) + 1e-6
                relative_burden = np.clip(centered_burden / scale, -2.0, 2.0)
                strength = float(np.clip(self.method.burden_recovery_strength, 0.0, 2.0))
                # Under-burdened agents accept more collision-avoidance routing;
                # over-burdened agents keep more goal progress unless they need
                # to escape a hazard.
                burden_factor = np.clip(1.0 - strength * relative_burden, 0.30, 1.85)
                burden_goal_weight = np.clip(0.25 + 0.18 * strength * np.maximum(relative_burden, 0.0), 0.18, 0.65)
                hazard_escape_factor = np.clip(1.0 + 0.30 * strength * np.maximum(relative_burden, 0.0), 1.0, 1.80)

        for i in range(self.n_agents):
            goal_dir = unit_vector(goals[i] - positions[i], fallback=actions_arr[i])
            recovery_vec = np.zeros(2, dtype=np.float64)
            route_target, route_strength = self._safefairbench_route_target(i, positions[i], goals[i])
            scenario = str(self.task.kwargs.get("scenario", ""))
            if scenario == "highway_merge" and self.method.use_burden_aware_recovery:
                route_target, route_strength = self._highway_fair_merge_assist_target(
                    i,
                    positions,
                    route_target,
                    route_strength,
                    relative_burden,
                )
            if route_strength > 0.0:
                route_dir = unit_vector(route_target - positions[i], fallback=goal_dir)
                recovery_vec += route_strength * route_dir
                scores[i] += route_strength

            for point, weight in ((positions[i], 1.35), (predicted[i], 1.0)):
                if hazards.size:
                    for hazard in hazards:
                        diff = np.asarray(point - hazard, dtype=np.float64)
                        dist = float(np.linalg.norm(diff))
                        limit = hazard_radius + margin
                        if dist < limit:
                            away = unit_vector(diff, fallback=goal_dir)
                            strength = hazard_escape_factor[i] * weight * (limit - dist) / max(limit, 1e-8)
                            recovery_vec += strength * away
                            scores[i] += strength

                for j in range(self.n_agents):
                    if i == j:
                        continue
                    diff = np.asarray(point - predicted[j], dtype=np.float64)
                    dist = float(np.linalg.norm(diff))
                    limit = collision_radius + margin
                    if dist < limit:
                        away = unit_vector(diff, fallback=positions[i] - positions[j])
                        strength = burden_factor[i] * 0.75 * weight * (limit - dist) / max(limit, 1e-8)
                        recovery_vec += strength * away
                        scores[i] += strength

            if scores[i] > 0.0:
                route_blend_floor = 0.0
                if route_strength > 0.0:
                    if scenario == "pogema_warehouse":
                        route_blend_floor = float(self.task.kwargs.get("pogema_route_blend_floor", 0.98))
                    elif scenario == "highway_merge":
                        route_blend_floor = float(self.task.kwargs.get("highway_route_blend_floor", 0.97))
                    elif scenario == "vmas_multigoal":
                        route_blend_floor = 0.90
                    elif scenario == "vmas_chargeshare":
                        route_blend_floor = float(self.task.kwargs.get("chargeshare_route_blend_floor", 0.78))
                local_blend = max(blend, route_blend_floor)
                target = unit_vector(recovery_vec + burden_goal_weight[i] * goal_dir, fallback=goal_dir) * speed
                recovered[i] = np.clip((1.0 - local_blend) * recovered[i] + local_blend * target, -1.0, 1.0)
                intervention[i] = 1.0

        info: Dict[str, Any] = {
            "safety_recovery_interventions": intervention.copy(),
            "safety_recovery_rate": float(np.mean(intervention)),
            "safety_recovery_score": scores.copy(),
            "safety_recovery_delta": float(np.mean(np.linalg.norm(recovered - actions_arr, axis=1))),
            "burden_recovery_factor": burden_factor.copy(),
        }
        if np.any(intervention > 0.0):
            info["executed_actions"] = [x.astype(np.float32) for x in recovered]
        return [x.astype(np.float32) for x in recovered], info

    def reset(self, seed: Optional[int] = None):
        self.logger.reset()
        try:
            out = self.env.reset(seed=seed)
        except TypeError:
            if seed is not None and hasattr(self.env, "seed"):
                self.env.seed(seed)
            out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self.last_obs = obs
        return normalize_obs_container(obs, self.n_agents), info

    def step(self, actions: Sequence[np.ndarray]):
        actions, recovery_info = self._safefairbench_recovery_actions(actions)
        packed = pack_actions_for_env(actions, self.action_space, env=self.env)
        out = self.env.step(packed)
        if len(out) == 6:
            next_obs, reward, costs, terminated, truncated, info = out
            done = normalize_done(terminated, truncated)
            if isinstance(info, dict):
                info = dict(info)
                info["costs"] = costs
            else:
                info = {"raw_info": info, "costs": costs}
        elif len(out) == 5:
            next_obs, reward, terminated, truncated, info = out
            done = normalize_done(terminated, truncated)
        elif len(out) == 4:
            next_obs, reward, done_raw, info = out
            done = normalize_done(done_raw, None)
        else:
            raise RuntimeError(f"Unexpected env.step output length: {len(out)}")
        if not isinstance(info, dict):
            info = {"raw_info": info}
        if recovery_info:
            info.update(recovery_info)
        fair_info = self.logger.step(self.last_obs, actions, next_obs, info)
        info.update(fair_info)
        info["safety_cost"] = scalar_cost_from_info(info)
        info["team_reward"] = scalar_reward(reward)
        self.last_obs = next_obs
        return normalize_obs_container(next_obs, self.n_agents), scalar_reward(reward), done, info

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

def make_raw_env(task: TaskSpec, seed: int = 0):
    if task.source == "builtin":
        return BottleneckFairEnv(
            n_agents=task.kwargs.get("n_agents", task.n_agents),
            n_hazards=task.kwargs.get("n_hazards", 2),
            episode_length=task.episode_length,
            goal_radius=task.kwargs.get("goal_radius", 0.12),
            seed=seed,
        )
    if task.source == "safefairbench_v0":
        return SafeFairBenchEnv(task, seed=seed)
    if task.source in {"safety_gymnasium", "safety_gymnasium_chargeshare"}:
        try:
            with prepend_sys_path([preferred_source_path("safety_gymnasium")]):
                import safety_gymnasium
        except Exception as exc:
            raise ImportError(
                f"safety_gymnasium import failed: {type(exc).__name__}: {exc}. "
                f"Expected vendored checkout at {preferred_source_path('safety_gymnasium')}."
            ) from exc
        env = safety_gymnasium.make(task.env_id)
        return SafetyGymTaskAdapter(env, task)
    if task.source == "safe_mamujoco":
        scenario = task.kwargs.get("sg_scenario", task.kwargs.get("scenario", task.env_id))
        if isinstance(scenario, str) and scenario.endswith("-v2"):
            scenario = scenario[:-3]
        agent_conf = task.kwargs.get("agent_conf", "2x4")
        agent_obsk = task.kwargs.get("agent_obsk", 1)
        try:
            with prepend_sys_path([preferred_source_path("safety_gymnasium")]):
                import safety_gymnasium
            env = safety_gymnasium.make_ma(
                scenario,
                agent_conf,
                agent_obsk=agent_obsk,
            )
            return env
        except Exception as sg_exc:
            print(
                f"[safe_mamujoco] Safety-Gymnasium make_ma fallback failed for "
                f"{scenario}/{agent_conf}: {type(sg_exc).__name__}: {sg_exc}"
            )
        try:
            from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import MujocoMulti
        except Exception:
            macpo_path = EXTERNAL_ROOT / "macpo" / "MACPO"
            with prepend_sys_path([macpo_path]):
                try:
                    from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import MujocoMulti
                except Exception:
                    safe_mamujoco_path = EXTERNAL_ROOT / "safe_mamujoco"
                    with prepend_sys_path([safe_mamujoco_path, safe_mamujoco_path / "safety_multi_agent_mujoco"]):
                        try:
                            from safety_ma_mujoco.safety_multiagent_mujoco.mujoco_multi import MujocoMulti
                        except Exception as exc2:
                            raise ImportError(
                                f"Could not import official Safe-MAMuJoCo wrapper from MACPO or Safe-Multi-Agent-Mujoco: "
                                f"{type(exc2).__name__}: {exc2}"
                            ) from exc2
        env_args = {
            "scenario": task.kwargs.get("scenario", task.env_id),
            "agent_conf": agent_conf,
            "agent_obsk": agent_obsk,
            "episode_limit": task.episode_length,
        }
        env = MujocoMulti(env_args=env_args)
        if hasattr(env, "seed"):
            env.seed(seed)
        return env
    if task.source == "safe_marobosuite":
        raise NotImplementedError("Safe Multi-Agent Robosuite is optional. Install the official repo and add its raw env constructor here.")
    raise ValueError(f"Unknown task source: {task.source}")

def make_fair_env(task: TaskSpec, method: MethodSpec, seed: int = 0) -> FairMultiAgentWrapper:
    raw = make_raw_env(task, seed=seed)
    if task.source == "safefairbench_v0" and bool(method.use_safety_recovery) and hasattr(raw, "use_recovery_projection"):
        raw.use_recovery_projection = True
    return FairMultiAgentWrapper(raw, task, method)

def infer_dims_from_env(env: FairMultiAgentWrapper, seed: int = 0) -> Tuple[List[int], List[int]]:
    obs, _ = env.reset(seed=seed)
    obs_dims = [int(np.asarray(o).reshape(-1).shape[0]) for o in obs]
    action_space = getattr(env, "action_space", None)
    act_dims = []
    if isinstance(action_space, (list, tuple)):
        for sp in action_space[:env.n_agents]:
            act_dims.append(int(np.prod(getattr(sp, "shape", (1,)))))
    elif isinstance(action_space, dict):
        for key in sorted(action_space.keys(), key=lambda x: str(x))[:env.n_agents]:
            act_dims.append(int(np.prod(getattr(action_space[key], "shape", (1,)))))
    elif callable(action_space) and hasattr(env.env, "possible_agents"):
        for key in list(getattr(env.env, "possible_agents"))[:env.n_agents]:
            sp = action_space(key)
            act_dims.append(int(np.prod(getattr(sp, "shape", (1,)))))
    elif hasattr(action_space, "shape"):
        shape = tuple(action_space.shape)
        if len(shape) >= 2 and shape[0] == env.n_agents:
            act_dims = [int(np.prod(shape[1:])) for _ in range(env.n_agents)]
        elif len(shape) == 1:
            split = int(np.prod(shape)) // env.n_agents
            act_dims = [split for _ in range(env.n_agents)]
    if not act_dims:
        act_dims = [2 for _ in range(env.n_agents)]
    return obs_dims, act_dims

smoke_env = make_fair_env(TASKS["smoke_bottleneck"], METHODS["safe_fair_macpo"], seed=0)
obs0, info0 = smoke_env.reset(seed=0)
obs_dims0, act_dims0 = infer_dims_from_env(smoke_env, seed=0)
print("Smoke obs dims:", obs_dims0, "act dims:", act_dims0)
smoke_env.close()


# =============================================================================
# 6. Networks and rollout buffer
# =============================================================================
def mlp(sizes: Sequence[int], activation: Callable[[], nn.Module] = nn.Tanh, output_activation: Optional[Callable[[], nn.Module]] = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)

class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (128, 128), log_std_init: float = -0.5):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes, act_dim])
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init, dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.rsample()
        logp = dist.log_prob(action).sum(dim=-1)
        return action, logp

    def log_prob_entropy(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        logp = dist.log_prob(action).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent

class CentralCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.net = mlp([input_dim, *hidden_sizes, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class MultiAgentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dims: Sequence[int],
        act_dims: Sequence[int],
        hidden_sizes: Tuple[int, int] = (128, 128),
        decentralized_critic: bool = False,
    ):
        super().__init__()
        self.n_agents = len(obs_dims)
        self.obs_dims = list(obs_dims)
        self.act_dims = list(act_dims)
        self.decentralized_critic = bool(decentralized_critic)
        self.actors = nn.ModuleList([GaussianActor(obs_dims[i], act_dims[i], hidden_sizes=hidden_sizes) for i in range(self.n_agents)])
        if self.decentralized_critic:
            self.v_reward = nn.ModuleList([CentralCritic(obs_dims[i], hidden_sizes=hidden_sizes) for i in range(self.n_agents)])
            self.v_safety = nn.ModuleList([CentralCritic(obs_dims[i], hidden_sizes=hidden_sizes) for i in range(self.n_agents)])
            self.v_fairness = nn.ModuleList([CentralCritic(obs_dims[i], hidden_sizes=hidden_sizes) for i in range(self.n_agents)])
        else:
            joint_dim = int(sum(obs_dims))
            self.v_reward = CentralCritic(joint_dim, hidden_sizes=hidden_sizes)
            self.v_safety = CentralCritic(joint_dim, hidden_sizes=hidden_sizes)
            self.v_fairness = CentralCritic(joint_dim, hidden_sizes=hidden_sizes)

    def joint_obs_tensor(self, obs_list: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
        arr = np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs_list], axis=0)
        return torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

    def _critic_mean_from_obs_list(self, critics: nn.ModuleList, obs_list: Sequence[np.ndarray], device: torch.device) -> float:
        vals = []
        for i, critic in enumerate(critics):
            obs_i = torch.as_tensor(np.asarray(obs_list[i], dtype=np.float32).reshape(1, -1), device=device)
            vals.append(critic(obs_i))
        return float(torch.stack(vals, dim=0).mean().cpu().item())

    def _critic_mean_from_batch(self, critics: nn.ModuleList, obs_agent_tensors: List[torch.Tensor]) -> torch.Tensor:
        vals = [critic(obs_agent_tensors[i]) for i, critic in enumerate(critics)]
        return torch.stack(vals, dim=0).mean(dim=0)

    @torch.no_grad()
    def logp_sum_for_actions(self, obs_list: Sequence[np.ndarray], actions: Sequence[np.ndarray], device: torch.device) -> float:
        logps = []
        for i, actor in enumerate(self.actors):
            obs_i = torch.as_tensor(np.asarray(obs_list[i], dtype=np.float32).reshape(1, -1), device=device)
            act_i = torch.as_tensor(np.asarray(actions[i], dtype=np.float32).reshape(1, -1), device=device)
            dist = actor.distribution(obs_i)
            logps.append(dist.log_prob(act_i).sum(dim=-1).squeeze(0).cpu().item())
        return float(np.sum(logps))

    @torch.no_grad()
    def act(self, obs_list: Sequence[np.ndarray], device: torch.device, deterministic: bool = False) -> Dict[str, Any]:
        actions = []
        logps = []
        for i, actor in enumerate(self.actors):
            obs_i = torch.as_tensor(np.asarray(obs_list[i], dtype=np.float32).reshape(1, -1), device=device)
            action_i, logp_i = actor(obs_i, deterministic=deterministic)
            actions.append(action_i.squeeze(0).cpu().numpy())
            logps.append(logp_i.squeeze(0).cpu().item())
        if self.decentralized_critic:
            value_reward = self._critic_mean_from_obs_list(self.v_reward, obs_list, device)
            value_safety = self._critic_mean_from_obs_list(self.v_safety, obs_list, device)
            value_fairness = self._critic_mean_from_obs_list(self.v_fairness, obs_list, device)
        else:
            joint = self.joint_obs_tensor(obs_list, device)
            value_reward = float(self.v_reward(joint).cpu().item())
            value_safety = float(self.v_safety(joint).cpu().item())
            value_fairness = float(self.v_fairness(joint).cpu().item())
        return {
            "actions": actions,
            "logp_sum": float(np.sum(logps)),
            "value_reward": value_reward,
            "value_safety": value_safety,
            "value_fairness": value_fairness,
        }

    def evaluate_batch(self, obs_agent_tensors: List[torch.Tensor], action_agent_tensors: List[torch.Tensor], joint_obs_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        logps, ents = [], []
        for i, actor in enumerate(self.actors):
            lp, ent = actor.log_prob_entropy(obs_agent_tensors[i], action_agent_tensors[i])
            logps.append(lp)
            ents.append(ent)
        if self.decentralized_critic:
            value_reward = self._critic_mean_from_batch(self.v_reward, obs_agent_tensors)
            value_safety = self._critic_mean_from_batch(self.v_safety, obs_agent_tensors)
            value_fairness = self._critic_mean_from_batch(self.v_fairness, obs_agent_tensors)
        else:
            value_reward = self.v_reward(joint_obs_tensor)
            value_safety = self.v_safety(joint_obs_tensor)
            value_fairness = self.v_fairness(joint_obs_tensor)
        return {
            "logp_sum": torch.stack(logps, dim=0).sum(dim=0),
            "entropy": torch.stack(ents, dim=0).sum(dim=0),
            "value_reward": value_reward,
            "value_safety": value_safety,
            "value_fairness": value_fairness,
        }

@dataclass
class RolloutBatch:
    obs_agents: List[np.ndarray]
    actions_agents: List[np.ndarray]
    joint_obs: np.ndarray
    old_logp_sum: np.ndarray
    rewards: np.ndarray
    safety_costs: np.ndarray
    fairness_costs: np.ndarray
    dones: np.ndarray
    values_reward: np.ndarray
    values_safety: np.ndarray
    values_fairness: np.ndarray
    last_values: Tuple[float, float, float]
    infos: List[Dict[str, Any]]

    def size(self) -> int:
        return len(self.rewards)

def compute_gae(x: np.ndarray, values: np.ndarray, last_value: float, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(x)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_value = last_value
        else:
            next_nonterminal = 1.0 - float(dones[t])
            next_value = values[t + 1]
        delta = x[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)

def normalize_advantage(adv: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    adv = np.asarray(adv, dtype=np.float32)
    return (adv - adv.mean()) / (adv.std() + eps)


# =============================================================================
# 7. Unified safe-fair trainer
# =============================================================================
class SafeFairTrainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.task = cfg.task
        self.method = cfg.method
        self.device = torch.device(cfg.device)
        set_global_seed(cfg.seed)
        self.env = make_fair_env(self.task, self.method, seed=cfg.seed)
        self.obs_dims, self.act_dims = infer_dims_from_env(self.env, seed=cfg.seed)
        self.ac = MultiAgentActorCritic(
            self.obs_dims,
            self.act_dims,
            hidden_sizes=cfg.hidden_sizes,
            decentralized_critic=self.method.decentralized_critic,
        ).to(self.device)
        if cfg.init_checkpoint:
            ckpt = Path(cfg.init_checkpoint).expanduser().resolve()
            if not ckpt.exists():
                raise FileNotFoundError(f"SAFE_FAIR_CFG_INIT_CHECKPOINT not found: {ckpt}")
            self.ac.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.lambda_safety = float(cfg.init_lambda_safety)
        self.lambda_fairness = float(cfg.init_lambda_fairness)
        self.global_step = 0
        self.update_idx = 0
        self.start_time = time.time()
        self.history: List[Dict[str, Any]] = []
        self.run_id = self.make_run_id()
        self.run_dir = LOG_ROOT / cfg.run_group / self.task.name / self.method.name / f"seed_{cfg.seed}_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_json(cfg.to_dict(), self.run_dir / "config.json")

    def make_run_id(self) -> str:
        obj = {
            "task": self.task.name,
            "method": self.method.name,
            "seed": self.cfg.seed,
            "quick": self.cfg.quick,
            "tag": self.cfg.run_tag,
            "time": now_stamp(),
        }
        return stable_hash(obj)

    def safety_target_per_step(self) -> float:
        return float(self.task.safety_budget_per_step * self.cfg.safety_target_scale)

    def fairness_target_per_step(self) -> float:
        init_scale = max(1.0, float(self.cfg.fairness_budget_scale_init))
        base_budget = float(self.task.fairness_budget_per_step)
        post_warmup_updates = max(0, self.update_idx - self.cfg.fairness_warmup_updates)
        if self.cfg.fairness_budget_ramp_updates <= 0:
            scale = 1.0 if self.update_idx >= self.cfg.fairness_warmup_updates else init_scale
        else:
            progress = min(1.0, post_warmup_updates / float(self.cfg.fairness_budget_ramp_updates))
            scale = init_scale + (1.0 - init_scale) * progress
        return float(base_budget * scale)

    def fairness_active_for_batch(self, batch: RolloutBatch) -> bool:
        if not self.method.use_fairness_constraint:
            return False
        if self.update_idx < self.cfg.fairness_warmup_updates:
            return False
        safety_error = float(np.mean(batch.safety_costs) - self.safety_target_per_step())
        return (not self.method.safety_first) or (safety_error <= -self.cfg.fairness_safety_margin)

    def collect_rollout(self) -> RolloutBatch:
        cfg = self.cfg
        obs, _ = self.env.reset(seed=cfg.seed + self.global_step)
        n = self.task.n_agents
        obs_agents: List[List[np.ndarray]] = [[] for _ in range(n)]
        actions_agents: List[List[np.ndarray]] = [[] for _ in range(n)]
        joint_obs_list: List[np.ndarray] = []
        logps, rewards, safety_costs, fairness_costs, dones = [], [], [], [], []
        values_r, values_s, values_f = [], [], []
        infos: List[Dict[str, Any]] = []
        for _ in range(cfg.rollout_steps):
            act_out = self.ac.act(obs, device=self.device, deterministic=False)
            actions = act_out["actions"]
            next_obs, reward, done, info = self.env.step(actions)
            executed_actions = info.get("executed_actions", actions) if isinstance(info, dict) else actions
            executed_actions = normalize_obs_container(executed_actions, n)
            for i in range(n):
                obs_agents[i].append(np.asarray(obs[i], dtype=np.float32).reshape(-1))
                actions_agents[i].append(np.asarray(executed_actions[i], dtype=np.float32).reshape(-1))
            joint_obs_list.append(np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs], axis=0))
            logps.append(self.ac.logp_sum_for_actions(obs, executed_actions, device=self.device))
            rewards.append(float(reward))
            safety_costs.append(float(info.get("safety_cost", 0.0)))
            fairness_costs.append(float(info.get("fairness_cost", 0.0)))
            dones.append(float(done))
            values_r.append(act_out["value_reward"])
            values_s.append(act_out["value_safety"])
            values_f.append(act_out["value_fairness"])
            infos.append(info)
            self.global_step += 1
            obs = next_obs
            if done:
                obs, _ = self.env.reset(seed=cfg.seed + self.global_step)
        last = self.ac.act(obs, device=self.device, deterministic=True)
        return RolloutBatch(
            obs_agents=[np.stack(x, axis=0).astype(np.float32) for x in obs_agents],
            actions_agents=[np.stack(x, axis=0).astype(np.float32) for x in actions_agents],
            joint_obs=np.stack(joint_obs_list, axis=0).astype(np.float32),
            old_logp_sum=np.asarray(logps, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            safety_costs=np.asarray(safety_costs, dtype=np.float32),
            fairness_costs=np.asarray(fairness_costs, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            values_reward=np.asarray(values_r, dtype=np.float32),
            values_safety=np.asarray(values_s, dtype=np.float32),
            values_fairness=np.asarray(values_f, dtype=np.float32),
            last_values=(last["value_reward"], last["value_safety"], last["value_fairness"]),
            infos=infos,
        )

    def _build_effective_advantage(self, adv_r: np.ndarray, adv_s: np.ndarray, adv_f: np.ndarray, batch: RolloutBatch) -> np.ndarray:
        method = self.method
        adv = adv_r.copy()
        if method.safety_reward_coeff > 0:
            adv = adv - method.safety_reward_coeff * adv_s
        if method.fairness_reward_coeff > 0:
            adv = adv - method.fairness_reward_coeff * adv_f
        if method.use_safety_constraint:
            adv = adv - self.lambda_safety * adv_s
        if method.use_fairness_constraint:
            fairness_active = self.fairness_active_for_batch(batch)
            if fairness_active:
                adv = adv - self.lambda_fairness * adv_f
        return normalize_advantage(adv)

    def update(self, batch: RolloutBatch) -> Dict[str, Any]:
        cfg = self.cfg
        T = batch.size()
        adv_r, ret_r = compute_gae(batch.rewards, batch.values_reward, batch.last_values[0], batch.dones, cfg.gamma, cfg.gae_lambda)
        adv_s, ret_s = compute_gae(batch.safety_costs, batch.values_safety, batch.last_values[1], batch.dones, cfg.gamma, cfg.gae_lambda)
        fairness_enabled = self.method.uses_fairness_objective
        adv_f, ret_f = compute_gae(batch.fairness_costs, batch.values_fairness, batch.last_values[2], batch.dones, cfg.gamma, cfg.gae_lambda)
        effective_adv = self._build_effective_advantage(adv_r, adv_s, adv_f, batch)

        obs_t = [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch.obs_agents]
        act_t = [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch.actions_agents]
        joint_t = torch.as_tensor(batch.joint_obs, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(batch.old_logp_sum, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(effective_adv, dtype=torch.float32, device=self.device)
        ret_r_t = torch.as_tensor(ret_r, dtype=torch.float32, device=self.device)
        ret_s_t = torch.as_tensor(ret_s, dtype=torch.float32, device=self.device)
        ret_f_t = torch.as_tensor(ret_f, dtype=torch.float32, device=self.device)

        idxs = np.arange(T)
        losses = []
        approx_kls = []
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                out = self.ac.evaluate_batch([x[mb_t] for x in obs_t], [x[mb_t] for x in act_t], joint_t[mb_t])
                ratio = torch.exp(out["logp_sum"] - old_logp_t[mb_t])
                unclipped = ratio * adv_t[mb_t]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_t[mb_t]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (
                    F.mse_loss(out["value_reward"], ret_r_t[mb_t])
                    + F.mse_loss(out["value_safety"], ret_s_t[mb_t])
                )
                if fairness_enabled:
                    value_loss = value_loss + F.mse_loss(out["value_fairness"], ret_f_t[mb_t])
                entropy = out["entropy"].mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = (old_logp_t[mb_t] - out["logp_sum"]).mean().abs().item()
                losses.append(float(loss.detach().cpu().item()))
                approx_kls.append(approx_kl)

        mean_safety = float(np.mean(batch.safety_costs))
        mean_fairness = float(np.mean(batch.fairness_costs))
        safety_target = self.safety_target_per_step()
        fairness_target = self.fairness_target_per_step()
        safety_error = mean_safety - safety_target
        fairness_error = mean_fairness - fairness_target
        fairness_active = self.fairness_active_for_batch(batch)
        safety_dual_lr = cfg.safety_dual_lr if cfg.safety_dual_lr is not None else cfg.dual_lr
        fairness_dual_lr = cfg.fairness_dual_lr if cfg.fairness_dual_lr is not None else cfg.dual_lr

        if self.method.use_safety_constraint:
            self.lambda_safety = min(cfg.max_lambda_safety, max(0.0, self.lambda_safety + safety_dual_lr * safety_error))
        if self.method.use_fairness_constraint:
            if not fairness_active:
                self.lambda_fairness = max(0.0, self.lambda_fairness * 0.995)
            else:
                self.lambda_fairness = min(cfg.max_lambda_fairness, max(0.0, self.lambda_fairness + fairness_dual_lr * fairness_error))

        jain_vals = [float(x.get("jain_burden", np.nan)) for x in batch.infos if isinstance(x, dict)]
        gini_vals = [float(x.get("gini_burden", np.nan)) for x in batch.infos if isinstance(x, dict)]
        envy_vals = [float(x.get("envy_gap", np.nan)) for x in batch.infos if isinstance(x, dict)]
        debt_vals = []
        recovery_rates = []
        recovery_deltas = []
        for x in batch.infos:
            if isinstance(x, dict) and "fairness_debt" in x:
                debt_vals.append(float(np.mean(np.abs(as_np(x["fairness_debt"], dtype=np.float64)))))
            if isinstance(x, dict) and "safety_recovery_rate" in x:
                recovery_rates.append(float(x["safety_recovery_rate"]))
            if isinstance(x, dict) and "safety_recovery_delta" in x:
                recovery_deltas.append(float(x["safety_recovery_delta"]))
        record = {
            "global_step": self.global_step,
            "loss": float(np.mean(losses)) if losses else np.nan,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else np.nan,
            "rollout_return_mean": float(np.mean(batch.rewards)),
            "rollout_safety_cost_mean": mean_safety,
            "rollout_fairness_cost_mean": mean_fairness,
            "safety_target_per_step": safety_target,
            "fairness_target_per_step": fairness_target,
            "fairness_active": float(fairness_active),
            "lambda_safety": self.lambda_safety,
            "lambda_fairness": self.lambda_fairness,
            "jain_burden": float(np.nanmean(jain_vals)) if jain_vals else np.nan,
            "gini_burden": float(np.nanmean(gini_vals)) if gini_vals else np.nan,
            "envy_gap": float(np.nanmean(envy_vals)) if envy_vals else np.nan,
            "debt_abs_mean": float(np.mean(debt_vals)) if debt_vals else np.nan,
            "safety_recovery_rate": float(np.mean(recovery_rates)) if recovery_rates else 0.0,
            "safety_recovery_delta": float(np.mean(recovery_deltas)) if recovery_deltas else 0.0,
            "task": self.task.name,
            "method": self.method.name,
            "seed": self.cfg.seed,
            "run_group": self.cfg.run_group,
            "wallclock_sec": float(time.time() - self.start_time),
        }
        self.history.append(record)
        self.update_idx += 1
        return record

    @torch.no_grad()
    def evaluate(self, episodes: Optional[int] = None, deterministic: bool = True, save_path: Optional[Path] = None) -> Dict[str, Any]:
        episodes = episodes or self.cfg.eval_episodes
        returns, safety_costs, fairness_costs, successes = [], [], [], []
        jains, ginis, envys, debt_aucs = [], [], [], []
        per_agent_burdens = []
        per_agent_burden_components = []
        recovery_rates, recovery_deltas = [], []
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.cfg.seed + 10_000 + ep)
            ep_ret, ep_safety, ep_fair = 0.0, 0.0, 0.0
            done = False
            steps = 0
            last_info: Dict[str, Any] = {}
            while not done and steps < self.task.episode_length:
                out = self.ac.act(obs, device=self.device, deterministic=deterministic)
                obs, reward, done, info = self.env.step(out["actions"])
                ep_ret += float(reward)
                ep_safety += float(info.get("safety_cost", 0.0))
                ep_fair += float(info.get("fairness_cost", 0.0))
                if isinstance(info, dict):
                    recovery_rates.append(float(info.get("safety_recovery_rate", 0.0)))
                    recovery_deltas.append(float(info.get("safety_recovery_delta", 0.0)))
                last_info = info
                steps += 1
            summary = self.env.logger.summary()
            returns.append(ep_ret)
            safety_costs.append(ep_safety)
            fairness_costs.append(ep_fair)
            successes.append(float(last_info.get("success", 0.0)) if isinstance(last_info, dict) else 0.0)
            jains.append(summary["jain_burden"])
            ginis.append(summary["gini_burden"])
            envys.append(summary["envy_gap_final"])
            debt_aucs.append(summary["debt_auc"])
            per_agent_burdens.append(self.env.logger.total_burden.copy())
            feature_arr = np.asarray(self.env.logger.feature_history, dtype=np.float64)
            if feature_arr.ndim == 3 and feature_arr.shape[1] == self.task.n_agents:
                per_agent_burden_components.append(feature_arr.sum(axis=0))

        per_agent_burdens = np.asarray(per_agent_burdens, dtype=np.float64)
        per_agent_burden_components = np.asarray(per_agent_burden_components, dtype=np.float64)
        result = {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "safety_cost_mean": float(np.mean(safety_costs)),
            "safety_cost_std": float(np.std(safety_costs)),
            "safety_cost_p50": float(np.percentile(safety_costs, 50)),
            "safety_cost_p90": float(np.percentile(safety_costs, 90)),
            "safety_cost_max": float(np.max(safety_costs)),
            "fairness_cost_mean": float(np.mean(fairness_costs)),
            "fairness_cost_std": float(np.std(fairness_costs)),
            "success_rate": float(np.mean(successes)),
            "jain_burden_mean": float(np.mean(jains)),
            "gini_burden_mean": float(np.mean(ginis)),
            "envy_gap_mean": float(np.mean(envys)),
            "debt_auc_mean": float(np.mean(debt_aucs)),
            "violation_rate": float(np.mean(np.asarray(safety_costs) > self.task.safety_budget_per_step * self.task.episode_length)),
            "hard_violation_rate": float(np.mean(np.asarray(safety_costs) > self.task.safety_budget_per_step * self.task.episode_length)),
            "per_agent_burden_mean": per_agent_burdens.mean(axis=0).tolist() if per_agent_burdens.size else [],
            "per_agent_burden_std": per_agent_burdens.std(axis=0).tolist() if per_agent_burdens.size else [],
            "burden_component_names": ["wait", "hazard", "yield", "detour", "resource"],
            "per_agent_burden_components_mean": per_agent_burden_components.mean(axis=0).tolist() if per_agent_burden_components.size else [],
            "per_agent_burden_components_std": per_agent_burden_components.std(axis=0).tolist() if per_agent_burden_components.size else [],
            "safety_recovery_rate_mean": float(np.mean(recovery_rates)) if recovery_rates else 0.0,
            "safety_recovery_delta_mean": float(np.mean(recovery_deltas)) if recovery_deltas else 0.0,
            "global_step": self.global_step,
            "task": self.task.name,
            "method": self.method.name,
            "seed": self.cfg.seed,
            "run_group": self.cfg.run_group,
            "run_tag": self.cfg.run_tag,
            "deterministic": bool(deterministic),
            "train_wallclock_sec": float(time.time() - self.start_time),
        }
        save_json(result, save_path or (self.run_dir / "eval_summary.json"))
        return result

    def train(self) -> Dict[str, Any]:
        cfg = self.cfg
        n_updates = max(1, int(math.ceil(cfg.total_steps / cfg.rollout_steps)))
        pbar = tqdm(range(n_updates), desc=f"{self.task.name}/{self.method.name}/seed{cfg.seed}")
        for _ in pbar:
            batch = self.collect_rollout()
            rec = self.update(batch)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"R": f"{rec['rollout_return_mean']:.3f}", "Cs": f"{rec['rollout_safety_cost_mean']:.3f}", "Cf": f"{rec['rollout_fairness_cost_mean']:.3f}", "ls": f"{self.lambda_safety:.2f}", "lf": f"{self.lambda_fairness:.2f}"})
            pd.DataFrame(self.history).to_csv(self.run_dir / "train_history.csv", index=False)
        torch.save(self.ac.state_dict(), self.run_dir / "model.pt")
        result = self.evaluate(save_path=self.run_dir / "eval_summary.json") if RUN_EVALUATION else {}
        if RUN_EVALUATION:
            self.evaluate(deterministic=False, save_path=self.run_dir / "eval_summary_stochastic.json")
        self.close()
        return result

    def close(self):
        self.env.close()

def run_single_experiment(cfg: RunConfig) -> Dict[str, Any]:
    trainer = SafeFairTrainer(cfg)
    if RUN_TRAINING:
        return trainer.train()
    else:
        print("RUN_TRAINING=False: initialized trainer and wrote config only.")
        print("Set RUN_TRAINING=True to run:", trainer.run_dir)
        trainer.close()
        return {"task": cfg.task.name, "method": cfg.method.name, "seed": cfg.seed, "status": "not-run", "run_dir": str(trainer.run_dir)}


# =============================================================================
# 8. Experiment registry builders
# =============================================================================
MAIN_METHOD_KEYS = ["ippo", "mappo", "macpo", "mappo_lagrangian", "dfrl_soto_ggf", "reward_shaped_mappo_fair", "reward_shaped_macpo_fair", "safe_fair_macpo"]
ABLATION_METHOD_KEYS = ["safe_fair_macpo", "safe_fair_no_fairness", "safe_fair_no_safety", "safe_fair_no_debt", "safe_fair_no_envy", "safe_fair_no_recovery", "safe_fair_local_k1", "safe_fair_per_agent_critic"]
FULL_TASK_KEYS = ["smamujoco_halfcheetah_2x3_fair", "smamujoco_ant_2x4_fair", "smamujoco_manyagent_ant_fair", "safetygym_point_multigoal1_fair", "safetygym_chargeshare_fair"]
QUICK_TASK_KEYS = ["smoke_bottleneck"]
WORKLOAD_IMBALANCE_TASK_KEYS = [
    "smamujoco_halfcheetah_2x3_fair",
    "smamujoco_halfcheetah_3x2_workload",
    "smamujoco_halfcheetah_6x1_workload",
    "smamujoco_ant_2x4_fair",
    "smamujoco_ant_2x4d_workload",
    "smamujoco_ant_4x2_workload",
    "smamujoco_ant_8x1_workload",
    "smamujoco_manyagent_ant_fair",
    "smamujoco_walker2d_2x3_workload",
]
WORKLOAD_IMBALANCE_METHOD_KEYS = ["ippo", "macpo", "mappo_lagrangian", "safe_fair_macpo"]
SAFEFAIRBENCH_TASK_KEYS = [
    "safefairbench_vmas_multigoal",
    "safefairbench_vmas_chargeshare",
    "safefairbench_pogema_warehouse",
    "safefairbench_highway_merge",
]
SAFEFAIRBENCH_METHOD_KEYS = [
    "mappo",
    "macpo",
    "mappo_lagrangian",
    "macpo_recovery",
    "mappo_lagrangian_recovery",
    "dfrl_soto_ggf",
    "reward_shaped_mappo_fair",
    "reward_shaped_macpo_fair",
    "safe_fair_macpo",
]
SAFEFAIRBENCH_ABLATION_METHOD_KEYS = [
    "safe_fair_macpo",
    "safe_fair_no_fairness",
    "safe_fair_no_safety",
    "safe_fair_no_recovery",
    "safe_fair_no_debt",
    "safe_fair_no_hazard_burden",
    "safe_fair_no_contested_access_burden",
    "safe_fair_no_yield_burden",
    "safe_fair_jain_only",
    "safe_fair_envy_only",
    "safe_fair_scalar_critic",
    "safe_fair_per_agent_critic",
]

def make_run_config(task_key: str, method_key: str, seed: int, run_group: str, quick: bool = False, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
    task = TASKS[task_key]
    method = METHODS[method_key]
    cfg = RunConfig(
        task=task,
        method=method,
        seed=seed,
        total_steps=12_000 if quick else 500_000,
        rollout_steps=512 if quick else 4096,
        ppo_epochs=3 if quick else 5,
        minibatch_size=256 if quick else 1024,
        eval_episodes=3 if quick else 10,
        hidden_sizes=(64, 64) if quick else (128, 128),
        run_group=run_group,
        quick=quick,
        device=str(DEVICE),
    )
    env_overrides = {
        "total_steps": env_int("SAFE_FAIR_CFG_TOTAL_STEPS"),
        "rollout_steps": env_int("SAFE_FAIR_CFG_ROLLOUT_STEPS"),
        "ppo_epochs": env_int("SAFE_FAIR_CFG_PPO_EPOCHS"),
        "minibatch_size": env_int("SAFE_FAIR_CFG_MINIBATCH_SIZE"),
        "eval_episodes": env_int("SAFE_FAIR_CFG_EVAL_EPISODES"),
        "entropy_coef": env_float("SAFE_FAIR_CFG_ENTROPY_COEF"),
        "dual_lr": env_float("SAFE_FAIR_CFG_DUAL_LR"),
        "init_lambda_safety": env_float("SAFE_FAIR_CFG_INIT_LAMBDA_SAFETY"),
        "init_lambda_fairness": env_float("SAFE_FAIR_CFG_INIT_LAMBDA_FAIRNESS"),
        "max_lambda_safety": env_float("SAFE_FAIR_CFG_MAX_LAMBDA_SAFETY"),
        "max_lambda_fairness": env_float("SAFE_FAIR_CFG_MAX_LAMBDA_FAIRNESS"),
        "safety_dual_lr": env_float("SAFE_FAIR_CFG_SAFETY_DUAL_LR"),
        "fairness_dual_lr": env_float("SAFE_FAIR_CFG_FAIRNESS_DUAL_LR"),
        "safety_target_scale": env_float("SAFE_FAIR_CFG_SAFETY_TARGET_SCALE"),
        "fairness_warmup_updates": env_int("SAFE_FAIR_CFG_FAIRNESS_WARMUP_UPDATES"),
        "fairness_safety_margin": env_float("SAFE_FAIR_CFG_FAIRNESS_SAFETY_MARGIN"),
        "fairness_budget_scale_init": env_float("SAFE_FAIR_CFG_FAIRNESS_BUDGET_SCALE_INIT"),
        "fairness_budget_ramp_updates": env_int("SAFE_FAIR_CFG_FAIRNESS_BUDGET_RAMP_UPDATES"),
        "init_checkpoint": env_str("SAFE_FAIR_CFG_INIT_CHECKPOINT"),
    }
    run_tag = env_str("SAFE_FAIR_CFG_RUN_TAG")
    if run_tag:
        env_overrides["run_tag"] = run_tag
    for k, v in env_overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    method_updates: Dict[str, Any] = {}
    method_recovery = env_optional_flag("SAFE_FAIR_METHOD_USE_SAFETY_RECOVERY")
    if method_recovery is not None:
        method_updates["use_safety_recovery"] = bool(method_recovery)
    method_recovery_margin = env_float("SAFE_FAIR_METHOD_SAFETY_RECOVERY_MARGIN")
    if method_recovery_margin is not None:
        method_updates["safety_recovery_margin"] = float(method_recovery_margin)
    method_recovery_blend = env_float("SAFE_FAIR_METHOD_SAFETY_RECOVERY_BLEND")
    if method_recovery_blend is not None:
        method_updates["safety_recovery_blend"] = float(method_recovery_blend)
    method_recovery_speed = env_float("SAFE_FAIR_METHOD_SAFETY_RECOVERY_SPEED")
    if method_recovery_speed is not None:
        method_updates["safety_recovery_speed"] = float(method_recovery_speed)
    method_burden_recovery = env_optional_flag("SAFE_FAIR_METHOD_USE_BURDEN_AWARE_RECOVERY")
    if method_burden_recovery is not None:
        method_updates["use_burden_aware_recovery"] = bool(method_burden_recovery)
    method_burden_recovery_strength = env_float("SAFE_FAIR_METHOD_BURDEN_RECOVERY_STRENGTH")
    if method_burden_recovery_strength is not None:
        method_updates["burden_recovery_strength"] = float(method_burden_recovery_strength)
    if method_updates:
        cfg.method = MethodSpec(**{**asdict(cfg.method), **method_updates})
    task_updates: Dict[str, Any] = {}
    task_safety_budget_per_step = env_float("SAFE_FAIR_TASK_SAFETY_BUDGET_PER_STEP")
    if task_safety_budget_per_step is not None:
        task_updates["safety_budget_per_step"] = float(task_safety_budget_per_step)
    task_safety_budget_scale = env_float("SAFE_FAIR_TASK_SAFETY_BUDGET_SCALE")
    if task_safety_budget_scale is not None:
        task_updates["safety_budget_per_step"] = float(cfg.task.safety_budget_per_step * task_safety_budget_scale)
    task_fairness_budget_per_step = env_float("SAFE_FAIR_TASK_FAIRNESS_BUDGET_PER_STEP")
    if task_fairness_budget_per_step is not None:
        task_updates["fairness_budget_per_step"] = float(task_fairness_budget_per_step)
    task_kwargs_json = env_str("SAFE_FAIR_TASK_KWARGS_JSON")
    if task_kwargs_json:
        try:
            extra_kwargs = json.loads(task_kwargs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SAFE_FAIR_TASK_KWARGS_JSON is not valid JSON: {exc}") from exc
        if not isinstance(extra_kwargs, dict):
            raise ValueError("SAFE_FAIR_TASK_KWARGS_JSON must decode to an object")
        task_updates["kwargs"] = {**cfg.task.kwargs, **extra_kwargs}
    if task_updates:
        cfg.task = TaskSpec(**{**asdict(cfg.task), **task_updates})
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    return cfg

def build_main_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = EXPERIMENT_SEEDS) -> List[RunConfig]:
    task_keys = QUICK_TASK_KEYS if quick else FULL_TASK_KEYS
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in task_keys:
        for method_key in MAIN_METHOD_KEYS:
            for seed in seeds:
                experiments.append(make_run_config(task_key, method_key, seed, "main", quick=quick))
    return experiments

def build_ablation_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = ABLATION_EXPERIMENT_SEEDS) -> List[RunConfig]:
    task_keys = QUICK_TASK_KEYS if quick else ["smamujoco_halfcheetah_2x3_fair", "smamujoco_ant_2x4_fair", "safetygym_point_multigoal1_fair", "safetygym_chargeshare_fair"]
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in task_keys:
        for method_key in ABLATION_METHOD_KEYS:
            for seed in seeds:
                experiments.append(make_run_config(task_key, method_key, seed, "ablation", quick=quick))
    return experiments

def build_fairness_budget_sweep(quick: bool = QUICK_TEST, seeds: Sequence[int] = (0, 1, 2)) -> List[RunConfig]:
    task_keys = QUICK_TASK_KEYS if quick else ["smamujoco_ant_2x4_fair", "safetygym_chargeshare_fair"]
    budgets = [0.025, 0.05, 0.10] if quick else [0.02, 0.04, 0.08, 0.16]
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in task_keys:
        for b in budgets:
            for seed in seeds:
                cfg = make_run_config(task_key, "safe_fair_macpo", seed, "fairness_budget_sweep", quick=quick)
                cfg.task = TaskSpec(**{**asdict(cfg.task), "fairness_budget_per_step": float(b)})
                experiments.append(cfg)
    return experiments

def build_heterogeneity_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = (0, 1, 2)) -> List[RunConfig]:
    task_keys = QUICK_TASK_KEYS if quick else ["safetygym_chargeshare_fair", "smamujoco_ant_2x4_fair"]
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in task_keys:
        task = TASKS[task_key]
        q = np.linspace(1.0, 2.0, task.n_agents)
        q = (q / q.sum()).tolist()
        method = MethodSpec(**{**asdict(METHODS["safe_fair_macpo"]), "name": "safe_fair_heterogeneous_q", "target_shares": q})
        METHODS["safe_fair_heterogeneous_q"] = method
        for seed in seeds:
            experiments.append(make_run_config(task_key, "safe_fair_heterogeneous_q", seed, "heterogeneity", quick=quick))
    return experiments

def build_scalability_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = (0, 1, 2)) -> List[RunConfig]:
    if quick:
        task = TaskSpec(**{**asdict(TASKS["smoke_bottleneck"]), "name": "smoke_bottleneck_8agents", "n_agents": 8, "kwargs": {"n_agents": 8, "n_hazards": 2, "goal_radius": 0.12}})
        TASKS[task.name] = task
        task_keys = [task.name]
        seeds = (0,)
    else:
        task_keys = ["smamujoco_manyagent_ant_fair"]
    experiments = []
    for task_key in task_keys:
        for method_key in ["safe_fair_macpo", "safe_fair_local_k1", "macpo"]:
            for seed in seeds:
                cfg = make_run_config(task_key, method_key, seed, "scalability", quick=quick)
                if quick:
                    cfg.total_steps = 15_000
                experiments.append(cfg)
    return experiments

def build_stress_shift_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = (0, 1, 2)) -> List[RunConfig]:
    task_keys = QUICK_TASK_KEYS if quick else ["safetygym_point_multigoal1_fair", "smamujoco_ant_2x4_fair"]
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in task_keys:
        for method_key in ["macpo", "reward_shaped_macpo_fair", "safe_fair_macpo"]:
            for seed in seeds:
                experiments.append(make_run_config(task_key, method_key, seed, "stress_shift", quick=quick))
    return experiments

def build_workload_imbalance_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = (0,)) -> List[RunConfig]:
    task_keys = QUICK_TASK_KEYS if quick else WORKLOAD_IMBALANCE_TASK_KEYS
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in task_keys:
        for method_key in WORKLOAD_IMBALANCE_METHOD_KEYS:
            for seed in seeds:
                cfg = make_run_config(task_key, method_key, seed, "workload_imbalance", quick=quick)
                if quick:
                    cfg.total_steps = 12_000
                experiments.append(cfg)
    return experiments

def build_safefairbench_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = SAFEFAIRBENCH_EXPERIMENT_SEEDS) -> List[RunConfig]:
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in SAFEFAIRBENCH_TASK_KEYS:
        for method_key in SAFEFAIRBENCH_METHOD_KEYS:
            for seed in seeds:
                cfg = make_run_config(task_key, method_key, seed, "safefairbench", quick=quick)
                if quick:
                    if env_int("SAFE_FAIR_CFG_TOTAL_STEPS") is None:
                        cfg.total_steps = 2_048
                    if env_int("SAFE_FAIR_CFG_ROLLOUT_STEPS") is None:
                        cfg.rollout_steps = 256
                    if env_int("SAFE_FAIR_CFG_PPO_EPOCHS") is None:
                        cfg.ppo_epochs = 2
                    if env_int("SAFE_FAIR_CFG_MINIBATCH_SIZE") is None:
                        cfg.minibatch_size = 128
                    if env_int("SAFE_FAIR_CFG_EVAL_EPISODES") is None:
                        cfg.eval_episodes = 2
                experiments.append(cfg)
    return experiments

def build_safefairbench_ablation_experiments(quick: bool = QUICK_TEST, seeds: Sequence[int] = SAFEFAIRBENCH_ABLATION_EXPERIMENT_SEEDS) -> List[RunConfig]:
    if quick:
        seeds = (0,)
    experiments = []
    for task_key in SAFEFAIRBENCH_TASK_KEYS:
        for method_key in SAFEFAIRBENCH_ABLATION_METHOD_KEYS:
            for seed in seeds:
                cfg = make_run_config(task_key, method_key, seed, "safefairbench_ablation", quick=quick)
                if quick:
                    if env_int("SAFE_FAIR_CFG_TOTAL_STEPS") is None:
                        cfg.total_steps = 2_048
                    if env_int("SAFE_FAIR_CFG_ROLLOUT_STEPS") is None:
                        cfg.rollout_steps = 256
                    if env_int("SAFE_FAIR_CFG_PPO_EPOCHS") is None:
                        cfg.ppo_epochs = 2
                    if env_int("SAFE_FAIR_CFG_MINIBATCH_SIZE") is None:
                        cfg.minibatch_size = 128
                    if env_int("SAFE_FAIR_CFG_EVAL_EPISODES") is None:
                        cfg.eval_episodes = 2
                experiments.append(cfg)
    return experiments

main_experiments = build_main_experiments(quick=QUICK_TEST)
ablation_experiments = build_ablation_experiments(quick=QUICK_TEST)
sweep_experiments = build_fairness_budget_sweep(quick=QUICK_TEST)
heterogeneity_experiments = build_heterogeneity_experiments(quick=QUICK_TEST)
scalability_experiments = build_scalability_experiments(quick=QUICK_TEST)
stress_experiments = build_stress_shift_experiments(quick=QUICK_TEST)
workload_imbalance_experiments = build_workload_imbalance_experiments(quick=QUICK_TEST)
safefairbench_experiments = build_safefairbench_experiments(quick=QUICK_TEST) if INCLUDE_SAFEFAIRBENCH else []
safefairbench_ablation_experiments = build_safefairbench_ablation_experiments(quick=QUICK_TEST) if INCLUDE_SAFEFAIRBENCH else []

all_experiments = main_experiments + ablation_experiments + sweep_experiments + heterogeneity_experiments + scalability_experiments + stress_experiments + workload_imbalance_experiments + safefairbench_experiments + safefairbench_ablation_experiments
manifest_df = pd.DataFrame([{"run_group": cfg.run_group, "task": cfg.task.name, "method": cfg.method.name, "seed": cfg.seed, "total_steps": cfg.total_steps, "rollout_steps": cfg.rollout_steps, "fair_budget": cfg.task.fairness_budget_per_step, "quick": cfg.quick} for cfg in all_experiments])
manifest_df.to_csv(CONFIG_ROOT / "experiment_manifest.csv", index=False)
print("Number of configured runs:", len(all_experiments))
manifest_df.head(20)


# =============================================================================
# 9. Run experiments
# =============================================================================
def write_run_configs(experiments: Sequence[RunConfig], root: Path = CONFIG_ROOT) -> None:
    for cfg in experiments:
        run_key = stable_hash({"group": cfg.run_group, "task": cfg.task.name, "method": cfg.method.name, "seed": cfg.seed, "fair_budget": cfg.task.fairness_budget_per_step})
        save_json(cfg.to_dict(), root / cfg.run_group / f"{cfg.task.name}__{cfg.method.name}__seed{cfg.seed}__{run_key}.json")

write_run_configs(all_experiments)

selected_experiments = main_experiments if QUICK_TEST else (main_experiments + ablation_experiments + sweep_experiments + heterogeneity_experiments + scalability_experiments + stress_experiments + workload_imbalance_experiments)
if INCLUDE_SAFEFAIRBENCH:
    selected_experiments = selected_experiments + safefairbench_experiments + safefairbench_ablation_experiments

if TASK_FILTERS:
    selected_experiments = [cfg for cfg in selected_experiments if cfg.task.name in TASK_FILTERS]
if METHOD_FILTERS:
    selected_experiments = [cfg for cfg in selected_experiments if cfg.method.name in METHOD_FILTERS]
if RUN_GROUP_FILTERS:
    selected_experiments = [cfg for cfg in selected_experiments if cfg.run_group in RUN_GROUP_FILTERS]
if SEED_FILTERS:
    selected_experiments = [cfg for cfg in selected_experiments if cfg.seed in SEED_FILTERS]
if FAIR_BUDGET_FILTERS:
    selected_experiments = [
        cfg
        for cfg in selected_experiments
        if any(abs(float(cfg.task.fairness_budget_per_step) - b) <= 1e-9 for b in FAIR_BUDGET_FILTERS)
    ]
if MAX_RUNS > 0:
    selected_experiments = selected_experiments[:MAX_RUNS]

print("Selected runs after filters:", len(selected_experiments))

results = []
if RUN_TRAINING:
    for cfg in selected_experiments:
        try:
            res = run_single_experiment(cfg)
            results.append(res)
            pd.DataFrame(results).to_csv(LOG_ROOT / f"latest_results_{now_stamp()}.csv", index=False)
        except Exception as exc:
            print(f"[error] {cfg.task.name}/{cfg.method.name}/seed{cfg.seed}: {exc}")
            results.append({"task": cfg.task.name, "method": cfg.method.name, "seed": cfg.seed, "status": f"failed: {type(exc).__name__}", "error": str(exc)})
else:
    print("RUN_TRAINING=False; writing configuration manifests only.")
    if selected_experiments:
        print("Example config:", selected_experiments[0].task.name, selected_experiments[0].method.name)
    else:
        print("No runs selected after filters.")
    # Uncomment to sanity-check one trainer constructor:
    # _ = run_single_experiment(selected_experiments[0])

pd.DataFrame(results).head()


# =============================================================================
# 10. Native upstream command builders
# =============================================================================
def macpo_native_command(cfg: RunConfig) -> Optional[str]:
    if cfg.task.source != "safe_mamujoco":
        return None
    macpo_root = EXTERNAL_ROOT / "macpo" / "MACPO" / "macpo"
    scenario = cfg.task.kwargs.get("scenario", cfg.task.env_id)
    agent_conf = cfg.task.kwargs.get("agent_conf", "2x4")
    agent_obsk = cfg.task.kwargs.get("agent_obsk", 1)
    return (
        f"cd {macpo_root}/scripts && "
        f"CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py "
        f"--env_name mujoco --algorithm_name macpo --experiment_name {cfg.run_group}_{cfg.method.name} "
        f"--scenario {scenario} --agent_conf {agent_conf} --agent_obsk {agent_obsk} "
        f"--seed {cfg.seed} --n_training_threads 4 --n_rollout_threads 16 "
        f"--num_mini_batch 40 --episode_length {cfg.task.episode_length} "
        f"--num_env_steps {cfg.total_steps} --ppo_epoch 1 "
        f"--use_value_active_masks --add_center_xy --use_state_agent "
        f"--safety_bound {cfg.task.safety_budget_per_step * cfg.task.episode_length:.6f} "
        f"--use_wandb False"
    )

def mappo_lagrangian_native_command(cfg: RunConfig) -> Optional[str]:
    if cfg.task.source != "safe_mamujoco":
        return None
    root = EXTERNAL_ROOT / "macpo" / "MAPPO-Lagrangian" / "mappo_lagrangian"
    scenario = cfg.task.kwargs.get("scenario", cfg.task.env_id)
    agent_conf = cfg.task.kwargs.get("agent_conf", "2x4")
    agent_obsk = cfg.task.kwargs.get("agent_obsk", 1)
    return (
        f"cd {root}/scripts && "
        f"CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py "
        f"--env_name mujoco --algorithm_name mappo_lagrangian --experiment_name {cfg.run_group}_{cfg.method.name} "
        f"--scenario {scenario} --agent_conf {agent_conf} --agent_obsk {agent_obsk} "
        f"--seed {cfg.seed} --n_training_threads 4 --n_rollout_threads 16 "
        f"--episode_length {cfg.task.episode_length} --num_env_steps {cfg.total_steps} "
        f"--safety_bound {cfg.task.safety_budget_per_step * cfg.task.episode_length:.6f} "
        f"--use_wandb False"
    )

def mappo_native_command(cfg: RunConfig) -> Optional[str]:
    root = EXTERNAL_ROOT / "mappo"
    return (
        f"# MAPPO native template; adapt env flags to the supported upstream environment.\n"
        f"cd {root} && "
        f"CUDA_VISIBLE_DEVICES=0 python onpolicy/scripts/train/train_mujoco.py "
        f"--algorithm_name rmappo --experiment_name {cfg.run_group}_{cfg.method.name} "
        f"--seed {cfg.seed} --num_env_steps {cfg.total_steps}"
    )

def dfrl_native_command(cfg: RunConfig) -> Optional[str]:
    root = EXTERNAL_ROOT / "dfrl"
    return f"# DFRL/SOTO-GGF native template; the official repo uses its own environments.\ncd {root} && python main.py --seed {cfg.seed}"

def def_marl_native_command(cfg: RunConfig) -> Optional[str]:
    root = EXTERNAL_ROOT / "def_marl"
    n = cfg.task.n_agents
    return f"# Def-MARL native command on its official MPE/Lidar tasks; not the MuJoCo fair wrapper.\ncd {root} && python train.py --env MPECorridor --algo def-marl -n {n} --obs 3 --seed {cfg.seed} --steps {cfg.total_steps}"

def native_command_for(cfg: RunConfig) -> Optional[str]:
    if cfg.method.name == "macpo":
        return macpo_native_command(cfg)
    if cfg.method.name == "mappo_lagrangian":
        return mappo_lagrangian_native_command(cfg)
    if cfg.method.name == "mappo":
        return mappo_native_command(cfg)
    if cfg.method.name == "dfrl_soto_ggf":
        return dfrl_native_command(cfg)
    if cfg.method.name == "def_marl":
        return def_marl_native_command(cfg)
    return None

def write_native_command_file(experiments: Sequence[RunConfig], path: Path = ROOT / "native_baseline_commands.sh") -> Path:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", "# Auto-generated native baseline command templates.", "# Review environment-specific flags before launching full native runs.", ""]
    for cfg in experiments:
        cmd = native_command_for(cfg)
        if cmd:
            lines.append(f"echo 'Running {cfg.task.name}/{cfg.method.name}/seed{cfg.seed}'")
            lines.append(cmd)
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path

native_script = write_native_command_file(main_experiments)
print("Wrote native command templates:", native_script)
print(native_script.read_text(encoding="utf-8")[:2000])


# =============================================================================
# 11. Result aggregation
# =============================================================================
def collect_eval_summaries(log_root: Path = LOG_ROOT) -> pd.DataFrame:
    rows = []
    for path in log_root.rglob("eval_summary.json"):
        try:
            rows.append(load_json(path))
        except Exception as exc:
            print("Could not read", path, exc)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["task", "method", "seed", "run_group", "run_tag", "return_mean", "safety_cost_mean", "violation_rate", "hard_violation_rate", "success_rate", "jain_burden_mean", "gini_burden_mean", "envy_gap_mean", "debt_auc_mean"])

def collect_train_histories(log_root: Path = LOG_ROOT) -> pd.DataFrame:
    frames = []
    for path in log_root.rglob("train_history.csv"):
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as exc:
            print("Could not read", path, exc)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

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
    run_tag_filter = os.environ.get("SAFE_FAIR_TABLE_RUN_TAG_FILTER", "").strip()
    if run_tag_filter:
        if "run_tag" not in sub.columns:
            return pd.DataFrame()
        sub = sub[sub["run_tag"].fillna("") == run_tag_filter].copy()
    metrics = ["return_mean", "safety_cost_mean", "hard_violation_rate", "success_rate", "jain_burden_mean", "gini_burden_mean", "envy_gap_mean", "debt_auc_mean"]
    rows = []
    for (task, method), g in sub.groupby(["task", "method"]):
        row = {"Environment": task, "Method": method}
        for m in metrics:
            if m == "hard_violation_rate" and (m not in g or pd.to_numeric(g[m], errors="coerce").dropna().empty) and "violation_rate" in g:
                row[m] = mean_std_str(g["violation_rate"])
            else:
                row[m] = mean_std_str(g[m]) if m in g else "--"
        rows.append(row)
    return pd.DataFrame(rows)

eval_df = collect_eval_summaries(LOG_ROOT)
train_df = collect_train_histories(LOG_ROOT)

main_table = make_summary_table(eval_df, "main")
ablation_table = make_summary_table(eval_df, "ablation")
sweep_table = make_summary_table(eval_df, "fairness_budget_sweep")

for name, table in [("main_results", main_table), ("ablation_results", ablation_table), ("fairness_sweep", sweep_table)]:
    path_csv = TABLE_ROOT / f"{name}.csv"
    path_tex = TABLE_ROOT / f"{name}.tex"
    table.to_csv(path_csv, index=False)
    try:
        table.to_latex(path_tex, index=False, escape=False)
    except Exception:
        pass
    print(name, "rows:", len(table), "->", path_csv)

main_table.head()


# =============================================================================
# 12. Plotting helpers
# =============================================================================
def save_tradeoff_plot(df: pd.DataFrame, run_group: str = "main", path: Path = FIG_ROOT / "tradeoff_return_safety_fairness.png") -> Optional[Path]:
    sub = df[df["run_group"] == run_group].copy() if not df.empty else df
    if sub.empty:
        print("No data for tradeoff plot.")
        return None
    agg = sub.groupby(["task", "method"], as_index=False).agg({"return_mean": "mean", "safety_cost_mean": "mean", "jain_burden_mean": "mean"})
    plt.figure(figsize=(7, 5))
    for method, g in agg.groupby("method"):
        plt.scatter(g["safety_cost_mean"], g["return_mean"], s=80, label=method, alpha=0.85)
        for _, row in g.iterrows():
            plt.annotate(f"{row['jain_burden_mean']:.2f}", (row["safety_cost_mean"], row["return_mean"]), fontsize=8)
    plt.xlabel("Safety cost (lower is better)")
    plt.ylabel("Episodic return (higher is better)")
    plt.title("Return–safety trade-off; annotations show Jain burden index")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path

def save_learning_curve(train_df: pd.DataFrame, metric: str = "debt_abs_mean", path: Path = FIG_ROOT / "learning_curve_debt.png") -> Optional[Path]:
    if train_df.empty or metric not in train_df:
        print("No data for learning curve.")
        return None
    plt.figure(figsize=(7, 5))
    for (task, method), g in train_df.groupby(["task", "method"]):
        gg = g.sort_values("global_step")
        plt.plot(gg["global_step"], gg[metric], label=f"{task}/{method}")
    plt.xlabel("Environment steps")
    plt.ylabel(metric)
    plt.title(f"Training curve: {metric}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path

def save_per_agent_burden_plot(df: pd.DataFrame, path: Path = FIG_ROOT / "per_agent_burden_histogram.png") -> Optional[Path]:
    if df.empty or "per_agent_burden_mean" not in df:
        print("No data for burden histogram.")
        return None
    sub = df[df["method"] == "safe_fair_macpo"].copy()
    if sub.empty:
        sub = df.copy()
    row = sub.iloc[0]
    burdens = row.get("per_agent_burden_mean", [])
    if isinstance(burdens, str):
        try:
            burdens = json.loads(burdens)
        except Exception:
            burdens = []
    if not burdens:
        print("No per-agent burdens found.")
        return None
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(burdens)), burdens)
    plt.xlabel("Agent")
    plt.ylabel("Cumulative burden")
    plt.title(f"Per-agent burden: {row.get('task')} / {row.get('method')}")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path

tradeoff_path = save_tradeoff_plot(eval_df)
curve_path = save_learning_curve(train_df)
burden_path = save_per_agent_burden_plot(eval_df)
print("Figures:", tradeoff_path, curve_path, burden_path)


# =============================================================================
# 13. Empty paper-table templates
# =============================================================================
def empty_main_template() -> pd.DataFrame:
    envs = ["smamujoco_halfcheetah_2x3_fair", "smamujoco_ant_2x4_fair", "safetygym_point_multigoal1_fair", "safetygym_chargeshare_fair"]
    methods = MAIN_METHOD_KEYS
    rows = []
    for env in envs:
        for m in methods:
            rows.append({"Environment": env, "Method": m, "Return ↑": "--", "Safety cost ↓": "--", "Hard violation ↓": "--", "Success ↑": "--", "Jain burden ↑": "--", "Gini burden ↓": "--", "Envy gap ↓": "--", "Debt AUC ↓": "--"})
    return pd.DataFrame(rows)

def empty_ablation_template() -> pd.DataFrame:
    rows = []
    variants = ["safe_fair_macpo", "safe_fair_no_debt", "safe_fair_no_envy", "safe_fair_no_recovery", "reward_shaped_macpo_fair", "safe_fair_heterogeneous_q", "safe_fair_local_k1"]
    for variant in variants:
        rows.append({"Variant": variant, "Return ↑": "--", "Safety cost ↓": "--", "Jain burden ↑": "--", "Gini burden ↓": "--", "Envy gap ↓": "--", "Debt AUC ↓": "--"})
    return pd.DataFrame(rows)

def empty_sweep_template() -> pd.DataFrame:
    rows = []
    for env in ["smamujoco_ant_2x4_fair", "safetygym_chargeshare_fair"]:
        for b in ["low", "medium", "high", "very_high"]:
            rows.append({"Environment": env, "Fairness budget": b, "Return ↑": "--", "Safety cost ↓": "--", "Jain burden ↑": "--", "Gini burden ↓": "--", "Envy gap ↓": "--", "Debt AUC ↓": "--"})
    return pd.DataFrame(rows)

templates = {
    "template_main_results.csv": empty_main_template(),
    "template_ablation.csv": empty_ablation_template(),
    "template_fairness_budget_sweep.csv": empty_sweep_template(),
}
for filename, table in templates.items():
    table.to_csv(TABLE_ROOT / filename, index=False)
    print(TABLE_ROOT / filename)

empty_main_template().head(12)


# =============================================================================
# 14. Environment availability checks
# =============================================================================
def check_environment_availability(task_keys: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key in task_keys:
        task = TASKS[key]
        method = METHODS["safe_fair_macpo"]
        try:
            env = make_fair_env(task, method, seed=0)
            obs, info = env.reset(seed=0)
            obs_dims, act_dims = infer_dims_from_env(env, seed=0)
            env.close()
            rows.append({"task": key, "available": True, "obs_dims": str(obs_dims), "act_dims": str(act_dims), "message": "ok"})
        except Exception as exc:
            rows.append({"task": key, "available": False, "obs_dims": "", "act_dims": "", "message": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)

availability_task_keys = list(QUICK_TASK_KEYS if QUICK_TEST else FULL_TASK_KEYS)
if INCLUDE_SAFEFAIRBENCH:
    availability_task_keys.extend(SAFEFAIRBENCH_TASK_KEYS)
availability_df = check_environment_availability(availability_task_keys)
availability_df.to_csv(ROOT / "environment_availability.csv", index=False)
availability_df


# =============================================================================
# 15. Stress-shift evaluation helpers
# =============================================================================
def make_shifted_task(task: TaskSpec, shift_name: str) -> TaskSpec:
    data = asdict(task)
    data["name"] = f"{task.name}__shift_{shift_name}"
    kwargs = dict(data.get("kwargs", {}))
    if shift_name == "more_hazards":
        kwargs["n_hazards"] = max(3, int(kwargs.get("n_hazards", 2)) + 1)
    elif shift_name == "tight_bottleneck":
        kwargs["goal_radius"] = float(kwargs.get("goal_radius", 0.12)) * 0.85
    elif shift_name == "scarce_station":
        kwargs["station_positions"] = [[0.0, 0.0]]
        kwargs["low_energy_threshold"] = 0.55
    data["kwargs"] = kwargs
    return TaskSpec(**data)

def evaluate_checkpoint_placeholder(run_dir: Path, shifted_task: TaskSpec, method: MethodSpec, seed: int = 0) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg_data = load_json(cfg_path)
    cfg = RunConfig(task=shifted_task, method=method, seed=seed, total_steps=1, rollout_steps=1, run_group="stress_eval", quick=cfg_data.get("quick", False), device=str(DEVICE))
    trainer = SafeFairTrainer(cfg)
    ckpt = run_dir / "model.pt"
    if ckpt.exists():
        trainer.ac.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    else:
        print("No checkpoint found; evaluating a freshly initialized policy:", ckpt)
    result = trainer.evaluate()
    trainer.close()
    return result

shift_examples = [make_shifted_task(TASKS[QUICK_TASK_KEYS[0]], s) for s in ["more_hazards", "tight_bottleneck", "scarce_station"]]
[(t.name, t.kwargs) for t in shift_examples]


# =============================================================================
# 16. Optional custom launch loop with checkpoints
# =============================================================================
def train_with_checkpoints(cfg: RunConfig, checkpoint_interval_updates: int = 10) -> Dict[str, Any]:
    trainer = SafeFairTrainer(cfg)
    n_updates = max(1, int(math.ceil(cfg.total_steps / cfg.rollout_steps)))
    for update_idx in tqdm(range(n_updates), desc=f"ckpt/{cfg.task.name}/{cfg.method.name}/seed{cfg.seed}"):
        batch = trainer.collect_rollout()
        rec = trainer.update(batch)
        if update_idx % checkpoint_interval_updates == 0 or update_idx == n_updates - 1:
            torch.save(trainer.ac.state_dict(), trainer.run_dir / "model.pt")
            pd.DataFrame(trainer.history).to_csv(trainer.run_dir / "train_history.csv", index=False)
    result = trainer.evaluate() if RUN_EVALUATION else {}
    torch.save(trainer.ac.state_dict(), trainer.run_dir / "model.pt")
    trainer.close()
    return result

# Example:
# cfg = make_run_config("smoke_bottleneck", "safe_fair_macpo", seed=0, run_group="debug", quick=True)
# RUN_TRAINING = True
# train_with_checkpoints(cfg)


# =============================================================================
# 17. Reproducibility manifest
# =============================================================================
def write_reproducibility_manifest(path: Path = ROOT / "reproducibility_manifest.json") -> Dict[str, Any]:
    manifest = {
        "created_at": now_stamp(),
        "host": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "root": str(ROOT),
        "quick_test": QUICK_TEST,
        "run_training": RUN_TRAINING,
        "install_external": INSTALL_EXTERNAL,
        "official_sources": source_manifest.to_dict(orient="records"),
        "experiment_manifest_csv": str(CONFIG_ROOT / "experiment_manifest.csv"),
        "tasks": {k: asdict(v) for k, v in TASKS.items()},
        "methods": {k: asdict(v) for k, v in METHODS.items()},
    }
    save_json(manifest, path)
    return manifest

manifest = write_reproducibility_manifest()
print(json.dumps({k: manifest[k] for k in ["created_at", "python", "torch", "cuda_available", "device", "quick_test"]}, indent=2))
