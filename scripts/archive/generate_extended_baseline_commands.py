#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "safe_fair_macpo_runs" / "official_sources"
OUT = ROOT / "safe_fair_macpo_runs" / "native_extended_baseline_commands.sh"


def happo_hatrpo_commands() -> str:
    root = SRC / "happo_hatrpo"
    lines = [
        "# HAPPO / HATRPO on MuJoCo-family tasks",
        f"cd {root}",
        (
            "CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py "
            "--env_name mujoco --algorithm_name happo --experiment_name external_happo_halfcheetah "
            "--scenario HalfCheetah-v2 --agent_conf 2x3 --agent_obsk 1 "
            "--seed 0 --n_training_threads 4 --n_rollout_threads 4 "
            "--num_mini_batch 40 --episode_length 1000 --num_env_steps 500000 "
            "--ppo_epoch 5 --lr 5e-6 --critic_lr 5e-3 "
            "--std_x_coef 1 --std_y_coef 5e-1 --use_value_active_masks --add_center_xy --use_state_agent --share_policy"
        ),
        (
            "CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py "
            "--env_name mujoco --algorithm_name happo --experiment_name external_happo_ant "
            "--scenario Ant-v2 --agent_conf 2x4 --agent_obsk 1 "
            "--seed 0 --n_training_threads 4 --n_rollout_threads 4 "
            "--num_mini_batch 40 --episode_length 1000 --num_env_steps 500000 "
            "--ppo_epoch 5 --lr 5e-6 --critic_lr 5e-3 "
            "--std_x_coef 1 --std_y_coef 5e-1 --use_value_active_masks --add_center_xy --use_state_agent --share_policy"
        ),
        (
            "CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py "
            "--env_name mujoco --algorithm_name hatrpo --experiment_name external_hatrpo_halfcheetah "
            "--scenario HalfCheetah-v2 --agent_conf 2x3 --agent_obsk 1 "
            "--seed 0 --n_training_threads 4 --n_rollout_threads 4 "
            "--num_mini_batch 40 --episode_length 1000 --num_env_steps 500000 "
            "--ppo_epoch 5 --kl_threshold 1e-4 "
            "--lr 5e-6 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 5e-1 "
            "--use_value_active_masks --add_center_xy --use_state_agent --share_policy"
        ),
        (
            "CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py "
            "--env_name mujoco --algorithm_name hatrpo --experiment_name external_hatrpo_ant "
            "--scenario Ant-v2 --agent_conf 2x4 --agent_obsk 1 "
            "--seed 0 --n_training_threads 4 --n_rollout_threads 4 "
            "--num_mini_batch 40 --episode_length 1000 --num_env_steps 500000 "
            "--ppo_epoch 5 --kl_threshold 1e-4 "
            "--lr 5e-6 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 5e-1 "
            "--use_value_active_masks --add_center_xy --use_state_agent --share_policy"
        ),
        "",
    ]
    return "\n".join(lines)


def def_marl_commands() -> str:
    root = SRC / "def_marl"
    lines = [
        "# Def-MARL native tasks (secondary evidence; not apples-to-apples with current MuJoCo/Safety-Gym wrappers)",
        f"cd {root}",
        "python train.py --env MPECorridor --algo def-marl -n 4 --obs 3 --seed 0 --steps 500000",
        "python train.py --env MPECorridor --algo informarl --cost-weight 0.0 -n 4 --obs 3 --seed 0 --steps 500000",
        "python train.py --env MPECorridor --algo informarl_lagr -n 4 --obs 3 --seed 0 --steps 500000",
        "",
    ]
    return "\n".join(lines)


def dec_safe_marl_commands() -> str:
    root = SRC / "dec_safe_marl"
    lines = [
        "# Decentralized safe-MARL with general utilities (secondary evidence on native tasks only)",
        f"cd {root}",
        "python main_piston.py",
        "python main_wireless.py",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Auto-generated extended baseline commands.",
            "# Tier A commands are the most relevant to the current paper environments.",
            "",
            happo_hatrpo_commands(),
            def_marl_commands(),
            dec_safe_marl_commands(),
        ]
    )
    OUT.write_text(text, encoding="utf-8")
    OUT.chmod(0o755)
    print(OUT)


if __name__ == "__main__":
    main()
