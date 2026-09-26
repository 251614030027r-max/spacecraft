"""Shared plumbing for the V3 post-training pipeline (M2 / M3 / M6 / M5).

Every script here rebuilds the environment through the training script's own
``accelerated_training_configs`` and refuses to run if the result differs from
the run's manifest, so value data and calibration are collected on exactly the
environment the policy was trained on.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from env.hybrid_env import PrecaptureHybridEnv
from train.train_hybrid import accelerated_training_configs

BranchSchedule = Callable[[int, np.ndarray], str]

FAILURE_KEYS = (
    "keepout_failure",
    "fov_failure",
    "outer_speed_failure",
    "outer_radial_failure",
    "transition_speed_failure",
    "terminal_constraint_failure",
    "distance_failure",
    "time_failure",
)


def load_manifest(run_dir: Path, *, require_completed: bool = True) -> dict[str, Any]:
    """``require_completed=False`` is for smoke tests on a checkpoint only."""

    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    if manifest.get("waypoint_parametrization") != "task_state_v3":
        raise ValueError("the V3 pipeline needs a task_state_v3 run")
    if require_completed and manifest.get("status") != "completed":
        raise ValueError(f"run {run_dir} is not completed (status={manifest.get('status')})")
    return manifest


def make_env_for_run(manifest: dict[str, Any]) -> PrecaptureHybridEnv:
    flags = manifest["evaluation_flags"].split()
    horizon = int(flags[flags.index("--horizon") + 1])
    environment_config, hybrid_config = accelerated_training_configs(
        horizon_steps=horizon,
        waypoint_parametrization="task_state_v3",
        execution_feedback="--execution-feedback" in flags,
        monotone_commit=bool(manifest["monotone_commit"]),
        baseline_anchored_residual=bool(manifest["baseline_anchored_residual"]),
        opportunity_task="--opportunity-task" in flags,
        adaptive_task="--adaptive-task" in flags,
    )
    if asdict(hybrid_config) != manifest["hybrid"]:
        raise ValueError(
            "this checkout builds a different hybrid config from the run's "
            "manifest; check out the run's code_commit"
        )
    env = PrecaptureHybridEnv(environment_config, hybrid_config)
    if int(env.observation_space.shape[0]) != int(manifest["observation_dimension"]):
        raise ValueError("observation dimension differs from the run's manifest")
    return env


def load_policy(run_dir: Path, model_name: str = "final_model.zip"):
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    return SAC.load(Path(run_dir) / model_name, device="cpu")


def run_episode(
    env: PrecaptureHybridEnv,
    policy: Any,
    seed: int,
    schedule: BranchSchedule,
) -> dict[str, Any]:
    """One episode; ``schedule(decision_index, observation)`` returns the branch.

    The learned branch executes the deterministic policy action. Branch
    switching goes through ``step_with_branch``, which enforces one-way
    learned -> baseline and resets the controller at the handoff.
    """

    observation, _ = env.reset(seed=int(seed))
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    branches: list[str] = []
    zero = np.zeros(env.action_space.shape, dtype=np.float64)
    info: dict[str, Any] = {}
    terminated = truncated = False
    decision = 0
    while not (terminated or truncated):
        branch = schedule(decision, observation)
        if branch == "learned":
            action, _ = policy.predict(observation, deterministic=True)
        else:
            action = zero
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        branches.append(branch)
        observation, reward, terminated, truncated, info = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch=branch
        )
        rewards.append(float(reward))
        decision += 1
    return {
        "seed": int(seed),
        "observations": np.stack(observations),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "branches": branches,
        "completed": bool(info.get("completed", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure": [key for key in FAILURE_KEYS if info.get(key)],
        "qp_zero_fallbacks": int(info.get("hybrid_v3_episode_qp_zero_fallbacks", -1)),
    }


def always(branch: str) -> BranchSchedule:
    return lambda decision, observation: branch


def learned_then_baseline(k: int) -> BranchSchedule:
    return lambda decision, observation: "learned" if decision < k else "baseline"


def parse_seed_range(text: str) -> list[int]:
    """'270000-270047' or '270000,270001' -> list of ints."""

    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = (int(v) for v in part.split("-"))
            seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    return seeds
