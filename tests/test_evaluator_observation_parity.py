"""The formal evaluator must feed the policy the observation training produces.

The evaluator runs each decision's control steps itself (to time them), so it
duplicates the environment's decision loop. Until 2026-09-26 the duplicate
never updated the execution-feedback block, and every policy trained with
feedback was evaluated on an all-zero feedback input. This test runs the same
policy and seed through both paths and requires identical observations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from train.train_hybrid import accelerated_training_configs
from env.hybrid_env import PrecaptureHybridEnv

REPOSITORY = Path(__file__).resolve().parents[1]
SEED = 263000
DECISIONS = 4


def _training_env() -> PrecaptureHybridEnv:
    environment_config, hybrid_config = accelerated_training_configs(
        horizon_steps=35,
        waypoint_parametrization="task_state_v3",
        execution_feedback=True,
        adaptive_task=True,
    )
    return PrecaptureHybridEnv(environment_config, hybrid_config)


def test_evaluator_and_training_path_see_the_same_observations(tmp_path: Path) -> None:
    env = _training_env()
    model = SAC("MlpPolicy", env, seed=0, device="cpu", policy_kwargs={"net_arch": [16]})
    model_path = tmp_path / "policy.zip"
    model.save(model_path)

    observation, _ = env.reset(seed=SEED)
    training_path = []
    feedback = env.policy_observation_slices()["execution_feedback"]
    for _ in range(DECISIONS):
        training_path.append(np.asarray(observation, dtype=np.float32).copy())
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch="learned"
        )
        assert not (terminated or truncated)
    env.close()
    # The feedback block is live after the first decision (co-rotation uses
    # the actuators), so an all-zero block in the evaluator could not pass.
    assert np.any(training_path[1][feedback] > 0.0)

    output = tmp_path / "eval.json"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "experiments.evaluate_hybrid_policy",
            "--episodes", "1",
            "--seed", str(SEED),
            "--horizon", "35",
            "--parametrization", "task_state_v3",
            "--phase-time-observation",
            "--execution-feedback",
            "--adaptive-task",
            "--model", str(model_path),
            "--max-decisions", str(DECISIONS),
            "--record-observations",
            "--output", str(output),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    record = json.loads(output.read_text())["records"][0]
    evaluator_path = np.asarray(record["policy_observations"], dtype=np.float32)
    assert evaluator_path.shape == (DECISIONS, training_path[0].size)
    np.testing.assert_array_equal(evaluator_path, np.stack(training_path))
