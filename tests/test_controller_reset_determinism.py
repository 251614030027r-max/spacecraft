"""A reset MPC must be the controller a fresh episode starts with.

cvxpy caches the previous solver object and warm-start updates it in place;
until the cache was dropped in ``reset`` the same seed replayed in a reused
environment drifted by ~5e-7 N from the first control step. The V3 value data
(M2) and calibration (M6) replay seeds in one environment and rely on exact
prefixes, and a V3 handback relies on reset meaning "Pure MPC from scratch".
"""

from __future__ import annotations

import numpy as np

from env.hybrid_env import PrecaptureHybridEnv
from train.train_hybrid import accelerated_training_configs


def _episode_prefix(env: PrecaptureHybridEnv, decisions: int) -> tuple[np.ndarray, np.ndarray]:
    observation, _ = env.reset(seed=263001)
    wrenches: list[np.ndarray] = []
    original = env.controller.command

    def recorded(*args, **kwargs):
        wrench, diagnostics = original(*args, **kwargs)
        wrenches.append(np.asarray(wrench, dtype=np.float64).copy())
        return wrench, diagnostics

    env.controller.command = recorded  # type: ignore[method-assign]
    try:
        for _ in range(decisions):
            observation, *_ = env.step_with_branch(np.zeros(2), branch="baseline")
    finally:
        env.controller.command = original  # type: ignore[method-assign]
    return np.asarray(observation), np.stack(wrenches)


def test_replaying_a_seed_in_a_reused_environment_is_bitwise_identical() -> None:
    environment_config, hybrid_config = accelerated_training_configs(
        horizon_steps=35,
        waypoint_parametrization="task_state_v3",
        execution_feedback=True,
        adaptive_task=True,
    )
    env = PrecaptureHybridEnv(environment_config, hybrid_config)
    first_observation, first_wrenches = _episode_prefix(env, 2)
    second_observation, second_wrenches = _episode_prefix(env, 2)
    env.close()
    np.testing.assert_array_equal(first_wrenches, second_wrenches)
    np.testing.assert_array_equal(first_observation, second_observation)
