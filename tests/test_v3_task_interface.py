from __future__ import annotations

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


def _v3_env(*, reference_step_max_m: float = 0.4) -> PrecaptureHybridEnv:
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="task_state_v3",
            runtime_diagnostics=False,
            v2_reference_step_max_m=reference_step_max_m,
        ),
    )
    env.reset(seed=262004)
    return env


def test_v3_zero_action_keeps_task_state_fixed_for_ten_decisions() -> None:
    env = _v3_env()
    try:
        initial = (env._task_progress_m, env._task_commitment)
        for _ in range(10):
            env.waypoint_from_action(np.zeros(2, dtype=np.float64))
            assert (env._task_progress_m, env._task_commitment) == initial
    finally:
        env.close()


def test_v3_unit_action_applies_configured_axis_increments_when_metric_allows() -> None:
    env = _v3_env(reference_step_max_m=2.0)
    try:
        env.waypoint_from_action(np.ones(2, dtype=np.float64))
        assert env._task_progress_m == 0.5
        assert env._task_commitment == 0.05
    finally:
        env.close()


def test_v3_terminal_task_state_is_exactly_the_desired_pose() -> None:
    env = _v3_env()
    try:
        desired = np.asarray(
            env.environment_config.precapture_task.desired_position,
            dtype=np.float64,
        )
        reference = env.reference_for_task_state(env.v2_progress_max_m, 1.0)
        assert np.linalg.norm(reference - desired) < 1.0e-12
    finally:
        env.close()
