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


def test_v3_memory_axis_blend_is_continuous_through_antipodal_direction() -> None:
    desired = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    radius_m = 15.0
    for commitment in (0.3, 0.5, 0.7):
        env = _v3_env()
        try:
            previous_hold = None
            previous_reference = None
            for angle_deg in np.linspace(170.0, 190.0, 81):
                angle = np.deg2rad(angle_deg)
                hold = np.array([np.cos(angle), np.sin(angle), 0.0])
                direction = env._v3_memory_axis_direction(
                    hold, desired, commitment
                )
                reference = radius_m * direction
                if previous_reference is not None and previous_hold is not None:
                    natural = radius_m * float(np.linalg.norm(hold - previous_hold))
                    jump = float(np.linalg.norm(reference - previous_reference))
                    assert jump <= 1.2 * natural + 0.40 + 1.0e-12
                previous_hold = hold
                previous_reference = reference
        finally:
            env.close()


def test_v3_memory_axis_blend_preserves_both_endpoints() -> None:
    env = _v3_env()
    try:
        desired = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hold = np.array(
            [np.cos(np.deg2rad(179.0)), np.sin(np.deg2rad(179.0)), 0.0]
        )
        assert np.linalg.norm(
            env._v3_memory_axis_direction(hold, desired, 0.0) - hold
        ) < 1.0e-12
        assert np.linalg.norm(
            env._v3_memory_axis_direction(hold, desired, 1.0) - desired
        ) < 1.0e-12
    finally:
        env.close()
