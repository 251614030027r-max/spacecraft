"""Adaptive sync-entry mainline task (2026-09-18).

Opened initial distribution, no hard far-range corridor: the capture opportunity
is a cost structure, not a legality gate, so Pure MPC always keeps a legal path
and the learned layer's job is the sync-vs-stage resource trade-off. These tests
pin that the config opens only the task-selection space, adds no new hard
constraint, and carries the staging observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import (
    precapture_adaptive_capture_environment_config,
    precapture_planning_environment_config,
)


def test_adaptive_config_opens_task_without_a_new_hard_constraint() -> None:
    base = precapture_planning_environment_config()
    adp = precapture_adaptive_capture_environment_config()
    # No outer-approach corridor (opportunity is a cost, not a gate).
    assert adp.precapture_task.outer_approach_half_angle_rad is None
    # Opened task-selection space.
    assert adp.precapture_initial_range_max_m == 28.0
    assert adp.max_distance_m == 35.0
    assert adp.precapture_initial_pointing_error_max_rad == pytest.approx(
        np.deg2rad(25.0)
    )
    # Frozen control difficulty.
    assert adp.max_force_per_axis_n == base.max_force_per_axis_n
    assert adp.max_torque_per_axis_nm == base.max_torque_per_axis_nm
    assert adp.dt_s == base.dt_s
    assert adp.phase2_target_tumble_scale == base.phase2_target_tumble_scale
    for field in (
        "corridor_half_angle_rad",
        "fov_half_angle_rad",
        "terminal_total_speed_limit_m_s",
        "keepout_radius_m",
        "entry_port_axial_distance_m",
    ):
        assert getattr(adp.precapture_task, field) == getattr(
            base.precapture_task, field
        )


def test_adaptive_env_carries_staging_observation_and_no_corridor_keys() -> None:
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=20,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_execution_feedback_observation=True,
            baseline_anchored_residual=True,
            include_staging_direction_observation=True,
        ),
    )
    plain = PrecaptureHybridEnv(
        environment_config=precapture_planning_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=20,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_execution_feedback_observation=True,
            baseline_anchored_residual=True,
        ),
    )
    # Staging direction adds exactly 3 dims over the same non-staging config.
    assert env.observation_space.shape[0] == plain.observation_space.shape[0] + 3
    _, info = env.reset(seed=262000)
    assert env.observation_space.shape == (38,)
    assert env.action_space.shape == (2,)
    assert env.env._staging_direction_inertial is not None
    assert env.environment_config.precapture_task.entry_phase_gate_cos <= -1.0
    # No far-range corridor was added, so no outer-approach violation counter.
    assert "outer_approach_violation_steps" not in info
    env.close()
    plain.close()
