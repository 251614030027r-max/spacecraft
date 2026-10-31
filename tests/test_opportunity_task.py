"""Opportunity task: outer frozen approach corridor + inner rotating capture.

The outer approach corridor is a counted, non-terminating safety-margin
constraint around the episode-frozen inertial staging direction ``s0``, active
pre-entry and inactive after the terminal region latches. It does not co-rotate,
so the body-fixed capture geometry sweeps in and out of it -- the "when to
close" decision. These tests pin the geometry and the guarding (the historical
task stays bitwise unchanged).
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics.relative import relative_state
from dynamics.types import SpacecraftState
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import (
    precapture_opportunity_environment_config,
    precapture_planning_environment_config,
)
from env.task import PrecaptureTaskConfig, compute_precapture_metrics


def _states(chaser_position: np.ndarray) -> tuple[SpacecraftState, SpacecraftState]:
    target = SpacecraftState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    chaser = SpacecraftState(np.eye(3), np.asarray(chaser_position, float), np.zeros(3), np.zeros(3))
    return target, chaser


def test_outer_approach_margin_tracks_angle_to_staging() -> None:
    task = PrecaptureTaskConfig(outer_approach_half_angle_rad=float(np.deg2rad(40.0)))
    position = np.array([12.0, 0.0, 0.0])
    target, chaser = _states(position)
    relative = relative_state(target, chaser)

    # Staging aligned with the chaser's displacement -> angle 0, full margin.
    aligned = compute_precapture_metrics(
        target, chaser, relative, task,
        staging_direction_inertial=position / np.linalg.norm(position),
    )
    assert aligned.outer_approach_angle_rad == pytest.approx(0.0, abs=1e-9)
    assert aligned.outer_approach_margin_rad == pytest.approx(np.deg2rad(40.0), abs=1e-9)

    # Staging 50 deg off -> angle 50 deg, margin negative (outside the corridor).
    off = np.array([np.cos(np.deg2rad(50.0)), np.sin(np.deg2rad(50.0)), 0.0])
    misaligned = compute_precapture_metrics(
        target, chaser, relative, task, staging_direction_inertial=off,
    )
    assert misaligned.outer_approach_angle_rad == pytest.approx(np.deg2rad(50.0), abs=1e-6)
    assert misaligned.outer_approach_margin_rad < 0.0


def test_outer_approach_inactive_when_disabled_or_terminal() -> None:
    position = np.array([12.0, 0.0, 0.0])
    target, chaser = _states(position)
    relative = relative_state(target, chaser)
    off = np.array([np.cos(np.deg2rad(80.0)), np.sin(np.deg2rad(80.0)), 0.0])

    # Corridor disabled (historical task): large positive margin, never binds.
    disabled = compute_precapture_metrics(
        target, chaser, relative, PrecaptureTaskConfig(),
        staging_direction_inertial=off,
    )
    assert disabled.outer_approach_margin_rad > 100.0

    # Enabled but terminal latched: outer approach is inactive.
    task = PrecaptureTaskConfig(outer_approach_half_angle_rad=float(np.deg2rad(40.0)))
    terminal = compute_precapture_metrics(
        target, chaser, relative, task,
        terminal_region_active=True, staging_direction_inertial=off,
    )
    assert terminal.outer_approach_margin_rad > 100.0


def test_opportunity_config_opens_task_and_keeps_control_frozen() -> None:
    base = precapture_planning_environment_config()
    opp = precapture_opportunity_environment_config()
    assert base.precapture_task.outer_approach_half_angle_rad is None
    assert opp.precapture_task.outer_approach_half_angle_rad == pytest.approx(
        np.deg2rad(40.0)
    )
    assert opp.precapture_initial_range_max_m == 28.0
    assert opp.max_distance_m == 35.0
    # Frozen control difficulty.
    assert opp.max_force_per_axis_n == base.max_force_per_axis_n
    assert opp.max_torque_per_axis_nm == base.max_torque_per_axis_nm
    assert opp.dt_s == base.dt_s
    assert opp.phase2_target_tumble_scale == base.phase2_target_tumble_scale
    for field in (
        "corridor_half_angle_rad",
        "fov_half_angle_rad",
        "terminal_total_speed_limit_m_s",
        "keepout_radius_m",
    ):
        assert getattr(opp.precapture_task, field) == getattr(base.precapture_task, field)


def _env(opportunity: bool) -> PrecaptureHybridEnv:
    base = (
        precapture_opportunity_environment_config()
        if opportunity
        else precapture_planning_environment_config()
    )
    return PrecaptureHybridEnv(
        environment_config=base,
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=20,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_execution_feedback_observation=True,
            baseline_anchored_residual=True,
            include_staging_direction_observation=opportunity,
        ),
    )


def test_staging_observation_adds_three_dims_and_starts_in_corridor() -> None:
    plain = _env(False)
    opp = _env(True)
    assert opp.observation_space.shape[0] == plain.observation_space.shape[0] + 3
    _, info = opp.reset(seed=262000)
    # The chaser starts on s0, so it is inside the outer corridor at reset.
    assert info["outer_approach_margin_rad"] > 0.0
    assert "outer_approach_violation_steps" in info
    assert info["outer_approach_violation_steps"] == 0
    # Historical task carries no outer-approach keys.
    _, plain_info = plain.reset(seed=262000)
    assert "outer_approach_violation_steps" not in plain_info
    plain.close()
    opp.close()
