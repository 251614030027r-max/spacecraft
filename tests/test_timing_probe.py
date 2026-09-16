"""Zero-training timing-value A/B scaffold: opened-distribution config and the
scripted timing (B) arm.

The probe asks whether timing the entry to a favourable phase helps, with the
strong MPC frozen. Two correctness properties matter here: the config opens only
the task-selection space and leaves every control-difficulty knob and the gate
default honest, and the scripted arm is a clean hold-until-favourable bang-bang
whose committed command is the immediate arm's.
"""

from __future__ import annotations

import numpy as np
import pytest
from argparse import Namespace

from dynamics.lie import so3_exp

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import (
    precapture_planning_environment_config,
    precapture_timing_probe_environment_config,
)
from experiments.evaluate_hybrid_policy import (
    ScriptedEntryTiming,
    _TIMED_COMMIT_ACTION,
    _TIMED_HOLD_ACTION,
    evaluation_environment_config,
)


def test_timing_probe_config_opens_only_task_selection() -> None:
    base = precapture_planning_environment_config()
    probe = precapture_timing_probe_environment_config()

    # Opened: initial range and pointing spread.
    assert probe.precapture_initial_range_max_m > base.precapture_initial_range_max_m
    assert probe.precapture_initial_range_min_m == base.precapture_initial_range_min_m
    assert (
        probe.precapture_initial_pointing_error_max_rad
        > base.precapture_initial_pointing_error_max_rad
    )
    # Strictly inside the frozen distance-failure boundary.
    assert probe.precapture_initial_range_max_m < probe.max_distance_m

    # Frozen control difficulty: authority, update rate, tumble, and the whole
    # constraint/geometry set are untouched.
    assert probe.max_force_per_axis_n == base.max_force_per_axis_n
    assert probe.max_torque_per_axis_nm == base.max_torque_per_axis_nm
    assert probe.dt_s == base.dt_s
    assert probe.phase2_target_tumble_scale == base.phase2_target_tumble_scale
    assert probe.max_time_s == base.max_time_s
    assert probe.max_distance_m == base.max_distance_m
    base_task = base.precapture_task
    probe_task = probe.precapture_task
    for field in (
        "corridor_half_angle_rad",
        "fov_half_angle_rad",
        "terminal_total_speed_limit_m_s",
        "keepout_radius_m",
        "entry_port_axial_distance_m",
        "outer_inertial_speed_limit_m_s",
    ):
        assert getattr(probe_task, field) == getattr(base_task, field)


def test_timing_probe_gate_off_by_default() -> None:
    # The primary A/B must not manufacture a difference with a hard gate.
    probe = precapture_timing_probe_environment_config()
    assert probe.precapture_task.entry_phase_gate_cos <= -1.0
    # A finite diagnostic gate is available but is a guardrail only.
    gated = precapture_timing_probe_environment_config(entry_phase_gate_deg=90.0)
    assert np.isclose(gated.precapture_task.entry_phase_gate_cos, 0.0, atol=1e-12)


def _timing_env() -> PrecaptureHybridEnv:
    env = PrecaptureHybridEnv(
        environment_config=precapture_timing_probe_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
        ),
    )
    env.reset(seed=262000)
    return env


def test_a_b_use_identical_cached_target_dynamics() -> None:
    options = dict(timing_probe=True, perception=False, entry_phase_gate_deg=180.0)
    config_a = evaluation_environment_config(Namespace(**options, control="desired_pose"))
    config_b = evaluation_environment_config(Namespace(**options, control="timed_entry"))
    assert config_a == config_b
    assert config_a.cache_target_trajectory
    a = PrecaptureHybridEnv(environment_config=config_a)
    b = PrecaptureHybridEnv(environment_config=config_b)
    try:
        a.reset(seed=262000)
        b.reset(seed=262000)
        assert a.env._target_trajectory is b.env._target_trajectory
        assert np.array_equal(a.env.target_state.rotation, b.env.target_state.rotation)
        for index in (0, 1, 10, 100):
            cached = a.env._target_trajectory[index]
            assert np.array_equal(cached.rotation, b.env._target_trajectory[index].rotation)
        # Cache index zero is the reset state, and index one is the next step.
        assert np.array_equal(a.env.target_state.rotation, a.env._target_trajectory[0].rotation)
    finally:
        a.close()
        b.close()


def test_hold_waypoint_follows_frozen_inertial_staging_direction() -> None:
    env = _timing_env()
    try:
        staging = np.asarray(env.env._staging_direction_inertial)
        target = env.env.target_state
        assert target is not None
        first = env.waypoint_from_action(_TIMED_HOLD_ACTION)
        first_inertial = target.rotation @ first
        assert first_inertial / np.linalg.norm(first_inertial) == pytest.approx(staging)
        target.rotation = target.rotation @ so3_exp(np.array([0.0, 0.0, 0.6]))
        second = env.waypoint_from_action(_TIMED_HOLD_ACTION)
        second_inertial = target.rotation @ second
        assert second_inertial / np.linalg.norm(second_inertial) == pytest.approx(staging)
        assert second_inertial == pytest.approx(first_inertial)
        commit = env.waypoint_from_action(_TIMED_COMMIT_ACTION)
        assert commit == pytest.approx(env.environment_config.precapture_task.desired_position)
    finally:
        env.close()


def test_timed_arm_holds_until_favourable_then_latches() -> None:
    env = _timing_env()
    info = {"target_center_distance_m": 20.0}

    # An unreachable threshold never commits (until the deadline, far away here).
    waiting = ScriptedEntryTiming(
        env, commit_favourability_cos=2.0, lead_speed_m_s=0.20
    )
    assert np.array_equal(waiting.action(info), _TIMED_HOLD_ACTION)
    assert waiting.committed is False
    assert waiting.commit_time_s is None

    # A threshold that is always cleared commits on the first decision and stays
    # committed, and records where it committed.
    committing = ScriptedEntryTiming(
        env, commit_favourability_cos=-2.0, lead_speed_m_s=0.20
    )
    assert np.array_equal(committing.action(info), _TIMED_COMMIT_ACTION)
    assert committing.committed is True
    assert committing.commit_time_s == 0.0
    assert committing.favourability_at_commit is not None
    assert -1.0 <= committing.favourability_at_commit <= 1.0
    # Latched: still commit on the next decision.
    assert np.array_equal(committing.action(info), _TIMED_COMMIT_ACTION)
    env.close()


def test_timed_arm_predicts_from_the_cached_truth_trajectory() -> None:
    # The oracle reads the target's actual future attitude from the cached truth
    # trajectory, not a constant-body-rate extrapolation.
    env = _timing_env()
    assert env.env._target_trajectory is not None
    timing = ScriptedEntryTiming(
        env, commit_favourability_cos=0.707, lead_speed_m_s=0.20
    )
    lead_s = 40.0
    dt_s = env.environment_config.dt_s
    future_step = env.env.step_count + int(round(lead_s / dt_s))
    task = env.environment_config.precapture_task
    staging = env.env._staging_direction_inertial
    expected_rotation = np.asarray(
        env.env._target_trajectory[future_step].rotation, dtype=np.float64
    )
    expected = float(
        np.clip((expected_rotation @ task.approach_axis) @ staging, -1.0, 1.0)
    )
    assert timing._predicted_favourability(lead_s) == pytest.approx(expected)
    env.close()


def test_timed_arm_force_commits_near_the_deadline() -> None:
    env = _timing_env()
    # Push the clock to the end of the episode so no wait can transit in time.
    env.env.time_seconds = env.environment_config.max_time_s - 5.0
    forced = ScriptedEntryTiming(
        env, commit_favourability_cos=2.0, lead_speed_m_s=0.20
    )
    action = forced.action({"target_center_distance_m": 20.0})
    assert np.array_equal(action, _TIMED_COMMIT_ACTION)
    assert forced.committed is True
    env.close()
