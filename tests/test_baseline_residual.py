"""Baseline-anchored task residual on the arrival_condition interface.

Single factor: a zero residual must recover the nominal (fixed-setpoint Pure
MPC) command bitwise, so the strong baseline is never lost; a non-zero residual
lets the policy pull back from full commit. Default off leaves arrival_condition
unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config


def _env(residual: bool, parametrization: str = "arrival_condition"):
    return PrecaptureHybridEnv(
        environment_config=precapture_planning_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization=parametrization,
            runtime_diagnostics=False,
            baseline_anchored_residual=residual,
        ),
    )


def test_zero_residual_recovers_nominal_bitwise():
    plain = _env(residual=False)
    plain.reset(seed=262000)
    nominal_wp = plain.waypoint_from_action(np.array([1.0, 0.0]))

    res = _env(residual=True)
    res.reset(seed=262000)
    residual_wp = res.waypoint_from_action(np.array([0.0, 0.0]))

    assert np.array_equal(residual_wp, nominal_wp)
    # and both are exactly the desired pose (the fixed-setpoint command)
    desired = np.asarray(
        plain.environment_config.precapture_task.desired_position, dtype=np.float64
    )
    assert np.allclose(residual_wp, desired)


def test_action_for_desired_pose_is_zero_residual():
    res = _env(residual=True)
    res.reset(seed=262000)
    desired = np.asarray(
        res.environment_config.precapture_task.desired_position, dtype=np.float64
    )
    action = res.action_for_waypoint(desired)
    assert np.array_equal(action, np.array([0.0, 0.0]))
    # and it maps back to the desired pose
    assert np.allclose(res.waypoint_from_action(action), desired)


def test_negative_commit_residual_pulls_back_from_full_commit():
    res = _env(residual=True)
    res.reset(seed=262000)
    desired = np.asarray(
        res.environment_config.precapture_task.desired_position, dtype=np.float64
    )
    # a negative commit residual on a fresh env (blend not yet ratcheted to 1)
    # must not command the desired pose -- it holds short of full commit.
    held = res.waypoint_from_action(np.array([-0.6, 0.0]))
    assert not np.allclose(held, desired)


def test_most_negative_residual_recovers_full_inertial_hold():
    # The commit channel uses gain 2 so the most negative residual reaches the
    # full inertial hold -- the same waypoint the plain arrival_condition
    # interface produces at a=(-1, 0). Without this the policy could not express
    # complete staging, only the mid-blend.
    plain = _env(residual=False)
    plain.reset(seed=262000)
    hold_wp = plain.waypoint_from_action(np.array([-1.0, 0.0]))

    res = _env(residual=True)
    res.reset(seed=262000)
    residual_hold_wp = res.waypoint_from_action(np.array([-1.0, 0.0]))
    assert np.array_equal(residual_hold_wp, hold_wp)
    # and the action that names the full hold is the most-negative commit residual
    assert np.allclose(res.action_for_waypoint(hold_wp), np.array([-1.0, 0.0]))


def test_residual_requires_arrival_condition():
    with pytest.raises(ValueError):
        PrecaptureHybridConfig(
            waypoint_parametrization="radial_local",
            baseline_anchored_residual=True,
        )


def test_default_off_leaves_arrival_unchanged():
    # residual defaults to False; a=(1,0) still commands the desired pose.
    plain = _env(residual=False)
    plain.reset(seed=262001)
    desired = np.asarray(
        plain.environment_config.precapture_task.desired_position, dtype=np.float64
    )
    assert np.allclose(plain.waypoint_from_action(np.array([1.0, 0.0])), desired)
    assert plain.hybrid_config.baseline_anchored_residual is False
