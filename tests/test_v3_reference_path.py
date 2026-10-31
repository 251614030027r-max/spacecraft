"""The MPC follows the V3 learned reference's true path over its horizon.

An inertially fixed hold is a circle in the target frame. Handing the MPC a
straight-line extrapolation of it (a chord) made the chaser push against the
centrifugal term, about omega^2 r of thrust the task never asked for.
"""

from __future__ import annotations

import numpy as np

from dynamics.lie import so3_exp
from env.hybrid_env import PrecaptureHybridEnv
from train.train_hybrid import accelerated_training_configs


def _env() -> PrecaptureHybridEnv:
    environment_config, hybrid_config = accelerated_training_configs(
        horizon_steps=35,
        waypoint_parametrization="task_state_v3",
        execution_feedback=True,
        adaptive_task=True,
    )
    return PrecaptureHybridEnv(environment_config, hybrid_config)


def test_an_inertial_hold_is_passed_as_the_rotating_frame_circle() -> None:
    env = _env()
    env.reset(seed=265000)
    env.step_with_branch(np.zeros(2), branch="learned")  # commit c = 0, progress 0
    path = env.v3_reference_path()
    assert path is not None and path.shape == (3, env.mpc_config.horizon_steps + 1)
    rotation = np.asarray(env.env.target_state.rotation)
    omega = np.asarray(env.env.target_state.omega)
    radius = float(np.linalg.norm(path[:, 0]))
    for index in (0, 10, 35):
        predicted = rotation @ so3_exp(index * env.environment_config.dt_s * omega)
        expected = radius * (predicted.T @ env._hold_inertial)
        np.testing.assert_allclose(path[:, index], expected, atol=1e-9)
    # a chord would leave the circle; the path keeps its radius
    np.testing.assert_allclose(np.linalg.norm(path, axis=0), radius, atol=1e-9)
    env.close()


def test_the_path_meets_the_next_committed_reference_when_the_task_state_is_held() -> None:
    env = _env()
    env.reset(seed=265001)
    env.step_with_branch(np.array([1.0, 1.0]), branch="learned")
    # At the decision boundary, the path's first point is where the reference
    # is predicted to be now; holding the task state commits the next one.
    predicted_now = env.v3_reference_path()[:, 0].copy()
    env.step_with_branch(np.zeros(2), branch="learned")
    committed = env._v3_applied_radius_m * env._v3_applied_direction
    # The committed direction was computed from the true target attitude at
    # this time; the path used the exact same attitude, so they agree.
    np.testing.assert_allclose(predicted_now, committed, atol=1e-9)
    env.close()


def test_there_is_no_path_on_the_baseline_branch() -> None:
    env = _env()
    env.reset(seed=265002)
    env.step_with_branch(np.zeros(2), branch="baseline")
    assert env.v3_reference_path() is None
    assert env.controller_reference(np.zeros(3))["external_reference_path"] is None
    env.close()
