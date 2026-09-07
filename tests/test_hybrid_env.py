"""What the coupling wrapper promises, pinned.

The three properties worth a test are the three that make a result mean
something: the wrapper does not change what the policy sees, the action is an
absolute waypoint in the frame the channel is lossless in, and one decision is
exactly the hold the optimiser was configured for.
"""

from __future__ import annotations

import numpy as np

from env.hybrid_env import (
    PrecaptureHybridConfig,
    PrecaptureHybridEnv,
    hybrid_mpc_config,
)
from env.phase2_env import precapture_planning_environment_config


def test_hybrid_action_is_three_dimensional_and_observation_is_unchanged() -> None:
    """The learned layer sees exactly what Pure SAC sees, and names a place."""

    env = PrecaptureHybridEnv()
    assert env.action_space.shape == (3,)
    assert env.observation_space == env.env.observation_space
    observation, _ = env.reset(seed=262000)
    assert env.observation_space.contains(observation.astype(np.float32))
    env.close()


def test_hybrid_pins_the_channel_the_diagnosis_measured_as_lossless() -> None:
    """Target body frame, frozen attitude, hold equal to the decision period.

    Under the inertially oriented reading the same channel could not carry the
    task's own desired pose at all, and a hold that disagreed with the decision
    period would adopt a waypoint part way through a decision.
    """

    hybrid = PrecaptureHybridConfig()
    config = hybrid_mpc_config(hybrid, precapture_planning_environment_config())
    assert config.reference_source == "external_local"
    assert config.external_reference_frame == "target"
    assert config.precapture_attitude_reference == "frozen"
    assert config.external_reference_hold_steps == hybrid.decision_period_steps


def test_waypoint_is_absolute_and_radially_clipped() -> None:
    """Absolute, so "hold out here" is as expressible as "close on the port".

    A displacement parametrisation would bias the policy toward moving, and
    holding position is one of the two answers the task poses.
    """

    env = PrecaptureHybridEnv()
    env.reset(seed=262000)
    hybrid = env.hybrid_config

    far = env.waypoint_from_action(np.array([1.0, 1.0, 1.0]))
    assert np.isclose(
        float(np.linalg.norm(far)), hybrid.maximum_waypoint_radius_m
    )
    assert np.all(far > 0.0)

    near = env.waypoint_from_action(np.array([0.01, 0.0, 0.0]))
    assert np.isclose(
        float(np.linalg.norm(near)), hybrid.minimum_waypoint_radius_m
    )
    assert near[0] > 0.0 and np.allclose(near[1:], 0.0)

    # Same action, same waypoint: the map carries no state of its own.
    action = np.array([-0.2, 0.1, 0.05])
    assert np.array_equal(
        env.waypoint_from_action(action), env.waypoint_from_action(action)
    )

    # A degenerate command must not ask for the target's own centre.
    degenerate = env.waypoint_from_action(np.zeros(3))
    assert float(np.linalg.norm(degenerate)) >= hybrid.minimum_waypoint_radius_m
    env.close()


def test_one_decision_is_the_configured_number_of_control_steps() -> None:
    env = PrecaptureHybridEnv()
    env.reset(seed=262000)
    started = env.env.time_seconds
    _, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0, 0.0]))
    assert not (terminated or truncated)
    assert info["hybrid_control_steps"] == env.hybrid_config.decision_period_steps
    assert np.isclose(
        env.env.time_seconds - started,
        env.hybrid_config.decision_period_steps * env.environment_config.dt_s,
    )
    assert np.isfinite(reward)
    # The waypoint actually flown is reported, so a trajectory can be audited
    # against the decision that produced it.
    assert np.allclose(
        info["hybrid_waypoint_target_frame"],
        env.waypoint_from_action(np.array([-0.2, 0.0, 0.0])),
    )
    env.close()


def test_hybrid_config_rejects_incoherent_settings() -> None:
    for bad in (
        {"decision_period_steps": 0},
        {"waypoint_scale_m": 0.0},
        {"minimum_waypoint_radius_m": 30.0},
        {"horizon_steps": 0},
    ):
        try:
            PrecaptureHybridConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
