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
    initial_potential = env._current_reward_potential()
    _, reward, terminated, truncated, info = env.step(np.array([-0.2, 0.0, 0.0]))
    assert not (terminated or truncated)
    assert info["hybrid_control_steps"] == env.hybrid_config.decision_period_steps
    assert np.isclose(
        env.env.time_seconds - started,
        env.hybrid_config.decision_period_steps * env.environment_config.dt_s,
    )
    assert np.isfinite(reward)
    expected_shaping = env.environment_config.precapture_reward.potential_weight * (
        env.hybrid_config.decision_discount_factor * info["reward_potential"]
        - initial_potential
    )
    assert np.isclose(info["hybrid_reward_shaping"], expected_shaping)
    assert np.isclose(
        reward,
        info["hybrid_integrated_reward_without_shaping"] + expected_shaping,
    )
    assert not np.isclose(
        info["hybrid_removed_micro_shaping"], expected_shaping
    )
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
        {"decision_discount_factor": 0.0},
        {"waypoint_scale_m": 0.0},
        {"minimum_waypoint_radius_m": 30.0},
        {"horizon_steps": 0},
    ):
        try:
            PrecaptureHybridConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_absolute_parametrisation_is_measured_unlearnable_and_kept_only_as_default() -> None:
    """The density that killed the first training run, pinned as a fact.

    Sampled uniformly, most of the absolute box commands a point further out
    than the chaser starts, and almost none of it commands anything near the
    goal. This is not a style complaint -- 400 episodes of SAC on it moved
    neither reward nor episode length. The test exists so the number cannot
    quietly drift back.
    """

    env = PrecaptureHybridEnv()
    env.reset(seed=262000)
    generator = np.random.default_rng(0)
    radii = np.array(
        [
            float(np.linalg.norm(env.waypoint_from_action(action)))
            for action in generator.uniform(-1.0, 1.0, size=(2000, 3))
        ]
    )
    assert float(np.mean(radii > 15.0)) > 0.6
    assert float(np.mean(radii < 6.0)) < 0.05
    env.close()


def test_radial_local_zero_action_holds_and_the_box_is_radially_unbiased() -> None:
    """Every command is a sane one, and none of them is a jump.

    Zero names the chaser's own present point, so the co-rotating hold is
    exactly the centre of the box and the inertially frozen hold -- the one
    that opens an entry window -- is the small lateral offset that undoes
    ``omega * 2 s``. Half the box commands inward, against 30% for the
    absolute parametrisation.
    """

    env = PrecaptureHybridEnv(
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="radial_local"
        )
    )
    env.reset(seed=262000)
    assert env.action_space.shape == (4,)
    position = env._current_position()
    assert np.allclose(env.waypoint_from_action(np.zeros(4)), position)

    radius = float(np.linalg.norm(position))
    inward = float(
        np.linalg.norm(env.waypoint_from_action(np.array([-1.0, 0.0, 0.0, 0.0])))
    )
    outward = float(
        np.linalg.norm(env.waypoint_from_action(np.array([1.0, 0.0, 0.0, 0.0])))
    )
    assert inward < radius < outward
    assert np.isclose(inward, radius * np.exp(-0.7), rtol=1e-9)

    generator = np.random.default_rng(0)
    radii = np.array(
        [
            float(np.linalg.norm(env.waypoint_from_action(action)))
            for action in generator.uniform(-1.0, 1.0, size=(2000, 4))
        ]
    )
    assert 0.4 < float(np.mean(radii < radius)) < 0.6
    env.close()


def test_radial_local_lateral_nudge_is_continuous_in_the_action() -> None:
    """No basis switching: a small action change is a small waypoint change.

    A lateral basis chosen from "the least aligned axis" would flip meaning
    across a switching surface, so the projection is taken directly against the
    current direction instead.
    """

    env = PrecaptureHybridEnv(
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="radial_local"
        )
    )
    env.reset(seed=262000)
    base = np.array([0.0, 0.3, -0.2, 0.1])
    previous = env.waypoint_from_action(base)
    for step in np.linspace(0.0, 0.4, 40)[1:]:
        current = env.waypoint_from_action(base + np.array([0.0, step, 0.0, 0.0]))
        assert float(np.linalg.norm(current - previous)) < 0.5
        previous = current
    env.close()


def test_action_for_waypoint_respects_each_parametrisation() -> None:
    absolute = PrecaptureHybridEnv(
        hybrid_config=PrecaptureHybridConfig(waypoint_parametrization="absolute")
    )
    absolute.reset(seed=262000)
    absolute_target = np.array([-6.0, 2.0, -1.0])
    assert np.allclose(
        absolute.waypoint_from_action(absolute.action_for_waypoint(absolute_target)),
        absolute_target,
    )
    absolute.close()

    radial = PrecaptureHybridEnv(
        hybrid_config=PrecaptureHybridConfig(waypoint_parametrization="radial_local")
    )
    radial.reset(seed=262000)
    current = radial._current_position()
    reachable_target = 0.75 * current
    action = radial.action_for_waypoint(reachable_target)
    assert action.shape == radial.action_space.shape == (4,)
    assert np.allclose(radial.waypoint_from_action(action), reachable_target)
    radial.close()


def test_hybrid_config_rejects_an_unknown_parametrisation() -> None:
    try:
        PrecaptureHybridConfig(waypoint_parametrization="polar")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
