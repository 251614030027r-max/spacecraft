"""What the coupling wrapper promises, pinned.

The three properties worth a test are the three that make a result mean
something: the wrapper does not change what the policy sees, the action is an
absolute waypoint in the frame the channel is lossless in, and one decision is
exactly the hold the optimiser was configured for.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from controllers.mpc.prediction import relative_to_vector
from env.hybrid_env import (
    PrecaptureHybridConfig,
    PrecaptureHybridEnv,
    hybrid_mpc_config,
)
from env.phase2_env import precapture_planning_environment_config
from env.se3_rendezvous_env import SE3RendezvousEnv
from experiments.evaluate_hybrid_policy import (
    summarize_entry_channel,
    truth_violation_step_counts,
)
from train.train_hybrid import accelerated_training_configs


def test_hybrid_action_is_three_dimensional_and_observation_is_unchanged() -> None:
    """The learned layer sees exactly what Pure SAC sees, and names a place."""

    env = PrecaptureHybridEnv()
    assert env.action_space.shape == (3,)
    assert env.observation_space == env.env.observation_space
    observation, _ = env.reset(seed=262000)
    assert env.observation_space.contains(observation.astype(np.float32))
    env.close()


def test_truth_violation_step_counts_preserve_missing_fields() -> None:
    counts = truth_violation_step_counts(
        {"keepout_violation_steps": 3, "fov_violation_steps": 0}
    )
    assert counts["keepout_violation_steps"] == 3
    assert counts["fov_violation_steps"] == 0
    assert counts["corridor_violation_steps"] is None


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


def test_runtime_diagnostics_switch_does_not_change_the_mpc_command() -> None:
    environment = replace(
        precapture_planning_environment_config(), cache_target_trajectory=False
    )
    enabled = PrecaptureHybridEnv(
        environment_config=environment,
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="radial_local", runtime_diagnostics=True
        ),
    )
    disabled = PrecaptureHybridEnv(
        environment_config=environment,
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="radial_local", runtime_diagnostics=False
        ),
    )
    try:
        _, enabled_info = enabled.reset(seed=262000)
        _, disabled_info = disabled.reset(seed=262000)
        action = np.zeros(4)
        enabled_waypoint = enabled.waypoint_from_action(action)
        disabled_waypoint = disabled.waypoint_from_action(action)
        assert np.array_equal(enabled_waypoint, disabled_waypoint)
        enabled_command, enabled_diagnostics = enabled.controller.command(
            relative_to_vector(enabled.env.relative),
            target_state=enabled.env.target_state,
            time_seconds=enabled.env.time_seconds,
            terminal_latched=bool(enabled_info["terminal_region_active"]),
            external_reference=enabled_waypoint,
        )
        disabled_command, disabled_diagnostics = disabled.controller.command(
            relative_to_vector(disabled.env.relative),
            target_state=disabled.env.target_state,
            time_seconds=disabled.env.time_seconds,
            terminal_latched=bool(disabled_info["terminal_region_active"]),
            external_reference=disabled_waypoint,
        )
        assert np.array_equal(enabled_command, disabled_command)
        assert enabled_diagnostics.status == disabled_diagnostics.status
        assert (
            enabled_diagnostics.used_zero_fallback
            == disabled_diagnostics.used_zero_fallback
        )
    finally:
        enabled.close()
        disabled.close()


def test_on_demand_target_propagation_matches_eager_cache_bitwise() -> None:
    base = replace(
        precapture_planning_environment_config(),
        max_time_s=0.3,
        phase2_target_phase_sampling=True,
    )
    eager = SE3RendezvousEnv(replace(base, cache_target_trajectory=True))
    on_demand = SE3RendezvousEnv(replace(base, cache_target_trajectory=False))
    try:
        eager.reset(seed=262000)
        on_demand.reset(seed=262000)
        action = np.zeros(6)
        for _ in range(3):
            _, _, _, _, eager_info = eager.step(action)
            _, _, _, _, on_demand_info = on_demand.step(action)
            # This is expected bitwise equality, not an empirical tolerance: the
            # two branches call the same propagate_rk45 with identical target
            # parameters, gravity, RK45Settings, dt_s and step time stamps.
            assert np.array_equal(eager.target_state.position, on_demand.target_state.position)
            assert np.array_equal(eager.target_state.velocity, on_demand.target_state.velocity)
            assert np.array_equal(eager.target_state.rotation, on_demand.target_state.rotation)
            assert np.array_equal(eager.target_state.omega, on_demand.target_state.omega)
            assert eager_info["target_nfev"] == 0
            assert on_demand_info["target_nfev"] > 0
    finally:
        eager.close()
        on_demand.close()


def test_hybrid_training_selects_only_engineering_acceleration_switches() -> None:
    environment, hybrid = accelerated_training_configs(
        horizon_steps=20, waypoint_parametrization="radial_local"
    )
    assert not environment.cache_target_trajectory
    assert not hybrid.runtime_diagnostics
    assert hybrid.include_target_phase_and_time_observation
    assert not hybrid_mpc_config(hybrid, environment).runtime_diagnostics
    canonical = precapture_planning_environment_config()
    assert canonical.cache_target_trajectory
    assert PrecaptureHybridConfig().runtime_diagnostics
    assert not PrecaptureHybridConfig().include_target_phase_and_time_observation


def test_v4_observation_adds_target_phase_and_remaining_time_only() -> None:
    environment = replace(
        precapture_planning_environment_config(), cache_target_trajectory=False
    )
    legacy = PrecaptureHybridEnv(
        environment_config=environment,
        hybrid_config=PrecaptureHybridConfig(waypoint_parametrization="radial_local"),
    )
    v4 = PrecaptureHybridEnv(
        environment_config=environment,
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="radial_local",
            include_target_phase_and_time_observation=True,
        ),
    )
    try:
        legacy_observation, _ = legacy.reset(seed=262000)
        v4_observation, _ = v4.reset(seed=262000)
        assert legacy_observation.shape == (24,)
        assert v4_observation.shape == (31,)
        np.testing.assert_array_equal(v4_observation[:24], legacy_observation)
        np.testing.assert_array_equal(
            v4_observation[24:30],
            v4.env.target_state.rotation[:, :2].reshape(-1).astype(np.float32),
        )
        assert v4_observation[-1] == 1.0
        next_observation, _, _, _, _ = v4.step(np.zeros(4))
        assert next_observation[-1] < 1.0
        assert v4.observation_space.contains(next_observation)
    finally:
        legacy.close()
        v4.close()


def test_v4_phase_encoding_distinguishes_sampled_absolute_target_attitude() -> None:
    config = PrecaptureHybridConfig(
        waypoint_parametrization="radial_local",
        include_target_phase_and_time_observation=True,
    )
    environment = replace(
        precapture_planning_environment_config(), cache_target_trajectory=False
    )
    env = PrecaptureHybridEnv(environment_config=environment, hybrid_config=config)
    try:
        first, _ = env.reset(seed=262000)
        second, _ = env.reset(seed=262001)
        assert not np.array_equal(first[24:30], second[24:30])
        assert first[-1] == second[-1] == 1.0
    finally:
        env.close()


def test_entry_channel_summary_applies_preregistered_retry_rule() -> None:
    records = [
        {
            "illegal_entry_crossing_count": 1,
            "illegal_exit_count": 1,
            "retry_crossing_count": 1,
            "entry_crossings": [
                {
                    "legal": False,
                    "retry_after_illegal": False,
                    "remaining_time_s": 220.0,
                    "violation_excess": {
                        "radial_distance_m": 1.0,
                        "target_frame_speed_m_s": -0.1,
                        "closing_speed_m_s": -0.05,
                    },
                },
                {
                    "legal": True,
                    "retry_after_illegal": True,
                    "remaining_time_s": 80.0,
                    "violation_excess": {
                        "radial_distance_m": -0.2,
                        "target_frame_speed_m_s": -0.1,
                        "closing_speed_m_s": -0.05,
                    },
                },
            ],
        }
    ]
    summary = summarize_entry_channel(
        records,
        {
            "radial_distance_m": 3.151,
            "target_frame_speed_m_s": 0.35,
            "closing_speed_m_s": 0.20,
        },
    )
    assert summary["primary_illegal_cause_by_preregistered_60pct_rule"] == (
        "radial_distance"
    )
    assert summary["episodes_with_exit_and_retry"] == 1
    assert summary["legal_retry_crossings"] == 1


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


def _arrival_env(**changes: object) -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="arrival_condition", **changes
        )
    )


def test_arrival_condition_is_two_dimensional() -> None:
    env = _arrival_env()
    assert env.action_space.shape == (2,)
    env.close()


def test_arrival_condition_commit_end_names_the_desired_pose() -> None:
    """The capability floor: the commit end is the fixed-setpoint controller.

    Every radius setting has to give the same point, because the hold radius
    is meaningless once the commit is complete. This is the property that
    makes ``a = (+1, *)`` reproduce the Pure MPC row rather than approximate
    it, and a wrench-level equivalence check rests on it.
    """

    env = _arrival_env()
    env.reset(seed=262000)
    desired = env.environment_config.precapture_task.desired_position
    for radius_action in (-1.0, -0.3, 0.0, 0.5, 1.0):
        waypoint = env.waypoint_from_action(np.array([1.0, radius_action]))
        assert np.array_equal(waypoint, desired)
    env.close()


def test_arrival_condition_hold_end_is_inertially_frozen() -> None:
    """Waiting has to be inertial, not target-frame.

    The approach axis is body-fixed, so a chaser holding a fixed target-frame
    point co-rotates with it and the entry geometry never changes -- there is
    no window to wait for, only fuel to spend. So the hold command has to be a
    *different* target-frame point on every decision, tracing one frozen
    inertial direction.
    """

    env = _arrival_env()
    env.reset(seed=262000)
    hold = np.array([-1.0, 0.0])
    directions_target = []
    directions_inertial = []
    for _ in range(3):
        waypoint = env.waypoint_from_action(hold)
        directions_target.append(waypoint / np.linalg.norm(waypoint))
        rotation = env.env.target_state.rotation
        directions_inertial.append(rotation @ directions_target[-1])
        env.step(hold)
    inertial = np.array(directions_inertial)
    target_frame = np.array(directions_target)
    # One frozen inertial direction ...
    assert np.allclose(inertial, inertial[0], atol=1.0e-9)
    # ... which is a moving point in the frame the channel speaks.
    assert not np.allclose(target_frame[-1], target_frame[0], atol=1.0e-3)
    env.close()


def test_arrival_condition_radius_channel_is_monotone_and_bracketed() -> None:
    env = _arrival_env()
    env.reset(seed=262000)
    radii = [
        float(np.linalg.norm(env.waypoint_from_action(np.array([-1.0, value]))))
        for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ]
    assert all(a < b for a, b in zip(radii, radii[1:]))
    assert radii[2] == max(
        min(
            float(np.linalg.norm(env._current_position())),
            env.hybrid_config.maximum_waypoint_radius_m,
        ),
        env.hybrid_config.minimum_waypoint_radius_m,
    )
    env.close()


def test_execution_feedback_is_appended_and_bounded() -> None:
    env = _arrival_env(include_execution_feedback_observation=True)
    base = PrecaptureHybridEnv(
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="arrival_condition"
        )
    )
    assert env.observation_space.shape[0] == base.observation_space.shape[0] + 3
    observation, _ = env.reset(seed=262000)
    assert np.array_equal(observation[-3:], np.zeros(3))
    for _ in range(3):
        observation, _, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        feedback = observation[-3:]
        assert np.all(feedback >= 0.0) and np.all(feedback <= 1.0)
        assert info["hybrid_feedback_fallback_fraction"] == feedback[0]
        assert info["hybrid_feedback_solved_peak_slack"] == feedback[1]
        assert info["hybrid_feedback_mean_actuator_usage"] == feedback[2]
        # The slack summary is only ever read from steps that actually solved;
        # on a fallback step the solver variable can still hold the previous
        # solve's numbers.
        if info["hybrid_feedback_solved_steps"] == 0:
            assert feedback[1] == 0.0
        if terminated or truncated:
            break
    env.close()
    base.close()


def test_feedback_flag_off_leaves_the_observation_untouched() -> None:
    without = _arrival_env()
    with_feedback = _arrival_env(include_execution_feedback_observation=True)
    plain, _ = without.reset(seed=262000)
    augmented, _ = with_feedback.reset(seed=262000)
    assert np.array_equal(augmented[: plain.size], plain)
    without.close()
    with_feedback.close()


def _ratchet_env(monotone: bool = True) -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(
        environment_config=precapture_planning_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=20,
            waypoint_parametrization="arrival_condition",
            monotone_commit=monotone,
            runtime_diagnostics=False,
        ),
    )


def test_commit_blend_only_advances() -> None:
    """The ratchet is the whole of S8: a commanded retreat is a hold."""

    env = _ratchet_env()
    env.reset(seed=262000)
    seen = []
    for action in (-0.5, 0.2, -0.9, 0.7, -1.0):
        observation, _, _, _, _ = env.step(
            np.array([action, 0.0], dtype=np.float32)
        )
        seen.append(float(observation[-1]))
    assert seen == sorted(seen)
    assert seen == pytest.approx([0.25, 0.6, 0.6, 0.85, 0.85], abs=1e-6)


def test_ratchet_level_is_in_the_observation_and_resets() -> None:
    """Without it the same action means different things and the MDP is broken."""

    env = _ratchet_env()
    observation, _ = env.reset(seed=262000)
    assert env.observation_space.shape == (observation.shape[0],)
    assert float(observation[-1]) == 0.0
    env.step(np.array([0.5, 0.0], dtype=np.float32))
    observation, _ = env.reset(seed=262001)
    assert float(observation[-1]) == 0.0


def test_disabling_the_ratchet_restores_the_old_observation_width() -> None:
    with_ratchet, _ = _ratchet_env(True).reset(seed=262000)
    without, _ = _ratchet_env(False).reset(seed=262000)
    assert with_ratchet.shape[0] == without.shape[0] + 1
    assert np.allclose(with_ratchet[:-1], without)


def test_the_ratchet_does_not_move_the_commit_corner() -> None:
    """``a = (+1, *)`` must still name the desired pose on every decision."""

    task = precapture_planning_environment_config().precapture_task
    for radius_action in (-1.0, 0.0, 1.0):
        env = _ratchet_env()
        env.reset(seed=262004)
        for _ in range(5):
            waypoint = env.waypoint_from_action(
                np.array([1.0, radius_action], dtype=np.float64)
            )
            assert np.allclose(waypoint, task.desired_position)
            env.step(np.array([1.0, radius_action], dtype=np.float32))
