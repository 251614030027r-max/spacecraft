import numpy as np
from dataclasses import replace

from dynamics.lie import make_transform
from dynamics.relative import RelativeState, reconstruct_chaser_state, relative_state
from env.phase2_env import precapture_planning_environment_config
from env.scenarios import (
    sample_precapture_planning_chaser_state,
    target_initial_state,
)
from env.task import (
    PrecaptureTaskConfig,
    compute_precapture_metrics,
    evaluate_terminal_entry_crossing,
)
from env.se3_rendezvous_env import SE3RendezvousEnv


def test_precapture_outer_braking_profile_is_self_consistent() -> None:
    task = PrecaptureTaskConfig()
    assert task.outer_inertial_speed_limit_m_s == 2.00
    assert task.entry_port_axial_distance_m == 4.5
    assert np.isclose(
        task.entry_disc_radius_m,
        task.entry_port_axial_distance_m * np.tan(task.corridor_half_angle_rad),
    )
    assert np.isclose(task.outer_radial_closing_speed_limit(20.0), np.sqrt(0.72))
    assert np.isclose(task.outer_radial_closing_speed_limit(15.0), np.sqrt(0.52))
    assert np.isclose(task.outer_radial_closing_speed_limit(10.0), np.sqrt(0.32))
    assert task.outer_radial_closing_speed_limit(6.0) == 0.4


def test_precapture_sampler_restores_non_corotating_outer_states() -> None:
    task = PrecaptureTaskConfig()
    target = target_initial_state(tumble_scale=0.20)
    target_frame_speeds = []
    relative_omega_errors = []
    for seed in range(100):
        chaser = sample_precapture_planning_chaser_state(
            np.random.default_rng(seed), target, task=task
        )
        relative = relative_state(target, chaser)
        metrics = compute_precapture_metrics(target, chaser, relative, task)
        assert np.all(np.isfinite(chaser.position))
        assert 15.0 <= metrics.target_center_distance_m <= 20.0
        assert metrics.port_axial_distance_m > task.entry_port_axial_distance_m
        assert metrics.port_axial_distance_m > 0.0
        assert metrics.corridor_lateral_margin_m < 0.0
        assert metrics.keepout_margin_m > 0.0
        assert metrics.fov_margin_rad >= 0.0
        assert metrics.outer_inertial_speed_margin_m_s >= 0.0
        assert metrics.active_constraints_satisfied
        target_frame_speeds.append(metrics.target_frame_speed_m_s)
        relative_omega_errors.append(metrics.angular_velocity_error_rad_s)

    assert np.median(target_frame_speeds) > 0.50
    assert np.median(relative_omega_errors) > task.completion_angular_velocity_rad_s


def test_outer_speed_uses_inertial_motion_not_rotating_frame_motion() -> None:
    task = PrecaptureTaskConfig()
    target = target_initial_state(tumble_scale=0.20)
    chaser = sample_precapture_planning_chaser_state(
        np.random.default_rng(260904), target, task=task
    )
    metrics = compute_precapture_metrics(
        target, chaser, relative_state(target, chaser), task
    )
    assert metrics.inertial_relative_speed_m_s <= 0.10
    assert metrics.target_frame_speed_m_s > metrics.inertial_relative_speed_m_s
    assert not metrics.transition_speed_active
    assert metrics.active_constraints_satisfied


def _corotating_chaser(target, position_target_m):
    relative = RelativeState(
        make_transform(np.eye(3), np.asarray(position_target_m, dtype=float)),
        np.zeros(6),
    )
    return reconstruct_chaser_state(target, relative)


def test_precapture_five_semantic_states() -> None:
    task = PrecaptureTaskConfig()
    target = target_initial_state(tumble_scale=0.20)
    outer = sample_precapture_planning_chaser_state(
        np.random.default_rng(10), target, task=task
    )
    outer_metrics = compute_precapture_metrics(
        target, outer, relative_state(target, outer), task
    )
    assert outer_metrics.active_constraints_satisfied
    assert not outer_metrics.terminal_region_active

    outer_fast = outer.copy()
    outer_fast.velocity += outer_fast.rotation.T @ np.array([2.1, 0.0, 0.0])
    outer_fast_metrics = compute_precapture_metrics(
        target, outer_fast, relative_state(target, outer_fast), task
    )
    assert outer_fast_metrics.outer_inertial_speed_margin_m_s < 0.0
    assert not outer_fast_metrics.active_constraints_satisfied

    terminal = _corotating_chaser(target, [-7.0, 0.0, 0.0])
    terminal_metrics = compute_precapture_metrics(
        target,
        terminal,
        relative_state(target, terminal),
        task,
        terminal_region_active=True,
    )
    assert terminal_metrics.active_constraints_satisfied

    terminal_bad = _corotating_chaser(target, [-7.0, 5.0, 0.0])
    terminal_bad_metrics = compute_precapture_metrics(
        target,
        terminal_bad,
        relative_state(target, terminal_bad),
        task,
        terminal_region_active=True,
    )
    assert terminal_bad_metrics.corridor_lateral_margin_m < 0.0
    assert not terminal_bad_metrics.active_constraints_satisfied

    completed = _corotating_chaser(target, task.desired_position)
    completed_metrics = compute_precapture_metrics(
        target,
        completed,
        relative_state(target, completed),
        task,
        terminal_region_active=True,
    )
    assert completed_metrics.instantaneous_completion
    assert completed_metrics.active_constraints_satisfied


def _entry_metrics(target, task, position, velocity):
    relative = RelativeState(
        make_transform(np.eye(3), np.asarray(position, dtype=float)),
        np.concatenate((np.zeros(3), np.asarray(velocity, dtype=float))),
    )
    chaser = reconstruct_chaser_state(target, relative)
    return compute_precapture_metrics(target, chaser, relative_state(target, chaser), task)


def test_entry_disc_crossing_semantics() -> None:
    task = PrecaptureTaskConfig()
    target = target_initial_state(tumble_scale=0.0)
    previous = _entry_metrics(target, task, [-6.01, 0.0, 0.0], [0.20, 0.0, 0.0])
    legal = _entry_metrics(target, task, [-5.99, 0.0, 0.0], [0.20, 0.0, 0.0])
    result = evaluate_terminal_entry_crossing(previous, legal, task)
    assert result.crossed and result.legal
    assert result.radial_distance_m == 0.0

    outside_disc_previous = _entry_metrics(
        target, task, [-6.01, task.entry_disc_radius_m + 0.1, 0.0], [0.20, 0.0, 0.0]
    )
    outside_disc_current = _entry_metrics(
        target, task, [-5.99, task.entry_disc_radius_m + 0.1, 0.0], [0.20, 0.0, 0.0]
    )
    illegal = evaluate_terminal_entry_crossing(
        outside_disc_previous, outside_disc_current, task
    )
    assert illegal.crossed and not illegal.legal

    already_inside = _entry_metrics(
        target, task, [-5.8, 0.0, 0.0], [0.10, 0.0, 0.0]
    )
    assert not evaluate_terminal_entry_crossing(legal, already_inside, task).crossed


def test_environment_latches_only_on_legal_entry_disc_crossing() -> None:
    config = replace(
        precapture_planning_environment_config(),
        cache_target_trajectory=False,
        phase2_target_phase_sampling=False,
        include_j2=False,
    )
    target = target_initial_state(tumble_scale=0.0)
    relative = RelativeState(
        make_transform(np.eye(3), np.array([-6.005, 0.0, 0.0])),
        np.array([0.0, 0.0, 0.0, 0.10, 0.0, 0.0]),
    )
    chaser = reconstruct_chaser_state(target, relative)
    env = SE3RendezvousEnv(config)
    _, reset_info = env.reset(
        seed=11, options={"target_state": target, "chaser_state": chaser}
    )
    assert not reset_info["terminal_region_active"]
    _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["terminal_region_active"]
    assert env._terminal_region_entered
    assert not terminated
    assert not truncated
    retreat = RelativeState(
        make_transform(np.eye(3), np.array([-6.10, 0.0, 0.0])),
        np.array([0.0, 0.0, 0.0, -0.10, 0.0, 0.0]),
    )
    env.chaser_state = reconstruct_chaser_state(env.target_state, retreat)
    env.relative = relative_state(env.target_state, env.chaser_state)
    _, _, _, _, retreat_info = env.step(np.zeros(6, dtype=np.float32))
    assert retreat_info["terminal_region_active"]
    assert env._terminal_region_entered
    env.close()


def test_illegal_entry_is_counted_without_latching_or_termination() -> None:
    config = replace(
        precapture_planning_environment_config(),
        cache_target_trajectory=False,
        phase2_target_phase_sampling=False,
        include_j2=False,
    )
    target = target_initial_state(tumble_scale=0.0)
    lateral = config.precapture_task.entry_disc_radius_m + 0.1
    relative = RelativeState(
        make_transform(np.eye(3), np.array([-6.005, lateral, 0.0])),
        np.array([0.0, 0.0, 0.0, 0.10, 0.0, 0.0]),
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(
            seed=12,
            options={
                "target_state": target,
                "chaser_state": reconstruct_chaser_state(target, relative),
            },
        )
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["illegal_terminal_entry"]
        assert info["illegal_terminal_entry_count"] == 1
        assert not info["terminal_region_active"]
        assert not terminated and not truncated
    finally:
        env.close()


def test_precapture_completion_requires_a_prior_legal_latch() -> None:
    config = replace(
        precapture_planning_environment_config(),
        cache_target_trajectory=False,
        phase2_target_phase_sampling=False,
        include_j2=False,
    )
    target = target_initial_state(tumble_scale=0.0)
    chaser = _corotating_chaser(target, config.precapture_task.desired_position)
    env = SE3RendezvousEnv(config)
    try:
        env.reset(
            seed=13,
            options={"target_state": target, "chaser_state": chaser},
        )
        for _ in range(config.precapture_task.completion_required_steps(config.dt_s) + 1):
            _, _, terminated, truncated, info = env.step(
                np.zeros(6, dtype=np.float32)
            )
            assert not terminated and not truncated
        assert info["joint_success"]
        assert not info["completed"]
        assert not info["terminal_region_active"]
    finally:
        env.close()


def test_precapture_episode_runs_at_the_configured_tumble_rate() -> None:
    """The task must tumble at its own config's rate, not the class default.

    ``_select_phase2_stage`` returns early for the precapture task, so the
    episode used to keep the constructor's ``target_tumble_scale`` (0.5) and run
    at 0.1031 rad/s while ``precapture_planning_environment_config`` asked for
    0.20. That is 2.5x the frozen S1-v2 rate, and it makes sustained co-rotation
    need ``m * omega^2 * r = 1.13 N`` per metre -- past the 8.66 N body diagonal
    beyond 7.7 m, so no controller can hold the outer region at all. Every
    documented number in this project is measured at 0.0412 rad/s.
    """

    config = precapture_planning_environment_config()
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=262000)
        assert env._episode_tumble_scale == config.phase2_target_tumble_scale
        rate = float(np.linalg.norm(env.target_state.omega))
        assert np.isclose(rate, 0.04123105625617661)
        assert env._episode_target_phase_seed == 262000
        # The co-rotation force the outer staging radius demands stays inside
        # the guaranteed single-axis authority, which is what makes the task
        # solvable by something other than a body-diagonal manoeuvre.
        chaser_mass = env.chaser_parameters.mass
        assert chaser_mass * rate * rate * 7.5 < config.max_force_per_axis_n
    finally:
        env.close()


def test_precapture_phase_sampling_uses_episode_seed_and_varies_beta() -> None:
    config = precapture_planning_environment_config()
    env = SE3RendezvousEnv(config)
    samples = []
    try:
        for seed in (262000, 262001, 262002):
            env.reset(seed=seed)
            assert env._episode_target_phase_seed == seed
            rate = float(np.linalg.norm(env.target_state.omega))
            assert np.isclose(rate, 0.04123105625617661)
            angular_momentum = env.target_state.rotation @ (
                env.target_parameters.inertia @ env.target_state.omega
            )
            angular_momentum /= np.linalg.norm(angular_momentum)
            axis_inertial = (
                env.target_state.rotation @ config.precapture_task.approach_axis
            )
            beta = float(
                np.arccos(np.clip(angular_momentum @ axis_inertial, -1.0, 1.0))
            )
            samples.append((env.target_state.rotation.copy(), beta))
        env.reset(seed=262000)
        assert np.array_equal(env.target_state.rotation, samples[0][0])
        assert len({round(beta, 10) for _, beta in samples}) == len(samples)
    finally:
        env.close()


def test_unsustainable_outer_rotation_ends_as_distance_failure() -> None:
    """At 16 m, omega'=0.08 needs more centripetal force than one 5 N axis."""

    config = replace(
        precapture_planning_environment_config(),
        max_time_s=60.0,
        cache_target_trajectory=False,
        phase2_target_phase_sampling=False,
        precapture_fov_violation_terminates=False,
        include_j2=False,
    )
    target = target_initial_state(tumble_scale=0.0)
    commanded_rate = 0.08
    relative = RelativeState(
        make_transform(np.eye(3), np.array([-16.0, 0.0, 0.0])),
        np.array([0.0, 0.0, commanded_rate, 0.0, -16.0 * commanded_rate, 0.0]),
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(
            seed=99,
            options={
                "target_state": target,
                "chaser_state": reconstruct_chaser_state(target, relative),
            },
        )
        # With no feasible centripetal command supplied, this is the limiting
        # outward branch a short-sighted controller can expose. The 2 m/s
        # numerical guard must not hide it as an outer-speed violation.
        action = np.zeros(6, dtype=np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(action)
        assert info["distance_failure"]
        assert not info["outer_speed_failure"]
        assert 15.0 <= info["time_seconds"] <= 30.0
    finally:
        env.close()


def test_open_loop_plan_decouples_outer_descent_from_coast_wait() -> None:
    from experiments.evaluate_precapture_oracle import (
        MATCH_TIME_S,
        OUTER_STAGING_RADIUS_M,
        CoastThenMatchPlan,
    )

    env = SE3RendezvousEnv(precapture_planning_environment_config())
    try:
        env.reset(seed=262000)
        plan = CoastThenMatchPlan(
            env,
            coast_min_time_s=160.0,
            coast_max_time_s=220.0,
            outer_descent_time_s=60.0,
        )
        assert plan.match_start_time_s >= 60.0
        assert plan.coast_time_s >= 160.0
        assert plan.coast_time_s <= 220.0
        assert plan.coast_time_s - plan.match_start_time_s == MATCH_TIME_S
        assert np.isclose(plan._radius(60.0), OUTER_STAGING_RADIUS_M)
        assert np.isclose(plan._radius(plan.match_start_time_s), OUTER_STAGING_RADIUS_M)
    finally:
        env.close()


def test_open_loop_plan_rejects_search_before_descent_and_match_fit() -> None:
    from experiments.evaluate_precapture_oracle import CoastThenMatchPlan

    env = SE3RendezvousEnv(precapture_planning_environment_config())
    try:
        env.reset(seed=262000)
        with np.testing.assert_raises(ValueError):
            CoastThenMatchPlan(
                env,
                coast_min_time_s=20.0,
                coast_max_time_s=100.0,
                outer_descent_time_s=80.0,
            )
    finally:
        env.close()
