import numpy as np
from dataclasses import replace

from dynamics.lie import make_transform
from dynamics.relative import RelativeState, reconstruct_chaser_state, relative_state
from env.phase2_env import precapture_planning_environment_config
from env.scenarios import (
    sample_precapture_planning_chaser_state,
    target_initial_state,
)
from env.task import PrecaptureTaskConfig, compute_precapture_metrics
from env.se3_rendezvous_env import SE3RendezvousEnv


def test_precapture_outer_braking_profile_is_self_consistent() -> None:
    task = PrecaptureTaskConfig()
    assert task.outer_inertial_speed_limit_m_s == 1.20
    assert task.terminal_activation_range_m == 6.0
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
        assert metrics.target_center_distance_m > task.terminal_activation_range_m
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
    outer_fast.velocity += outer_fast.rotation.T @ np.array([1.3, 0.0, 0.0])
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


def test_environment_latches_terminal_region_on_entry() -> None:
    config = replace(
        precapture_planning_environment_config(), cache_target_trajectory=False
    )
    target = target_initial_state(tumble_scale=config.phase2_target_tumble_scale)
    chaser = _corotating_chaser(target, [-5.5, 0.0, 0.0])
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
        expected = target_initial_state(
            tumble_scale=config.phase2_target_tumble_scale
        )
        assert np.allclose(env.target_state.omega, expected.omega)
        rate = float(np.linalg.norm(env.target_state.omega))
        assert np.isclose(rate, 0.04123105625617661)
        # The co-rotation force the outer staging radius demands stays inside
        # the guaranteed single-axis authority, which is what makes the task
        # solvable by something other than a body-diagonal manoeuvre.
        chaser_mass = env.chaser_parameters.mass
        assert chaser_mass * rate * rate * 7.5 < config.max_force_per_axis_n
    finally:
        env.close()
