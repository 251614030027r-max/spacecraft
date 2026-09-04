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


def test_precapture_speed_transition_is_continuous_and_configured() -> None:
    task = PrecaptureTaskConfig()
    assert task.target_frame_speed_limit(14.0) == 1.10
    assert np.isclose(task.target_frame_speed_limit(11.0), 0.725)
    assert task.target_frame_speed_limit(8.0) == 0.35
    assert task.target_frame_speed_limit(20.0) == 1.10
    assert task.target_frame_speed_limit(3.0) == 0.35


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
    outer_fast.velocity += outer_fast.rotation.T @ np.array([0.6, 0.0, 0.0])
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
    chaser = _corotating_chaser(target, [-7.5, 0.0, 0.0])
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
