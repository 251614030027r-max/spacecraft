import numpy as np

from dynamics.relative import relative_state
from env.scenarios import (
    sample_precapture_planning_chaser_state,
    target_initial_state,
)
from env.task import PrecaptureTaskConfig, compute_precapture_metrics


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
