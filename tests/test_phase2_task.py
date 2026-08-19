import numpy as np

from dynamics.lie import make_transform, so3_exp
from dynamics.relative import RelativeState
from env.task import Phase2MissionConfig, Phase2TaskConfig, compute_mission_metrics, compute_task_metrics


def _relative(position, rotation=None, omega=None, position_rate=None):
    rotation = np.eye(3) if rotation is None else rotation
    omega = np.zeros(3) if omega is None else omega
    position_rate = np.zeros(3) if position_rate is None else position_rate
    body_velocity = rotation.T @ position_rate
    return RelativeState(
        make_transform(rotation, np.asarray(position, dtype=float)),
        np.concatenate((omega, body_velocity)),
    )


def test_desired_state_is_feasible_and_complete() -> None:
    task = Phase2TaskConfig()
    metrics = compute_task_metrics(_relative(task.desired_position), task)
    assert metrics.constraints_satisfied
    assert metrics.instantaneous_completion
    assert np.allclose(metrics.desired_error_coordinates, 0.0)


def test_constraint_metric_signs_detect_independent_violations() -> None:
    task = Phase2TaskConfig()
    corridor_bad = compute_task_metrics(_relative([-6.0, 5.0, 0.0]), task)
    fov_bad = compute_task_metrics(
        _relative([-6.0, 0.0, 0.0], rotation=so3_exp([0.0, 0.0, np.pi])), task
    )
    speed_bad = compute_task_metrics(
        _relative([-6.0, 0.0, 0.0], position_rate=[0.5, 0.0, 0.0]), task
    )
    assert corridor_bad.corridor_lateral_margin_m < 0.0
    assert fov_bad.fov_margin_rad < 0.0
    assert speed_bad.total_speed_margin_m_s < 0.0
    assert speed_bad.closing_speed_margin_m_s < 0.0


def test_position_rate_matches_transform_finite_difference() -> None:
    rotation = so3_exp(np.array([0.2, -0.1, 0.3]))
    relative = _relative([-5.0, 1.0, -0.5], rotation=rotation, position_rate=[0.2, -0.1, 0.05])
    metrics = compute_task_metrics(relative)
    dt = 1.0e-7
    position_next = relative.position + dt * rotation @ relative.velocity
    finite_difference = (position_next - relative.position) / dt
    assert np.allclose(metrics.position_rate_target_m_s, finite_difference, atol=1e-8)


def test_closing_envelope_clamps_negative_remaining_distance() -> None:
    task = Phase2TaskConfig()
    assert task.closing_speed_limit(-2.0) == task.closing_speed_min_m_s
    assert task.closing_speed_limit(100.0) == task.closing_speed_max_m_s
    assert task.completion_required_steps(0.1) == 10


def test_gate_condition_is_a_state_event_not_terminal_completion() -> None:
    mission = Phase2MissionConfig()
    relative = _relative(mission.gate_position)
    metrics = compute_mission_metrics(relative, mission)
    assert mission.gate_satisfied(metrics)
    assert not compute_task_metrics(relative).instantaneous_completion


def test_acquisition_v2_gate_uses_position_speed_and_fov_only() -> None:
    mission = Phase2MissionConfig(
        gate_semantics="acquisition_v2",
        gate_position_tolerance_m=1.5,
        gate_speed_tolerance_m_s=0.30,
    )
    relative = _relative(
        mission.gate_position,
        rotation=so3_exp([0.0, np.deg2rad(35.0), 0.0]),
        omega=[0.10, 0.0, 0.0],
    )
    metrics = compute_mission_metrics(relative, mission)
    assert metrics.attitude_error_rad > mission.gate_attitude_tolerance_rad
    assert metrics.angular_velocity_error_rad_s > (
        mission.gate_angular_velocity_tolerance_rad_s
    )
    assert mission.gate_satisfied(metrics, fov_angle_rad=np.deg2rad(40.0))
    assert not mission.gate_satisfied(metrics, fov_angle_rad=np.deg2rad(50.0))


def test_manifest_lists_are_canonicalized_for_config_equality() -> None:
    task = Phase2TaskConfig(
        port_position_target_m=[-1.5, 0.0, 0.0],
        desired_position_target_m=[-3.0, 0.0, 0.0],
        approach_axis_target=[-1.0, 0.0, 0.0],
        camera_boresight_chaser=[1.0, 0.0, 0.0],
    )
    mission = Phase2MissionConfig(gate_position_target_m=[-8.0, 0.0, 0.0])
    assert task == Phase2TaskConfig()
    assert mission == Phase2MissionConfig()


def test_phase1_catastrophic_guard_is_outside_soft_speed_region() -> None:
    mission = Phase2MissionConfig()
    assert mission.gate_speed_tolerance_m_s < mission.phase1_soft_speed_m_s
    assert mission.phase1_soft_speed_m_s < mission.phase1_catastrophic_speed_limit_m_s
    assert mission.phase1_failure_penalty == mission.catastrophic_failure_penalty
    assert mission.phase1_failure_penalty < -mission.gate_reward
