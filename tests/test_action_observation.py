import numpy as np

from dynamics.lie import make_transform, se3_exp
from dynamics.relative import RelativeState
from env.action import normalized_to_wrench, wrench_to_normalized
from env.observation import (
    build_observation,
    build_phase2_mission_observation,
    build_phase2_observation,
)
from env.reward import Phase2MissionReward
from env.task import Phase2TaskConfig


def test_action_mapping_roundtrip_and_clipping() -> None:
    action = np.array([0.5, -0.5, 2.0, -2.0, 0.25, -0.25])
    wrench = normalized_to_wrench(action)
    assert np.allclose(wrench.torque, [0.3, -0.3, 0.6])
    assert np.allclose(wrench.force, [-5.0, 1.25, -1.25])
    assert np.allclose(
        wrench_to_normalized(wrench), np.clip(action, -1.0, 1.0)
    )


def test_observation_is_finite_bounded_and_monotone() -> None:
    relative = RelativeState(
        se3_exp(np.array([0.2, -0.1, 0.3, 20.0, -5.0, 2.0])),
        np.array([0.1, -0.05, 0.02, 1.0, -0.3, 0.2]),
    )
    observation = build_observation(
        relative,
        attitude_scale_rad=np.deg2rad(30.0),
        distance_scale_m=3.0,
        angular_velocity_scale_rad_s=0.05,
        velocity_scale_m_s=0.2,
    )
    assert observation.shape == (12,)
    assert np.all(np.isfinite(observation))
    assert np.max(np.abs(observation)) < 10.0


def _phase2_observation(relative: RelativeState, target_omega: np.ndarray) -> np.ndarray:
    return build_phase2_observation(
        relative,
        task=Phase2TaskConfig(),
        attitude_scale_rad=np.deg2rad(60.0),
        distance_scale_m=5.0,
        angular_velocity_scale_rad_s=0.03,
        velocity_scale_m_s=0.35,
        target_angular_velocity_rad_s=target_omega,
    )


def test_phase2_v2_1_observation_exposes_direction_and_target_motion() -> None:
    left = RelativeState(
        make_transform(np.eye(3), np.array([-5.0, 0.2, 0.0])), np.zeros(6)
    )
    right = RelativeState(
        make_transform(np.eye(3), np.array([-5.0, -0.2, 0.0])), np.zeros(6)
    )
    zero = np.zeros(3)
    left_observation = _phase2_observation(left, zero)
    right_observation = _phase2_observation(right, zero)
    moving_target = _phase2_observation(left, np.array([0.01, -0.02, 0.03]))
    assert left_observation.shape == (23,)
    assert np.allclose(left_observation[15:19], -right_observation[15:19])
    assert np.allclose(left_observation[12:15], 0.0)
    assert not np.allclose(moving_target[12:15], 0.0)


def test_phase2_v2_1_desired_pose_has_zero_task_error_channels() -> None:
    task = Phase2TaskConfig()
    desired = RelativeState(task.desired_transform, np.zeros(6))
    observation = _phase2_observation(desired, np.zeros(3))
    assert np.allclose(observation[:12], 0.0, atol=1.0e-7)
    assert np.all(np.isfinite(observation))


def test_phase2_v2_1_observation_stays_bounded_after_port_plane_escape() -> None:
    escaped = RelativeState(
        make_transform(np.eye(3), np.array([2.0, 1.0e3, -1.0e3])), np.zeros(6)
    )
    observation = _phase2_observation(escaped, np.zeros(3))
    assert observation.shape == (23,)
    assert np.all(np.isfinite(observation))
    assert np.max(np.abs(observation)) < 10.0


def test_phase2_mission_observation_has_active_reference_and_phase_flag() -> None:
    task = Phase2TaskConfig()
    relative = RelativeState(
        make_transform(np.eye(3), np.array([-8.0, 0.0, 0.0])), np.zeros(6)
    )
    common = dict(
        relative=relative,
        task=task,
        attitude_scale_rad=np.deg2rad(60.0),
        distance_scale_m=10.0,
        angular_velocity_scale_rad_s=0.03,
        velocity_scale_m_s=0.50,
    )
    phase1 = build_phase2_mission_observation(
        **common,
        active_reference_position_m=np.array([-8.0, 0.0, 0.0]),
        mission_phase=0,
    )
    phase2 = build_phase2_mission_observation(
        **common,
        active_reference_position_m=task.desired_position,
        mission_phase=1,
    )
    assert phase1.shape == phase2.shape == (24,)
    assert np.allclose(phase1[3:6], 0.0)
    assert not np.allclose(phase2[3:6], 0.0)
    assert phase1[-1] == 0.0 and phase2[-1] == 1.0
    assert np.all(np.isfinite(phase1)) and np.all(np.isfinite(phase2))


def test_phase2_mission_translation_is_expressed_in_chaser_body_frame() -> None:
    rotation = se3_exp(np.array([0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0]))[:3, :3]
    position_target = np.array([0.0, 2.0, 0.0])
    rate_target = np.array([0.0, 0.5, 0.0])
    relative = RelativeState(
        make_transform(rotation, position_target),
        np.concatenate((np.zeros(3), rotation.T @ rate_target)),
    )
    observation = build_phase2_mission_observation(
        relative,
        task=Phase2TaskConfig(),
        active_reference_position_m=np.zeros(3),
        mission_phase=0,
        attitude_scale_rad=1.0,
        distance_scale_m=1.0,
        angular_velocity_scale_rad_s=1.0,
        velocity_scale_m_s=1.0,
    )
    expected_position = rotation.T @ position_target
    expected_rate = rotation.T @ rate_target
    expected = 10.0 * np.concatenate((expected_position, expected_rate)) / (
        10.0 + np.abs(np.concatenate((expected_position, expected_rate)))
    )
    assert np.allclose(observation[3:9], expected, atol=1.0e-7)


def test_phase2_mission_translation_is_unchanged_at_equal_attitude() -> None:
    position_target = np.array([1.0, -2.0, 0.5])
    rate_target = np.array([-0.2, 0.1, 0.3])
    relative = RelativeState(
        make_transform(np.eye(3), position_target),
        np.concatenate((np.zeros(3), rate_target)),
    )
    common = dict(
        relative=relative,
        task=Phase2TaskConfig(),
        active_reference_position_m=np.zeros(3),
        mission_phase=0,
        attitude_scale_rad=1.0,
        distance_scale_m=1.0,
        angular_velocity_scale_rad_s=1.0,
        velocity_scale_m_s=1.0,
    )
    body = build_phase2_mission_observation(**common)
    target = build_phase2_mission_observation(
        **common, translational_observation_frame="target"
    )
    assert np.allclose(body[3:9], target[3:9], atol=1.0e-7)


def test_velocity_tracking_error_is_zero_at_phase1_desired_velocity() -> None:
    task = Phase2TaskConfig()
    reward = Phase2MissionReward(task=task, phase1_cruise_speed_m_s=0.15)
    reference = np.array([-8.0, 0.0, 0.0])
    position = np.array([-10.0, 0.0, 0.0])
    stationary = RelativeState(
        make_transform(np.eye(3), position), np.zeros(6)
    )
    desired = reward.phase1_desired_velocity(stationary, reference)
    relative = RelativeState(
        make_transform(np.eye(3), position),
        np.concatenate((np.zeros(3), desired)),
    )
    observation = build_phase2_mission_observation(
        relative,
        task=task,
        active_reference_position_m=reference,
        active_reference_velocity_target_m_s=desired,
        mission_phase=0,
        attitude_scale_rad=1.0,
        distance_scale_m=1.0,
        angular_velocity_scale_rad_s=1.0,
        velocity_scale_m_s=1.0,
        translational_velocity_observation="tracking_error",
    )
    assert np.allclose(observation[6:9], 0.0, atol=1.0e-7)


def test_body_velocity_tracking_error_uses_verified_inverse_rotation() -> None:
    task = Phase2TaskConfig()
    rotation = se3_exp(
        np.array([0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])
    )[:3, :3]
    desired_target = np.array([0.15, 0.0, 0.0])
    actual_target = np.array([0.30, 0.0, 0.0])
    relative = RelativeState(
        make_transform(rotation, np.array([-10.0, 0.0, 0.0])),
        np.concatenate((np.zeros(3), rotation.T @ actual_target)),
    )
    observation = build_phase2_mission_observation(
        relative,
        task=task,
        active_reference_position_m=np.array([-8.0, 0.0, 0.0]),
        active_reference_velocity_target_m_s=desired_target,
        mission_phase=0,
        attitude_scale_rad=1.0,
        distance_scale_m=1.0,
        angular_velocity_scale_rad_s=1.0,
        velocity_scale_m_s=1.0,
        translational_velocity_observation="tracking_error",
    )
    expected_raw = rotation.T @ (actual_target - desired_target)
    expected = 10.0 * expected_raw / (10.0 + np.abs(expected_raw))
    assert np.allclose(observation[6:9], expected, atol=1.0e-7)
    assert observation[7] < 0.0


def test_terminal_velocity_reference_closes_inside_the_speed_limits() -> None:
    """The constrained leg has a real guidance law, admissible by construction."""

    task = Phase2TaskConfig()
    reward = Phase2MissionReward(task=task)
    axis = task.approach_axis
    for range_m in (12.0, 8.0, 5.0, 3.5):
        position = task.desired_position + (range_m - 3.0) * axis
        relative = RelativeState(make_transform(np.eye(3), position), np.zeros(6))
        desired = reward.active_desired_velocity(
            relative, task.desired_position, terminal_constraints_active=True
        )
        axial_remaining = float(axis @ (position - task.desired_position))
        closing = float(-axis @ desired)
        assert closing > 0.0, "the reference must approach the desired pose"
        assert closing <= task.closing_speed_limit(axial_remaining)
        assert float(np.linalg.norm(desired)) <= task.total_speed_limit_m_s

    # At the desired pose the reference is stationary, so the leg has a fixed point.
    at_goal = RelativeState(
        make_transform(np.eye(3), task.desired_position), np.zeros(6)
    )
    assert np.allclose(
        reward.active_desired_velocity(
            at_goal, task.desired_position, terminal_constraints_active=True
        ),
        0.0,
    )


def test_phase_guidance_reference_switches_with_the_phase() -> None:
    """Phase 0 aims at its own reference; the terminal leg aims at the port axis."""

    task = Phase2TaskConfig()
    reward = Phase2MissionReward(task=task)
    waypoint = np.array([-8.0, 0.0, 0.0])
    position = np.array([-12.0, 1.0, 0.0])
    relative = RelativeState(make_transform(np.eye(3), position), np.zeros(6))
    approach = reward.active_desired_velocity(
        relative, waypoint, terminal_constraints_active=False
    )
    terminal = reward.active_desired_velocity(
        relative, waypoint, terminal_constraints_active=True
    )
    assert not np.allclose(approach, terminal)
    # Phase 0 heads straight at the Waypoint; the terminal law also kills the lateral
    # offset from the approach axis, so it carries a component the other lacks.
    assert abs(float(terminal[1])) > abs(float(approach[1]))


def test_terminal_reference_retreats_when_the_chaser_overshoots() -> None:
    """Past the desired pose the reference must back off, not hold.

    The corridor radius is the distance ahead of the port times tan(35 deg),
    so it pinches shut on a chaser that drifts toward the port. A reference
    that commands zero axial velocity there offers nothing to push back with.
    """

    task = Phase2TaskConfig()
    reward = Phase2MissionReward(task=task)
    axis = task.approach_axis
    for overshoot_m in (0.25, 0.75, 1.2):
        position = task.desired_position - overshoot_m * axis
        relative = RelativeState(make_transform(np.eye(3), position), np.zeros(6))
        desired = reward.active_desired_velocity(
            relative, task.desired_position, terminal_constraints_active=True
        )
        # Positive along the approach axis is away from the port.
        assert float(axis @ desired) > 0.0, "overshoot must command a retreat"
        assert float(np.linalg.norm(desired)) <= task.total_speed_limit_m_s
