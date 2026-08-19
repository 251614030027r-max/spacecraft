import numpy as np

from dynamics.lie import se3_exp
from dynamics.relative import RelativeState
from env.reward import Phase2MissionReward, Phase2TaskReward, PhysicalStorageReward
from env.task import Phase2TaskConfig


def _relative(position_m: float, velocity_m_s: float = 0.0) -> RelativeState:
    return RelativeState(
        se3_exp(np.array([0.0, 0.0, 0.0, position_m, 0.0, 0.0])),
        np.array([0.0, 0.0, 0.0, velocity_m_s, 0.0, 0.0]),
    )


def _reward() -> PhysicalStorageReward:
    return PhysicalStorageReward(
        characteristic_length_m=0.4482671036630004,
        time_constant_s=30.0,
        time_step_s=0.1,
        max_distance_m=30.0,
    )


def test_goal_has_zero_storage_reward_without_actuation() -> None:
    reward = _reward()
    goal = _relative(0.0)
    reward.reset(goal)
    breakdown = reward.compute(goal, np.zeros(6))
    assert breakdown.storage_energy == 0.0
    assert breakdown.storage_potential == 0.0
    assert breakdown.total == 0.0


def test_moving_toward_goal_produces_storage_dissipation() -> None:
    reward = _reward()
    reward.reset(_relative(4.0, 0.1))
    breakdown = reward.compute(_relative(3.0, 0.05), np.zeros(6))
    assert breakdown.storage_dissipation > 0.0
    assert breakdown.storage_penalty < 0.0
    assert breakdown.total > 0.0


def test_actuation_term_penalizes_normalized_utilization() -> None:
    reward = _reward()
    state = _relative(1.0)
    reward.reset(state)
    breakdown = reward.compute(state, np.ones(6))
    assert breakdown.actuation_penalty == -reward.step_weight


def test_safety_barrier_increases_near_distance_limit() -> None:
    reward = _reward()
    assert reward.safety_barrier(_relative(29.0)) > reward.safety_barrier(
        _relative(5.0)
    )


def test_phase2_cost_uses_declared_time_constant() -> None:
    reward = Phase2TaskReward(
        task=Phase2TaskConfig(),
        time_step_s=0.1,
        time_constant_s=30.0,
        attitude_scale_rad=np.deg2rad(30.0),
        distance_scale_m=3.0,
        angular_velocity_scale_rad_s=0.05,
        velocity_scale_m_s=0.2,
    )
    assert np.isclose(reward.step_weight, 0.1 / 30.0)


def test_phase2_constraint_cost_is_smooth_inside_and_bounded_outside() -> None:
    reward = Phase2TaskReward(
        task=Phase2TaskConfig(),
        time_step_s=0.1,
        time_constant_s=30.0,
        attitude_scale_rad=np.deg2rad(60.0),
        distance_scale_m=5.0,
        angular_velocity_scale_rad_s=0.03,
        velocity_scale_m_s=0.35,
    )
    safe = reward._proximity_cost(0.5, 1.0)
    near = reward._proximity_cost(0.0, 1.0)
    violated = reward._proximity_cost(-0.1, 1.0)
    extreme = reward._proximity_cost(-1.0e6, 1.0)
    assert safe > near > violated
    assert extreme == -reward.maximum_normalized_constraint_cost


def test_mission_reward_progress_and_terminal_warning_are_well_scaled() -> None:
    task = Phase2TaskConfig()
    reward = Phase2MissionReward(task=task)
    far = _relative(-12.0)
    closer = _relative(-10.0)
    reward.reset(far, np.array([-8.0, 0.0, 0.0]))
    progress = reward.compute(
        closer, np.zeros(6), terminal_constraints_active=False
    )
    assert progress.progress > 0.0
    assert progress.constraint_warning <= 0.0

    near_boundary = RelativeState(
        se3_exp(np.array([0.0, 0.0, 0.0, -6.0, 3.0, 0.0])),
        np.zeros(6),
    )
    reward.reset(near_boundary, task.desired_position)
    warning = reward.compute(
        near_boundary, np.zeros(6), terminal_constraints_active=True
    )
    assert np.isfinite(warning.total)
    assert warning.constraint_warning < 0.0


def test_phase1_soft_speed_cost_starts_before_gate_failure_region() -> None:
    task = Phase2TaskConfig()
    reward = Phase2MissionReward(task=task)
    safe = _relative(-10.0, 0.16)
    moderate = _relative(-10.0, 0.30)
    high = _relative(-10.0, 0.50)
    reward.reset(safe, np.array([-8.0, 0.0, 0.0]))
    safe_result = reward.compute(
        safe, np.zeros(6), terminal_constraints_active=False
    )
    reward.reset(moderate, np.array([-8.0, 0.0, 0.0]))
    moderate_result = reward.compute(
        moderate, np.zeros(6), terminal_constraints_active=False
    )
    reward.reset(high, np.array([-8.0, 0.0, 0.0]))
    high_result = reward.compute(
        high, np.zeros(6), terminal_constraints_active=False
    )
    assert np.isclose(safe_result.constraint_warning, 0.0)
    assert high_result.constraint_warning < moderate_result.constraint_warning < 0.0
    assert high_result.constraint_warning >= -(
        reward.phase1_speed_weight
        + reward.phase1_velocity_tracking_weight
        + reward.phase1_attitude_weight
        + reward.phase1_angular_velocity_weight
    )


def test_phase1_potential_is_bounded_and_has_no_absolute_state_cost() -> None:
    reward = Phase2MissionReward(task=Phase2TaskConfig())
    reference = np.array([-8.0, 0.0, 0.0])
    extreme = _relative(1.0e9, 1.0e6)
    bounded = reward.bounded_potential(extreme, reference)
    assert 0.0 <= bounded <= 2.1
    reward.reset(extreme, reference)
    result = reward.compute(
        extreme, np.zeros(6), terminal_constraints_active=False
    )
    assert result.state_penalty == 0.0
    assert np.isfinite(result.total)


def test_phase1_stationary_progress_uses_sac_discount() -> None:
    reward = Phase2MissionReward(task=Phase2TaskConfig(), discount_factor=0.997)
    reference = np.array([-8.0, 0.0, 0.0])
    state = _relative(-12.0)
    reward.reset(state, reference)
    result = reward.compute(state, np.zeros(6), terminal_constraints_active=False)
    expected = reward.progress_weight * (1.0 - reward.discount_factor) * result.potential
    assert np.isclose(result.progress, expected)


def test_phase1_velocity_field_points_to_gate_and_brakes_near_it() -> None:
    reward = Phase2MissionReward(task=Phase2TaskConfig())
    reference = np.array([-8.0, 0.0, 0.0])
    far = reward.phase1_desired_velocity(_relative(-18.0), reference)
    near = reward.phase1_desired_velocity(_relative(-9.0), reference)
    assert np.allclose(far, [reward.phase1_cruise_speed_m_s, 0.0, 0.0])
    assert 0.0 < near[0] < far[0]
    assert np.isclose(near[0], reward.phase1_braking_gain_per_s)


def test_phase1_reward_prefers_gate_directed_velocity_over_reverse_velocity() -> None:
    reward = Phase2MissionReward(task=Phase2TaskConfig())
    reference = np.array([-8.0, 0.0, 0.0])
    initial = _relative(-12.0, 0.0)
    toward = _relative(-12.0, 0.18)
    reverse = _relative(-12.0, -0.18)
    reward.reset(initial, reference)
    toward_result = reward.compute(
        toward, np.zeros(6), terminal_constraints_active=False
    )
    reward.reset(initial, reference)
    reverse_result = reward.compute(
        reverse, np.zeros(6), terminal_constraints_active=False
    )
    assert toward_result.total > reverse_result.total
