from __future__ import annotations

import numpy as np
import pytest

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


def _v3_env(
    *,
    reference_step_max_m: float = 0.4,
    decision_period_steps: int = 20,
    horizon_steps: int = 20,
) -> PrecaptureHybridEnv:
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="task_state_v3",
            runtime_diagnostics=False,
            v2_reference_step_max_m=reference_step_max_m,
            decision_period_steps=decision_period_steps,
            horizon_steps=horizon_steps,
        ),
    )
    env.reset(seed=262004)
    return env


def test_v3_zero_action_keeps_task_state_fixed_for_ten_decisions() -> None:
    env = _v3_env()
    try:
        initial = (env._task_progress_m, env._task_commitment)
        for _ in range(10):
            env.waypoint_from_action(np.zeros(2, dtype=np.float64))
            assert (env._task_progress_m, env._task_commitment) == initial
    finally:
        env.close()


def test_v3_unit_action_applies_configured_axis_increments_when_metric_allows() -> None:
    env = _v3_env(reference_step_max_m=2.0)
    try:
        env.waypoint_from_action(np.ones(2, dtype=np.float64))
        assert env._task_progress_m == 0.5
        assert env._task_commitment == 0.05
    finally:
        env.close()


def test_v3_terminal_task_state_is_exactly_the_desired_pose() -> None:
    env = _v3_env()
    try:
        desired = np.asarray(
            env.environment_config.precapture_task.desired_position,
            dtype=np.float64,
        )
        reference = env.reference_for_task_state(env.v2_progress_max_m, 1.0)
        assert np.linalg.norm(reference - desired) < 1.0e-12
    finally:
        env.close()


def test_v3_direction_step_is_pure_and_handles_antipodes() -> None:
    start = np.array([1.0, 0.0, 0.0])
    goal = -start
    first = PrecaptureHybridEnv._direction_step(start, goal, 0.2)
    second = PrecaptureHybridEnv._direction_step(start, goal, 0.2)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert np.arccos(np.clip(first @ start, -1.0, 1.0)) == pytest.approx(0.2)


@pytest.mark.parametrize("closest_deg", [0.0, 3.0, 90.0, 177.0, 180.0])
def test_v3_direction_state_obeys_dynamic_bound_across_singular_geometries(
    closest_deg: float,
) -> None:
    radius = 25.0
    omega = 0.05
    period = 2.0
    natural_angle = 1.2 * omega * period
    action_budget = 0.4
    applied = np.array([0.0, 1.0, 0.0])
    previous_reference = radius * applied
    for phase_deg in np.linspace(closest_deg - 25.0, closest_deg + 25.0, 101):
        phase = np.deg2rad(phase_deg)
        goal = np.array([np.cos(phase), np.sin(phase), 0.0])
        applied = PrecaptureHybridEnv._direction_step(
            applied, goal, natural_angle + action_budget / radius
        )
        reference = radius * applied
        jump = float(np.linalg.norm(reference - previous_reference))
        assert jump <= action_budget + natural_angle * radius + 1.0e-12
        previous_reference = reference


def test_v3_candidate_previews_are_read_only_and_commit_once() -> None:
    env = _v3_env()
    try:
        assert env._v3_applied_direction is None
        assert env._v3_direction_commit_count == 0
        env._limit_task_state(0.5, 0.05)
        assert env._v3_applied_direction is None
        assert env._v3_direction_commit_count == 0

        env.waypoint_from_action(np.ones(2, dtype=np.float64))
        assert env._v3_applied_direction is not None
        assert env._v3_direction_commit_count == 1
        committed = env._v3_applied_direction.copy()
        env._limit_task_state(1.0, 0.10)
        assert np.array_equal(env._v3_applied_direction, committed)
        assert env._v3_direction_commit_count == 1
    finally:
        env.close()


def test_v3_direction_state_reaches_both_task_endpoints_exactly() -> None:
    env = _v3_env()
    try:
        desired = np.asarray(
            env.environment_config.precapture_task.desired_position,
            dtype=np.float64,
        )
        desired_direction = desired / np.linalg.norm(desired)
        for _ in range(100):
            env._commit_v3_reference(env.v2_progress_max_m, 1.0)
        assert np.linalg.norm(env._v3_applied_direction - desired_direction) < 1.0e-12
        assert np.linalg.norm(
            env.reference_for_task_state(env.v2_progress_max_m, 1.0) - desired
        ) < 1.0e-12

        env._v3_applied_direction = None
        env._v3_applied_radius_m = None
        env._commit_v3_reference(0.0, 0.0)
        assert np.linalg.norm(
            env._v3_applied_direction - env._v3_observation_direction()
        ) < 1.0e-12
    finally:
        env.close()


def test_v3_reference_jump_monitor_distinguishes_target_and_inertial_frames() -> None:
    env = _v3_env()
    try:
        first = np.array([4.0, 0.0, 0.0], dtype=np.float64)
        target_jump, inertial_jump = env._record_v3_reference_jump(first)
        assert target_jump == 0.0
        assert inertial_jump == 0.0

        assert env.env.target_state is not None
        rotation = env.env.target_state.rotation
        second = np.array([4.0, 0.3, 0.0], dtype=np.float64)
        target_jump, inertial_jump = env._record_v3_reference_jump(second)
        assert target_jump == pytest.approx(np.linalg.norm(second - first))
        assert inertial_jump == pytest.approx(
            np.linalg.norm(rotation @ (second - first))
        )
    finally:
        env.close()


def test_v3_learned_to_baseline_resets_controller_and_is_one_way() -> None:
    env = _v3_env()
    try:
        env._select_v3_branch("learned")
        env.controller._nominal_controls.fill(2.0)
        env.controller._drift.fill(3.0)
        env.controller._control_step = 7
        env.controller._held_external_reference = np.ones(3)
        env.controller._held_external_velocity.fill(4.0)

        env._select_v3_branch("baseline")
        assert np.array_equal(
            env.controller._nominal_controls,
            np.zeros_like(env.controller._nominal_controls),
        )
        assert np.array_equal(env.controller._drift, np.zeros_like(env.controller._drift))
        assert env.controller._control_step == 0
        assert env.controller._held_external_reference is None
        assert np.array_equal(
            env.controller._held_external_velocity,
            np.zeros_like(env.controller._held_external_velocity),
        )
        assert env._hybrid_branch_switches == 1
        with pytest.raises(ValueError, match="baseline.*learned"):
            env._select_v3_branch("learned")
    finally:
        env.close()


def test_v3_branch_and_switch_count_are_reported_without_observation_bit() -> None:
    env = _v3_env(decision_period_steps=1, horizon_steps=2)
    try:
        observation_shape = env.observation_space.shape
        _, _, terminated, truncated, info = env.step_with_branch(
            np.zeros(2), branch="learned"
        )
        assert not (terminated or truncated)
        assert info["hybrid_branch"] == "learned"
        assert info["hybrid_branch_switches"] == 0
        assert info["hybrid_task_action_raw"] == [0.0, 0.0]
        assert info["hybrid_v3_direction_commit_count"] == 1
        assert env.observation_space.shape == observation_shape
    finally:
        env.close()


def test_v3_baseline_decision_does_not_commit_direction_state() -> None:
    env = _v3_env(decision_period_steps=1, horizon_steps=2)
    try:
        _, _, _, _, info = env.step_with_branch(np.zeros(2), branch="baseline")
        assert env._v3_applied_direction is None
        assert info["hybrid_v3_direction_commit_count"] == 0
    finally:
        env.close()


def test_v3_observation_appends_applied_direction_without_branch_bit() -> None:
    environment = precapture_adaptive_capture_environment_config()
    common = dict(
        runtime_diagnostics=False,
        include_target_phase_and_time_observation=True,
        include_staging_direction_observation=True,
    )
    v2 = PrecaptureHybridEnv(
        environment_config=environment,
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="task_state_v2", **common
        ),
    )
    v3 = PrecaptureHybridEnv(
        environment_config=environment,
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="task_state_v3", **common
        ),
    )
    try:
        v2_observation, _ = v2.reset(seed=262004)
        v3_observation, _ = v3.reset(seed=262004)
        assert v3.observation_space.shape == (v2.observation_space.shape[0] + 3,)
        assert np.array_equal(v3_observation[:-3], v2_observation)
        assert np.allclose(
            v3_observation[-3:], v3._v3_observation_direction(), atol=1.0e-7
        )
        v3._select_v3_branch("baseline")
        assert np.array_equal(
            v3._policy_observation(v3.env._observation())[:-3],
            v3_observation[:-3],
        )
    finally:
        v2.close()
        v3.close()
