"""Pre-registered acceptance tests for the V2 two-axis task interface.

These start as strict xfails.  Once the V2 interface exists each XPASS is a
suite failure, forcing the implementing change to remove the corresponding
marker rather than silently inheriting an untested fix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


def _v2_env() -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="task_state_v2",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_staging_direction_observation=True,
        ),
    )


def _set_task_state(env: PrecaptureHybridEnv, rho_m: float, commitment: float) -> None:
    env._task_progress_m = rho_m
    env._task_commitment = commitment


def test_t1_distance_axis_is_effective_and_allows_retreat() -> None:
    env = _v2_env()
    try:
        env.reset(seed=262000)
        _set_task_state(env, 5.0, 0.5)
        advanced = env.waypoint_from_action(env.action_for_task_state(5.2, 0.5))
        advanced_progress = env._task_progress_m
        retreated = env.waypoint_from_action(env.action_for_task_state(5.1, 0.5))
        retreated_progress = env._task_progress_m
        assert not np.allclose(advanced, retreated)
        assert retreated_progress < advanced_progress  # rho can decrease
        assert retreated_progress >= 0.0  # retreat is finite and bounded
    finally:
        env.close()


def test_t2_commitment_axis_is_effective_and_never_latches() -> None:
    env = _v2_env()
    try:
        env.reset(seed=262001)
        _set_task_state(env, 10.0, 0.5)
        high = env.waypoint_from_action(env.action_for_task_state(10.0, 1.0))
        committed = env._task_commitment
        low = env.waypoint_from_action(env.action_for_task_state(10.0, 0.0))
        retreated = env._task_commitment
        assert not np.allclose(high, low)
        assert committed > 0.5
        assert 0.0 <= retreated < committed
    finally:
        env.close()


def test_t3_rejected_proposal_has_no_persistent_side_effect() -> None:
    env = _v2_env()
    try:
        env.reset(seed=262002)
        _set_task_state(env, 11.0, 0.4)
        before = (env._task_progress_m, env._task_commitment)
        rejected = env.waypoint_from_action(
            env.action_for_task_state(3.0, 1.0), proposal_accepted=False
        )
        assert (env._task_progress_m, env._task_commitment) == before
        desired = np.asarray(env.environment_config.precapture_task.desired_position)
        assert np.array_equal(rejected, desired)
        staged = env.waypoint_from_action(
            env.action_for_task_state(12.0, 0.0), proposal_accepted=True
        )
        assert not np.array_equal(staged, desired)
        assert (env._task_progress_m, env._task_commitment) != before
    finally:
        env.close()


def _record_commands(env: PrecaptureHybridEnv) -> list[np.ndarray]:
    commands: list[np.ndarray] = []
    original = env.controller.command

    def wrapped(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        commands.append(np.asarray(result[0]).copy())
        return result

    env.controller.command = wrapped  # type: ignore[method-assign]
    return commands


def test_t4_reject_all_proposals_is_bitwise_pure_mpc_closed_loop() -> None:
    baseline = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
        ),
    )
    v2 = _v2_env()
    try:
        baseline.reset(seed=262003)
        v2.reset(seed=262003)
        baseline_commands = _record_commands(baseline)
        v2_commands = _record_commands(v2)
        desired = np.asarray(baseline.environment_config.precapture_task.desired_position)
        for proposal in ([-1.0, -1.0], [1.0, 1.0], [0.25, -0.75]):
            baseline.step(baseline.action_for_waypoint(desired))
            v2.step_with_proposal(np.asarray(proposal), proposal_accepted=False)
        assert len(baseline_commands) == len(v2_commands) > 0
        assert np.array_equal(np.asarray(v2_commands), np.asarray(baseline_commands))
    finally:
        baseline.close()
        v2.close()


def test_t5_all_legal_task_states_are_geometrically_acceptable_without_far_gate() -> None:
    env = _v2_env()
    try:
        env.reset(seed=262004)
        task = env.environment_config.precapture_task
        references = []
        for rho_m in np.linspace(0.0, env.v2_progress_max_m, 9):
            for commitment in np.linspace(0.0, 1.0, 9):
                point = env.reference_for_task_state(rho_m, commitment)
                assert np.all(np.isfinite(point))
                assert np.linalg.norm(point) >= task.keepout_radius_m
                references.append((rho_m, commitment, point))
        # No new far-range corridor/gate: commitment remains effective at the
        # largest radius instead of every far proposal collapsing to one point.
        far = [point for rho, _, point in references if rho == 0.0]
        assert len(np.unique(np.round(far, 9), axis=0)) == 9
    finally:
        env.close()


def test_t6_limiter_is_monotonic_and_has_no_scanned_dead_zone() -> None:
    env = _v2_env()
    try:
        env.reset(seed=262005)
        # T6 isolates the independent axis limiter in a geometry where the
        # new metric reference cap is inactive. Large-angle metric saturation
        # is covered separately by T7 without weakening either assertion.
        assert env.env.target_state is not None
        commit = np.asarray(env.environment_config.precapture_task.desired_position)
        commit /= np.linalg.norm(commit)
        angle = np.deg2rad(10.0)
        hold = np.array(
            [
                np.cos(angle) * commit[0] - np.sin(angle) * commit[1],
                np.sin(angle) * commit[0] + np.cos(angle) * commit[1],
                commit[2],
            ]
        )
        env._hold_inertial = np.asarray(env.env.target_state.rotation) @ hold
        _set_task_state(env, 5.0, 0.5)
        radius_outputs = []
        for rho_m in np.linspace(4.6, 5.4, 21):
            _set_task_state(env, 5.0, 0.5)
            env.waypoint_from_action(env.action_for_task_state(rho_m, 0.5))
            radius_outputs.append(env._task_progress_m)
        assert radius_outputs == sorted(radius_outputs)
        assert len(np.unique(np.round(radius_outputs, 12))) == 21

        commitment_outputs = []
        # Scan the complete unsaturated asymmetric limiter interval. Values
        # beyond it are intentionally clipped; the V1 defect was a dead half
        # *inside* the nominal action range, not physical rate saturation.
        for commitment in np.linspace(0.48, 0.55, 21):
            _set_task_state(env, 5.0, 0.5)
            env.waypoint_from_action(env.action_for_task_state(5.0, commitment))
            commitment_outputs.append(env._task_commitment)
        assert commitment_outputs == sorted(commitment_outputs)
        assert len(np.unique(np.round(commitment_outputs, 12))) == 21
    finally:
        env.close()


def test_t7_reference_displacement_is_bounded_in_metres() -> None:
    env = _v2_env()
    try:
        env.reset(seed=262000)
        assert env.env.target_state is not None
        rotation = np.asarray(env.env.target_state.rotation)
        commit = np.asarray(env.environment_config.precapture_task.desired_position)
        commit /= np.linalg.norm(commit)
        for radius_m in (12.0, 15.0, 19.0):
            for angle_deg in (45.0, 90.0, 120.0, 180.0):
                # Keep the requested 180-degree corner while avoiding the
                # undefined interpolation axis of an exactly antipodal pair.
                angle = np.deg2rad(min(angle_deg, 180.0 - 1.0e-6))
                hold = np.array(
                    [
                        np.cos(angle) * commit[0] - np.sin(angle) * commit[1],
                        np.sin(angle) * commit[0] + np.cos(angle) * commit[1],
                        commit[2],
                    ]
                )
                hold /= np.linalg.norm(hold)
                for proposal in ("both_maximum", "commitment_only", "distance_only"):
                    env._hold_radius_m = radius_m
                    env._hold_inertial = rotation @ hold
                    env._task_progress_m = 0.0
                    env._task_commitment = 0.0
                    before = env.reference_for_task_state(0.0, 0.0)
                    target_progress = (
                        env.v2_progress_max_m
                        if proposal in {"both_maximum", "distance_only"}
                        else 0.0
                    )
                    target_commitment = (
                        1.0
                        if proposal in {"both_maximum", "commitment_only"}
                        else 0.0
                    )
                    after = env.waypoint_from_action(
                        env.action_for_task_state(
                            target_progress, target_commitment
                        )
                    )
                    displacement = float(np.linalg.norm(after - before))
                    assert (
                        displacement
                        <= env.hybrid_config.v2_reference_step_max_m + 1.0e-9
                    )
    finally:
        env.close()
