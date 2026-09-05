import numpy as np
import pytest

from controllers.mpc import LocalRelativePredictionModel, MPCConfig, MPCController, RelativePredictionModel, constrained_mpc_nominal_config, corridor_tracking_mpc_config
from controllers.mpc.constraints import linearize_constraint_margins, normalized_constraint_margins
from controllers.mpc.linearization import central_difference_linearization
from controllers.mpc.prediction import relative_to_vector
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from env.scenarios import chaser_parameters, target_parameters


def _reference(config: SE3RendezvousConfig) -> RelativePredictionModel:
    return RelativePredictionModel(
        target_parameters(),
        chaser_parameters(),
        config.dt_s,
        GravityOptions(include_j2=config.include_j2),
        RK45Settings(config.solver_rtol, config.solver_atol, config.dt_s),
    )


def test_reference_prediction_matches_environment_one_step() -> None:
    config = SE3RendezvousConfig(
        curriculum_enabled=False,
        max_time_s=1.0,
        cache_target_trajectory=False,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=21)
        assert env.relative is not None and env.target_state is not None
        x = relative_to_vector(env.relative)
        target = env.target_state.copy()
        wrench = GeneralizedForce.from_vector(
            np.array([0.1, -0.2, 0.05, 1.0, -2.0, 0.5])
        )
        predicted, _ = _reference(config).predict(x, wrench.vector, target, 0.0)
        env.step(wrench_to_normalized(wrench))
        assert env.relative is not None
        actual = relative_to_vector(env.relative)
    finally:
        env.close()
    assert np.allclose(predicted, actual, rtol=2.0e-8, atol=2.0e-8)


def test_local_jacobian_is_finite_and_scale_stable() -> None:
    model = LocalRelativePredictionModel(chaser_parameters())
    config = MPCConfig()
    state = np.array([0.2, -0.1, 0.05, 2.0, -1.0, 0.5, 0.01, -0.02, 0.005, 0.1, -0.05, 0.02])
    control = np.zeros(6)
    first = central_difference_linearization(
        model.predict,
        state,
        control,
        state_scales=config.state_scales,
        input_scales=config.input_scales,
        relative_step=1.0e-4,
    )
    second = central_difference_linearization(
        model.predict,
        state,
        control,
        state_scales=config.state_scales,
        input_scales=config.input_scales,
        relative_step=1.0e-5,
    )
    assert all(np.all(np.isfinite(value)) for value in first)
    assert np.allclose(first[0], second[0], rtol=2.0e-3, atol=2.0e-5)
    assert np.allclose(first[1], second[1], rtol=2.0e-3, atol=2.0e-5)


def test_mpc_command_obeys_physical_input_bounds() -> None:
    config = SE3RendezvousConfig(curriculum_enabled=False, max_time_s=1.0)
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=8)
        assert env.relative is not None and env.target_state is not None
        mpc_config = MPCConfig(horizon_steps=5, outer_iterations=1, linearization_stride=2)
        controller = MPCController(
            mpc_config,
            LocalRelativePredictionModel(chaser_parameters()),
            _reference(config),
        )
        command, diagnostics = controller.command(
            relative_to_vector(env.relative),
            target_state=env.target_state,
            time_seconds=0.0,
        )
    finally:
        env.close()
    assert np.all(np.isfinite(command))
    assert np.all(np.abs(command) <= mpc_config.input_scales + 1.0e-9)
    assert not diagnostics.used_zero_fallback


def test_exact_mpc_linearization_is_finite() -> None:
    config = SE3RendezvousConfig(curriculum_enabled=False, max_time_s=1.0)
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=19)
        assert env.relative is not None and env.target_state is not None
        mpc_config = MPCConfig(
            horizon_steps=5,
            outer_iterations=1,
            exact_linearization_refresh_steps=1,
        )
        controller = MPCController(
            mpc_config,
            LocalRelativePredictionModel(chaser_parameters()),
            _reference(config),
        )
        command, diagnostics = controller.command(
            relative_to_vector(env.relative),
            target_state=env.target_state,
            time_seconds=0.0,
        )
    finally:
        env.close()
    assert np.all(np.isfinite(command))
    assert diagnostics.status == "optimal"
    assert not diagnostics.used_zero_fallback


def test_phase2_constraint_margins_and_linearization_are_consistent() -> None:
    env_config = SE3RendezvousConfig(phase2_enabled=True, curriculum_enabled=False)
    env = SE3RendezvousEnv(env_config)
    try:
        env.reset(seed=41)
        assert env.relative is not None
        state = relative_to_vector(env.relative)
    finally:
        env.close()
    config = constrained_mpc_nominal_config(horizon_steps=5)
    margins = normalized_constraint_margins(
        state, config.task, corridor_facets=config.corridor_facets
    )
    jacobian, offset = linearize_constraint_margins(
        state,
        config.task,
        corridor_facets=config.corridor_facets,
    )
    assert np.min(margins) >= 0.0
    assert np.allclose(jacobian @ state - offset, margins)
    perturbation = 1.0e-5 * config.state_scales
    actual = normalized_constraint_margins(
        state + perturbation,
        config.task,
        corridor_facets=config.corridor_facets,
    )
    predicted = margins + jacobian @ perturbation
    assert np.allclose(actual, predicted, atol=2.0e-5)


def test_phase2_constrained_mpc_targets_desired_pose_without_fallback() -> None:
    config = SE3RendezvousConfig(
        phase2_enabled=True,
        curriculum_enabled=False,
        max_time_s=1.0,
        cache_target_trajectory=False,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=42)
        assert env.relative is not None and env.target_state is not None
        mpc_config = constrained_mpc_nominal_config(
            horizon_steps=8,
            outer_iterations=1,
            exact_linearization_refresh_steps=1,
        )
        controller = MPCController(
            mpc_config,
            LocalRelativePredictionModel(chaser_parameters()),
            _reference(config),
        )
        command, diagnostics = controller.command(
            relative_to_vector(env.relative),
            target_state=env.target_state,
            time_seconds=0.0,
        )
    finally:
        env.close()
    assert np.all(np.isfinite(command))
    assert not diagnostics.used_zero_fallback
    assert diagnostics.maximum_slack <= mpc_config.constraint_slack_limit
    assert np.isfinite(diagnostics.predicted_minimum_margin)


def _random_relative_states(count: int, seed: int) -> list[np.ndarray]:
    """States spanning the corridor geometry, not only the nominal envelope."""

    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            (
                rng.normal(scale=0.5, size=3),
                rng.normal(scale=4.0, size=3),
                rng.normal(scale=0.02, size=3),
                rng.normal(scale=0.2, size=3),
            )
        )
        for _ in range(count)
    ]


def test_margin_functions_agree_with_the_task_module() -> None:
    """The lean margins must be the task's margins, not a second definition.

    ``constraints`` stopped routing through ``compute_task_metrics`` because a
    margin needs three of its twenty fields and was paying for an ``se3_log``,
    an ``so3_log`` and an SVD it never reads. The values have to be unchanged,
    so compare them against the task module itself rather than a reimplementation.
    """

    from controllers.mpc.constraints import normalized_truth_margins
    from dynamics.lie import se3_exp
    from dynamics.relative import RelativeState
    from env.task import Phase2TaskConfig, compute_task_metrics

    task = Phase2TaskConfig()
    for state in _random_relative_states(64, seed=7):
        metrics = compute_task_metrics(
            RelativeState(se3_exp(state[:6]), state[6:]), task
        )
        expected = np.array(
            [
                min(
                    metrics.corridor_axial_margin_m,
                    metrics.corridor_lateral_margin_m,
                )
                / 3.0,
                metrics.fov_margin_rad / task.fov_half_angle_rad,
                metrics.total_speed_margin_m_s / task.total_speed_limit_m_s,
                metrics.closing_speed_margin_m_s / task.closing_speed_max_m_s,
            ]
        )
        assert np.array_equal(normalized_truth_margins(state, task), expected)

        facets = normalized_constraint_margins(state, task, corridor_facets=8)
        assert facets[9] == expected[1]
        assert facets[10] == expected[2]
        assert facets[11] == expected[3]


def test_constraint_jacobian_is_analytic_and_matches_central_differences() -> None:
    """The Jacobian is closed-form; central differences are the ground truth.

    Finite differencing the margins was 74% of a control step, so the Jacobian
    is derived instead. A derivative error here would silently mis-steer the
    QP's constraints, which is exactly the failure a numerical check catches.
    """

    from controllers.mpc.constraints import _pose
    from env.task import Phase2TaskConfig

    task = Phase2TaskConfig()
    facets = 8
    step = 1.0e-6

    def central(state: np.ndarray) -> np.ndarray:
        columns = []
        for index in range(12):
            direction = np.zeros(12)
            direction[index] = step
            columns.append(
                (
                    normalized_constraint_margins(
                        state + direction, task, corridor_facets=facets
                    )
                    - normalized_constraint_margins(
                        state - direction, task, corridor_facets=facets
                    )
                )
                / (2.0 * step)
            )
        return np.stack(columns, axis=1)

    checked = 0
    for state in _random_relative_states(48, seed=11):
        # The closing-speed envelope has a kink where it meets its floor and its
        # ceiling; neither one-sided derivative is "the" derivative there, so a
        # central difference is not a fair reference within a step of it.
        remaining = float(
            task.approach_axis @ (_pose(state)[1] - task.desired_position)
        )
        envelope = (
            task.closing_speed_min_m_s
            + task.closing_speed_slope_per_s * remaining
        )
        if abs(remaining) < 1.0e-3 or abs(envelope - task.closing_speed_max_m_s) < 1.0e-3:
            continue
        jacobian, offset = linearize_constraint_margins(
            state, task, corridor_facets=facets
        )
        assert np.max(np.abs(jacobian - central(state))) < 1.0e-7
        # No margin is a function of the relative angular velocity.
        assert np.array_equal(jacobian[:, 6:9], np.zeros((facets + 4, 3)))
        # The affine form the QP consumes has to reproduce the margins exactly.
        assert np.allclose(
            jacobian @ state - offset,
            normalized_constraint_margins(state, task, corridor_facets=facets),
            atol=1.0e-12,
        )
        checked += 1
    assert checked >= 40


def test_fixed_reference_is_the_construction_time_broadcast() -> None:
    """The default reference must stay the fixed setpoint, bitwise.

    Making the reference a per-stage parameter is what let the corridor path in;
    the terminal-only MPC evidence was measured before it existed, so the
    "fixed" default has to remain exactly ``reference_state`` at every stage.
    """


    config = constrained_mpc_nominal_config(horizon_steps=6)
    controller = MPCController(
        config,
        LocalRelativePredictionModel(chaser_parameters()),
        _reference(SE3RendezvousConfig(curriculum_enabled=False, max_time_s=1.0)),
    )
    rng = np.random.default_rng(5)
    state = np.concatenate(
        (rng.normal(scale=0.4, size=3), rng.normal(scale=4.0, size=3), np.zeros(6))
    )
    trajectory = controller._reference_trajectory(state)
    expected = np.tile(config.reference_state.reshape(12, 1), (1, config.horizon_steps + 1))
    assert np.array_equal(trajectory, expected)


def test_corridor_guidance_reference_tracks_the_shared_law() -> None:
    """The guidance reference must be the one law, not a fourth copy of it.

    Its velocity column has to equal ``env.task.corridor_guidance_velocity`` at
    the current position -- the same law the reward, the observation and the
    scripted controller use -- with an identity attitude reference, and the path
    has to advance toward the desired pose rather than sit still.
    """

    from dynamics.lie import se3_exp
    from env.task import corridor_guidance_velocity

    config = corridor_tracking_mpc_config(horizon_steps=8)
    controller = MPCController(
        config,
        LocalRelativePredictionModel(chaser_parameters()),
        _reference(SE3RendezvousConfig(curriculum_enabled=False, max_time_s=1.0)),
    )
    rng = np.random.default_rng(9)
    state = np.concatenate(
        (
            rng.normal(scale=0.3, size=3),
            rng.normal(scale=3.0, size=3),
            rng.normal(scale=0.02, size=3),
            rng.normal(scale=0.2, size=3),
        )
    )
    trajectory = controller._reference_trajectory(state)
    assert trajectory.shape == (12, config.horizon_steps + 1)

    position = se3_exp(state[:6])[:3, 3]
    assert np.allclose(trajectory[3:6, 0], position)
    assert np.allclose(
        trajectory[9:12, 0], corridor_guidance_velocity(position, config.task)
    )
    # Attitude reference is identity: exponential rotation and angular-rate
    # reference are zero at every stage.
    assert np.array_equal(trajectory[:3, :], np.zeros((3, config.horizon_steps + 1)))
    assert np.array_equal(trajectory[6:9, :], np.zeros((3, config.horizon_steps + 1)))
    # Each stage is the previous one advanced by its guidance velocity.
    for index in range(config.horizon_steps):
        step = trajectory[3:6, index] + trajectory[9:12, index] * config.dt_s
        assert np.allclose(trajectory[3:6, index + 1], step)


def test_corridor_guidance_reference_requires_a_task() -> None:
    from controllers.mpc import MPCConfig

    with pytest.raises(ValueError):
        MPCConfig(reference_source="corridor_guidance")


def test_mpc_evaluate_runs_on_the_single_phase_benchmark() -> None:
    """Pure MPC must fill a main-table row on the same task the other rows use.

    The three-way comparison is measured on the single-phase benchmark, so the
    MPC path has to run against that environment (10-14 m, constrained from step
    one) and emit the shared main_table block, not only the terminal shell it was
    first measured on.
    """

    from dataclasses import replace

    from controllers.mpc import corridor_tracking_mpc_config
    from env.phase2_env import phase2_environment_config
    from experiments.evaluate_mpc import evaluate

    env_config = replace(
        phase2_environment_config("single_phase"), max_time_s=1.0
    )
    result = evaluate(
        corridor_tracking_mpc_config(horizon_steps=8),
        episodes=1,
        seed=262000,
        environment_config=env_config,
    )
    assert result["mpc_config"]["reference_source"] == "corridor_guidance"
    table = result["main_table"]
    assert table["episodes"] == 1
    assert set(table["worst_constraint_margin"]) == {
        "corridor_axial_margin_m",
        "corridor_lateral_margin_m",
        "fov_margin_rad",
        "total_speed_margin_m_s",
        "closing_speed_margin_m_s",
    }
    assert table["per_step_compute_s"]["control_period_s"] == pytest.approx(0.1)


def test_single_phase_defaults_to_the_fair_corridor_reference() -> None:
    """The benchmark row must not be measured with a myopic fixed setpoint.

    A fixed reference on the 95 s single_phase task is a myopic regulator, so
    the task defaults to the shared corridor-guidance path; the terminal shell
    keeps the fixed setpoint its historical evidence was measured on. An
    explicit choice overrides either default. This resolver is what stops the
    Pure MPC row from being assembled under the wrong reference by an omitted
    flag.
    """

    from experiments.evaluate_mpc import resolve_reference_source

    assert resolve_reference_source("single_phase", None) == "corridor_guidance"
    assert resolve_reference_source("terminal", None) == "fixed"
    assert resolve_reference_source("single_phase", "fixed") == "fixed"
    assert resolve_reference_source("terminal", "corridor_guidance") == "corridor_guidance"


def test_precapture_constraint_jacobian_matches_central_differences() -> None:
    from controllers.mpc.constraints import (
        linearize_precapture_constraint_margins,
        normalized_precapture_constraint_margins,
    )
    from env.task import PrecaptureTaskConfig

    task = PrecaptureTaskConfig()
    target_omega = np.array([0.01, 0.04, -0.005])
    facets = 8
    states = [
        np.array([0.1, -0.05, 0.02, -17.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.05]),
        np.array([0.05, 0.02, -0.03, -10.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.1, -0.1, 0.02]),
        np.array([0.02, -0.01, 0.03, -6.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.05, -0.03, 0.01]),
    ]
    step = 1.0e-6
    for state in states:
        terminal_latched = bool(np.linalg.norm(state[3:6]) < 8.0)
        jacobian, offset = linearize_precapture_constraint_margins(
            state,
            task,
            target_angular_velocity_rad_s=target_omega,
            terminal_latched=terminal_latched,
            corridor_facets=facets,
        )
        columns = []
        for index in range(12):
            direction = np.zeros(12)
            direction[index] = step
            plus = normalized_precapture_constraint_margins(
                state + direction,
                task,
                target_angular_velocity_rad_s=target_omega,
                terminal_latched=terminal_latched,
                corridor_facets=facets,
            )
            minus = normalized_precapture_constraint_margins(
                state - direction,
                task,
                target_angular_velocity_rad_s=target_omega,
                terminal_latched=terminal_latched,
                corridor_facets=facets,
            )
            columns.append((plus - minus) / (2.0 * step))
        numerical = np.stack(columns, axis=1)
        assert np.max(np.abs(jacobian - numerical)) < 2.0e-7
        nominal = normalized_precapture_constraint_margins(
            state,
            task,
            target_angular_velocity_rad_s=target_omega,
            terminal_latched=terminal_latched,
            corridor_facets=facets,
        )
        assert np.allclose(jacobian @ state - offset, nominal, atol=1.0e-12)


def test_precapture_mpc_requires_explicit_latch_and_solves() -> None:
    from controllers.mpc import precapture_mpc_config
    from env.phase2_env import precapture_planning_environment_config

    env_config = precapture_planning_environment_config()
    env = SE3RendezvousEnv(env_config)
    try:
        env.reset(seed=260904)
        assert env.relative is not None and env.target_state is not None
        controller = MPCController(
            precapture_mpc_config(horizon_steps=3),
            LocalRelativePredictionModel(chaser_parameters()),
        )
        with pytest.raises(ValueError, match="terminal_latched"):
            controller.command(
                relative_to_vector(env.relative),
                target_state=env.target_state,
                time_seconds=0.0,
            )
        command, diagnostics = controller.command(
            relative_to_vector(env.relative),
            target_state=env.target_state,
            time_seconds=0.0,
            terminal_latched=False,
        )
    finally:
        env.close()
    assert np.all(np.isfinite(command))
    assert diagnostics.status == "optimal"
    assert not diagnostics.used_zero_fallback


def test_precapture_constraint_rows_follow_only_the_explicit_one_way_latch() -> None:
    from controllers.mpc.constraints import normalized_precapture_constraint_margins
    from env.task import PrecaptureTaskConfig

    task = PrecaptureTaskConfig()
    target_omega = np.array([0.0, 0.04, 0.0])
    outside = np.array([0.0, 0.0, 0.0, -9.0, 0.0, 0.0, *([0.0] * 6)])
    inside = outside.copy()
    inside[3] = -5.5
    outer = normalized_precapture_constraint_margins(
        outside,
        task,
        target_angular_velocity_rad_s=target_omega,
        terminal_latched=False,
        corridor_facets=8,
    )
    unlatched_inside = normalized_precapture_constraint_margins(
        inside,
        task,
        target_angular_velocity_rad_s=target_omega,
        terminal_latched=False,
        corridor_facets=8,
    )
    latched_outside = normalized_precapture_constraint_margins(
        outside,
        task,
        target_angular_velocity_rad_s=target_omega,
        terminal_latched=True,
        corridor_facets=8,
    )
    assert np.all(outer[2:4] < 1.0e3) and np.all(outer[4:] == 1.0e3)
    assert np.all(unlatched_inside[2:4] < 1.0e3)
    assert np.all(unlatched_inside[4:] == 1.0e3)
    assert latched_outside[2] == 1.0e3 and np.all(latched_outside[4:] < 1.0e3)


def test_external_local_waypoint_is_inertially_oriented() -> None:
    from controllers.mpc import precapture_mpc_config
    from dynamics.lie import se3_exp, so3_exp
    from env.scenarios import target_initial_state

    config = precapture_mpc_config(
        horizon_steps=5, reference_source="external_local"
    )
    controller = MPCController(
        config, LocalRelativePredictionModel(chaser_parameters())
    )
    target = target_initial_state(tumble_scale=0.20)
    waypoint_inertial = np.array([-12.0, 3.0, 1.0])
    reference = controller._reference_trajectory(
        np.zeros(12),
        target_state=target,
        external_reference=waypoint_inertial,
    )
    for index in range(config.horizon_steps + 1):
        target_rotation = target.rotation @ so3_exp(
            index * config.dt_s * target.omega
        )
        reference_transform = se3_exp(reference[:6, index])
        assert np.allclose(
            target_rotation @ reference_transform[:3, 3], waypoint_inertial
        )


def test_external_local_previews_motion_from_consecutive_3d_waypoints() -> None:
    from controllers.mpc import precapture_mpc_config
    from dynamics.lie import se3_exp, so3_exp
    from env.scenarios import target_initial_state

    config = precapture_mpc_config(
        horizon_steps=5,
        reference_source="external_local",
        external_reference_hold_steps=2,
    )
    controller = MPCController(
        config, LocalRelativePredictionModel(chaser_parameters())
    )
    target = target_initial_state(tumble_scale=0.20)
    initial = np.array([-12.0, 3.0, 1.0])
    velocity = np.array([0.0, 0.2, -0.1])
    controller._reference_trajectory(
        np.zeros(12),
        target_state=target,
        external_reference=initial,
        terminal_latched=False,
    )
    controller._control_step = config.external_reference_hold_steps
    position = initial + 2.0 * config.dt_s * velocity
    reference = controller._reference_trajectory(
        np.zeros(12),
        target_state=target,
        external_reference=position,
        terminal_latched=False,
    )
    for index in range(config.horizon_steps + 1):
        target_rotation = target.rotation @ so3_exp(
            index * config.dt_s * target.omega
        )
        reference_transform = se3_exp(reference[:6, index])
        expected = position + index * config.dt_s * velocity
        assert np.allclose(
            target_rotation @ reference_transform[:3, 3], expected
        )


def test_precapture_outer_reference_points_camera_at_port_and_holds_waypoint() -> None:
    from controllers.mpc import precapture_mpc_config
    from dynamics.lie import se3_exp
    from env.scenarios import target_initial_state

    config = precapture_mpc_config(horizon_steps=3, reference_source="external_local")
    controller = MPCController(config, LocalRelativePredictionModel(chaser_parameters()))
    target = target_initial_state(tumble_scale=0.20)
    state = np.zeros(12)
    state[3:6] = np.array([-12.0, 4.0, 1.0])
    first_waypoint = np.array([-11.0, 2.0, 0.5])
    reference = controller._reference_trajectory(
        state,
        target_state=target,
        external_reference=first_waypoint,
        terminal_latched=False,
    )
    pose = se3_exp(reference[:6, 0])
    line_of_sight = config.precapture_task.port_position - state[3:6]
    line_of_sight /= np.linalg.norm(line_of_sight)
    assert np.allclose(
        pose[:3, :3] @ config.precapture_task.camera_boresight,
        line_of_sight,
    )
    controller._control_step = 1
    held = controller._reference_trajectory(
        state,
        target_state=target,
        external_reference=np.array([4.0, 5.0, 6.0]),
        terminal_latched=False,
    )
    held_pose = se3_exp(held[:6, 0])
    assert np.allclose(target.rotation @ held_pose[:3, 3], first_waypoint)


def test_two_stage_precapture_guidance_latches_only_after_slow_staging() -> None:
    from env.phase2_env import precapture_planning_environment_config
    from env.scenarios import target_initial_state
    from experiments.evaluate_mpc import _two_stage_precapture_reference

    environment = precapture_planning_environment_config()
    target = target_initial_state(tumble_scale=0.20)
    state = np.zeros(12)
    state[3] = -10.0
    state[9] = 0.60
    staging, latched = _two_stage_precapture_reference(
        state, target, environment, final_stage_latched=False
    )
    assert not latched
    assert np.allclose(
        staging,
        target.rotation @ (10.0 * environment.precapture_task.approach_axis),
    )

    state[9] = 0.40
    final, latched = _two_stage_precapture_reference(
        state, target, environment, final_stage_latched=False
    )
    assert latched
    assert np.allclose(
        final, target.rotation @ environment.precapture_task.desired_position
    )

    state[3] = -12.0
    final_after_departure, latched = _two_stage_precapture_reference(
        state, target, environment, final_stage_latched=latched
    )
    assert latched
    assert np.allclose(final_after_departure, final)


def test_oracle_plan_guidance_matches_the_offline_path_through_the_waypoint_interface() -> None:
    """The oracle-plan row must deliver the offline path, unaltered, as waypoints.

    The declared learned-policy action is one target-centred, inertially
    oriented 3D waypoint. This row replays the offline coast-then-match path
    through exactly that interface, so the only thing separating it from the
    fixed-setpoint row is the reference -- same horizon, weights, solver and
    constraints. If the conversion drifted from the oracle's own trajectory the
    row would stop measuring what a perfect upper layer is worth.
    """

    from env.phase2_env import precapture_planning_environment_config
    from env.se3_rendezvous_env import SE3RendezvousEnv
    from experiments.evaluate_precapture_oracle import CoastThenMatchPlan

    env = SE3RendezvousEnv(precapture_planning_environment_config())
    try:
        env.reset(seed=262000)
        plan = CoastThenMatchPlan(env)
        for time_s in (0.0, 20.0, plan.coast_time_s, plan.coast_time_s + 30.0):
            body = plan.body_position(time_s)
            rotation = plan._target_rotation(time_s)
            waypoint = rotation @ body
            # Round trip through the interface returns the same body-frame point.
            assert np.allclose(rotation.T @ waypoint, body, atol=1.0e-12)
        # The path starts where the chaser is and ends at the desired pose.
        assert np.allclose(
            plan.body_position(0.0), env.relative.position, atol=1.0e-9
        )
        task = env.config.precapture_task
        assert np.allclose(
            plan.body_position(plan.coast_time_s + 200.0),
            task.desired_position,
            atol=1.0e-9,
        )
    finally:
        env.close()
