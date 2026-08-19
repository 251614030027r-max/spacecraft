import numpy as np

from controllers.mpc import LocalRelativePredictionModel, MPCConfig, MPCController, RelativePredictionModel, constrained_mpc_nominal_config
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
        state_scales=config.state_scales,
        relative_step=config.difference_relative_step,
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
