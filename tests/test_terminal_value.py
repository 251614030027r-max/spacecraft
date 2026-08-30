"""Tests for the learned convex-quadratic terminal value (SAC-MPC coupling)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from controllers.mpc import (
    ConvexQuadraticTerminalValue,
    LocalRelativePredictionModel,
    MPCConfig,
    MPCController,
    RelativePredictionModel,
    constrained_mpc_nominal_config,
    corridor_tracking_mpc_config,
    fit_convex_quadratic,
    learned_terminal_mpc_config,
)
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from controllers.mpc.prediction import relative_to_vector
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


def _sample_terminal_value(seed: int = 0) -> ConvexQuadraticTerminalValue:
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(12, 12))
    weight = root @ root.T + np.eye(12)  # symmetric positive definite
    return ConvexQuadraticTerminalValue(
        center=rng.normal(size=12),
        weight_matrix=weight,
        linear=rng.normal(size=12),
        constant=1.5,
    )


def test_default_terminal_cost_source_is_fixed_quadratic() -> None:
    assert MPCConfig().terminal_cost_source == "fixed_quadratic"
    assert constrained_mpc_nominal_config().terminal_cost_source == "fixed_quadratic"


def test_learned_convex_requires_a_terminal_value() -> None:
    with pytest.raises(ValueError):
        MPCConfig(terminal_cost_source="learned_convex")


def test_terminal_value_rejects_non_psd_weight() -> None:
    indefinite = np.diag([1.0, -2.0] + [1.0] * 10)
    with pytest.raises(ValueError):
        ConvexQuadraticTerminalValue(
            center=np.zeros(12), weight_matrix=indefinite, linear=np.zeros(12)
        )


def test_value_matches_the_quadratic_formula() -> None:
    value = _sample_terminal_value(1)
    rng = np.random.default_rng(2)
    x = rng.normal(size=12)
    z = x - value.center
    expected = float(z @ value.weight_matrix @ z) + float(value.linear @ z) + value.constant
    assert value.value(x) == pytest.approx(expected)


def test_cholesky_factor_reproduces_the_weight() -> None:
    value = _sample_terminal_value(3)
    factor = value.cholesky_factor()
    reconstructed = factor.T @ factor
    # The factor carries a tiny PSD jitter, so match to that floor, not exactly.
    assert np.allclose(reconstructed, value.weight_matrix, atol=1.0e-6)


def test_save_and_load_round_trip(tmp_path) -> None:
    value = _sample_terminal_value(4)
    path = tmp_path / "terminal_value.json"
    value.save(path)
    loaded = ConvexQuadraticTerminalValue.load(path)
    assert np.allclose(loaded.center, value.center)
    assert np.allclose(loaded.weight_matrix, value.weight_matrix)
    assert np.allclose(loaded.linear, value.linear)
    assert loaded.constant == pytest.approx(value.constant)


def test_fit_recovers_a_convex_quadratic_and_is_psd() -> None:
    """A quadratic-generated target is recovered; the fit is PSD by construction."""

    rng = np.random.default_rng(5)
    root = rng.normal(size=(12, 12))
    true_weight = root @ root.T + np.eye(12)
    true_linear = rng.normal(size=12)
    center = rng.normal(size=12)
    states = center + rng.normal(scale=2.0, size=(4000, 12))
    z = states - center
    targets = np.einsum("ni,ij,nj->n", z, true_weight, z) + z @ true_linear + 3.0

    fitted = fit_convex_quadratic(states, targets, center=center)

    eigenvalues = np.linalg.eigvalsh(fitted.weight_matrix)
    assert float(eigenvalues.min()) >= 0.0
    predictions = np.array([fitted.value(x) for x in states])
    residual = targets - predictions
    r_squared = 1.0 - np.sum(residual**2) / np.sum((targets - targets.mean()) ** 2)
    assert r_squared > 0.999


def test_fit_projects_a_concave_direction_to_psd() -> None:
    """A target curving downward in one direction is flattened, not left concave."""

    rng = np.random.default_rng(6)
    center = np.zeros(12)
    states = rng.normal(scale=1.5, size=(3000, 12))
    # Concave along axis 0, convex elsewhere.
    weight = np.diag([-1.0] + [1.0] * 11)
    targets = np.einsum("ni,ij,nj->n", states, weight, states)
    fitted = fit_convex_quadratic(states, targets, center=center)
    assert float(np.linalg.eigvalsh(fitted.weight_matrix).min()) >= 0.0


def test_fixed_quadratic_command_is_unchanged_by_the_new_branch() -> None:
    """The default terminal cost must be bitwise the historical fixed penalty."""

    config = SE3RendezvousConfig(
        phase2_enabled=True, curriculum_enabled=False, max_time_s=1.0,
        cache_target_trajectory=False,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=42)
        assert env.relative is not None and env.target_state is not None
        state = relative_to_vector(env.relative)
        target = env.target_state.copy()
    finally:
        env.close()

    mpc_config = constrained_mpc_nominal_config(
        horizon_steps=8, exact_linearization_refresh_steps=1
    )
    baseline = MPCController(
        mpc_config, LocalRelativePredictionModel(chaser_parameters()), _reference(config)
    )
    command_baseline, _ = baseline.command(
        state, target_state=target, time_seconds=0.0
    )

    explicit = MPCController(
        replace(mpc_config, terminal_cost_source="fixed_quadratic"),
        LocalRelativePredictionModel(chaser_parameters()),
        _reference(config),
    )
    command_explicit, _ = explicit.command(
        state, target_state=target, time_seconds=0.0
    )
    assert np.array_equal(command_baseline, command_explicit)


def test_learned_convex_qp_solves_without_fallback() -> None:
    """A learned terminal value must keep the lower layer a solvable convex QP."""

    config = SE3RendezvousConfig(
        phase2_enabled=True, curriculum_enabled=False, max_time_s=1.0,
        cache_target_trajectory=False,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=42)
        assert env.relative is not None and env.target_state is not None
        state = relative_to_vector(env.relative)
        target = env.target_state.copy()
    finally:
        env.close()

    # A benign terminal value centred at the corridor task's desired pose.
    task = corridor_tracking_mpc_config(horizon_steps=8).task
    from dynamics.lie import se3_log

    center = np.concatenate((se3_log(task.desired_transform), np.zeros(6)))
    terminal_value = ConvexQuadraticTerminalValue(
        center=center,
        weight_matrix=100.0 * np.eye(12),
        linear=np.zeros(12),
    )
    mpc_config = learned_terminal_mpc_config(
        terminal_value, horizon_steps=8, exact_linearization_refresh_steps=1
    )
    controller = MPCController(
        mpc_config, LocalRelativePredictionModel(chaser_parameters()), _reference(config)
    )
    command, diagnostics = controller.command(
        state, target_state=target, time_seconds=0.0
    )
    assert np.all(np.isfinite(command))
    assert not diagnostics.used_zero_fallback
    assert diagnostics.status in {"optimal", "optimal_inaccurate"}
    assert np.isfinite(diagnostics.predicted_cost)


def test_learned_convex_evaluate_serialises_the_config() -> None:
    """The evaluate path must run and JSON-serialise a learned-terminal config."""

    import json
    from dataclasses import replace as dc_replace

    from env.phase2_env import phase2_environment_config
    from experiments.evaluate_mpc import evaluate
    from dynamics.lie import se3_log

    task = corridor_tracking_mpc_config(horizon_steps=8).task
    center = np.concatenate((se3_log(task.desired_transform), np.zeros(6)))
    terminal_value = ConvexQuadraticTerminalValue(
        center=center, weight_matrix=50.0 * np.eye(12), linear=np.zeros(12)
    )
    env_config = dc_replace(
        phase2_environment_config("single_phase"), max_time_s=1.0
    )
    result = evaluate(
        learned_terminal_mpc_config(terminal_value, horizon_steps=8),
        episodes=1,
        seed=262000,
        environment_config=env_config,
    )
    assert result["mpc_config"]["terminal_cost_source"] == "learned_convex"
    # The nested terminal value survives asdict + the numpy JSON encoder.
    rendered = json.dumps(
        result,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v,
    )
    assert "terminal_cost_source" in rendered
