import numpy as np

from dynamics.gravity import GravityOptions
from dynamics.spacecraft import (
    body_dynamics_rhs,
    momentum_dynamics_rhs,
    state_derivative,
)
from dynamics.types import GeneralizedForce, SpacecraftParameters, SpacecraftState


def _sample_state_and_parameters() -> tuple[SpacecraftState, SpacecraftParameters]:
    state = SpacecraftState(
        np.eye(3),
        np.array([7.0e6, 1.0e5, -2.0e5]),
        np.array([0.2, -0.1, 0.3]),
        np.array([2.0, -1.0, 0.5]),
    )
    parameters = SpacecraftParameters(
        120.0,
        np.array([[20.0, -0.3, 0.1], [-0.3, 22.0, -0.2], [0.1, -0.2, 25.0]]),
    )
    return state, parameters


def test_eq6_matches_unified_eq11_sign_convention() -> None:
    state, parameters = _sample_state_and_parameters()
    wrench = GeneralizedForce(
        np.array([0.5, -0.2, 0.1]),
        np.array([1.0, -2.0, 0.3]),
    )
    omega_dot, velocity_dot = body_dynamics_rhs(state, parameters, wrench)
    unified = momentum_dynamics_rhs(state, parameters, wrench)
    assert np.allclose(unified[:3], omega_dot, atol=1e-13)
    assert np.allclose(unified[3:], velocity_dot, atol=1e-13)


def test_simple_force_and_torque_response() -> None:
    state = SpacecraftState(np.eye(3), np.array([7e6, 0.0, 0.0]), np.zeros(3), np.zeros(3))
    parameters = SpacecraftParameters(100.0, np.diag([2.0, 4.0, 5.0]))
    wrench = GeneralizedForce(np.array([2.0, 4.0, 5.0]), np.array([100.0, 0.0, -50.0]))
    omega_dot, velocity_dot = body_dynamics_rhs(state, parameters, wrench)
    assert np.allclose(omega_dot, np.ones(3))
    assert np.allclose(velocity_dot, np.array([1.0, 0.0, -0.5]))


def test_zero_wrench_stationary_state_is_equilibrium_without_gravity() -> None:
    state = SpacecraftState(np.eye(3), np.array([7e6, 0.0, 0.0]), np.zeros(3), np.zeros(3))
    parameters = SpacecraftParameters(100.0, np.diag([2.0, 4.0, 5.0]))
    derivative = state_derivative(
        state,
        parameters,
        GeneralizedForce.zeros(),
        GeneralizedForce.zeros(),
        GravityOptions(include_central=False),
    )
    assert np.allclose(derivative.rotation, np.zeros((3, 3)))
    assert np.allclose(derivative.position, np.zeros(3))
    assert np.allclose(derivative.omega, np.zeros(3))
    assert np.allclose(derivative.velocity, np.zeros(3))
