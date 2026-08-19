import numpy as np

from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings, propagate_rk45
from dynamics.lie import is_so3, so3_exp
from dynamics.relative import relative_state
from dynamics.types import GeneralizedForce, SpacecraftParameters, SpacecraftState


NO_GRAVITY = GravityOptions(include_central=False)


def test_uncontrolled_linear_motion_without_gravity() -> None:
    state = SpacecraftState(np.eye(3), np.array([1.0, 2.0, 3.0]), np.zeros(3), np.array([4.0, -2.0, 1.0]))
    parameters = SpacecraftParameters(100.0, 10.0 * np.eye(3))
    result, diagnostics = propagate_rk45(
        state,
        parameters,
        0.0,
        0.5,
        gravity_options=NO_GRAVITY,
    )
    assert np.allclose(result.position, state.position + 0.5 * state.velocity, atol=1e-11)
    assert np.allclose(result.velocity, state.velocity, atol=1e-12)
    assert diagnostics.nfev > 0


def test_rotation_is_orthogonal_and_matches_constant_spherical_body_rate() -> None:
    omega = np.array([0.1, -0.2, 0.3])
    state = SpacecraftState(np.eye(3), np.array([7e6, 0.0, 0.0]), omega, np.zeros(3))
    parameters = SpacecraftParameters(100.0, 10.0 * np.eye(3))
    result, _ = propagate_rk45(
        state,
        parameters,
        0.0,
        1.0,
        gravity_options=NO_GRAVITY,
    )
    assert is_so3(result.rotation, atol=1e-12)
    assert np.allclose(result.rotation, so3_exp(omega), atol=2e-10)


def test_zero_relative_error_remains_equilibrium_for_identical_models() -> None:
    state = SpacecraftState(
        np.eye(3),
        np.array([7.0e6, 1.0e6, 2.0e6]),
        np.array([0.05, -0.03, 0.02]),
        np.array([5000.0, 3000.0, -200.0]),
    )
    parameters = SpacecraftParameters(120.0, np.diag([20.0, 22.0, 25.0]))
    a, _ = propagate_rk45(state, parameters, 0.0, 0.1)
    b, _ = propagate_rk45(state.copy(), parameters, 0.0, 0.1)
    relative = relative_state(a, b)
    assert np.allclose(relative.transform, np.eye(4), atol=2e-9)
    assert np.allclose(relative.twist, np.zeros(6), atol=2e-10)


def test_rk45_tolerance_convergence() -> None:
    state = SpacecraftState(
        np.eye(3),
        np.array([7.0e6, 1.0e6, 2.0e6]),
        np.array([0.1, 0.2, -0.1]),
        np.array([5000.0, 3000.0, -200.0]),
    )
    parameters = SpacecraftParameters(120.0, np.diag([20.0, 22.0, 25.0]))
    loose, _ = propagate_rk45(
        state,
        parameters,
        0.0,
        0.1,
        settings=RK45Settings(rtol=1e-7, atol=1e-9, max_step=0.1),
    )
    tight, _ = propagate_rk45(
        state,
        parameters,
        0.0,
        0.1,
        settings=RK45Settings(rtol=1e-10, atol=1e-12, max_step=0.05),
    )
    assert np.linalg.norm(loose.position - tight.position) < 1e-5
    assert np.linalg.norm(loose.twist - tight.twist) < 1e-8
