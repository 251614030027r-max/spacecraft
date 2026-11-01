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


def test_constant_wrench_fast_path_is_bitwise_equal_to_the_reference_rhs() -> None:
    """The constant-control fast path (compute only) returns the reference derivative bit for bit, and a
    propagation through it equals one through the generic callable path bit for bit."""
    from dynamics.disturbance import zero_disturbance
    from dynamics.integrator import _constant_wrench_rhs
    from dynamics.spacecraft import pack_derivative, pack_state, state_derivative, unpack_state

    rng = np.random.default_rng(0)
    for i in range(300):
        a = rng.normal(size=(3, 3))
        parameters = SpacecraftParameters(rng.uniform(50, 300), np.diag(rng.uniform(5, 60, 3)) + 0.1 * (a + a.T))
        state = SpacecraftState(so3_exp(rng.normal(size=3)), rng.normal(size=3) * 4e6 + np.array([6.9e6, 0.0, 0.0]),
                                rng.normal(size=3) * 0.05, rng.normal(size=3) * 7e3)
        control = GeneralizedForce(rng.uniform(-0.6, 0.6, 3), rng.uniform(-2, 2, 3)) if i % 3 else GeneralizedForce.zeros()
        options = GravityOptions(include_central=bool(i % 7), include_second_moment=bool(i % 2), include_j2=bool(i % 5))
        y = pack_state(state) + rng.normal(size=18) * 1e-3
        reference = pack_derivative(state_derivative(unpack_state(y), parameters, control, zero_disturbance(0.0), options))
        assert np.array_equal(reference.view(np.uint64), _constant_wrench_rhs(parameters, control, options)(0.0, y).view(np.uint64))
        if i < 20:
            fast, _ = propagate_rk45(state, parameters, 0.0, 0.1, control=control, gravity_options=options)
            slow, _ = propagate_rk45(state, parameters, 0.0, 0.1, control=lambda _t, _s: control, gravity_options=options)
            for key in ("rotation", "position", "omega", "velocity"):
                assert np.array_equal(getattr(fast, key).view(np.uint64), getattr(slow, key).view(np.uint64))


def test_is_so3_matches_numpy_allclose_semantics() -> None:
    rng = np.random.default_rng(1)

    def reference(m, atol):
        v = np.asarray(m, dtype=np.float64)
        return bool(v.shape == (3, 3) and np.allclose(v.T @ v, np.eye(3), atol=atol) and np.isclose(np.linalg.det(v), 1.0, atol=atol))
    for i in range(5000):
        m = so3_exp(rng.normal(size=3) * 3) + rng.normal(size=(3, 3)) * 10 ** rng.uniform(-12, -3)
        if i % 50 == 0:
            m[rng.integers(3), rng.integers(3)] = (np.nan, np.inf, -np.inf)[i % 3]
        if i % 77 == 0:
            m = -m
        for atol in (1e-7, 1e-9, 1e-12):
            assert is_so3(m, atol) == reference(m, atol)
    assert not is_so3(np.eye(4))
