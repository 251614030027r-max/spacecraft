"""RK45 propagation of absolute spacecraft states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .constants import EARTH
from .disturbance import zero_disturbance
from .gravity import GravityOptions, second_moment_from_inertia
from .spacecraft import pack_derivative, pack_state, state_derivative, unpack_state
from .types import GeneralizedForce, SpacecraftParameters, SpacecraftState


ControlLaw = Callable[[float, SpacecraftState], GeneralizedForce]
DisturbanceLaw = Callable[[float], GeneralizedForce]


@dataclass(frozen=True)
class RK45Settings:
    rtol: float = 1e-9
    atol: float = 1e-11
    max_step: float | None = None


@dataclass(frozen=True)
class IntegrationDiagnostics:
    nfev: int
    njev: int
    status: int
    message: str


def _as_control_law(
    control: GeneralizedForce | ControlLaw,
) -> ControlLaw:
    if isinstance(control, GeneralizedForce):
        return lambda _time, _state: control
    if callable(control):
        return control
    raise TypeError("control 必须是 GeneralizedForce 或可调用控制律")


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross for two 3-vectors with the same products and differences (bitwise equal, no axis handling)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _hat3(x, y, z) -> np.ndarray:
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _constant_wrench_rhs(parameters: SpacecraftParameters, control: GeneralizedForce, gravity_options: GravityOptions):
    """The right-hand side of ``state_derivative`` for a constant control and no disturbance, without the per-call
    object construction and validation. Every arithmetic operation is the reference path's, in the same order and on
    freshly copied (contiguous) arrays, so the derivative is bitwise equal (tests/test_t2_stage.py checks it); a
    non-finite state or a radius <= 1 m still raises ValueError, as the reference path's validation does."""
    inertia, mass = parameters.inertia, parameters.mass
    mu, j2, re = EARTH.mu, EARTH.j2, EARTH.equatorial_radius
    moment = second_moment_from_inertia(inertia)
    c_torque, c_force, zero = control.torque, control.force, np.zeros(3)
    opts = gravity_options

    def rhs(_time: float, packed: np.ndarray) -> np.ndarray:
        packed = np.asarray(packed, dtype=np.float64)
        if packed.shape != (18,) or not np.isfinite(packed).all():
            raise ValueError("积分状态必须为有限的 shape=(18,)")
        rotation = packed[:9].reshape(3, 3).copy()
        position, omega, velocity = packed[9:12].copy(), packed[12:15].copy(), packed[15:18].copy()
        if opts.include_central:
            radius = float(np.linalg.norm(position))
            if radius <= 1.0:
                raise ValueError("orbital radius is too small")
            position_body = rotation.T @ position
            torque = 3.0 * mu / radius**5 * (_hat3(*position_body) @ inertia @ position_body)
            force = -mu * float(mass) / radius**3 * position_body
            if opts.include_second_moment:
                quadratic = float(position_body @ moment @ position_body)
                force = (
                    force
                    - 3.0 * mu / radius**5 * (moment @ position_body)
                    + 7.5 * mu * quadratic / radius**7 * position_body
                )
            if opts.include_j2:
                x, y, z = position
                z_ratio = z * z / (radius * radius)
                inertial_shape = np.array(
                    [x * (1.0 - 5.0 * z_ratio), y * (1.0 - 5.0 * z_ratio), z * (3.0 - 5.0 * z_ratio)], dtype=np.float64)
                coefficient = -1.5 * mu * float(mass) * j2 * re**2 / radius**5
                force += coefficient * (rotation.T @ inertial_shape)
        else:
            torque, force = np.zeros(3), np.zeros(3)
        total_torque = (torque + c_torque) + zero
        total_force = (force + c_force) + zero
        angular_rhs = -_cross3(omega, inertia @ omega) + total_torque
        omega_dot = np.linalg.solve(inertia, angular_rhs)
        velocity_dot = -_cross3(omega, velocity) + (total_force / mass)
        return np.concatenate(((rotation @ _hat3(*omega)).reshape(9), rotation @ velocity, omega_dot, velocity_dot))
    return rhs


def propagate_rk45(
    state: SpacecraftState,
    parameters: SpacecraftParameters,
    time_seconds: float,
    duration: float,
    *,
    control: GeneralizedForce | ControlLaw | None = None,
    disturbance: DisturbanceLaw = zero_disturbance,
    gravity_options: GravityOptions = GravityOptions(),
    settings: RK45Settings = RK45Settings(),
) -> tuple[SpacecraftState, IntegrationDiagnostics]:
    if duration <= 0.0:
        raise ValueError("duration 必须为正数")
    control_law = _as_control_law(control or GeneralizedForce.zeros())
    t0 = float(time_seconds)
    t1 = t0 + float(duration)

    def rhs(current_time: float, packed: np.ndarray) -> np.ndarray:
        current_state = unpack_state(packed)
        derivative = state_derivative(
            current_state,
            parameters,
            control_law(current_time, current_state),
            disturbance(current_time),
            gravity_options,
        )
        return pack_derivative(derivative)

    if isinstance(control, GeneralizedForce) and disturbance is zero_disturbance:
        rhs = _constant_wrench_rhs(parameters, control, gravity_options)   # same arithmetic, less overhead
    max_step = duration if settings.max_step is None else settings.max_step
    solution = solve_ivp(
        rhs,
        (t0, t1),
        pack_state(state),
        method="RK45",
        rtol=settings.rtol,
        atol=settings.atol,
        max_step=max_step,
        t_eval=[t1],
    )
    if not solution.success or solution.y.shape[1] != 1:
        raise RuntimeError(f"RK45 传播失败: {solution.message}")
    propagated = unpack_state(solution.y[:, -1], project_rotation=True)
    diagnostics = IntegrationDiagnostics(
        nfev=int(solution.nfev),
        njev=int(solution.njev),
        status=int(solution.status),
        message=str(solution.message),
    )
    return propagated, diagnostics
