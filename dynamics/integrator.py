"""RK45 propagation of absolute spacecraft states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .disturbance import zero_disturbance
from .gravity import GravityOptions
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
