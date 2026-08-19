"""Rigid-body spacecraft dynamics in body coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gravity import GravityOptions, gravity_wrench
from .lie import ad6, hat3, project_to_so3
from .types import GeneralizedForce, SpacecraftParameters, SpacecraftState


FloatArray = NDArray[np.float64]


@dataclass
class StateDerivative:
    rotation: FloatArray
    position: FloatArray
    omega: FloatArray
    velocity: FloatArray


def body_dynamics_rhs(
    state: SpacecraftState,
    parameters: SpacecraftParameters,
    applied_wrench: GeneralizedForce,
) -> tuple[FloatArray, FloatArray]:
    """Compute angular and translational body-frame accelerations."""

    angular_rhs = (
        -np.cross(state.omega, parameters.inertia @ state.omega)
        + applied_wrench.torque
    )
    omega_dot = np.linalg.solve(parameters.inertia, angular_rhs)
    velocity_dot = -np.cross(state.omega, state.velocity) + (
        applied_wrench.force / parameters.mass
    )
    return omega_dot, velocity_dot


def momentum_dynamics_rhs(
    state: SpacecraftState,
    parameters: SpacecraftParameters,
    applied_wrench: GeneralizedForce,
) -> FloatArray:
    """Equivalent generalized-momentum dynamics for cross-validation."""

    inertia6 = parameters.generalized_inertia
    momentum_rhs = ad6(state.twist).T @ (inertia6 @ state.twist)
    momentum_rhs += applied_wrench.vector
    return np.linalg.solve(inertia6, momentum_rhs)


def state_derivative(
    state: SpacecraftState,
    parameters: SpacecraftParameters,
    control: GeneralizedForce,
    disturbance: GeneralizedForce,
    gravity_options: GravityOptions = GravityOptions(),
) -> StateDerivative:
    total = gravity_wrench(state, parameters, gravity_options) + control + disturbance
    omega_dot, velocity_dot = body_dynamics_rhs(state, parameters, total)
    return StateDerivative(
        rotation=state.rotation @ hat3(state.omega),
        position=state.rotation @ state.velocity,
        omega=omega_dot,
        velocity=velocity_dot,
    )


def pack_state(state: SpacecraftState) -> FloatArray:
    return np.concatenate(
        (
            state.rotation.reshape(9),
            state.position,
            state.omega,
            state.velocity,
        )
    )


def unpack_state(value: ArrayLike, *, project_rotation: bool = False) -> SpacecraftState:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (18,):
        raise ValueError("积分状态必须为 shape=(18,)")
    rotation = array[:9].reshape(3, 3)
    if project_rotation:
        rotation = project_to_so3(rotation)
    return SpacecraftState(rotation, array[9:12], array[12:15], array[15:18])


def pack_derivative(derivative: StateDerivative) -> FloatArray:
    return np.concatenate(
        (
            derivative.rotation.reshape(9),
            derivative.position,
            derivative.omega,
            derivative.velocity,
        )
    )
