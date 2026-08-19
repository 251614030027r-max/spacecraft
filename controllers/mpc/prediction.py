"""Truth-consistent and fast local relative prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynamics.disturbance import zero_disturbance
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings, propagate_rk45
from dynamics.lie import se3_exp, se3_log
from dynamics.relative import (
    RelativeState,
    reconstruct_chaser_state,
    relative_state,
)
from dynamics.types import GeneralizedForce, SpacecraftParameters, SpacecraftState


FloatArray = NDArray[np.float64]


def relative_to_vector(relative: RelativeState) -> FloatArray:
    return np.concatenate((relative.exponential_coordinates, relative.twist))


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape=({size},) vector")
    return array


@dataclass(frozen=True)
class RelativePredictionModel:
    """One-step map built from the same absolute dynamics as the environment."""

    target_parameters: SpacecraftParameters
    chaser_parameters: SpacecraftParameters
    dt_s: float = 0.1
    gravity_options: GravityOptions = GravityOptions(include_j2=True)
    solver_settings: RK45Settings = RK45Settings(
        rtol=1.0e-7, atol=1.0e-9, max_step=0.1
    )

    def reconstruct_chaser(
        self, target: SpacecraftState, relative_vector: ArrayLike
    ) -> SpacecraftState:
        x = _vector(relative_vector, 12, "relative_vector")
        relative_transform = se3_exp(x[:6])
        return reconstruct_chaser_state(
            target, RelativeState(relative_transform, x[6:])
        )

    def propagate_target(
        self, target: SpacecraftState, time_seconds: float
    ) -> SpacecraftState:
        target_next, _ = propagate_rk45(
            target,
            self.target_parameters,
            time_seconds,
            self.dt_s,
            control=GeneralizedForce.zeros(),
            disturbance=zero_disturbance,
            gravity_options=self.gravity_options,
            settings=self.solver_settings,
        )
        return target_next

    def predict_given_target_next(
        self,
        relative_vector: ArrayLike,
        wrench_vector: ArrayLike,
        target: SpacecraftState,
        target_next: SpacecraftState,
        time_seconds: float,
    ) -> FloatArray:
        x = _vector(relative_vector, 12, "relative_vector")
        u = _vector(wrench_vector, 6, "wrench_vector")
        chaser = self.reconstruct_chaser(target, x)
        chaser_next, _ = propagate_rk45(
            chaser,
            self.chaser_parameters,
            time_seconds,
            self.dt_s,
            control=GeneralizedForce.from_vector(u),
            disturbance=zero_disturbance,
            gravity_options=self.gravity_options,
            settings=self.solver_settings,
        )
        return relative_to_vector(relative_state(target_next, chaser_next))

    def predict(
        self,
        relative_vector: ArrayLike,
        wrench_vector: ArrayLike,
        target: SpacecraftState,
        time_seconds: float,
    ) -> tuple[FloatArray, SpacecraftState]:
        target_next = self.propagate_target(target, time_seconds)
        prediction = self.predict_given_target_next(
            relative_vector,
            wrench_vector,
            target,
            target_next,
            time_seconds,
        )
        return prediction, target_next


@dataclass(frozen=True)
class LocalRelativePredictionModel:
    """Fast SE(3)-kinematic predictor used inside the online optimizer.

    Absolute orbital and target-tumble drift is supplied as a measured one-step
    affine bias by the controller. Input acceleration remains physical and uses
    the chaser mass and inertia.
    """

    chaser_parameters: SpacecraftParameters
    dt_s: float = 0.1

    def predict(self, relative_vector: ArrayLike, wrench_vector: ArrayLike) -> FloatArray:
        x = _vector(relative_vector, 12, "relative_vector")
        u = _vector(wrench_vector, 6, "wrench_vector")
        acceleration = np.linalg.solve(
            self.chaser_parameters.generalized_inertia, u
        )
        next_twist = x[6:] + self.dt_s * acceleration
        midpoint_twist = x[6:] + 0.5 * self.dt_s * acceleration
        next_transform = se3_exp(x[:6]) @ se3_exp(self.dt_s * midpoint_twist)
        return np.concatenate((se3_log(next_transform, project=True), next_twist))
