"""Multiplicative-error EKF for the 12D relative SE(3) state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from controllers.mpc.prediction import (
    LocalRelativePredictionModel,
    RelativePredictionModel,
    relative_to_vector,
)
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.lie import make_transform, se3_exp, so3_exp, so3_log
from dynamics.relative import RelativeState, reconstruct_target_state
from dynamics.types import SpacecraftParameters, SpacecraftState
from env.perception import (
    FeatureMeasurement,
    PerceptionConfig,
    project_feature_points,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RelativeEKFConfig:
    """Fixed A1 covariance and finite-difference settings."""

    initial_attitude_std_rad: float = float(np.deg2rad(5.0))
    initial_position_std_m: float = 0.50
    initial_angular_velocity_std_rad_s: float = 0.005
    initial_velocity_std_m_s: float = 0.05
    process_attitude_std_rad: float = 2.0e-4
    process_position_std_m: float = 1.0e-3
    process_angular_velocity_std_rad_s: float = 1.0e-4
    process_velocity_std_m_s: float = 5.0e-4
    attitude_difference_step_rad: float = 1.0e-6
    position_difference_step_m: float = 1.0e-5
    angular_velocity_difference_step_rad_s: float = 1.0e-7
    velocity_difference_step_m_s: float = 1.0e-6

    def __post_init__(self) -> None:
        covariance_scales = np.concatenate(
            (self.initial_std_vector, self.process_std_vector)
        )
        if not np.all(np.isfinite(covariance_scales)) or min(covariance_scales) <= 0.0:
            raise ValueError("EKF covariance scales must be positive")
        if (
            not np.all(np.isfinite(self.difference_steps))
            or min(self.difference_steps) <= 0.0
        ):
            raise ValueError("EKF finite-difference steps must be positive")

    @property
    def initial_block_stds(self) -> FloatArray:
        return np.array(
            [
                self.initial_attitude_std_rad,
                self.initial_position_std_m,
                self.initial_angular_velocity_std_rad_s,
                self.initial_velocity_std_m_s,
            ],
            dtype=np.float64,
        )

    @property
    def initial_std_vector(self) -> FloatArray:
        return np.repeat(self.initial_block_stds, 3)

    @property
    def process_std_vector(self) -> FloatArray:
        return np.repeat(
            np.array(
                [
                    self.process_attitude_std_rad,
                    self.process_position_std_m,
                    self.process_angular_velocity_std_rad_s,
                    self.process_velocity_std_m_s,
                ],
                dtype=np.float64,
            ),
            3,
        )

    @property
    def difference_steps(self) -> FloatArray:
        return np.repeat(
            np.array(
                [
                    self.attitude_difference_step_rad,
                    self.position_difference_step_m,
                    self.angular_velocity_difference_step_rad_s,
                    self.velocity_difference_step_m_s,
                ],
                dtype=np.float64,
            ),
            3,
        )


def inject_error(relative: RelativeState, error: ArrayLike) -> RelativeState:
    """Apply [dtheta, dp, domega, dv] to a nominal relative state."""

    delta = np.asarray(error, dtype=np.float64)
    if delta.shape != (12,) or not np.all(np.isfinite(delta)):
        raise ValueError("relative-state error must be a finite 12-vector")
    transform = make_transform(
        relative.rotation @ so3_exp(delta[:3]),
        relative.position + delta[3:6],
    )
    return RelativeState(transform, relative.twist + delta[6:])


def local_error(nominal: RelativeState, value: RelativeState) -> FloatArray:
    """Express ``value`` in the same local error coordinates as injection."""

    return np.concatenate(
        (
            so3_log(nominal.rotation.T @ value.rotation, project=True),
            value.position - nominal.position,
            value.twist - nominal.twist,
        )
    )


class RelativeStateEKF:
    """Finite-difference EKF whose mean uses the existing local predictor."""

    def __init__(
        self,
        chaser_parameters: SpacecraftParameters,
        perception_config: PerceptionConfig,
        *,
        target_parameters: SpacecraftParameters,
        dt_s: float = 0.1,
        gravity_options: GravityOptions = GravityOptions(include_j2=True),
        solver_settings: RK45Settings = RK45Settings(
            rtol=1.0e-7, atol=1.0e-9, max_step=0.1
        ),
        config: RelativeEKFConfig | None = None,
    ) -> None:
        self.config = config or RelativeEKFConfig()
        self.perception_config = perception_config
        self._local_prediction = LocalRelativePredictionModel(
            chaser_parameters, dt_s=dt_s
        )
        self._mean_prediction = RelativePredictionModel(
            target_parameters=target_parameters,
            chaser_parameters=chaser_parameters,
            dt_s=dt_s,
            gravity_options=gravity_options,
            solver_settings=solver_settings,
        )
        self._state: RelativeState | None = None
        self._covariance: FloatArray | None = None

    @property
    def state(self) -> RelativeState:
        if self._state is None:
            raise RuntimeError("EKF has not been initialized")
        return self._state

    @property
    def covariance(self) -> FloatArray:
        if self._covariance is None:
            raise RuntimeError("EKF has not been initialized")
        return self._covariance.copy()

    def initialize(
        self, truth: RelativeState, rng: np.random.Generator
    ) -> RelativeState:
        initial_std = self.config.initial_std_vector
        self._state = inject_error(truth, rng.normal(size=12) * initial_std)
        self._covariance = np.diag(initial_std * initial_std)
        return self.state

    def _propagate_local(
        self, relative: RelativeState, wrench_vector: ArrayLike
    ) -> RelativeState:
        prediction = self._local_prediction.predict(
            relative_to_vector(relative), wrench_vector
        )
        return RelativeState(se3_exp(prediction[:6]), prediction[6:])

    def _propagate_mean(
        self,
        relative: RelativeState,
        wrench_vector: ArrayLike,
        chaser_state: SpacecraftState,
        time_seconds: float,
    ) -> RelativeState:
        target_estimate = reconstruct_target_state(chaser_state, relative)
        prediction, _ = self._mean_prediction.predict(
            relative_to_vector(relative),
            wrench_vector,
            target_estimate,
            time_seconds,
        )
        return RelativeState(se3_exp(prediction[:6]), prediction[6:])

    def predict(
        self,
        wrench_vector: ArrayLike,
        *,
        chaser_state: SpacecraftState,
        time_seconds: float = 0.0,
    ) -> RelativeState:
        state = self.state
        covariance = self.covariance
        # G0 correction: the mean now follows the same absolute rigid-body and
        # gravity equations as truth, reconstructed only from the estimated
        # relative state and known ownship state. Covariance propagation keeps
        # the frozen A1 local Jacobian and process noise, so this is one model
        # correction rather than noise tuning.
        nominal_next = self._propagate_mean(
            state, wrench_vector, chaser_state, time_seconds
        )
        local_nominal_next = self._propagate_local(state, wrench_vector)
        transition = np.empty((12, 12), dtype=np.float64)
        for index, step in enumerate(self.config.difference_steps):
            delta = np.zeros(12, dtype=np.float64)
            delta[index] = step
            plus = self._propagate_local(inject_error(state, delta), wrench_vector)
            minus = self._propagate_local(inject_error(state, -delta), wrench_vector)
            transition[:, index] = (
                local_error(local_nominal_next, plus)
                - local_error(local_nominal_next, minus)
            ) / (2.0 * step)
        process_std = self.config.process_std_vector
        self._state = nominal_next
        self._covariance = (
            transition @ covariance @ transition.T
            + np.diag(process_std * process_std)
        )
        return self.state

    def update(self, measurement: FeatureMeasurement) -> bool:
        if measurement.visible_count == 0:
            return False
        state = self.state
        covariance = self.covariance
        indices = measurement.feature_indices
        predicted = project_feature_points(
            state, indices, self.perception_config
        ).reshape(-1)
        jacobian = np.empty((predicted.size, 12), dtype=np.float64)
        for index, step in enumerate(self.config.difference_steps):
            delta = np.zeros(12, dtype=np.float64)
            delta[index] = step
            plus = project_feature_points(
                inject_error(state, delta), indices, self.perception_config
            ).reshape(-1)
            minus = project_feature_points(
                inject_error(state, -delta), indices, self.perception_config
            ).reshape(-1)
            jacobian[:, index] = (plus - minus) / (2.0 * step)
        measurement_variance = self.perception_config.pixel_noise_std**2
        measurement_covariance = measurement_variance * np.eye(predicted.size)
        innovation_covariance = (
            jacobian @ covariance @ jacobian.T + measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            jacobian @ covariance,
        ).T
        innovation = measurement.image_points_px.reshape(-1) - predicted
        self._state = inject_error(state, gain @ innovation)
        identity = np.eye(12)
        residual_map = identity - gain @ jacobian
        updated_covariance = (
            residual_map @ covariance @ residual_map.T
            + gain @ measurement_covariance @ gain.T
        )
        self._covariance = 0.5 * (updated_covariance + updated_covariance.T)
        return True
