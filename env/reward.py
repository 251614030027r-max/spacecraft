"""Physical SE(3) storage reward used by the clean SAC baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from dynamics.relative import RelativeState
from dynamics.lie import so3_log
from env.task import Phase2TaskConfig, TaskMetrics, compute_task_metrics


@dataclass(frozen=True)
class RewardBreakdown:
    storage_dissipation: float
    storage_penalty: float
    actuation_penalty: float
    storage_energy: float
    storage_potential: float
    safety_barrier: float

    @property
    def total(self) -> float:
        return (
            self.storage_dissipation
            + self.storage_penalty
            + self.actuation_penalty
        )


class PhysicalStorageReward:
    """Dimensionless reward with three physically interpretable terms.

    The storage energy is

    ``V = ||[ell*phi, rho]||^2
           + tau^2 ||[ell*(omega+phi/tau), v+rho/tau]||^2``.

    The potential is ``Psi = log(1+V/ell^2)-log(1-(d/d_max)^2)``.
    Reward equals potential dissipation, time-integrated storage, and
    normalized actuator utilization. There are no terminal bonuses, fitted
    channel weights, or failure-penalty fallbacks.
    """

    def __init__(
        self,
        *,
        characteristic_length_m: float,
        time_constant_s: float,
        time_step_s: float,
        max_distance_m: float,
        barrier_epsilon: float = 1.0e-6,
    ) -> None:
        if min(
            characteristic_length_m,
            time_constant_s,
            time_step_s,
            max_distance_m,
        ) <= 0.0:
            raise ValueError("physical reward scales must be positive")
        if not 0.0 < barrier_epsilon < 1.0:
            raise ValueError("barrier_epsilon must lie in (0, 1)")
        self.characteristic_length_m = float(characteristic_length_m)
        self.time_constant_s = float(time_constant_s)
        self.time_step_s = float(time_step_s)
        self.max_distance_m = float(max_distance_m)
        self.barrier_epsilon = float(barrier_epsilon)
        self.step_weight = self.time_step_s / self.time_constant_s
        self._previous_potential: float | None = None

    def energy(self, relative: RelativeState) -> float:
        coordinates = relative.exponential_coordinates
        phi, rho = coordinates[:3], coordinates[3:]
        ell = self.characteristic_length_m
        tau = self.time_constant_s
        pose_displacement = np.concatenate((ell * phi, rho))
        flow_residual = np.concatenate(
            (ell * (relative.omega + phi / tau), relative.velocity + rho / tau)
        )
        return float(
            pose_displacement @ pose_displacement
            + tau * tau * (flow_residual @ flow_residual)
        )

    def safety_barrier(self, relative: RelativeState) -> float:
        ratio = float(np.linalg.norm(relative.position)) / self.max_distance_m
        margin = max(1.0 - ratio * ratio, self.barrier_epsilon)
        return -float(np.log(margin))

    def potential(self, relative: RelativeState) -> float:
        normalized_energy = self.energy(relative) / self.characteristic_length_m**2
        return float(np.log1p(normalized_energy) + self.safety_barrier(relative))

    def reset(self, relative: RelativeState) -> None:
        potential = self.potential(relative)
        if not np.isfinite(potential):
            raise ValueError("initial storage potential must be finite")
        self._previous_potential = potential

    def compute(
        self, relative: RelativeState, normalized_action: ArrayLike
    ) -> RewardBreakdown:
        if self._previous_potential is None:
            raise RuntimeError("reset() must be called before compute()")
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("normalized_action must be a finite shape=(6,) vector")
        energy = self.energy(relative)
        barrier = self.safety_barrier(relative)
        potential = float(
            np.log1p(energy / self.characteristic_length_m**2) + barrier
        )
        effort = float(np.mean(np.clip(action, -1.0, 1.0) ** 2))
        breakdown = RewardBreakdown(
            storage_dissipation=self._previous_potential - potential,
            storage_penalty=-self.step_weight * potential,
            actuation_penalty=-self.step_weight * effort,
            storage_energy=energy,
            storage_potential=potential,
            safety_barrier=barrier,
        )
        self._previous_potential = potential
        return breakdown


@dataclass(frozen=True)
class Phase2RewardBreakdown:
    task_dissipation: float
    task_penalty: float
    actuation_penalty: float
    corridor_penalty: float
    fov_penalty: float
    total_speed_penalty: float
    closing_speed_penalty: float
    task_potential: float

    @property
    def constraint_penalty(self) -> float:
        return (
            self.corridor_penalty
            + self.fov_penalty
            + self.total_speed_penalty
            + self.closing_speed_penalty
        )

    @property
    def total(self) -> float:
        return (
            self.task_dissipation
            + self.task_penalty
            + self.actuation_penalty
            + self.constraint_penalty
        )


class Phase2TaskReward:
    """Finite desired-pose storage reward with explicit signed-margin costs."""

    def __init__(
        self,
        *,
        task: Phase2TaskConfig,
        time_step_s: float,
        time_constant_s: float,
        attitude_scale_rad: float,
        distance_scale_m: float,
        angular_velocity_scale_rad_s: float,
        velocity_scale_m_s: float,
        constraint_buffer_fraction: float = 0.10,
        constraint_softness: float = 0.05,
        maximum_normalized_constraint_cost: float = 4.0,
    ) -> None:
        self.task = task
        self.step_weight = time_step_s / time_constant_s
        self.scales = np.array(
            [
                *([attitude_scale_rad] * 3),
                *([distance_scale_m] * 3),
                *([angular_velocity_scale_rad_s] * 3),
                *([velocity_scale_m_s] * 3),
            ],
            dtype=np.float64,
        )
        if min(time_step_s, time_constant_s, *self.scales) <= 0.0:
            raise ValueError("Phase-2 reward scales must be positive")
        if min(
            constraint_buffer_fraction,
            constraint_softness,
            maximum_normalized_constraint_cost,
        ) <= 0.0:
            raise ValueError("Phase-2 proximity parameters must be positive")
        self.constraint_buffer_fraction = float(constraint_buffer_fraction)
        self.constraint_softness = float(constraint_softness)
        self.maximum_normalized_constraint_cost = float(
            maximum_normalized_constraint_cost
        )
        self._previous_potential: float | None = None

    def potential(self, relative: RelativeState) -> float:
        metrics = compute_task_metrics(relative, self.task)
        error = np.concatenate(
            (metrics.desired_error_coordinates, relative.twist)
        )
        return float(np.log1p(np.mean((error / self.scales) ** 2)))

    def reset(self, relative: RelativeState) -> None:
        self._previous_potential = self.potential(relative)

    def _proximity_cost(self, margin: float, scale: float) -> float:
        """Bounded smooth cost that becomes visible before the boundary."""

        normalized_margin = float(margin) / max(float(scale), np.finfo(float).eps)
        argument = (
            self.constraint_buffer_fraction - normalized_margin
        ) / self.constraint_softness
        cost = self.constraint_softness * float(np.logaddexp(0.0, argument))
        return -min(cost, self.maximum_normalized_constraint_cost)

    def compute(
        self, relative: RelativeState, normalized_action: ArrayLike
    ) -> Phase2RewardBreakdown:
        if self._previous_potential is None:
            raise RuntimeError("reset() must be called before compute()")
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("normalized_action must be a finite shape=(6,) vector")
        metrics = compute_task_metrics(relative, self.task)
        potential = self.potential(relative)
        result = Phase2RewardBreakdown(
            task_dissipation=self._previous_potential - potential,
            task_penalty=-self.step_weight * potential,
            actuation_penalty=-self.step_weight
            * float(np.mean(np.clip(action, -1.0, 1.0) ** 2)),
            corridor_penalty=self.step_weight
            * self._proximity_cost(
                metrics.corridor_lateral_margin_m,
                max(
                    metrics.port_axial_distance_m
                    * np.tan(self.task.corridor_half_angle_rad),
                    1.0e-6,
                ),
            ),
            fov_penalty=self.step_weight
            * self._proximity_cost(
                metrics.fov_margin_rad, self.task.fov_half_angle_rad
            ),
            total_speed_penalty=self.step_weight
            * self._proximity_cost(
                metrics.total_speed_margin_m_s, self.task.total_speed_limit_m_s
            ),
            closing_speed_penalty=self.step_weight
            * self._proximity_cost(
                metrics.closing_speed_margin_m_s,
                self.task.closing_speed_max_m_s,
            ),
            task_potential=potential,
        )
        self._previous_potential = potential
        return result


@dataclass(frozen=True)
class Phase2MissionRewardBreakdown:
    progress: float
    state_penalty: float
    actuation_penalty: float
    constraint_warning: float
    event_reward: float
    potential: float

    @property
    def total(self) -> float:
        return (
            self.progress
            + self.state_penalty
            + self.actuation_penalty
            + self.constraint_warning
            + self.event_reward
        )


class Phase2MissionReward:
    """Unified active-reference reward for the canonical two-phase mission."""

    def __init__(
        self,
        *,
        task: Phase2TaskConfig,
        position_scale_m: float = 10.0,
        attitude_scale_rad: float = float(np.deg2rad(45.0)),
        velocity_scale_m_s: float = 0.20,
        angular_velocity_scale_rad_s: float = 0.03,
        progress_weight: float = 4.0,
        state_weight: float = 0.005,
        actuation_weight: float = 0.002,
        warning_weight: float = 1.0,
        phase1_soft_speed_m_s: float = 0.25,
        phase1_speed_span_m_s: float = 0.25,
        phase1_cruise_speed_m_s: float = 0.18,
        phase1_braking_gain_per_s: float = 0.08,
        phase1_speed_weight: float = 0.02,
        phase1_velocity_tracking_weight: float = 0.02,
        phase1_attitude_weight: float = 0.005,
        phase1_angular_velocity_weight: float = 0.005,
        discount_factor: float = 0.997,
    ) -> None:
        values = (
            position_scale_m,
            attitude_scale_rad,
            velocity_scale_m_s,
            angular_velocity_scale_rad_s,
            progress_weight,
            state_weight,
            actuation_weight,
            warning_weight,
            phase1_soft_speed_m_s,
            phase1_speed_span_m_s,
            phase1_cruise_speed_m_s,
            phase1_braking_gain_per_s,
            phase1_speed_weight,
            phase1_velocity_tracking_weight,
            phase1_attitude_weight,
            phase1_angular_velocity_weight,
            discount_factor,
        )
        if min(values) <= 0.0:
            raise ValueError("mission reward settings must be positive")
        self.task = task
        self.scales = np.array(
            [position_scale_m, attitude_scale_rad, velocity_scale_m_s, angular_velocity_scale_rad_s],
            dtype=np.float64,
        )
        self.progress_weight = float(progress_weight)
        self.state_weight = float(state_weight)
        self.actuation_weight = float(actuation_weight)
        self.warning_weight = float(warning_weight)
        self.phase1_soft_speed_m_s = float(phase1_soft_speed_m_s)
        self.phase1_speed_span_m_s = float(phase1_speed_span_m_s)
        self.phase1_cruise_speed_m_s = float(phase1_cruise_speed_m_s)
        self.phase1_braking_gain_per_s = float(phase1_braking_gain_per_s)
        self.phase1_speed_weight = float(phase1_speed_weight)
        self.phase1_velocity_tracking_weight = float(
            phase1_velocity_tracking_weight
        )
        self.phase1_attitude_weight = float(phase1_attitude_weight)
        self.phase1_angular_velocity_weight = float(
            phase1_angular_velocity_weight
        )
        self.discount_factor = float(discount_factor)
        if self.discount_factor > 1.0:
            raise ValueError("discount_factor must not exceed one")
        self._previous_potential: float | None = None
        self._previous_bounded_potential: float | None = None
        self._reference = np.zeros(3, dtype=np.float64)

    def potential(self, relative: RelativeState, reference_position_m: ArrayLike) -> float:
        reference = np.asarray(reference_position_m, dtype=np.float64)
        if reference.shape != (3,):
            raise ValueError("reference_position_m must have shape=(3,)")
        position_error = np.linalg.norm(relative.position - reference)
        attitude_error = np.linalg.norm(so3_log(relative.rotation, project=True))
        position_rate = np.linalg.norm(relative.rotation @ relative.velocity)
        angular_rate = np.linalg.norm(relative.omega)
        normalized = np.array(
            [position_error, attitude_error, position_rate, angular_rate],
            dtype=np.float64,
        ) / self.scales
        weights = np.array([1.0, 0.5, 0.25, 0.25], dtype=np.float64)
        return float(weights @ (normalized**2))

    def bounded_potential(
        self, relative: RelativeState, reference_position_m: ArrayLike
    ) -> float:
        """Bounded Phase-I cost potential used for discount-consistent shaping."""

        reference = np.asarray(reference_position_m, dtype=np.float64)
        if reference.shape != (3,):
            raise ValueError("reference_position_m must have shape=(3,)")
        position_error = np.linalg.norm(relative.position - reference)
        attitude_error = np.linalg.norm(so3_log(relative.rotation, project=True))
        velocity_error = np.linalg.norm(
            relative.rotation @ relative.velocity
            - self.phase1_desired_velocity(relative, reference)
        )
        angular_rate = np.linalg.norm(relative.omega)
        normalized_squared = (
            np.array(
                [position_error, attitude_error, velocity_error, angular_rate],
                dtype=np.float64,
            )
            / self.scales
        ) ** 2
        bounded = normalized_squared / (1.0 + normalized_squared)
        weights = np.array([1.0, 0.35, 0.5, 0.25], dtype=np.float64)
        return float(weights @ bounded)

    def phase1_desired_velocity(
        self, relative: RelativeState, reference_position_m: ArrayLike
    ) -> np.ndarray:
        """Smooth target-frame velocity field for Gate approach and braking."""

        reference = np.asarray(reference_position_m, dtype=np.float64)
        error = relative.position - reference
        distance = float(np.linalg.norm(error))
        if distance <= np.finfo(float).eps:
            return np.zeros(3, dtype=np.float64)
        desired_speed = min(
            self.phase1_cruise_speed_m_s,
            self.phase1_braking_gain_per_s * distance,
        )
        return -desired_speed * error / distance

    def active_desired_velocity(
        self,
        relative: RelativeState,
        reference_position_m: ArrayLike,
        *,
        terminal_constraints_active: bool,
    ) -> np.ndarray:
        """Formal target-frame velocity reference shared by reward and observation."""

        if terminal_constraints_active:
            # The current Phase-II active pose reference is stationary; its reward
            # penalizes the actual relative velocity norm with no separate guidance law.
            return np.zeros(3, dtype=np.float64)
        return self.phase1_desired_velocity(relative, reference_position_m)

    @staticmethod
    def _bounded_square(value: float) -> float:
        squared = float(value) ** 2
        return squared / (1.0 + squared)

    def reset(self, relative: RelativeState, reference_position_m: ArrayLike) -> None:
        self._reference = np.asarray(reference_position_m, dtype=np.float64).copy()
        self._previous_potential = self.potential(relative, self._reference)
        self._previous_bounded_potential = self.bounded_potential(
            relative, self._reference
        )

    @staticmethod
    def _warning(margin: float, scale: float, warning_fraction: float = 0.10) -> float:
        normalized_margin = float(margin) / max(float(scale), np.finfo(float).eps)
        if normalized_margin >= warning_fraction:
            return 0.0
        ratio = 1.0 - max(normalized_margin, 0.0) / warning_fraction
        return ratio * ratio

    def compute(
        self,
        relative: RelativeState,
        normalized_action: ArrayLike,
        *,
        terminal_constraints_active: bool,
        event_reward: float = 0.0,
    ) -> Phase2MissionRewardBreakdown:
        if self._previous_potential is None or self._previous_bounded_potential is None:
            raise RuntimeError("reset() must be called before compute()")
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("normalized_action must be finite and shape=(6,)")
        potential = self.potential(relative, self._reference)
        bounded_potential = self.bounded_potential(relative, self._reference)
        warning_penalty = 0.0
        if terminal_constraints_active:
            metrics = compute_task_metrics(relative, self.task)
            warning = sum(
                (
                    self._warning(
                        metrics.corridor_lateral_margin_m,
                        max(
                            metrics.port_axial_distance_m
                            * np.tan(self.task.corridor_half_angle_rad),
                            1.0e-6,
                        ),
                    ),
                    self._warning(metrics.fov_margin_rad, self.task.fov_half_angle_rad),
                    self._warning(
                        metrics.total_speed_margin_m_s,
                        self.task.total_speed_limit_m_s,
                    ),
                    self._warning(
                        metrics.closing_speed_margin_m_s,
                        self.task.closing_speed_max_m_s,
                    ),
                )
            )
            warning_penalty = -self.warning_weight * warning
        else:
            target_velocity = relative.rotation @ relative.velocity
            phase1_speed = float(np.linalg.norm(target_velocity))
            speed_excess = max(phase1_speed - self.phase1_soft_speed_m_s, 0.0)
            desired_velocity = self.phase1_desired_velocity(
                relative, self._reference
            )
            velocity_error = float(np.linalg.norm(target_velocity - desired_velocity))
            attitude_error = float(
                np.linalg.norm(so3_log(relative.rotation, project=True))
            )
            angular_velocity = float(np.linalg.norm(relative.omega))
            warning_penalty = -(
                self.phase1_speed_weight
                * self._bounded_square(speed_excess / self.phase1_speed_span_m_s)
                + self.phase1_velocity_tracking_weight
                * self._bounded_square(velocity_error / self.scales[2])
                + self.phase1_attitude_weight
                * self._bounded_square(attitude_error / self.scales[1])
                + self.phase1_angular_velocity_weight
                * self._bounded_square(angular_velocity / self.scales[3])
            )
        progress = (
            self.progress_weight * (self._previous_potential - potential)
            if terminal_constraints_active
            else self.progress_weight
            * (
                self._previous_bounded_potential
                - self.discount_factor * bounded_potential
            )
        )
        result = Phase2MissionRewardBreakdown(
            progress=progress,
            state_penalty=(
                -self.state_weight * potential if terminal_constraints_active else 0.0
            ),
            actuation_penalty=-self.actuation_weight
            * float(np.mean(np.clip(action, -1.0, 1.0) ** 2)),
            constraint_warning=warning_penalty,
            event_reward=float(event_reward),
            potential=(potential if terminal_constraints_active else bounded_potential),
        )
        self._previous_potential = potential
        self._previous_bounded_potential = bounded_potential
        return result
