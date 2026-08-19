"""Controller-independent Phase-2 task geometry and constraint metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dynamics.lie import inverse_transform, make_transform, se3_log, so3_log
from dynamics.relative import RelativeState


FloatArray = NDArray[np.float64]


def _vector3(value: tuple[float, float, float], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-vector")
    return array


@dataclass(frozen=True)
class Phase2TaskConfig:
    """Frozen nominal V1 geometry, constraints, and completion thresholds."""

    port_position_target_m: tuple[float, float, float] = (-1.5, 0.0, 0.0)
    desired_position_target_m: tuple[float, float, float] = (-3.0, 0.0, 0.0)
    approach_axis_target: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    camera_boresight_chaser: tuple[float, float, float] = (1.0, 0.0, 0.0)
    corridor_half_angle_rad: float = float(np.deg2rad(35.0))
    fov_half_angle_rad: float = float(np.deg2rad(50.0))
    total_speed_limit_m_s: float = 0.35
    closing_speed_min_m_s: float = 0.08
    closing_speed_slope_per_s: float = 0.04
    closing_speed_max_m_s: float = 0.30
    completion_position_m: float = 0.25
    completion_attitude_rad: float = float(np.deg2rad(10.0))
    completion_speed_m_s: float = 0.05
    completion_angular_velocity_rad_s: float = 0.02
    completion_hold_s: float = 1.0
    constraint_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in (
            "port_position_target_m",
            "desired_position_target_m",
            "approach_axis_target",
            "camera_boresight_chaser",
        ):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        axis = _vector3(self.approach_axis_target, "approach_axis_target")
        boresight = _vector3(self.camera_boresight_chaser, "camera_boresight_chaser")
        if not np.isclose(np.linalg.norm(axis), 1.0, atol=1.0e-12):
            raise ValueError("approach axis must be unit length")
        if not np.isclose(np.linalg.norm(boresight), 1.0, atol=1.0e-12):
            raise ValueError("camera boresight must be unit length")
        positive = (
            self.corridor_half_angle_rad,
            self.fov_half_angle_rad,
            self.total_speed_limit_m_s,
            self.closing_speed_min_m_s,
            self.closing_speed_slope_per_s,
            self.closing_speed_max_m_s,
            self.completion_position_m,
            self.completion_attitude_rad,
            self.completion_speed_m_s,
            self.completion_angular_velocity_rad_s,
            self.completion_hold_s,
            self.constraint_tolerance,
        )
        if min(positive) <= 0.0:
            raise ValueError("task scales and tolerances must be positive")
        if max(self.corridor_half_angle_rad, self.fov_half_angle_rad) >= np.pi / 2:
            raise ValueError("corridor and FOV half angles must be below 90 degrees")
        if self.closing_speed_min_m_s > self.closing_speed_max_m_s:
            raise ValueError("closing speed minimum cannot exceed maximum")

    @property
    def port_position(self) -> FloatArray:
        return _vector3(self.port_position_target_m, "port_position_target_m")

    @property
    def desired_position(self) -> FloatArray:
        return _vector3(self.desired_position_target_m, "desired_position_target_m")

    @property
    def approach_axis(self) -> FloatArray:
        return _vector3(self.approach_axis_target, "approach_axis_target")

    @property
    def camera_boresight(self) -> FloatArray:
        return _vector3(self.camera_boresight_chaser, "camera_boresight_chaser")

    @property
    def desired_transform(self) -> FloatArray:
        return make_transform(np.eye(3), self.desired_position)

    def closing_speed_limit(self, axial_remaining_m: float) -> float:
        return float(
            min(
                self.closing_speed_max_m_s,
                self.closing_speed_min_m_s
                + self.closing_speed_slope_per_s * max(float(axial_remaining_m), 0.0),
            )
        )

    def completion_required_steps(self, dt_s: float) -> int:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        return int(np.ceil(self.completion_hold_s / dt_s - 1.0e-12))


@dataclass(frozen=True)
class Phase2MissionConfig:
    """Canonical two-phase mission settings outside the terminal task."""

    gate_position_target_m: tuple[float, float, float] = (-8.0, 0.0, 0.0)
    gate_semantics: Literal["legacy_regulation", "acquisition_v2"] = (
        "legacy_regulation"
    )
    gate_position_tolerance_m: float = 1.0
    gate_attitude_tolerance_rad: float = float(np.deg2rad(30.0))
    gate_speed_tolerance_m_s: float = 0.20
    gate_angular_velocity_tolerance_rad_s: float = 0.03
    gate_fov_tolerance_rad: float = float(np.deg2rad(45.0))
    initial_distance_min_m: float = 12.0
    initial_distance_max_m: float = 18.0
    initial_direction_half_angle_rad: float = float(np.deg2rad(20.0))
    initial_attitude_limit_rad: float = float(np.deg2rad(45.0))
    initial_speed_limit_m_s: float = 0.20
    initial_angular_velocity_component_limit_rad_s: float = 0.02
    # Retained for historical-manifest compatibility; no longer terminates Phase I.
    phase1_speed_limit_m_s: float = 0.50
    phase1_soft_speed_m_s: float = 0.25
    phase1_cruise_speed_m_s: float = 0.18
    phase1_catastrophic_speed_limit_m_s: float = 1.00
    premature_entry_distance_m: float = 6.0
    gate_reward: float = 5.0
    final_success_reward: float = 20.0
    terminal_constraint_failure_penalty: float = -10.0
    phase1_failure_penalty: float = -15.0
    catastrophic_failure_penalty: float = -15.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gate_position_target_m",
            tuple(float(x) for x in self.gate_position_target_m),
        )
        _vector3(self.gate_position_target_m, "gate_position_target_m")
        positive = (
            self.gate_position_tolerance_m,
            self.gate_attitude_tolerance_rad,
            self.gate_speed_tolerance_m_s,
            self.gate_angular_velocity_tolerance_rad_s,
            self.gate_fov_tolerance_rad,
            self.initial_distance_min_m,
            self.initial_distance_max_m,
            self.initial_direction_half_angle_rad,
            self.initial_attitude_limit_rad,
            self.initial_speed_limit_m_s,
            self.initial_angular_velocity_component_limit_rad_s,
            self.phase1_speed_limit_m_s,
            self.phase1_soft_speed_m_s,
            self.phase1_cruise_speed_m_s,
            self.phase1_catastrophic_speed_limit_m_s,
            self.premature_entry_distance_m,
        )
        if min(positive) <= 0.0:
            raise ValueError("mission scales and thresholds must be positive")
        if self.initial_distance_min_m >= self.initial_distance_max_m:
            raise ValueError("mission initial distance bounds are invalid")
        if self.gate_semantics not in {"legacy_regulation", "acquisition_v2"}:
            raise ValueError("unsupported Gate semantics")
        if max(
            self.initial_direction_half_angle_rad,
            self.initial_attitude_limit_rad,
        ) >= np.pi:
            raise ValueError("mission angle bounds must be below pi")
        if self.phase1_soft_speed_m_s >= self.phase1_catastrophic_speed_limit_m_s:
            raise ValueError("Phase-I catastrophic speed must exceed the soft speed")
        if self.phase1_cruise_speed_m_s >= self.phase1_catastrophic_speed_limit_m_s:
            raise ValueError("Phase-I cruise speed must be below catastrophic speed")
        if min(
            self.gate_reward,
            self.final_success_reward,
            -self.terminal_constraint_failure_penalty,
            -self.phase1_failure_penalty,
            -self.catastrophic_failure_penalty,
        ) <= 0.0:
            raise ValueError("mission event rewards must have declared signs")

    @property
    def gate_position(self) -> FloatArray:
        return _vector3(self.gate_position_target_m, "gate_position_target_m")

    def gate_satisfied(
        self,
        metrics: "MissionMetrics",
        *,
        fov_angle_rad: float | None = None,
    ) -> bool:
        if self.gate_semantics == "acquisition_v2":
            if fov_angle_rad is None:
                raise ValueError("acquisition_v2 Gate requires the FOV angle")
            return bool(
                metrics.gate_position_error_m <= self.gate_position_tolerance_m
                and metrics.total_speed_m_s <= self.gate_speed_tolerance_m_s
                and fov_angle_rad <= self.gate_fov_tolerance_rad
            )
        return bool(
            metrics.gate_position_error_m <= self.gate_position_tolerance_m
            and metrics.attitude_error_rad <= self.gate_attitude_tolerance_rad
            and metrics.total_speed_m_s <= self.gate_speed_tolerance_m_s
            and metrics.angular_velocity_error_rad_s
            <= self.gate_angular_velocity_tolerance_rad_s
        )


@dataclass(frozen=True)
class MissionMetrics:
    gate_position_error_m: float
    target_center_distance_m: float
    attitude_error_rad: float
    total_speed_m_s: float
    angular_velocity_error_rad_s: float


def compute_mission_metrics(
    relative: RelativeState,
    mission: Phase2MissionConfig = Phase2MissionConfig(),
) -> MissionMetrics:
    position_rate = relative.rotation @ relative.velocity
    return MissionMetrics(
        gate_position_error_m=float(
            np.linalg.norm(relative.position - mission.gate_position)
        ),
        target_center_distance_m=float(np.linalg.norm(relative.position)),
        attitude_error_rad=float(
            np.linalg.norm(so3_log(relative.rotation, project=True))
        ),
        total_speed_m_s=float(np.linalg.norm(position_rate)),
        angular_velocity_error_rad_s=float(np.linalg.norm(relative.omega)),
    )


@dataclass(frozen=True)
class TaskMetrics:
    position_target_m: FloatArray
    position_rate_target_m_s: FloatArray
    desired_error_coordinates: FloatArray
    axial_remaining_m: float
    port_axial_distance_m: float
    corridor_radial_distance_m: float
    corridor_axial_margin_m: float
    corridor_lateral_margin_m: float
    fov_angle_rad: float
    fov_margin_rad: float
    total_speed_m_s: float
    total_speed_margin_m_s: float
    closing_speed_m_s: float
    closing_speed_limit_m_s: float
    closing_speed_margin_m_s: float
    position_error_m: float
    attitude_error_rad: float
    angular_velocity_error_rad_s: float
    instantaneous_completion: bool
    constraints_satisfied: bool


def orthogonal_plane_basis(unit_axis: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return a deterministic right-handed basis orthogonal to a unit axis."""

    axis = np.asarray(unit_axis, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("unit_axis must be a finite three-vector")
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1.0e-12):
        raise ValueError("unit_axis must have unit length")
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def compute_task_metrics(
    relative: RelativeState,
    config: Phase2TaskConfig = Phase2TaskConfig(),
) -> TaskMetrics:
    """Compute the single source of truth for Phase-2 task constraints."""

    rotation = relative.rotation
    position = relative.position
    position_rate = rotation @ relative.velocity
    desired_error = inverse_transform(config.desired_transform) @ relative.transform
    desired_coordinates = se3_log(desired_error, project=True)

    axis = config.approach_axis
    port_displacement = position - config.port_position
    port_axial_distance = float(axis @ port_displacement)
    lateral = port_displacement - port_axial_distance * axis
    radial_distance = float(np.linalg.norm(lateral))
    axial_margin = port_axial_distance
    lateral_margin = float(
        port_axial_distance * np.tan(config.corridor_half_angle_rad)
        - radial_distance
    )

    line_of_sight_target = config.port_position - position
    line_of_sight_chaser = rotation.T @ line_of_sight_target
    los_norm = float(np.linalg.norm(line_of_sight_chaser))
    if los_norm <= np.finfo(np.float64).eps:
        fov_angle = float(np.pi)
    else:
        unit_los = line_of_sight_chaser / los_norm
        cosine = float(np.clip(config.camera_boresight @ unit_los, -1.0, 1.0))
        sine = float(
            np.linalg.norm(np.cross(config.camera_boresight, unit_los))
        )
        fov_angle = float(np.arctan2(sine, cosine))
    fov_margin = float(config.fov_half_angle_rad - fov_angle)

    axial_remaining = float(axis @ (position - config.desired_position))
    total_speed = float(np.linalg.norm(position_rate))
    closing_speed = float(-axis @ position_rate)
    closing_limit = config.closing_speed_limit(axial_remaining)
    total_speed_margin = float(config.total_speed_limit_m_s - total_speed)
    closing_margin = float(closing_limit - closing_speed)

    position_error = float(np.linalg.norm(position - config.desired_position))
    attitude_error = float(np.linalg.norm(so3_log(desired_error[:3, :3], project=True)))
    angular_velocity_error = float(np.linalg.norm(relative.omega))
    instantaneous_completion = bool(
        position_error <= config.completion_position_m
        and attitude_error <= config.completion_attitude_rad
        and total_speed <= config.completion_speed_m_s
        and angular_velocity_error <= config.completion_angular_velocity_rad_s
    )
    tolerance = config.constraint_tolerance
    constraints_satisfied = bool(
        axial_margin >= -tolerance
        and lateral_margin >= -tolerance
        and fov_margin >= -tolerance
        and total_speed_margin >= -tolerance
        and closing_margin >= -tolerance
    )
    return TaskMetrics(
        position_target_m=position.copy(),
        position_rate_target_m_s=position_rate.copy(),
        desired_error_coordinates=desired_coordinates,
        axial_remaining_m=axial_remaining,
        port_axial_distance_m=port_axial_distance,
        corridor_radial_distance_m=radial_distance,
        corridor_axial_margin_m=axial_margin,
        corridor_lateral_margin_m=lateral_margin,
        fov_angle_rad=fov_angle,
        fov_margin_rad=fov_margin,
        total_speed_m_s=total_speed,
        total_speed_margin_m_s=total_speed_margin,
        closing_speed_m_s=closing_speed,
        closing_speed_limit_m_s=closing_limit,
        closing_speed_margin_m_s=closing_margin,
        position_error_m=position_error,
        attitude_error_rad=attitude_error,
        angular_velocity_error_rad_s=angular_velocity_error,
        instantaneous_completion=instantaneous_completion,
        constraints_satisfied=constraints_satisfied,
    )
