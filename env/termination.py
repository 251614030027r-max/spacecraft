"""Success metrics and episode termination conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics.relative import RelativeState


@dataclass(frozen=True)
class SuccessThresholds:
    attitude_rad: float = float(np.deg2rad(15.0))
    angular_velocity_rad_s: float = 0.03
    position_m: float = 0.5
    translational_velocity_m_s: float = 0.1


@dataclass(frozen=True)
class ErrorMetrics:
    attitude_rad: float
    angular_velocity_rad_s: float
    position_m: float
    translational_velocity_m_s: float
    direct_relative_distance_m: float
    attitude_success: bool
    position_success: bool
    joint_success: bool


def compute_error_metrics(
    relative: RelativeState,
    thresholds: SuccessThresholds = SuccessThresholds(),
) -> ErrorMetrics:
    coordinates = relative.exponential_coordinates
    attitude_error = float(np.linalg.norm(coordinates[:3]))
    position_error = float(np.linalg.norm(coordinates[3:]))
    angular_velocity_error = float(np.linalg.norm(relative.omega))
    velocity_error = float(np.linalg.norm(relative.velocity))
    distance = float(np.linalg.norm(relative.position))
    attitude_success = (
        attitude_error <= thresholds.attitude_rad
        and angular_velocity_error <= thresholds.angular_velocity_rad_s
    )
    position_success = (
        position_error <= thresholds.position_m
        and velocity_error <= thresholds.translational_velocity_m_s
    )
    return ErrorMetrics(
        attitude_rad=attitude_error,
        angular_velocity_rad_s=angular_velocity_error,
        position_m=position_error,
        translational_velocity_m_s=velocity_error,
        direct_relative_distance_m=distance,
        attitude_success=attitude_success,
        position_success=position_success,
        joint_success=attitude_success and position_success,
    )
