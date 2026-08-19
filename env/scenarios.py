"""Nominal tumbling-target scenario and bounded initial-state sampling."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from dynamics.constants import EARTH
from dynamics.lie import make_transform, so3_exp, so3_log
from dynamics.relative import RelativeState, reconstruct_chaser_state
from dynamics.types import SpacecraftParameters, SpacecraftState
from env.task import (
    Phase2MissionConfig,
    Phase2TaskConfig,
    compute_mission_metrics,
    compute_task_metrics,
    orthogonal_plane_basis,
)


TARGET_INERTIA_KG_M2 = np.array(
    [[42.7, -0.4, -0.6], [-0.4, 40.3, -0.5], [-0.6, -0.5, 37.9]],
    dtype=np.float64,
)
CHASER_INERTIA_KG_M2 = np.array(
    [[20.3, -0.3, -0.2], [-0.3, 19.9, -0.1], [-0.2, -0.1, 23.7]],
    dtype=np.float64,
)

TARGET_ORBIT_ALTITUDE_M = 500.0e3
TARGET_ORBIT_INCLINATION_RAD = float(np.deg2rad(45.0))
TARGET_BASE_TUMBLE_RAD_S = np.array([0.05, 0.2, 0.0], dtype=np.float64)


def target_initial_state(*, tumble_scale: float) -> SpacecraftState:
    if tumble_scale < 0.0:
        raise ValueError("tumble_scale must be non-negative")
    rotation = np.eye(3)
    orbital_radius = EARTH.equatorial_radius + TARGET_ORBIT_ALTITUDE_M
    circular_speed = np.sqrt(EARTH.mu / orbital_radius)
    inclination = TARGET_ORBIT_INCLINATION_RAD
    position_eci = np.array([orbital_radius, 0.0, 0.0])
    velocity_eci = np.array(
        [
            0.0,
            circular_speed * np.cos(inclination),
            circular_speed * np.sin(inclination),
        ]
    )
    return SpacecraftState(
        rotation=rotation,
        position=position_eci,
        omega=tumble_scale * TARGET_BASE_TUMBLE_RAD_S,
        velocity=rotation.T @ velocity_eci,
    )


def target_parameters() -> SpacecraftParameters:
    return SpacecraftParameters(225.0, TARGET_INERTIA_KG_M2)


def chaser_parameters() -> SpacecraftParameters:
    return SpacecraftParameters(106.0, CHASER_INERTIA_KG_M2)


def sample_bounded_rotation(rng: Generator, max_angle_rad: float) -> np.ndarray:
    if not 0.0 < max_angle_rad <= np.pi:
        raise ValueError("max_angle_rad must lie in (0, pi]")
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return so3_exp(axis * rng.uniform(0.0, max_angle_rad))


def sample_volume_uniform_shell(
    rng: Generator, inner_radius_m: float, outer_radius_m: float
) -> np.ndarray:
    if inner_radius_m < 0.0 or outer_radius_m <= inner_radius_m:
        raise ValueError("shell requires 0 <= inner_radius < outer_radius")
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    radius = rng.uniform(inner_radius_m**3, outer_radius_m**3) ** (1.0 / 3.0)
    return radius * direction


def sample_chaser_state(
    rng: Generator,
    target: SpacecraftState,
    *,
    shell_inner_m: float,
    shell_outer_m: float,
    max_relative_attitude_rad: float,
    inertial_velocity_component_limit_m_s: float,
    relative_angular_velocity_component_limit_rad_s: float,
) -> SpacecraftState:
    if inertial_velocity_component_limit_m_s <= 0.0:
        raise ValueError("inertial velocity limit must be positive")
    if relative_angular_velocity_component_limit_rad_s <= 0.0:
        raise ValueError("relative angular velocity limit must be positive")
    relative_rotation = sample_bounded_rotation(rng, max_relative_attitude_rad)
    rotation = target.rotation @ relative_rotation
    delta_position_eci = sample_volume_uniform_shell(
        rng, shell_inner_m, shell_outer_m
    )
    delta_velocity_eci = rng.uniform(
        -inertial_velocity_component_limit_m_s,
        inertial_velocity_component_limit_m_s,
        size=3,
    )
    inertial_velocity = target.rotation @ target.velocity + delta_velocity_eci
    omega = relative_rotation.T @ target.omega + rng.uniform(
        -relative_angular_velocity_component_limit_rad_s,
        relative_angular_velocity_component_limit_rad_s,
        size=3,
    )
    return SpacecraftState(
        rotation=rotation,
        position=target.position + delta_position_eci,
        omega=omega,
        velocity=rotation.T @ inertial_velocity,
    )


def sample_phase2_chaser_state(
    rng: Generator,
    target: SpacecraftState,
    *,
    task: Phase2TaskConfig = Phase2TaskConfig(),
    axial_remaining_min_m: float = 3.0,
    axial_remaining_max_m: float = 7.0,
    max_relative_attitude_rad: float = float(np.deg2rad(60.0)),
    relative_angular_velocity_component_limit_rad_s: float = 0.03,
    near_boundary_probability: float = 0.25,
    corridor_recovery_acceleration_m_s2: float = 0.03,
    initial_total_speed_limit_m_s: float | None = None,
    maximum_interior_radial_fraction: float = 0.75,
    minimum_fov_margin_rad: float = 0.0,
) -> SpacecraftState:
    """Sample a feasible Phase-2 state directly in task coordinates.

    The sampler covers the corridor cross-section by area, deliberately adds a
    controlled near-boundary fraction, and rejects attitudes outside the FOV.
    Returned samples satisfy the exact metrics used by the environment.
    """

    if not 0.0 <= near_boundary_probability <= 1.0:
        raise ValueError("near_boundary_probability must lie in [0, 1]")
    if not 0.0 < axial_remaining_min_m < axial_remaining_max_m:
        raise ValueError("axial remaining bounds must satisfy 0 < min < max")
    if not 0.0 < max_relative_attitude_rad <= np.pi:
        raise ValueError("max relative attitude must lie in (0, pi]")
    if relative_angular_velocity_component_limit_rad_s <= 0.0:
        raise ValueError("relative angular velocity limit must be positive")
    if corridor_recovery_acceleration_m_s2 <= 0.0:
        raise ValueError("corridor recovery acceleration must be positive")
    speed_limit = (
        task.total_speed_limit_m_s
        if initial_total_speed_limit_m_s is None
        else float(initial_total_speed_limit_m_s)
    )
    if not 0.0 < speed_limit <= task.total_speed_limit_m_s:
        raise ValueError("initial speed limit must lie in (0, task limit]")
    if not 0.0 < maximum_interior_radial_fraction <= 0.75:
        raise ValueError("maximum interior radial fraction must lie in (0, 0.75]")
    if not 0.0 <= minimum_fov_margin_rad < task.fov_half_angle_rad:
        raise ValueError("minimum FOV margin must lie in [0, FOV half angle)")

    axis = task.approach_axis
    plane_1, plane_2 = orthogonal_plane_basis(axis)
    axial_remaining = rng.uniform(axial_remaining_min_m, axial_remaining_max_m)
    axial_from_port = float(
        axis @ (task.desired_position - task.port_position) + axial_remaining
    )
    corridor_radius = axial_from_port * np.tan(task.corridor_half_angle_rad)
    if rng.random() < near_boundary_probability:
        radial_fraction = rng.uniform(0.75, 0.95)
    else:
        radial_fraction = maximum_interior_radial_fraction * np.sqrt(rng.random())
    azimuth = rng.uniform(-np.pi, np.pi)
    lateral_direction = np.cos(azimuth) * plane_1 + np.sin(azimuth) * plane_2
    position = (
        task.desired_position
        + axial_remaining * axis
        + radial_fraction * corridor_radius * lateral_direction
    )

    relative_rotation = None
    for _ in range(256):
        candidate = sample_bounded_rotation(rng, max_relative_attitude_rad)
        candidate_state = RelativeState(
            make_transform(candidate, position), np.zeros(6)
        )
        if (
            compute_task_metrics(candidate_state, task).fov_margin_rad
            >= minimum_fov_margin_rad
        ):
            relative_rotation = candidate
            break
    if relative_rotation is None:
        raise RuntimeError("failed to sample a Phase-2 attitude inside the FOV")

    closing_limit = task.closing_speed_limit(axial_remaining)
    closing_speed = rng.uniform(-0.02, 0.85 * closing_limit)
    closing_speed = min(closing_speed, 0.85 * speed_limit)
    lateral_speed_limit = np.sqrt(max(speed_limit**2 - closing_speed**2, 0.0))
    radial_direction = lateral_direction
    tangent_direction = np.cross(axis, radial_direction)
    corridor_margin = (1.0 - radial_fraction) * corridor_radius
    maximum_outward_radial_rate = (
        -np.tan(task.corridor_half_angle_rad) * closing_speed
        + np.sqrt(
            2.0
            * corridor_recovery_acceleration_m_s2
            * max(corridor_margin, 0.0)
        )
    )
    radial_upper = min(0.85 * lateral_speed_limit, maximum_outward_radial_rate)
    radial_lower = -0.85 * lateral_speed_limit
    if radial_upper < radial_lower:
        closing_speed = min(
            closing_speed,
            np.sqrt(
                2.0
                * corridor_recovery_acceleration_m_s2
                * max(corridor_margin, 0.0)
            )
            / np.tan(task.corridor_half_angle_rad),
        )
        lateral_speed_limit = np.sqrt(max(speed_limit**2 - closing_speed**2, 0.0))
        radial_lower = -0.85 * lateral_speed_limit
        radial_upper = min(
            0.85 * lateral_speed_limit,
            -np.tan(task.corridor_half_angle_rad) * closing_speed
            + np.sqrt(
                2.0
                * corridor_recovery_acceleration_m_s2
                * max(corridor_margin, 0.0)
            ),
        )
    radial_rate = rng.uniform(radial_lower, radial_upper)
    tangential_limit = np.sqrt(
        max((0.85 * lateral_speed_limit) ** 2 - radial_rate**2, 0.0)
    )
    tangential_rate = rng.uniform(-tangential_limit, tangential_limit)
    lateral_rate = (
        radial_rate * radial_direction + tangential_rate * tangent_direction
    )
    position_rate = -closing_speed * axis + lateral_rate
    relative_velocity = relative_rotation.T @ position_rate
    relative_omega = rng.uniform(
        -relative_angular_velocity_component_limit_rad_s,
        relative_angular_velocity_component_limit_rad_s,
        size=3,
    )
    relative = RelativeState(
        make_transform(relative_rotation, position),
        np.concatenate((relative_omega, relative_velocity)),
    )
    metrics = compute_task_metrics(relative, task)
    if not metrics.constraints_satisfied:
        raise RuntimeError("Phase-2 sampler produced an infeasible state")
    if np.linalg.norm(so3_log(relative_rotation, project=True)) > (
        max_relative_attitude_rad + 1.0e-12
    ):
        raise RuntimeError("Phase-2 sampler exceeded its attitude envelope")
    return reconstruct_chaser_state(target, relative)


def sample_phase2_mission_chaser_state(
    rng: Generator,
    target: SpacecraftState,
    *,
    mission: Phase2MissionConfig = Phase2MissionConfig(),
    difficulty: float = 1.0,
    easy_distance_min_m: float = 9.02,
    easy_distance_max_m: float = 9.05,
    easy_direction_half_angle_rad: float = float(np.deg2rad(0.02)),
    easy_attitude_limit_rad: float = float(np.deg2rad(2.0)),
    easy_speed_limit_m_s: float = 0.03,
    easy_angular_velocity_limit_rad_s: float = 0.001,
) -> SpacecraftState:
    """Sample a continuous easy-to-canonical Phase-I acquisition envelope."""

    difficulty = float(difficulty)
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("mission difficulty must lie in [0, 1]")

    def lerp(easy: float, full: float) -> float:
        return (1.0 - difficulty) * easy + difficulty * full

    axis = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    plane_1, plane_2 = orthogonal_plane_basis(axis)
    direction_half_angle = lerp(
        easy_direction_half_angle_rad, mission.initial_direction_half_angle_rad
    )
    cosine = rng.uniform(np.cos(direction_half_angle), 1.0)
    sine = np.sqrt(max(1.0 - cosine * cosine, 0.0))
    azimuth = rng.uniform(-np.pi, np.pi)
    direction = (
        cosine * axis
        + sine * (np.cos(azimuth) * plane_1 + np.sin(azimuth) * plane_2)
    )
    distance_min = lerp(easy_distance_min_m, mission.initial_distance_min_m)
    distance_max = lerp(easy_distance_max_m, mission.initial_distance_max_m)
    radius = rng.uniform(distance_min**3, distance_max**3) ** (1.0 / 3.0)
    position = radius * direction
    attitude_limit = lerp(
        easy_attitude_limit_rad, mission.initial_attitude_limit_rad
    )
    rotation = sample_bounded_rotation(rng, attitude_limit)

    velocity_direction = rng.normal(size=3)
    velocity_direction /= np.linalg.norm(velocity_direction)
    speed_limit = lerp(easy_speed_limit_m_s, mission.initial_speed_limit_m_s)
    full_speed = mission.initial_speed_limit_m_s * rng.random() ** (1.0 / 3.0)
    full_position_rate = full_speed * velocity_direction
    gate_error = position - mission.gate_position
    gate_direction = -gate_error / np.linalg.norm(gate_error)
    easy_position_rate = easy_speed_limit_m_s * gate_direction
    position_rate = (
        (1.0 - difficulty) * easy_position_rate
        + difficulty * full_position_rate
    )
    angular_velocity_limit = lerp(
        easy_angular_velocity_limit_rad_s,
        mission.initial_angular_velocity_component_limit_rad_s,
    )
    relative_omega = rng.uniform(-angular_velocity_limit, angular_velocity_limit, size=3)
    relative = RelativeState(
        make_transform(rotation, position),
        np.concatenate((relative_omega, rotation.T @ position_rate)),
    )
    metrics = compute_mission_metrics(relative, mission)
    if not distance_min <= metrics.target_center_distance_m <= distance_max:
        raise RuntimeError("mission sampler exceeded its distance envelope")
    if metrics.total_speed_m_s > speed_limit + 1.0e-12:
        raise RuntimeError("mission sampler exceeded its speed envelope")
    return reconstruct_chaser_state(target, relative)
