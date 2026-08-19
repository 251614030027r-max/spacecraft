"""Dimensionless observations for legacy Phase-1 and canonical Phase-2."""

import numpy as np

from dynamics.lie import inverse_transform, make_transform, se3_log
from dynamics.relative import RelativeState
from env.task import Phase2TaskConfig, compute_task_metrics, orthogonal_plane_basis


PHASE2_OBSERVATION_SCHEMA = "phase2_v2_1_23d"
PHASE2_MISSION_TARGET_TRANSLATION_OBSERVATION_SCHEMA = "phase2_mission_v1_24d"
PHASE2_MISSION_BODY_TRANSLATION_OBSERVATION_SCHEMA = (
    "phase2_mission_v2_body_translation_24d"
)
PHASE2_MISSION_OBSERVATION_SCHEMA = (
    "phase2_mission_v3_body_velocity_error_24d"
)


def _softsign(raw: np.ndarray, limit: float) -> np.ndarray:
    return limit * raw / (limit + np.abs(raw))


def build_observation(
    relative: RelativeState,
    *,
    attitude_scale_rad: float,
    distance_scale_m: float,
    angular_velocity_scale_rad_s: float,
    velocity_scale_m_s: float,
    softsign_limit: float = 10.0,
) -> np.ndarray:
    scales = (
        attitude_scale_rad,
        distance_scale_m,
        angular_velocity_scale_rad_s,
        velocity_scale_m_s,
        softsign_limit,
    )
    if min(scales) <= 0.0:
        raise ValueError("observation scales must be positive")
    coordinates = relative.exponential_coordinates
    raw = np.concatenate(
        (
            coordinates[:3] / attitude_scale_rad,
            coordinates[3:] / distance_scale_m,
            relative.omega / angular_velocity_scale_rad_s,
            relative.velocity / velocity_scale_m_s,
        )
    )
    bounded = _softsign(raw, softsign_limit)
    return bounded.astype(np.float32)


def build_phase2_observation(
    relative: RelativeState,
    *,
    task: Phase2TaskConfig,
    attitude_scale_rad: float,
    distance_scale_m: float,
    angular_velocity_scale_rad_s: float,
    velocity_scale_m_s: float,
    softsign_limit: float = 10.0,
    target_angular_velocity_rad_s: np.ndarray | None = None,
    target_angular_velocity_scale_rad_s: float = 0.05,
) -> np.ndarray:
    """Direct 23D task geometry, motion, and signed-margin observation."""

    metrics = compute_task_metrics(relative, task)
    target_omega = (
        np.zeros(3, dtype=np.float64)
        if target_angular_velocity_rad_s is None
        else np.asarray(target_angular_velocity_rad_s, dtype=np.float64)
    )
    scales = (
        attitude_scale_rad,
        distance_scale_m,
        angular_velocity_scale_rad_s,
        velocity_scale_m_s,
        target_angular_velocity_scale_rad_s,
        softsign_limit,
    )
    if min(scales) <= 0.0 or target_omega.shape != (3,):
        raise ValueError("Phase-2 observation scales and target omega are invalid")

    position_error = metrics.position_target_m - task.desired_position
    core_raw = np.concatenate(
        (
            metrics.desired_error_coordinates[:3] / attitude_scale_rad,
            position_error / distance_scale_m,
            metrics.position_rate_target_m_s / velocity_scale_m_s,
            relative.omega / angular_velocity_scale_rad_s,
            target_omega / target_angular_velocity_scale_rad_s,
        )
    )
    plane_1, plane_2 = orthogonal_plane_basis(task.approach_axis)
    port_displacement = metrics.position_target_m - task.port_position
    lateral = port_displacement - metrics.port_axial_distance_m * task.approach_axis
    geometric_corridor_radius = (
        metrics.port_axial_distance_m * np.tan(task.corridor_half_angle_rad)
    )
    corridor_radius = max(abs(geometric_corridor_radius), 0.25)
    corridor_direction = np.array(
        [plane_1 @ lateral, plane_2 @ lateral], dtype=np.float64
    ) / corridor_radius

    line_of_sight_target = task.port_position - metrics.position_target_m
    line_of_sight_chaser = relative.rotation.T @ line_of_sight_target
    los_norm = max(float(np.linalg.norm(line_of_sight_chaser)), np.finfo(float).eps)
    unit_los = line_of_sight_chaser / los_norm
    camera_plane_1, camera_plane_2 = orthogonal_plane_basis(task.camera_boresight)
    fov_direction = np.array(
        [camera_plane_1 @ unit_los, camera_plane_2 @ unit_los], dtype=np.float64
    )
    margins = np.array(
        [
            metrics.corridor_lateral_margin_m / corridor_radius,
            metrics.fov_margin_rad / task.fov_half_angle_rad,
            metrics.total_speed_margin_m_s / task.total_speed_limit_m_s,
            metrics.closing_speed_margin_m_s / task.closing_speed_max_m_s,
        ],
        dtype=np.float64,
    )
    observation = np.concatenate(
        (
            _softsign(core_raw, softsign_limit),
            _softsign(corridor_direction, softsign_limit),
            fov_direction,
            _softsign(margins, softsign_limit),
        )
    )
    if observation.shape != (23,) or not np.all(np.isfinite(observation)):
        raise RuntimeError("invalid Phase-2 V2.1 observation")
    return observation.astype(np.float32)


def build_phase2_mission_observation(
    relative: RelativeState,
    *,
    task: Phase2TaskConfig,
    active_reference_position_m: np.ndarray,
    mission_phase: int,
    attitude_scale_rad: float,
    distance_scale_m: float,
    angular_velocity_scale_rad_s: float,
    velocity_scale_m_s: float,
    softsign_limit: float = 10.0,
    target_angular_velocity_rad_s: np.ndarray | None = None,
    target_angular_velocity_scale_rad_s: float = 0.05,
    translational_observation_frame: str = "chaser_body",
    translational_velocity_observation: str = "actual",
    active_reference_velocity_target_m_s: np.ndarray | None = None,
) -> np.ndarray:
    """Canonical 24D mission observation with an explicit phase flag."""

    reference = np.asarray(active_reference_position_m, dtype=np.float64)
    if reference.shape != (3,) or mission_phase not in (0, 1):
        raise ValueError("active reference and mission phase are invalid")
    if translational_observation_frame not in {"target", "chaser_body"}:
        raise ValueError("unsupported translational observation frame")
    if translational_velocity_observation not in {"actual", "tracking_error"}:
        raise ValueError("unsupported translational velocity observation")
    metrics = compute_task_metrics(relative, task)
    target_omega = (
        np.zeros(3, dtype=np.float64)
        if target_angular_velocity_rad_s is None
        else np.asarray(target_angular_velocity_rad_s, dtype=np.float64)
    )
    if target_omega.shape != (3,):
        raise ValueError("target omega must have shape=(3,)")
    reference_error = inverse_transform(make_transform(np.eye(3), reference)) @ relative.transform
    reference_coordinates = se3_log(reference_error, project=True)
    scales = (
        attitude_scale_rad,
        distance_scale_m,
        angular_velocity_scale_rad_s,
        velocity_scale_m_s,
        target_angular_velocity_scale_rad_s,
        softsign_limit,
    )
    if min(scales) <= 0.0:
        raise ValueError("mission observation scales must be positive")
    position_error_target = relative.position - reference
    position_rate_target = metrics.position_rate_target_m_s
    if translational_velocity_observation == "tracking_error":
        if active_reference_velocity_target_m_s is None:
            raise ValueError("tracking-error observation requires reference velocity")
        desired_velocity = np.asarray(
            active_reference_velocity_target_m_s, dtype=np.float64
        )
        if desired_velocity.shape != (3,) or not np.all(np.isfinite(desired_velocity)):
            raise ValueError("active reference velocity must be a finite three-vector")
        velocity_target = position_rate_target - desired_velocity
    else:
        velocity_target = position_rate_target
    if translational_observation_frame == "chaser_body":
        # relative.rotation = R_target^T R_chaser maps chaser-body vectors to
        # target-frame vectors, so its transpose is the verified inverse map.
        position_error = relative.rotation.T @ position_error_target
        velocity_observation = relative.rotation.T @ velocity_target
    else:
        position_error = position_error_target
        velocity_observation = velocity_target
    core_raw = np.concatenate(
        (
            reference_coordinates[:3] / attitude_scale_rad,
            position_error / distance_scale_m,
            velocity_observation / velocity_scale_m_s,
            relative.omega / angular_velocity_scale_rad_s,
            target_omega / target_angular_velocity_scale_rad_s,
        )
    )
    plane_1, plane_2 = orthogonal_plane_basis(task.approach_axis)
    port_displacement = metrics.position_target_m - task.port_position
    lateral = port_displacement - metrics.port_axial_distance_m * task.approach_axis
    geometric_corridor_radius = metrics.port_axial_distance_m * np.tan(
        task.corridor_half_angle_rad
    )
    corridor_radius = max(abs(geometric_corridor_radius), 0.25)
    corridor_direction = np.array(
        [plane_1 @ lateral, plane_2 @ lateral], dtype=np.float64
    ) / corridor_radius
    line_of_sight_target = task.port_position - metrics.position_target_m
    line_of_sight_chaser = relative.rotation.T @ line_of_sight_target
    los_norm = max(float(np.linalg.norm(line_of_sight_chaser)), np.finfo(float).eps)
    unit_los = line_of_sight_chaser / los_norm
    camera_plane_1, camera_plane_2 = orthogonal_plane_basis(task.camera_boresight)
    fov_direction = np.array(
        [camera_plane_1 @ unit_los, camera_plane_2 @ unit_los], dtype=np.float64
    )
    # Terminal margins remain visible in Phase I as bounded future-task information;
    # only the environment decides whether they are active constraints.
    margins = np.array(
        [
            metrics.corridor_lateral_margin_m / corridor_radius,
            metrics.fov_margin_rad / task.fov_half_angle_rad,
            metrics.total_speed_margin_m_s / task.total_speed_limit_m_s,
            metrics.closing_speed_margin_m_s / task.closing_speed_max_m_s,
        ],
        dtype=np.float64,
    )
    observation = np.concatenate(
        (
            _softsign(core_raw, softsign_limit),
            _softsign(corridor_direction, softsign_limit),
            fov_direction,
            _softsign(margins, softsign_limit),
            np.array([float(mission_phase)], dtype=np.float64),
        )
    )
    if observation.shape != (24,) or not np.all(np.isfinite(observation)):
        raise RuntimeError("invalid Phase-2 mission observation")
    return observation.astype(np.float32)
