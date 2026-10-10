"""T1 near-field precapture task: geometry, truth metrics and the intent interface.

T1 is a separate, versioned task. Nothing here changes the F01 precapture task:
``T1TaskConfig`` subclasses ``PrecaptureTaskConfig`` only so that the shared
MPC can read the same port, desired pose, camera and corridor fields; every
T1-specific rule lives in this module or behind an ``isinstance`` dispatch.

Safety semantics are state based. There is no infinite entry plane and no
latch: outside the operation sphere ``R_KOS`` the chaser may move freely
(subject to the radial braking envelope); inside it the chaser centre must be
inside the port cone. Capture is the current state inside the cone, within the
completion thresholds, held for ``completion_hold_s``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from dynamics.lie import so3_exp
from dynamics.relative import relative_state
from dynamics.types import SpacecraftParameters, SpacecraftState
from env.scenarios import (
    TARGET_ORBIT_ALTITUDE_M,
    TARGET_ORBIT_INCLINATION_RAD,
    _uniform_rotation,
)
from dynamics.constants import EARTH
from env.task import PrecaptureTaskConfig, compute_precapture_metrics

FloatArray = NDArray[np.float64]

#: Uniform box of 225 kg whose principal inertia is diag(30, 39, 48) kg m^2.
T1_TARGET_MASS_KG = 225.0
T1_TARGET_INERTIA_KG_M2 = np.diag([30.0, 39.0, 48.0])
T1_TARGET_BOX_HALF_EXTENTS_M = (
    0.5 * float(np.sqrt(12 * (39.0 + 48.0 - 30.0) / (2 * T1_TARGET_MASS_KG))),
    0.5 * float(np.sqrt(12 * (30.0 + 48.0 - 39.0) / (2 * T1_TARGET_MASS_KG))),
    0.5 * float(np.sqrt(12 * (30.0 + 39.0 - 48.0) / (2 * T1_TARGET_MASS_KG))),
)


@dataclass(frozen=True)
class T1TaskConfig(PrecaptureTaskConfig):
    """T1 rules. Inherited fields keep their F01 meaning where T1 uses them.

    Used from the parent: ``port_position_target_m``,
    ``desired_position_target_m``, ``approach_axis_target``,
    ``camera_boresight_chaser``, ``corridor_half_angle_rad`` (35 deg),
    ``fov_half_angle_rad`` (50 deg), ``terminal_total_speed_limit_m_s``
    (0.35 m/s) and the completion thresholds. Not used: the entry plane, the
    2 m keep-out sphere, the 2 m/s outer inertial speed limit, the phase gate
    and the outer approach corridor.
    """

    operation_sphere_radius_m: float = 6.0
    #: Look-ahead band outside the sphere in which a chaser already inside the
    #: cone is treated as entering through the mouth (cone rules, axial
    #: envelope) instead of approaching the sphere (radial envelope).
    mouth_band_m: float = 0.5
    port_axial_clearance_m: float = 1.0
    envelope_final_speed_m_s: float = 0.05
    #: eta * F_axis / m = 0.5 * 2 N / 106 kg.
    envelope_deceleration_m_s2: float = 0.5 * 2.0 / 106.0
    chaser_envelope_radius_m: float = 0.95
    grab_rod_radius_m: float = 0.05
    max_distance_m: float = 30.0
    #: Truth-violation tolerances. The shared QP holds these rows as soft
    #: constraints; a breach must exceed a physically meaningful size to end
    #: an episode (a 9e-6 m/s envelope excursion is solver noise, not a
    #: safety event). Collision stays exact.
    speed_tolerance_m_s: float = 1.0e-3
    geometry_tolerance_m: float = 1.0e-3
    angle_tolerance_rad: float = 1.0e-3

    #: Knots (m) of the piecewise-linear braking envelope. Between knots the
    #: envelope is the chord of sqrt(v_f^2 + 2 a_eff d); a chord of a concave
    #: function lies below it, so the envelope is never looser than constant
    #: deceleration a_eff, and the MPC's per-segment linearization is exact
    #: (a tangent of the sqrt itself lies above the curve and let the
    #: optimiser predict feasibility while the truth was violated).
    envelope_knots_m: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    def _sqrt_envelope(self, distance_m: float) -> float:
        return float(
            np.sqrt(
                self.envelope_final_speed_m_s**2
                + 2.0 * self.envelope_deceleration_m_s2 * max(float(distance_m), 0.0)
            )
        )

    def _envelope_segment(self, distance_m: float) -> tuple[float, float, float]:
        """(left knot, value there, slope) of the chord containing ``distance_m``."""

        knots = self.envelope_knots_m
        d = max(float(distance_m), 0.0)
        index = int(np.clip(np.searchsorted(knots, d, side="right") - 1, 0, len(knots) - 2))
        left, right = knots[index], knots[index + 1]
        value_left = self._sqrt_envelope(left)
        slope = (self._sqrt_envelope(right) - value_left) / (right - left)
        return left, value_left, slope

    def braking_speed_limit(self, distance_m: float) -> float:
        """Piecewise-linear inner approximation of sqrt(v_f^2 + 2 a_eff d), d >= 0."""

        left, value, slope = self._envelope_segment(distance_m)
        return float(value + slope * (max(float(distance_m), 0.0) - left))

    def braking_speed_slope(self, distance_m: float) -> float:
        """d(limit)/dd of the active chord; zero for d <= 0 (limit held at v_f)."""

        if float(distance_m) <= 0.0:
            return 0.0
        return self._envelope_segment(distance_m)[2]

    def closing_speed_limit(self, axial_remaining_m: float) -> float:  # noqa: D401
        return self.braking_speed_limit(axial_remaining_m)

    def outer_radial_closing_speed_limit(self, range_m: float) -> float:
        return self.braking_speed_limit(float(range_m) - self.operation_sphere_radius_m)


def t1_target_parameters() -> SpacecraftParameters:
    return SpacecraftParameters(T1_TARGET_MASS_KG, T1_TARGET_INERTIA_KG_M2)


# --------------------------------------------------------------------------- #
# Truth geometry
# --------------------------------------------------------------------------- #


def cone_quantities(position: FloatArray, task: T1TaskConfig) -> tuple[float, float, float]:
    """Return (axial distance from the port, lateral distance, cone margin)."""

    axis = task.approach_axis
    displacement = np.asarray(position, dtype=np.float64) - task.port_position
    axial = float(axis @ displacement)
    lateral = float(np.linalg.norm(displacement - axial * axis))
    margin = min(
        axial - task.port_axial_clearance_m,
        axial * np.tan(task.corridor_half_angle_rad) - lateral,
    )
    return axial, lateral, float(margin)


def in_mouth_region(position: FloatArray, task: T1TaskConfig) -> bool:
    """Cone rules apply: inside the cone and no farther than the mouth band."""

    _, _, margin = cone_quantities(position, task)
    return bool(
        margin >= 0.0
        and float(np.linalg.norm(position))
        < task.operation_sphere_radius_m + task.mouth_band_m
    )


def collision_clearance(position: FloatArray, task: T1TaskConfig) -> float:
    """Distance from the chaser envelope sphere to the box plus grab rod (<0: contact)."""

    p = np.asarray(position, dtype=np.float64)
    half = np.asarray(T1_TARGET_BOX_HALF_EXTENTS_M)
    box_distance = float(np.linalg.norm(np.maximum(np.abs(p) - half, 0.0)))
    start = np.array([-half[0], 0.0, 0.0])
    end = np.asarray(task.port_position, dtype=np.float64)
    segment = end - start
    t = float(np.clip((p - start) @ segment / (segment @ segment), 0.0, 1.0))
    rod_distance = float(np.linalg.norm(p - (start + t * segment))) - task.grab_rod_radius_m
    return min(box_distance, rod_distance) - task.chaser_envelope_radius_m


@dataclass(frozen=True)
class T1Metrics:
    position_target_m: FloatArray
    range_m: float
    axial_from_port_m: float
    lateral_m: float
    cone_margin_m: float
    in_cone: bool
    mouth_region: bool
    union_margin_m: float
    collision_clearance_m: float
    fov_margin_rad: float
    radial_closing_speed_m_s: float
    radial_envelope_margin_m_s: float
    target_frame_speed_m_s: float
    total_speed_margin_m_s: float
    axial_closing_speed_m_s: float
    axial_envelope_margin_m_s: float
    position_error_m: float
    attitude_error_rad: float
    angular_velocity_error_rad_s: float
    instantaneous_capture: bool
    violations: dict[str, float]

    @property
    def hard_violation(self) -> bool:
        return bool(self.violations)


def compute_t1_metrics(
    target: SpacecraftState, chaser: SpacecraftState, task: T1TaskConfig
) -> T1Metrics:
    relative = relative_state(target, chaser)
    base = compute_precapture_metrics(
        target, chaser, relative, task, terminal_region_active=False
    )
    p = np.asarray(base.position_target_m, dtype=np.float64)
    range_m = float(np.linalg.norm(p))
    axial, lateral, cone_margin = cone_quantities(p, task)
    in_cone = cone_margin >= 0.0
    mouth = bool(in_cone and range_m < task.operation_sphere_radius_m + task.mouth_band_m)
    union_margin = max(range_m - task.operation_sphere_radius_m, cone_margin)
    clearance = collision_clearance(p, task)
    inertial = np.asarray(base.inertial_relative_velocity_m_s, dtype=np.float64)
    radial_closing = float(-(p / max(range_m, 1e-12)) @ inertial)
    rate = np.asarray(base.position_rate_target_m_s, dtype=np.float64)
    speed = float(np.linalg.norm(rate))
    axial_closing = float(-task.approach_axis @ rate)
    axial_remaining = float(task.approach_axis @ (p - task.desired_position))
    if mouth:
        radial_margin = np.inf
        total_margin = task.terminal_total_speed_limit_m_s - speed
        axial_margin = task.closing_speed_limit(axial_remaining) - axial_closing
    else:
        radial_margin = task.outer_radial_closing_speed_limit(range_m) - radial_closing
        total_margin = np.inf
        axial_margin = np.inf
    violations: dict[str, float] = {}
    for name, margin, tol in (
        ("collision", clearance, task.constraint_tolerance),
        ("operation_sphere_cone", union_margin, task.geometry_tolerance_m),
        ("fov", float(base.fov_margin_rad), task.angle_tolerance_rad),
        ("radial_envelope", radial_margin, task.speed_tolerance_m_s),
        ("total_speed", total_margin, task.speed_tolerance_m_s),
        ("axial_envelope", axial_margin, task.speed_tolerance_m_s),
    ):
        if margin < -tol:
            violations[name] = float(-margin)
    capture = bool(
        in_cone
        and base.position_error_m <= task.completion_position_m
        and base.attitude_error_rad <= task.completion_attitude_rad
        and speed <= task.completion_speed_m_s
        and base.angular_velocity_error_rad_s <= task.completion_angular_velocity_rad_s
    )
    return T1Metrics(
        position_target_m=p,
        range_m=range_m,
        axial_from_port_m=axial,
        lateral_m=lateral,
        cone_margin_m=float(cone_margin),
        in_cone=bool(in_cone),
        mouth_region=mouth,
        union_margin_m=float(union_margin),
        collision_clearance_m=float(clearance),
        fov_margin_rad=float(base.fov_margin_rad),
        radial_closing_speed_m_s=radial_closing,
        radial_envelope_margin_m_s=float(radial_margin),
        target_frame_speed_m_s=speed,
        total_speed_margin_m_s=float(total_margin),
        axial_closing_speed_m_s=axial_closing,
        axial_envelope_margin_m_s=float(axial_margin),
        position_error_m=float(base.position_error_m),
        attitude_error_rad=float(base.attitude_error_rad),
        angular_velocity_error_rad_s=float(base.angular_velocity_error_rad_s),
        instantaneous_capture=capture,
        violations=violations,
    )


# --------------------------------------------------------------------------- #
# Initial states
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class T1InitialDistribution:
    tumble_rate_deg_s: tuple[float, float] = (2.1, 2.7)
    tumble_offset_from_z_deg: tuple[float, float] = (10.0, 20.0)
    range_m: tuple[float, float] = (9.0, 15.0)
    inertial_speed_max_m_s: float = 0.10
    pointing_error_max_deg: float = 15.0
    chaser_omega_component_max_rad_s: float = 0.005


def lvlh_basis(target: SpacecraftState) -> FloatArray:
    """Columns: radial, along-track, orbit normal (ECI)."""

    r = target.position / np.linalg.norm(target.position)
    v_inertial = target.rotation @ target.velocity
    h = np.cross(target.position, v_inertial)
    h /= np.linalg.norm(h)
    return np.column_stack((r, np.cross(h, r), h))


def sample_t1_initial_states(
    rng: Generator, distribution: T1InitialDistribution, task: T1TaskConfig
) -> tuple[SpacecraftState, SpacecraftState]:
    """Target (uniform SO(3), near-z spin) and chaser (LVLH, independent)."""

    d = distribution
    orbital_radius = EARTH.equatorial_radius + TARGET_ORBIT_ALTITUDE_M
    circular_speed = np.sqrt(EARTH.mu / orbital_radius)
    inc = TARGET_ORBIT_INCLINATION_RAD
    position_eci = np.array([orbital_radius, 0.0, 0.0])
    velocity_eci = circular_speed * np.array([0.0, np.cos(inc), np.sin(inc)])
    rotation = _uniform_rotation(rng)
    rate = np.deg2rad(rng.uniform(*d.tumble_rate_deg_s))
    offset = np.deg2rad(rng.uniform(*d.tumble_offset_from_z_deg))
    azimuth = rng.uniform(-np.pi, np.pi)
    omega = rate * np.array(
        [np.sin(offset) * np.cos(azimuth), np.sin(offset) * np.sin(azimuth), np.cos(offset)]
    )
    target = SpacecraftState(
        rotation=rotation,
        position=position_eci,
        omega=omega,
        velocity=rotation.T @ velocity_eci,
    )

    basis = lvlh_basis(target)
    direction_lvlh = rng.normal(size=3)
    direction_lvlh /= np.linalg.norm(direction_lvlh)
    range_m = rng.uniform(*d.range_m)
    chaser_position = position_eci + basis @ (range_m * direction_lvlh)
    velocity_direction = rng.normal(size=3)
    velocity_direction /= np.linalg.norm(velocity_direction)
    speed = d.inertial_speed_max_m_s * rng.random() ** (1.0 / 3.0)
    chaser_velocity = velocity_eci + speed * velocity_direction

    port_eci = position_eci + rotation @ task.port_position
    boresight = port_eci - chaser_position
    boresight /= np.linalg.norm(boresight)
    reference = np.array([0.0, 0.0, 1.0]) if abs(boresight[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference, boresight)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(boresight, y_axis)
    aimed = np.column_stack((boresight, y_axis, z_axis))
    roll = so3_exp(np.array([rng.uniform(-np.pi, np.pi), 0.0, 0.0]))
    tilt_axis = np.array([0.0, *rng.normal(size=2)])
    tilt_axis /= np.linalg.norm(tilt_axis)
    tilt = so3_exp(tilt_axis * np.deg2rad(rng.uniform(0.0, d.pointing_error_max_deg)))
    chaser_rotation = aimed @ tilt @ roll
    chaser = SpacecraftState(
        rotation=chaser_rotation,
        position=chaser_position,
        omega=rng.uniform(
            -d.chaser_omega_component_max_rad_s, d.chaser_omega_component_max_rad_s, size=3
        ),
        velocity=chaser_rotation.T @ chaser_velocity,
    )
    return target, chaser


# --------------------------------------------------------------------------- #
# Intent interface G
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class T1IntentConfig:
    staging_radius_m: float = 7.0
    final_radius_m: float = 3.0


@dataclass(frozen=True)
class EpisodeAnchor:
    """Fixed per episode: angular-momentum axis H0 and the longitude base e1 (ECI)."""

    h_axis: FloatArray
    e1: FloatArray

    @property
    def e2(self) -> FloatArray:
        return np.cross(self.h_axis, self.e1)


def episode_anchor(target: SpacecraftState, chaser: SpacecraftState) -> EpisodeAnchor:
    momentum = target.rotation @ (T1_TARGET_INERTIA_KG_M2 @ target.omega)
    h = momentum / np.linalg.norm(momentum)
    los = chaser.position - target.position
    los /= np.linalg.norm(los)
    e1 = los - float(los @ h) * h
    if np.linalg.norm(e1) < 1e-6:  # line of sight along H: deterministic fallback
        fallback = np.array([1.0, 0.0, 0.0]) if abs(h[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = fallback - float(fallback @ h) * h
    return EpisodeAnchor(h_axis=h, e1=e1 / np.linalg.norm(e1))


def staging_direction(anchor: EpisodeAnchor, latitude_rad: float) -> FloatArray:
    return np.cos(latitude_rad) * anchor.e1 + np.sin(latitude_rad) * anchor.h_axis


def _slerp_deterministic(start: FloatArray, goal: FloatArray, fraction: float) -> FloatArray:
    a = start / np.linalg.norm(start)
    b = goal / np.linalg.norm(goal)
    cosine = float(np.clip(a @ b, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return a.copy()
    if np.pi - angle < 1e-6:
        reference = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, reference)
        axis /= np.linalg.norm(axis)
        theta = fraction * angle
        return np.cos(theta) * a + np.sin(theta) * np.cross(axis, a)
    sine = np.sin(angle)
    return (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / sine


def intent_from_action(action: FloatArray) -> tuple[float, float]:
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
    return 0.5 * np.pi * float(a[0]), 0.5 * (float(a[1]) + 1.0)


def reference_position_target_frame(
    target_rotation: FloatArray,
    anchor: EpisodeAnchor,
    latitude_rad: float,
    sigma: float,
    task: T1TaskConfig,
    intent: T1IntentConfig,
) -> FloatArray:
    """G at one instant, written in the target body frame."""

    n_p = task.approach_axis
    if sigma <= 0.5:
        u_target = target_rotation.T @ staging_direction(anchor, latitude_rad)
        return intent.staging_radius_m * _slerp_deterministic(u_target, n_p, 2.0 * sigma)
    radius = intent.staging_radius_m - 2.0 * (sigma - 0.5) * (
        intent.staging_radius_m - intent.final_radius_m
    )
    return radius * n_p


__all__ = [
    "EpisodeAnchor",
    "T1IntentConfig",
    "T1InitialDistribution",
    "T1Metrics",
    "T1TaskConfig",
    "T1_TARGET_BOX_HALF_EXTENTS_M",
    "T1_TARGET_INERTIA_KG_M2",
    "T1_TARGET_MASS_KG",
    "collision_clearance",
    "compute_t1_metrics",
    "cone_quantities",
    "episode_anchor",
    "in_mouth_region",
    "intent_from_action",
    "lvlh_basis",
    "reference_position_target_frame",
    "sample_t1_initial_states",
    "staging_direction",
    "t1_target_parameters",
]
