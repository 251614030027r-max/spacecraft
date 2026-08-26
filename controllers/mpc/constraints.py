"""Normalized truth margins and their analytic Jacobian for constrained MPC.

Both halves of this module exist because the constraint linearisation, not the
QP, is what put the MPC controller over its control period. Profiled at
``horizon_steps = 50``, ``linearize_constraint_margins`` was 74% of
``MPCController.command`` and the QP was 9%, and two thirds of the constraint
work was inside ``env.task.compute_task_metrics``:

* **The margins need three of its twenty fields.** ``compute_task_metrics`` also
  builds the desired-pose error coordinates and the attitude error, so every
  margin evaluation paid for an ``se3_log``, an ``so3_log`` and the SVD inside
  ``project_to_so3`` -- none of which any margin reads. The margin functions
  here compute only what a margin is a function of, from the same expressions,
  so the values are unchanged; a test pins them against the task module.
* **The Jacobian was ten finite differences of that.** It is analytic here.
  Every margin is a function of the relative position ``p``, the target-frame
  relative velocity ``R v`` and, for the field of view, the chaser-frame line of
  sight ``R^T (p_port - p)``; each of those has a closed-form derivative with
  respect to the exponential coordinates, so the whole Jacobian follows by the
  chain rule with no perturbation at all.

The state is the same 12-vector the prediction model uses: ``x[:6]`` are the
SE(3) exponential coordinates ``(phi, rho)`` of the relative transform and
``x[6:]`` is the relative body twist ``(omega, v)``. With ``T = exp(x[:6]^)``
this repository's convention gives ``R = exp(phi^)`` and ``p = J_l(phi) rho``,
which is what the derivatives below differentiate.

Margins do not depend on the relative angular velocity, so those three columns
are exactly zero, as they were under finite differencing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynamics.lie import _SMALL_ANGLE, hat3, left_jacobian_so3, so3_exp
from env.task import Phase2TaskConfig


FloatArray = NDArray[np.float64]

# Below this the geometry is degenerate -- a zero line of sight, a zero relative
# speed, a boresight exactly along the line of sight -- and the quantity is not
# differentiable there. The Jacobian row is zeroed rather than allowed to blow
# up: zero is inside the subdifferential and keeps the QP well posed.
_DEGENERATE = 1.0e-12


@dataclass(frozen=True)
class _CorridorGeometry:
    """Facet directions and the constant corridor gradient, built once per task.

    Every corridor margin is exactly affine in the relative position, so its
    gradient with respect to ``p`` is a constant matrix -- there is nothing to
    recompute per state, per horizon index or per control step.
    """

    axis: FloatArray
    first: FloatArray
    second: FloatArray
    directions: FloatArray          # (facets, 3), each orthogonal to the axis
    position_gradient: FloatArray   # (1 + facets, 3), d(margin)/d(p)


@lru_cache(maxsize=16)
def _corridor_geometry(
    task: Phase2TaskConfig, corridor_facets: int, distance_scale_m: float
) -> _CorridorGeometry:
    axis = task.approach_axis
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = 2.0 * np.pi * np.arange(corridor_facets) / corridor_facets
    directions = (
        np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    )
    # margin_axial   = (a . q) / s
    # margin_facet_k = (a . q) tan(half) cos(pi/F) / s - (d_k . lateral) / s,
    # and d_k is orthogonal to a, so d_k . lateral = d_k . q.
    slope = float(np.tan(task.corridor_half_angle_rad) * np.cos(np.pi / corridor_facets))
    gradient = np.empty((1 + corridor_facets, 3), dtype=np.float64)
    gradient[0] = axis / distance_scale_m
    gradient[1:] = (slope * axis[None, :] - directions) / distance_scale_m
    return _CorridorGeometry(axis, first, second, directions, gradient)


def _state(state: ArrayLike) -> FloatArray:
    x = np.asarray(state, dtype=np.float64)
    if x.shape != (12,) or not np.all(np.isfinite(x)):
        raise ValueError("constraint state must be finite and shape=(12,)")
    return x


def _pose(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(R, p, J_l(phi))`` for the relative transform ``exp(x[:6]^)``."""

    phi, rho = x[:3], x[3:6]
    jacobian = left_jacobian_so3(phi)
    return so3_exp(phi), jacobian @ rho, jacobian


def _fov_angle(
    rotation: FloatArray, position: FloatArray, task: Phase2TaskConfig
) -> tuple[float, FloatArray, float]:
    """Return ``(angle, chaser-frame line of sight, its norm)``.

    Same expression as ``env.task.compute_task_metrics``: the angle between the
    boresight and the line of sight, via ``arctan2`` so it stays accurate near
    both zero and pi.
    """

    line_of_sight = rotation.T @ (task.port_position - position)
    norm = float(np.linalg.norm(line_of_sight))
    if norm <= np.finfo(np.float64).eps:
        return float(np.pi), line_of_sight, norm
    unit = line_of_sight / norm
    boresight = task.camera_boresight
    cosine = float(np.clip(boresight @ unit, -1.0, 1.0))
    sine = float(np.linalg.norm(np.cross(boresight, unit)))
    return float(np.arctan2(sine, cosine)), line_of_sight, norm


def normalized_truth_margins(
    state: ArrayLike, task: Phase2TaskConfig
) -> FloatArray:
    """Four task-truth margins in dimensionless units."""

    x = _state(state)
    rotation, position, _ = _pose(x)
    axis = task.approach_axis
    displacement = position - task.port_position
    axial = float(axis @ displacement)
    lateral = displacement - axial * axis
    corridor = min(
        axial,
        axial * np.tan(task.corridor_half_angle_rad) - float(np.linalg.norm(lateral)),
    )
    angle, _, _ = _fov_angle(rotation, position, task)
    position_rate = rotation @ x[9:]
    total_speed = float(np.linalg.norm(position_rate))
    axial_remaining = float(axis @ (position - task.desired_position))
    closing_speed = float(-axis @ position_rate)
    return np.asarray(
        [
            corridor / 3.0,
            (task.fov_half_angle_rad - angle) / task.fov_half_angle_rad,
            (task.total_speed_limit_m_s - total_speed) / task.total_speed_limit_m_s,
            (task.closing_speed_limit(axial_remaining) - closing_speed)
            / task.closing_speed_max_m_s,
        ],
        dtype=np.float64,
    )


def normalized_constraint_margins(
    state: ArrayLike,
    task: Phase2TaskConfig,
    *,
    corridor_facets: int,
    distance_scale_m: float = 3.0,
) -> FloatArray:
    """Positive means feasible; corridor uses an inscribed regular polygon."""

    x = _state(state)
    if corridor_facets < 4 or distance_scale_m <= 0.0:
        raise ValueError("invalid constraint geometry settings")
    geometry = _corridor_geometry(task, corridor_facets, distance_scale_m)
    rotation, position, _ = _pose(x)
    axis = geometry.axis
    displacement = position - task.port_position
    axial = float(axis @ displacement)
    lateral = displacement - axial * axis
    polygon_radius = (
        axial
        * np.tan(task.corridor_half_angle_rad)
        * np.cos(np.pi / corridor_facets)
    )
    corridor = np.empty(1 + corridor_facets, dtype=np.float64)
    corridor[0] = axial / distance_scale_m
    corridor[1:] = (polygon_radius - geometry.directions @ lateral) / distance_scale_m
    angle, _, _ = _fov_angle(rotation, position, task)
    position_rate = rotation @ x[9:]
    axial_remaining = float(axis @ (position - task.desired_position))
    return np.concatenate(
        (
            corridor,
            [
                (task.fov_half_angle_rad - angle) / task.fov_half_angle_rad,
                (task.total_speed_limit_m_s - float(np.linalg.norm(position_rate)))
                / task.total_speed_limit_m_s,
                (
                    task.closing_speed_limit(axial_remaining)
                    - float(-axis @ position_rate)
                )
                / task.closing_speed_max_m_s,
            ],
        )
    )


def _position_rotation_gradient(phi: FloatArray, rho: FloatArray) -> FloatArray:
    """Return ``d(J_l(phi) rho) / d(phi)``, the one term without a free form.

    With ``J_l = I + b phi^ + c (phi^)^2`` the position is
    ``p = rho + b (phi x rho) + c phi x (phi x rho)``, so differentiating the
    two cross products and the two scalar coefficients gives the result in
    closed form. The small-angle branch matches ``left_jacobian_so3``'s own
    series so the two agree where they meet.
    """

    theta = float(np.linalg.norm(phi))
    cross = np.cross(phi, rho)
    double_cross = np.cross(phi, cross)
    if theta < _SMALL_ANGLE:
        theta2 = theta * theta
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        c = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
        # d b / d theta and d c / d theta from the same series.
        b_prime = -theta / 12.0 + theta * theta2 / 180.0
        c_prime = -theta / 60.0 + theta * theta2 / 1260.0
    else:
        sine, cosine = np.sin(theta), np.cos(theta)
        theta2 = theta * theta
        b = (1.0 - cosine) / theta2
        c = (theta - sine) / (theta2 * theta)
        b_prime = (theta * sine - 2.0 * (1.0 - cosine)) / (theta2 * theta)
        c_prime = ((1.0 - cosine) * theta - 3.0 * (theta - sine)) / (theta2 * theta2)
    rho_hat = hat3(rho)
    gradient = -b * rho_hat - c * (hat3(cross) + hat3(phi) @ rho_hat)
    if theta > _DEGENERATE:
        unit = phi / theta
        gradient = gradient + np.outer(b_prime * cross + c_prime * double_cross, unit)
    return gradient


def linearize_constraint_margins(
    state: ArrayLike,
    task: Phase2TaskConfig,
    *,
    corridor_facets: int,
    distance_scale_m: float = 3.0,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(jacobian, offset)`` with ``margins ~= jacobian @ x - offset``.

    Analytic. Each margin is a function of the relative position ``p``, the
    target-frame relative velocity ``s = R v`` and the chaser-frame line of
    sight ``z = R^T (p_port - p)``, and each of those differentiates in closed
    form against the exponential coordinates:

        d p = (d p / d phi) d phi + J_l(phi) d rho
        d s = -R [v]_x J_r(phi) d phi + R d v
        d z = [z]_x J_r(phi) d phi - R^T d p

    with ``J_r(phi) = J_l(phi)^T``. Only ``d p / d phi`` needs work of its own,
    and ``_position_rotation_gradient`` supplies it.

    The closing-speed limit has a kink where the range-dependent envelope meets
    its floor and its ceiling; the right-hand derivative is taken there, which
    is what the one-sided finite difference this replaces also selected.
    """

    x = _state(state)
    if corridor_facets < 4 or distance_scale_m <= 0.0:
        raise ValueError("invalid constraint geometry settings")
    geometry = _corridor_geometry(task, corridor_facets, distance_scale_m)
    phi, rho, velocity = x[:3], x[3:6], x[9:]
    rotation, position, left_jacobian = _pose(x)
    right_jacobian = left_jacobian.T
    axis = geometry.axis
    count = corridor_facets + 4

    # d(margin)/d(p): the corridor block is constant, the field of view enters
    # through the line of sight and the closing speed through its envelope.
    position_gradient = np.zeros((count, 3), dtype=np.float64)
    position_gradient[: corridor_facets + 1] = geometry.position_gradient

    # d(margin)/d(s = R v): total speed and closing speed only.
    rate_gradient = np.zeros((count, 3), dtype=np.float64)
    position_rate = rotation @ velocity
    speed = float(np.linalg.norm(position_rate))
    if speed > _DEGENERATE:
        rate_gradient[corridor_facets + 2] = (
            -position_rate / speed / task.total_speed_limit_m_s
        )
    rate_gradient[corridor_facets + 3] = axis / task.closing_speed_max_m_s

    # The closing-speed limit is min(max, min_ + slope * max(remaining, 0)), so
    # it moves with the position only on the interior branch.
    axial_remaining = float(axis @ (position - task.desired_position))
    if (
        axial_remaining > 0.0
        and task.closing_speed_min_m_s
        + task.closing_speed_slope_per_s * axial_remaining
        < task.closing_speed_max_m_s
    ):
        position_gradient[corridor_facets + 3] = (
            task.closing_speed_slope_per_s / task.closing_speed_max_m_s
        ) * axis

    # Field of view: the angle between a fixed boresight and z depends on the
    # direction of z alone, so its gradient is the boresight component
    # perpendicular to z, scaled by |z| and by sin(angle).
    angle, line_of_sight, line_of_sight_norm = _fov_angle(rotation, position, task)
    rotation_gradient = np.zeros((count, 3), dtype=np.float64)
    sine = float(np.sin(angle))
    if line_of_sight_norm > _DEGENERATE and abs(sine) > _DEGENERATE:
        unit = line_of_sight / line_of_sight_norm
        boresight = task.camera_boresight
        perpendicular = boresight - float(boresight @ unit) * unit
        # d(margin_fov)/d(z), then chain it onto both of z's dependencies.
        gradient = perpendicular / (
            line_of_sight_norm * sine * task.fov_half_angle_rad
        )
        position_gradient[corridor_facets + 1] = -gradient @ rotation.T
        rotation_gradient[corridor_facets + 1] = gradient @ hat3(line_of_sight)

    jacobian = np.zeros((count, 12), dtype=np.float64)
    jacobian[:, :3] = (
        position_gradient @ _position_rotation_gradient(phi, rho)
        + rotation_gradient @ right_jacobian
        - rate_gradient @ (rotation @ hat3(velocity) @ right_jacobian)
    )
    jacobian[:, 3:6] = position_gradient @ left_jacobian
    # Columns 6:9 stay zero: no margin is a function of the angular velocity.
    jacobian[:, 9:] = rate_gradient @ rotation
    nominal = normalized_constraint_margins(
        x, task, corridor_facets=corridor_facets, distance_scale_m=distance_scale_m
    )
    return jacobian, jacobian @ x - nominal
