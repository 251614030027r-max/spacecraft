"""Normalized truth margins and local affine models for constrained MPC."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynamics.lie import se3_exp
from dynamics.relative import RelativeState
from env.task import Phase2TaskConfig, compute_task_metrics


FloatArray = NDArray[np.float64]


def normalized_truth_margins(
    state: ArrayLike, task: Phase2TaskConfig
) -> FloatArray:
    """Four task-truth margins in dimensionless units."""

    x = np.asarray(state, dtype=np.float64)
    if x.shape != (12,) or not np.all(np.isfinite(x)):
        raise ValueError("constraint state must be finite and shape=(12,)")
    metrics = compute_task_metrics(RelativeState(se3_exp(x[:6]), x[6:]), task)
    return np.asarray(
        [
            min(
                metrics.corridor_axial_margin_m,
                metrics.corridor_lateral_margin_m,
            )
            / 3.0,
            metrics.fov_margin_rad / task.fov_half_angle_rad,
            metrics.total_speed_margin_m_s / task.total_speed_limit_m_s,
            metrics.closing_speed_margin_m_s / task.closing_speed_max_m_s,
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

    x = np.asarray(state, dtype=np.float64)
    if x.shape != (12,) or not np.all(np.isfinite(x)):
        raise ValueError("constraint state must be finite and shape=(12,)")
    if corridor_facets < 4 or distance_scale_m <= 0.0:
        raise ValueError("invalid constraint geometry settings")
    relative = RelativeState(se3_exp(x[:6]), x[6:])
    metrics = compute_task_metrics(relative, task)
    axis = task.approach_axis
    displacement = relative.position - task.port_position
    axial = float(axis @ displacement)
    lateral = displacement - axial * axis
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    polygon_radius = (
        axial
        * np.tan(task.corridor_half_angle_rad)
        * np.cos(np.pi / corridor_facets)
    )
    corridor = [axial / distance_scale_m]
    for index in range(corridor_facets):
        angle = 2.0 * np.pi * index / corridor_facets
        direction = np.cos(angle) * first + np.sin(angle) * second
        corridor.append((polygon_radius - float(direction @ lateral)) / distance_scale_m)
    return np.asarray(
        [
            *corridor,
            metrics.fov_margin_rad / task.fov_half_angle_rad,
            metrics.total_speed_margin_m_s / task.total_speed_limit_m_s,
            metrics.closing_speed_margin_m_s / task.closing_speed_max_m_s,
        ],
        dtype=np.float64,
    )


def linearize_constraint_margins(
    state: ArrayLike,
    task: Phase2TaskConfig,
    *,
    corridor_facets: int,
    state_scales: ArrayLike,
    relative_step: float,
) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(state, dtype=np.float64)
    scales = np.asarray(state_scales, dtype=np.float64)
    nominal = normalized_constraint_margins(
        x, task, corridor_facets=corridor_facets
    )
    jacobian = np.empty((nominal.size, 12), dtype=np.float64)
    # Margins do not depend on angular velocity. A one-sided local derivative
    # halves geometry evaluations while retaining a step far below task tolerances.
    for index in (*range(6), *range(9, 12)):
        scale = scales[index]
        delta = relative_step * scale
        direction = np.zeros(12)
        direction[index] = delta
        jacobian[:, index] = (
            normalized_constraint_margins(
                x + direction, task, corridor_facets=corridor_facets
            )
            - nominal
        ) / delta
    jacobian[:, 6:9] = 0.0
    return jacobian, jacobian @ x - nominal
