"""Map normalized policy actions to physical actuator commands."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynamics.constants import MAX_FORCE_PER_AXIS_N, MAX_TORQUE_PER_AXIS_NM
from dynamics.types import GeneralizedForce


FloatArray = NDArray[np.float64]


def normalized_to_wrench(
    action: ArrayLike,
    *,
    max_torque_per_axis_nm: float = MAX_TORQUE_PER_AXIS_NM,
    max_force_per_axis_n: float = MAX_FORCE_PER_AXIS_N,
) -> GeneralizedForce:
    if max_torque_per_axis_nm <= 0.0 or max_force_per_axis_n <= 0.0:
        raise ValueError("actuator limits must be positive")
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("action must be a finite shape=(6,) vector")
    clipped = np.clip(value, -1.0, 1.0)
    return GeneralizedForce(
        torque=max_torque_per_axis_nm * clipped[:3],
        force=max_force_per_axis_n * clipped[3:],
    )


def wrench_to_normalized(
    wrench: GeneralizedForce,
    *,
    max_torque_per_axis_nm: float = MAX_TORQUE_PER_AXIS_NM,
    max_force_per_axis_n: float = MAX_FORCE_PER_AXIS_N,
) -> FloatArray:
    if max_torque_per_axis_nm <= 0.0 or max_force_per_axis_n <= 0.0:
        raise ValueError("actuator limits must be positive")
    normalized = np.concatenate(
        (
            wrench.torque / max_torque_per_axis_nm,
            wrench.force / max_force_per_axis_n,
        )
    )
    return np.clip(normalized, -1.0, 1.0)
