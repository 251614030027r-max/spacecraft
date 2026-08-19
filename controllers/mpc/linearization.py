"""Scale-aware central-difference linearization."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def central_difference_linearization(
    function: Callable[[FloatArray, FloatArray], FloatArray],
    state: ArrayLike,
    control: ArrayLike,
    *,
    state_scales: ArrayLike,
    input_scales: ArrayLike,
    relative_step: float = 1.0e-4,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.asarray(state, dtype=np.float64)
    u = np.asarray(control, dtype=np.float64)
    sx = np.asarray(state_scales, dtype=np.float64)
    su = np.asarray(input_scales, dtype=np.float64)
    if x.shape != (12,) or u.shape != (6,):
        raise ValueError("linearization state/input shapes must be (12,) and (6,)")
    if sx.shape != x.shape or su.shape != u.shape or relative_step <= 0.0:
        raise ValueError("invalid difference scales")
    a = np.empty((12, 12), dtype=np.float64)
    b = np.empty((12, 6), dtype=np.float64)
    for index, scale in enumerate(sx):
        delta = relative_step * scale
        direction = np.zeros_like(x)
        direction[index] = delta
        a[:, index] = (
            function(x + direction, u) - function(x - direction, u)
        ) / (2.0 * delta)
    for index, scale in enumerate(su):
        delta = relative_step * scale
        direction = np.zeros_like(u)
        direction[index] = delta
        b[:, index] = (
            function(x, u + direction) - function(x, u - direction)
        ) / (2.0 * delta)
    nominal = function(x, u)
    c = nominal - a @ x - b @ u
    if not all(np.all(np.isfinite(value)) for value in (a, b, c)):
        raise FloatingPointError("non-finite MPC linearization")
    return a, b, c
