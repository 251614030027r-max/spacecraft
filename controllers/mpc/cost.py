"""Dimensionless MPC cost helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def nonlinear_rollout_cost(
    states: list[np.ndarray],
    controls: ArrayLike,
    *,
    state_scales: ArrayLike,
    input_scales: ArrayLike,
    state_weight: float,
    input_weight: float,
    terminal_weight: float,
    reference_state: ArrayLike | None = None,
) -> float:
    u = np.asarray(controls, dtype=np.float64)
    sx = np.asarray(state_scales, dtype=np.float64)
    su = np.asarray(input_scales, dtype=np.float64)
    # A single reference is broadcast to every stage (the fixed-setpoint case);
    # a (len(states), 12) array is a per-stage reference trajectory, which the
    # objective and this diagnostic have to share or the number is meaningless.
    if reference_state is None:
        reference = np.zeros((len(states), sx.size), dtype=np.float64)
    else:
        reference = np.asarray(reference_state, dtype=np.float64)
        if reference.ndim == 1:
            reference = np.tile(reference, (len(states), 1))
    if reference.shape != (len(states), sx.size):
        raise ValueError("reference must match state scales and rollout length")
    stage = sum(
        state_weight * float(np.sum(np.square((x - reference[index]) / sx)))
        + input_weight * float(np.sum(np.square(control / su)))
        for index, (x, control) in enumerate(zip(states[:-1], u))
    )
    return stage + terminal_weight * float(
        np.sum(np.square((states[-1] - reference[-1]) / sx))
    )
