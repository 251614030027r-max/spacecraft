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
    reference = (
        np.zeros_like(sx)
        if reference_state is None
        else np.asarray(reference_state, dtype=np.float64)
    )
    if reference.shape != sx.shape:
        raise ValueError("reference state must match state scales")
    stage = sum(
        state_weight * float(np.sum(np.square((x - reference) / sx)))
        + input_weight * float(np.sum(np.square(control / su)))
        for x, control in zip(states[:-1], u)
    )
    return stage + terminal_weight * float(
        np.sum(np.square((states[-1] - reference) / sx))
    )
