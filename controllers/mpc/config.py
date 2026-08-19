"""Declared, dimensionless MPC-only configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics.lie import se3_log
from env.task import Phase2TaskConfig


def _state_scales() -> np.ndarray:
    return np.array(
        [
            *([np.deg2rad(75.0)] * 3),
            *([3.0] * 3),
            *([0.05] * 3),
            *([0.2] * 3),
        ],
        dtype=np.float64,
    )


def _input_scales() -> np.ndarray:
    return np.array([*([0.6] * 3), *([5.0] * 3)], dtype=np.float64)


@dataclass(frozen=True)
class MPCConfig:
    dt_s: float = 0.1
    horizon_steps: int = 50
    outer_iterations: int = 1
    linearization_stride: int = 5
    state_scales: np.ndarray = field(default_factory=_state_scales)
    input_scales: np.ndarray = field(default_factory=_input_scales)
    state_weight: float = 1.0
    input_weight: float = 0.01
    terminal_weight: float = 100.0
    difference_relative_step: float = 1.0e-4
    drift_refresh_steps: int = 10
    linearization_source: str = "exact"
    exact_linearization_refresh_steps: int = 10
    solver: str = "OSQP"
    solver_max_iter: int = 10_000
    solver_eps_abs: float = 1.0e-4
    solver_eps_rel: float = 1.0e-4
    reference_state: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.float64)
    )
    task: Phase2TaskConfig | None = None
    corridor_facets: int = 8
    constraint_slack_weight: float = 1.0e4
    constraint_slack_limit: float = 2.0
    constraint_tightening: float = 0.02

    def __post_init__(self) -> None:
        state_scales = np.asarray(self.state_scales, dtype=np.float64)
        input_scales = np.asarray(self.input_scales, dtype=np.float64)
        reference_state = np.asarray(self.reference_state, dtype=np.float64)
        if state_scales.shape != (12,) or input_scales.shape != (6,):
            raise ValueError("MPC scales must have shapes (12,) and (6,)")
        if reference_state.shape != (12,) or not np.all(np.isfinite(reference_state)):
            raise ValueError("MPC reference state must be finite and shape=(12,)")
        if np.min(state_scales) <= 0.0 or np.min(input_scales) <= 0.0:
            raise ValueError("MPC scales must be positive")
        if min(self.dt_s, self.difference_relative_step) <= 0.0:
            raise ValueError("time and difference steps must be positive")
        if min(
            self.horizon_steps,
            self.outer_iterations,
            self.linearization_stride,
            self.drift_refresh_steps,
            self.exact_linearization_refresh_steps,
        ) <= 0:
            raise ValueError("MPC integer settings must be positive")
        if self.linearization_source not in {"exact", "local"}:
            raise ValueError("linearization_source must be 'exact' or 'local'")
        if self.corridor_facets < 4:
            raise ValueError("corridor_facets must be at least four")
        if min(self.constraint_slack_weight, self.constraint_slack_limit) <= 0.0:
            raise ValueError("constraint slack settings must be positive")
        if self.constraint_tightening < 0.0:
            raise ValueError("constraint tightening must be non-negative")
        object.__setattr__(self, "state_scales", state_scales.copy())
        object.__setattr__(self, "input_scales", input_scales.copy())
        object.__setattr__(self, "reference_state", reference_state.copy())


def constrained_mpc_nominal_config(**changes: object) -> MPCConfig:
    """Return the single canonical constrained MPC baseline configuration."""
    task = Phase2TaskConfig()
    reference = np.concatenate((se3_log(task.desired_transform), np.zeros(6)))
    values = {
        "task": task,
        "reference_state": reference,
        # The Phase-2 soft-constraint QP is numerically ill-conditioned in
        # OSQP for viable near-boundary states; CLARABEL solved the same
        # formulation without fallback while preserving the truth constraints.
        "solver": "CLARABEL",
    }
    values.update(changes)
    return MPCConfig(**values)


# Historical name retained only for loading old evidence/scripts. Current code
# imports constrained_mpc_nominal_config explicitly.
phase2_mpc_config = constrained_mpc_nominal_config
