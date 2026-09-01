"""Declared, dimensionless MPC-only configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamics.lie import se3_log
from env.task import Phase2TaskConfig

from .terminal_value import ConvexQuadraticTerminalValue


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
    # "fixed" holds the reference at reference_state for the whole horizon -- a
    # myopic regulator. "corridor_guidance" rolls env.task.corridor_guidance_velocity
    # forward from the current relative position, giving the QP the same path
    # plan the reward and the scripted controller use, so the comparison is not
    # decided by one method having a reference and the other not.
    reference_source: str = "fixed"
    # Fraction of the total-speed limit the corridor-guidance reference aims for
    # (the coefficient inside env.task.corridor_guidance_velocity, default 0.6).
    # Lowering it flies the co-rotation more conservatively, buying total_speed
    # margin -- the knob for the observation-error margin control (fly at 0.5x /
    # 0.4x). It changes only the MPC's own reference, not the task or reward, so
    # it stays a clean single factor. Applies only under corridor_guidance.
    corridor_speed_fraction: float = 0.6
    # "fixed_quadratic" keeps the historical terminal penalty
    # terminal_weight * ||S^-1 (x - r)||^2 -- a diagonal cost on the deviation
    # from the terminal reference, bitwise unchanged. "learned_convex" replaces
    # it with the fitted convex quadratic in `terminal_value`, the single
    # interpretable factor of the SAC-MPC coupling minimal experiment.
    terminal_cost_source: str = "fixed_quadratic"
    terminal_value: ConvexQuadraticTerminalValue | None = None
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
        if self.reference_source not in {"fixed", "corridor_guidance"}:
            raise ValueError("reference_source must be 'fixed' or 'corridor_guidance'")
        if self.reference_source == "corridor_guidance" and self.task is None:
            raise ValueError("corridor_guidance reference requires a task")
        if self.terminal_cost_source not in {"fixed_quadratic", "learned_convex"}:
            raise ValueError(
                "terminal_cost_source must be 'fixed_quadratic' or 'learned_convex'"
            )
        if self.terminal_cost_source == "learned_convex" and not isinstance(
            self.terminal_value, ConvexQuadraticTerminalValue
        ):
            raise ValueError(
                "learned_convex terminal cost requires a ConvexQuadraticTerminalValue"
            )
        if self.corridor_facets < 4:
            raise ValueError("corridor_facets must be at least four")
        if not 0.0 < self.corridor_speed_fraction <= 1.0:
            raise ValueError("corridor_speed_fraction must be in (0, 1]")
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


def corridor_tracking_mpc_config(**changes: object) -> MPCConfig:
    """Constrained MPC that tracks the corridor-guidance path, not a setpoint.

    Same task, solver and horizon as the nominal config; the only change is that
    the reference is the corridor-guidance trajectory the reward and the scripted
    controller already use, so Pure MPC is compared on how it flies a shared
    reference rather than on whether it has one. This is the fair Pure MPC row
    for the single-phase task; the fixed-setpoint config stays the record for
    the terminal-only evidence measured before it existed.
    """

    return constrained_mpc_nominal_config(
        reference_source="corridor_guidance", **changes
    )


def learned_terminal_mpc_config(
    terminal_value: ConvexQuadraticTerminalValue,
    *,
    horizon_steps: int = 10,
    **changes: object,
) -> MPCConfig:
    """Short-horizon constrained MPC with a learned convex terminal value.

    This is the (c) row of the minimal SAC-MPC coupling experiment: the same
    task, solver and corridor-guidance reference as the fair Pure MPC baseline,
    but a shortened horizon whose terminal cost is the fitted convex quadratic
    rather than the fixed diagonal penalty. The single interpretable factor
    against the matched short-horizon "fixed_quadratic" row (b) is
    ``terminal_cost_source``.
    """

    values: dict[str, object] = {
        "reference_source": "corridor_guidance",
        "horizon_steps": horizon_steps,
        "terminal_cost_source": "learned_convex",
        "terminal_value": terminal_value,
    }
    values.update(changes)
    return constrained_mpc_nominal_config(**values)


# Historical name retained only for loading old evidence/scripts. Current code
# imports constrained_mpc_nominal_config explicitly.
phase2_mpc_config = constrained_mpc_nominal_config
