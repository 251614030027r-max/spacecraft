"""Successive-linearization, input-constrained MPC-only controller."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynamics.types import SpacecraftState

from .config import MPCConfig
from .constraints import (
    linearize_constraint_margins,
    normalized_constraint_margins,
    normalized_truth_margins,
)
from .cost import nonlinear_rollout_cost
from .linearization import central_difference_linearization
from .prediction import LocalRelativePredictionModel, RelativePredictionModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MPCStepDiagnostics:
    status: str
    solve_time_s: float
    solver_iterations: int
    outer_iterations: int
    objective: float
    predicted_cost: float
    used_zero_fallback: bool
    constraint_linearization_time_s: float
    maximum_slack: float
    total_slack: float
    predicted_minimum_margin: float
    predicted_first_step_margins: tuple[float, ...]


class MPCController:
    def __init__(
        self,
        config: MPCConfig,
        local_model: LocalRelativePredictionModel,
        reference_model: RelativePredictionModel | None = None,
    ) -> None:
        self.config = config
        self.local_model = local_model
        self.reference_model = reference_model
        n = config.horizon_steps
        self._x = cp.Variable((12, n + 1))
        self._u = cp.Variable((6, n))
        self._x0 = cp.Parameter(12)
        self._a = [cp.Parameter((12, 12)) for _ in range(n)]
        self._b = [cp.Parameter((12, 6)) for _ in range(n)]
        self._c = [cp.Parameter(12) for _ in range(n)]
        self._constraint_jacobians: list[cp.Parameter] = []
        self._constraint_offsets: list[cp.Parameter] = []
        self._slack: cp.Variable | None = None
        sx_inverse = np.diag(1.0 / config.state_scales)
        su_inverse = np.diag(1.0 / config.input_scales)
        constraints: list[cp.Constraint] = [self._x[:, 0] == self._x0]
        objective = 0.0
        reference = config.reference_state
        constraint_count = config.corridor_facets + 4
        if config.task is not None:
            self._slack = cp.Variable((constraint_count, n), nonneg=True)
            self._constraint_jacobians = [
                cp.Parameter((constraint_count, 12)) for _ in range(n)
            ]
            self._constraint_offsets = [
                cp.Parameter(constraint_count) for _ in range(n)
            ]
        for index in range(n):
            constraints.extend(
                [
                    self._x[:, index + 1]
                    == self._a[index] @ self._x[:, index]
                    + self._b[index] @ self._u[:, index]
                    + self._c[index],
                    self._u[:, index] <= config.input_scales,
                    self._u[:, index] >= -config.input_scales,
                ]
            )
            objective += config.state_weight * cp.sum_squares(
                sx_inverse @ (self._x[:, index] - reference)
            ) + config.input_weight * cp.sum_squares(
                su_inverse @ self._u[:, index]
            )
            if self._slack is not None:
                constraints.extend(
                    [
                        self._constraint_jacobians[index]
                        @ self._x[:, index + 1]
                        + self._slack[:, index]
                        >= self._constraint_offsets[index],
                        self._slack[:, index] <= config.constraint_slack_limit,
                    ]
                )
                objective += config.constraint_slack_weight * cp.sum_squares(
                    self._slack[:, index]
                )
        objective += config.terminal_weight * cp.sum_squares(
            sx_inverse @ (self._x[:, n] - reference)
        )
        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        self._nominal_controls = np.zeros((n, 6), dtype=np.float64)
        self._drift = np.zeros(12, dtype=np.float64)
        self._exact_linearization: tuple[
            FloatArray, FloatArray, FloatArray
        ] | None = None
        self._control_step = 0

    def reset(self) -> None:
        self._nominal_controls.fill(0.0)
        self._drift.fill(0.0)
        self._exact_linearization = None
        self._control_step = 0

    def _predict(self, state: FloatArray, control: FloatArray) -> FloatArray:
        return self.local_model.predict(state, control) + self._drift

    def _rollout(self, initial: FloatArray, controls: FloatArray) -> list[FloatArray]:
        states = [initial.copy()]
        for control in controls:
            states.append(self._predict(states[-1], control))
        return states

    def _refresh_drift(
        self, state: FloatArray, target: SpacecraftState, time_seconds: float
    ) -> None:
        if self.reference_model is None:
            self._drift.fill(0.0)
            return
        reference, _ = self.reference_model.predict(
            state, np.zeros(6), target, time_seconds
        )
        self._drift = reference - self.local_model.predict(state, np.zeros(6))

    def _refresh_exact_linearization(
        self, state: FloatArray, target: SpacecraftState, time_seconds: float
    ) -> None:
        if self.reference_model is None:
            raise RuntimeError("exact MPC linearization requires reference model")
        target_next = self.reference_model.propagate_target(target, time_seconds)

        def exact_map(x: FloatArray, u: FloatArray) -> FloatArray:
            assert self.reference_model is not None
            return self.reference_model.predict_given_target_next(
                x, u, target, target_next, time_seconds
            )

        self._exact_linearization = central_difference_linearization(
            exact_map,
            state,
            self._nominal_controls[0],
            state_scales=self.config.state_scales,
            input_scales=self.config.input_scales,
            relative_step=self.config.difference_relative_step,
        )

    def command(
        self,
        relative_vector: ArrayLike,
        *,
        target_state: SpacecraftState,
        time_seconds: float,
    ) -> tuple[FloatArray, MPCStepDiagnostics]:
        state = np.asarray(relative_vector, dtype=np.float64)
        if state.shape != (12,) or not np.all(np.isfinite(state)):
            raise ValueError("MPC state must be finite and shape=(12,)")
        use_exact = self.config.linearization_source == "exact"
        if use_exact and (
            self._exact_linearization is None
            or self._control_step
            % self.config.exact_linearization_refresh_steps
            == 0
        ):
            self._refresh_exact_linearization(state, target_state, time_seconds)
        elif not use_exact and self._control_step % self.config.drift_refresh_steps == 0:
            self._refresh_drift(state, target_state, time_seconds)
        controls = np.vstack((self._nominal_controls[1:], np.zeros((1, 6))))
        status = "not_solved"
        total_solve_time = 0.0
        constraint_linearization_time = 0.0
        solver_iterations = 0
        objective = float("nan")
        completed_outer = 0
        try:
            for outer in range(self.config.outer_iterations):
                nominal_states = self._rollout(state, controls)
                cached_linearization: tuple[FloatArray, FloatArray, FloatArray] | None = None
                for index in range(self.config.horizon_steps):
                    if use_exact:
                        assert self._exact_linearization is not None
                        cached_linearization = self._exact_linearization
                    elif index % self.config.linearization_stride == 0:
                        cached_linearization = central_difference_linearization(
                            self._predict,
                            nominal_states[index],
                            controls[index],
                            state_scales=self.config.state_scales,
                            input_scales=self.config.input_scales,
                            relative_step=self.config.difference_relative_step,
                        )
                    assert cached_linearization is not None
                    self._a[index].value, self._b[index].value, self._c[index].value = (
                        cached_linearization
                    )
                    if self.config.task is not None:
                        started_constraints = perf_counter()
                        jacobian, offset = linearize_constraint_margins(
                            nominal_states[index + 1],
                            self.config.task,
                            corridor_facets=self.config.corridor_facets,
                            state_scales=self.config.state_scales,
                            relative_step=self.config.difference_relative_step,
                        )
                        constraint_linearization_time += (
                            perf_counter() - started_constraints
                        )
                        self._constraint_jacobians[index].value = jacobian
                        self._constraint_offsets[index].value = (
                            offset + self.config.constraint_tightening
                        )
                self._x0.value = state
                started = perf_counter()
                solver_options = dict(
                    solver=self.config.solver,
                    warm_start=True,
                    verbose=False,
                    max_iter=self.config.solver_max_iter,
                )
                if self.config.solver == "OSQP":
                    solver_options.update(
                        eps_abs=self.config.solver_eps_abs,
                        eps_rel=self.config.solver_eps_rel,
                        polishing=True,
                    )
                elif self.config.solver == "CLARABEL":
                    solver_options.update(
                        tol_gap_abs=self.config.solver_eps_abs,
                        tol_gap_rel=self.config.solver_eps_rel,
                        tol_feas=self.config.solver_eps_abs,
                    )
                self._problem.solve(**solver_options)
                total_solve_time += perf_counter() - started
                status = str(self._problem.status)
                completed_outer = outer + 1
                stats = self._problem.solver_stats
                solver_iterations += int(stats.num_iters or 0)
                if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or self._u.value is None:
                    raise RuntimeError(f"QP status {status}")
                controls = np.asarray(self._u.value.T, dtype=np.float64)
                objective = float(self._problem.value)
            controls = np.clip(
                controls, -self.config.input_scales, self.config.input_scales
            )
            command = controls[0].copy()
            self._nominal_controls = controls
            used_fallback = False
        except (cp.SolverError, RuntimeError, FloatingPointError, ValueError):
            command = np.zeros(6, dtype=np.float64)
            self._nominal_controls.fill(0.0)
            used_fallback = True
        predicted_states = self._rollout(state, self._nominal_controls)
        if self._slack is not None and self._slack.value is not None:
            slack_values = np.asarray(self._slack.value, dtype=np.float64)
            maximum_slack = float(np.max(slack_values))
            total_slack = float(np.sum(slack_values))
        else:
            maximum_slack = total_slack = 0.0
        predicted_minimum_margin = (
            min(
                float(np.min(normalized_truth_margins(item, self.config.task)))
                for item in predicted_states[1:]
            )
            if self.config.task is not None
            else float("nan")
        )
        predicted_first_step_margins = (
            tuple(
                float(value)
                for value in normalized_truth_margins(
                    predicted_states[1], self.config.task
                )
            )
            if self.config.task is not None
            else ()
        )
        predicted_cost = nonlinear_rollout_cost(
            predicted_states,
            self._nominal_controls,
            state_scales=self.config.state_scales,
            input_scales=self.config.input_scales,
            state_weight=self.config.state_weight,
            input_weight=self.config.input_weight,
            terminal_weight=self.config.terminal_weight,
            reference_state=self.config.reference_state,
        )
        self._control_step += 1
        return command, MPCStepDiagnostics(
            status=status,
            solve_time_s=total_solve_time,
            solver_iterations=solver_iterations,
            outer_iterations=completed_outer,
            objective=objective,
            predicted_cost=predicted_cost,
            used_zero_fallback=used_fallback,
            constraint_linearization_time_s=constraint_linearization_time,
            maximum_slack=maximum_slack,
            total_slack=total_slack,
            predicted_minimum_margin=predicted_minimum_margin,
            predicted_first_step_margins=predicted_first_step_margins,
        )
