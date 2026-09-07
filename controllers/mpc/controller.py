"""Successive-linearization, input-constrained MPC-only controller."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from dynamics.lie import (
    left_jacobian_so3,
    make_transform,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)
from dynamics.types import SpacecraftState
from env.task import corridor_guidance_velocity

from .config import MPCConfig
from .constraints import (
    linearize_constraint_margins,
    linearize_precapture_constraint_margins,
    normalized_constraint_margins,
    normalized_precapture_truth_margins,
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
    model_linearization_time_s: float
    rollout_time_s: float
    exact_refresh_time_s: float
    maximum_slack: float
    total_slack: float
    predicted_minimum_margin: float
    predicted_first_step_margins: tuple[float, ...]


def _unit_or_none(vector: FloatArray) -> FloatArray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


def _align_rotation(start: FloatArray, goal: FloatArray) -> FloatArray:
    """Minimal rotation carrying unit vector ``start`` onto unit vector ``goal``."""

    cosine = float(np.clip(start @ goal, -1.0, 1.0))
    cross = np.cross(start, goal)
    sine = float(np.linalg.norm(cross))
    if sine <= 1.0e-12:
        if cosine > 0.0:
            return np.eye(3)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(seed @ start)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        axis = np.cross(start, seed)
        axis /= np.linalg.norm(axis)
        return so3_exp(np.pi * axis)
    return so3_exp(np.arctan2(sine, cosine) * (cross / sine))


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
        # Per-stage reference parameter. Filled each step: broadcast from the
        # fixed setpoint under "fixed", or a corridor-guidance path under
        # "corridor_guidance". A parameter, not a constant, so the trajectory
        # can move without rebuilding the problem.
        self._reference = cp.Parameter((12, n + 1))
        self._reference.value = np.tile(
            config.reference_state.reshape(12, 1), (1, n + 1)
        )
        constraint_count = (
            config.corridor_facets + 7
            if config.precapture_task is not None
            else config.corridor_facets + 4
        )
        if config.task is not None or config.precapture_task is not None:
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
                sx_inverse @ (self._x[:, index] - self._reference[:, index])
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
        if config.terminal_cost_source == "learned_convex":
            # A learned convex quadratic V(x) = (x-c)^T H (x-c) + g^T (x-c),
            # entering as ||L (x-c)||^2 + g^T (x-c) with L^T L = H. Every term is
            # a numpy constant baked in here, so the terminal cost is a fixed
            # affine-of-variable expression -- trivially DCP/DPP, no neural
            # network and no manifold Jacobian in the solver. It replaces the
            # fixed diagonal penalty; the reference tracked over the horizon is
            # unchanged, only the last-stage cost differs.
            terminal_value = config.terminal_value
            assert terminal_value is not None
            factor = terminal_value.cholesky_factor()
            center = terminal_value.center
            deviation = self._x[:, n] - center
            objective += cp.sum_squares(factor @ deviation) + (
                terminal_value.linear @ deviation
            )
        else:
            objective += config.terminal_weight * cp.sum_squares(
                sx_inverse @ (self._x[:, n] - self._reference[:, n])
            )
        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        self._nominal_controls = np.zeros((n, 6), dtype=np.float64)
        self._drift = np.zeros(12, dtype=np.float64)
        self._exact_linearization: tuple[
            FloatArray, FloatArray, FloatArray
        ] | None = None
        self._control_step = 0
        self._episode_plan_initial_state: FloatArray | None = None
        self._episode_plan_start_time_s: float | None = None
        self._held_external_reference: FloatArray | None = None
        self._held_external_velocity = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self._nominal_controls.fill(0.0)
        self._drift.fill(0.0)
        self._exact_linearization = None
        self._control_step = 0
        self._episode_plan_initial_state = None
        self._episode_plan_start_time_s = None
        self._held_external_reference = None
        self._held_external_velocity.fill(0.0)

    def _precapture_los_rotation(self, state: FloatArray) -> FloatArray:
        """Nearest current-roll attitude whose camera boresight points at port."""

        task = self.config.precapture_task
        if task is None:
            raise ValueError("LOS reference requires a precapture task")
        transform = se3_exp(state[:6])
        current_rotation = transform[:3, :3]
        line_of_sight = task.port_position - transform[:3, 3]
        line_of_sight /= np.linalg.norm(line_of_sight)
        current_boresight = current_rotation @ task.camera_boresight
        cosine = float(np.clip(current_boresight @ line_of_sight, -1.0, 1.0))
        cross = np.cross(current_boresight, line_of_sight)
        sine = float(np.linalg.norm(cross))
        if sine <= 1.0e-12:
            if cosine > 0.0:
                return current_rotation
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(reference @ current_boresight)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            axis = np.cross(current_boresight, reference)
            axis /= np.linalg.norm(axis)
            return so3_exp(np.pi * axis) @ current_rotation
        axis = cross / sine
        return so3_exp(np.arctan2(sine, cosine) * axis) @ current_rotation

    def _precapture_reference_rotation(
        self, state: FloatArray, *, terminal_latched: bool | None
    ) -> FloatArray:
        """Relative attitude the reference asks for: LOS outside, docking inside."""

        if self.config.precapture_task is None or bool(terminal_latched):
            return np.eye(3)
        return self._precapture_los_rotation(state)

    def _precapture_reference_rotations(
        self,
        state: FloatArray,
        positions: FloatArray,
        *,
        terminal_latched: bool | None,
    ) -> FloatArray:
        """One reference attitude per horizon index, under the configured mode.

        Reference positions are written in the *target* body frame, so a
        waypoint held fixed in inertial space sweeps through that frame at the
        target's tumble rate -- 0.0412 rad/s, 2.36 deg/s here. The three modes
        differ only in whether the attitude half of the reference is allowed to
        follow that sweep:

        ``frozen``
            One attitude, aimed from the *state*, held across the horizon, and
            (through ``_reference_angular_rates``) a zero relative rate. This
            is the behaviour every recorded precapture row was measured under
            and stays the default, so no prior number moves.
        ``swept``
            Index 0 as in ``frozen``, later indices carried by the rotation the
            *reference* sightline itself undergoes. Pose and rate then describe
            the same motion, but only for a chaser that is on its reference:
            under a large position lag the swept rate aims from where the
            reference is, not from where the chaser is.
        ``aimed``
            Every index aimed from its own reference position, so the reference
            is one self-consistent trajectory -- at the cost of a parallax step
            at index 0 whenever the chaser is not yet on the waypoint.
        """

        count = int(positions.shape[1])
        rotations = np.zeros((count, 3, 3), dtype=np.float64)
        if self.config.precapture_task is None or bool(terminal_latched):
            rotations[:] = np.eye(3)
            return rotations
        mode = self.config.precapture_attitude_reference
        task = self.config.precapture_task
        rotation = self._precapture_los_rotation(state)
        if mode == "frozen":
            rotations[:] = rotation
            return rotations
        previous = _unit_or_none(task.port_position - positions[:, 0])
        for index in range(count):
            if mode == "aimed":
                # Aim index 0 too: the parallax between the chaser and its own
                # waypoint belongs in the tracking error, not in the first rate
                # row, where it would appear as a step of order 1 rad/s.
                rotation = self._aim_rotation(rotation, positions[:, index])
            elif index > 0:
                current = _unit_or_none(task.port_position - positions[:, index])
                if previous is not None and current is not None:
                    rotation = _align_rotation(previous, current) @ rotation
                if current is not None:
                    previous = current
            rotations[index] = rotation
        return rotations

    def _aim_rotation(
        self, rotation: FloatArray, position: FloatArray
    ) -> FloatArray:
        """Nearest rotation to ``rotation`` whose boresight points at the port."""

        task = self.config.precapture_task
        if task is None:
            raise ValueError("LOS reference requires a precapture task")
        line_of_sight = _unit_or_none(task.port_position - position)
        if line_of_sight is None:
            return rotation
        boresight = _unit_or_none(rotation @ task.camera_boresight)
        if boresight is None:
            return rotation
        return _align_rotation(boresight, line_of_sight) @ rotation

    def _reference_angular_rates(
        self, rotations: FloatArray
    ) -> FloatArray:
        """Body-frame relative rate the reference attitude sequence implies."""

        count = int(rotations.shape[0])
        rates = np.zeros((3, count), dtype=np.float64)
        if count < 2:
            return rates
        for index in range(count - 1):
            rates[:, index] = (
                so3_log(rotations[index].T @ rotations[index + 1])
                / self.config.dt_s
            )
        rates[:, count - 1] = rates[:, count - 2]
        return rates

    def _apply_precapture_attitude_reference(
        self,
        reference: FloatArray,
        state: FloatArray,
        *,
        terminal_latched: bool | None,
    ) -> FloatArray:
        if self.config.precapture_task is None or bool(terminal_latched):
            return reference
        positions = np.zeros((3, reference.shape[1]), dtype=np.float64)
        for index in range(reference.shape[1]):
            positions[:, index] = se3_exp(reference[:6, index])[:3, 3]
        rotations = self._precapture_reference_rotations(
            state, positions, terminal_latched=terminal_latched
        )
        adjusted = reference.copy()
        for index in range(reference.shape[1]):
            adjusted[:6, index] = se3_log(
                make_transform(rotations[index], positions[:, index])
            )
        adjusted[6:9, :] = self._reference_angular_rates(rotations)
        return adjusted

    def _endpoint_plan(
        self,
        initial_state: FloatArray,
        sample_times_s: FloatArray,
    ) -> FloatArray:
        """Return a closed-form minimum-jerk path using endpoints only."""

        initial_transform = se3_exp(initial_state[:6])
        desired_transform = se3_exp(self.config.reference_state[:6])
        initial_rotation = initial_transform[:3, :3]
        initial_position = initial_transform[:3, 3]
        desired_position = desired_transform[:3, 3]
        rotation_to_goal = so3_log(initial_rotation.T @ desired_transform[:3, :3])
        distance = float(np.linalg.norm(desired_position - initial_position))
        duration = max(
            self.config.planning_min_duration_s,
            1.875 * distance / self.config.planning_speed_m_s,
        )
        clipped_times = np.clip(sample_times_s, 0.0, duration)

        def quintic(
            start: FloatArray, goal: FloatArray, initial_rate: FloatArray
        ) -> tuple[FloatArray, FloatArray]:
            delta = goal - start
            a3 = (10.0 * delta - 6.0 * initial_rate * duration) / duration**3
            a4 = (-15.0 * delta + 8.0 * initial_rate * duration) / duration**4
            a5 = (6.0 * delta - 3.0 * initial_rate * duration) / duration**5
            t = clipped_times[:, None]
            value = start + initial_rate * t + a3 * t**3 + a4 * t**4 + a5 * t**5
            rate = initial_rate + 3.0 * a3 * t**2 + 4.0 * a4 * t**3 + 5.0 * a5 * t**4
            return value, rate

        positions, position_rates = quintic(
            initial_position,
            desired_position,
            initial_rotation @ initial_state[9:12],
        )
        rotation_vectors, rotation_rates = quintic(
            np.zeros(3, dtype=np.float64),
            rotation_to_goal,
            initial_state[6:9],
        )
        reference = np.zeros((12, sample_times_s.size), dtype=np.float64)
        for index, (rotation_vector, rotation_rate) in enumerate(
            zip(rotation_vectors, rotation_rates)
        ):
            rotation = initial_rotation @ so3_exp(rotation_vector)
            position = positions[index]
            position_rate = position_rates[index]
            reference[:6, index] = se3_log(make_transform(rotation, position))
            reference[6:9, index] = left_jacobian_so3(-rotation_vector) @ rotation_rate
            reference[9:12, index] = rotation.T @ position_rate
        return reference

    def _reference_trajectory(
        self,
        state: FloatArray,
        time_seconds: float = 0.0,
        *,
        target_state: SpacecraftState | None = None,
        external_reference: ArrayLike | None = None,
        terminal_latched: bool | None = None,
    ) -> FloatArray:
        """Return the ``(12, horizon+1)`` reference the objective tracks.

        Under ``"fixed"`` every stage is ``reference_state`` -- the myopic
        regulator, and the value is bitwise the construction-time broadcast, so
        prior evidence is unchanged. Under ``"corridor_guidance"`` the reference
        is a path: roll ``corridor_guidance_velocity`` forward from the current
        relative position, one guidance step per horizon step. The desired
        attitude is identity, so a reference pose ``(I, p)`` has exponential
        coordinates ``(0, p)`` and a target-frame velocity ``v`` is already the
        body-frame reference there -- the columns are ``[0, p, 0, v]``.
        """

        n = self.config.horizon_steps
        if self.config.reference_source == "external_local":
            if target_state is None:
                raise ValueError("external_local reference requires target_state")
            waypoint = np.asarray(external_reference, dtype=np.float64)
            if waypoint.shape != (3,) or not np.all(np.isfinite(waypoint)):
                raise ValueError(
                    "external_local reference must be a finite inertial-oriented "
                    "3D waypoint"
                )
            if (
                self._held_external_reference is None
                or self._control_step % self.config.external_reference_hold_steps == 0
            ):
                if self._held_external_reference is None:
                    self._held_external_velocity.fill(0.0)
                else:
                    hold_time_s = (
                        self.config.external_reference_hold_steps
                        * self.config.dt_s
                    )
                    self._held_external_velocity = (
                        waypoint - self._held_external_reference
                    ) / hold_time_s
                self._held_external_reference = waypoint.copy()
            waypoint = self._held_external_reference
            waypoint_velocity = self._held_external_velocity
            # The state is (se3_log(T_rel), twist), so the reference has to be
            # written in those coordinates too: rows 3:6 are the *exponential*
            # translation rho = J_l(phi)^-1 p, not the position, and rows 9:12
            # are the relative velocity in the *chaser body* frame, not the
            # target-frame position rate. Assigning p into rho and differencing
            # it left the velocity reference expressed in the target frame; the
            # chaser's roll at reset is uniform on [-pi, pi], so that was an
            # arbitrarily rotated velocity command -- up to 2 * omega * r of
            # pure error, which at 16 m saturated the first step and pushed the
            # optimiser off any waypoint it was given. This follows the same
            # convention as _endpoint_plan.
            positions = np.zeros((3, n + 1), dtype=np.float64)
            for index in range(n + 1):
                offset = index * self.config.dt_s
                if self.config.external_reference_frame == "target":
                    # The waypoint is already a target-body-frame point, so it
                    # is constant in the frame the state is written in. Under
                    # the "inertial" contract the same body-fixed goal has to
                    # be re-issued as a rotating inertial point, and the map
                    # below turns it into a circular reference the optimiser
                    # then tracks with a standing lag -- 1.3 m at the 3 m
                    # desired pose, against a 0.25 m completion tolerance.
                    positions[:, index] = waypoint + offset * waypoint_velocity
                    continue
                target_rotation = target_state.rotation @ so3_exp(
                    offset * target_state.omega
                )
                inertial_position = waypoint + offset * waypoint_velocity
                positions[:, index] = target_rotation.T @ inertial_position
            position_rates = np.zeros((3, n + 1), dtype=np.float64)
            position_rates[:, :-1] = np.diff(positions, axis=1) / self.config.dt_s
            position_rates[:, -1] = position_rates[:, -2]
            rotations = self._precapture_reference_rotations(
                state, positions, terminal_latched=terminal_latched
            )
            reference = np.zeros((12, n + 1), dtype=np.float64)
            for index in range(n + 1):
                reference[:6, index] = se3_log(
                    make_transform(rotations[index], positions[:, index])
                )
                reference[9:12, index] = (
                    rotations[index].T @ position_rates[:, index]
                )
            # Rows 6:9 carry whatever relative rate that attitude sequence
            # implies -- zero under "frozen", the sightline sweep otherwise.
            reference[6:9, :] = self._reference_angular_rates(rotations)
            return reference
        if self.config.reference_source == "fixed":
            reference = np.tile(
                self.config.reference_state.reshape(12, 1), (1, n + 1)
            )
            return self._apply_precapture_attitude_reference(
                reference, state, terminal_latched=terminal_latched
            )
        sample_offsets = np.arange(n + 1, dtype=np.float64) * self.config.dt_s
        if self.config.reference_source == "receding_plan":
            return self._endpoint_plan(state, sample_offsets)
        if self.config.reference_source == "episode_plan":
            if self._episode_plan_initial_state is None:
                self._episode_plan_initial_state = state.copy()
                self._episode_plan_start_time_s = float(time_seconds)
            assert self._episode_plan_start_time_s is not None
            elapsed = max(0.0, float(time_seconds) - self._episode_plan_start_time_s)
            return self._endpoint_plan(
                self._episode_plan_initial_state,
                elapsed + sample_offsets,
            )
        if self.config.reference_source == "online_endpoint":
            initial_transform = se3_exp(state[:6])
            desired_transform = se3_exp(self.config.reference_state[:6])
            initial_rotation = initial_transform[:3, :3]
            initial_position = initial_transform[:3, 3]
            desired_position = desired_transform[:3, 3]
            position_error = initial_position - desired_position
            distance = float(np.linalg.norm(position_error))
            position_rate = min(
                1.0 / self.config.planning_min_duration_s,
                self.config.planning_speed_m_s / max(distance, 1.0e-12),
            )
            rotation_to_goal = so3_log(
                initial_rotation.T @ desired_transform[:3, :3]
            )
            angle = float(np.linalg.norm(rotation_to_goal))
            rotation_rate = min(
                1.0 / self.config.planning_min_duration_s,
                self.config.planning_angular_speed_rad_s / max(angle, 1.0e-12),
            )
            reference = np.zeros((12, n + 1), dtype=np.float64)
            for index, sample_time in enumerate(sample_offsets):
                position_decay = float(np.exp(-position_rate * sample_time))
                rotation_decay = float(np.exp(-rotation_rate * sample_time))
                position = desired_position + position_decay * position_error
                target_position_rate = -position_rate * position_decay * position_error
                rotation_vector = (1.0 - rotation_decay) * rotation_to_goal
                rotation = initial_rotation @ so3_exp(rotation_vector)
                rotation_vector_rate = (
                    rotation_rate * rotation_decay * rotation_to_goal
                )
                reference[:6, index] = se3_log(
                    make_transform(rotation, position)
                )
                reference[6:9, index] = (
                    left_jacobian_so3(-rotation_vector) @ rotation_vector_rate
                )
                reference[9:12, index] = rotation.T @ target_position_rate
            return reference
        assert self.config.task is not None
        position = se3_exp(state[:6])[:3, 3]
        reference = np.zeros((12, n + 1), dtype=np.float64)
        for index in range(n + 1):
            velocity = corridor_guidance_velocity(
                position,
                self.config.task,
                total_speed_fraction=self.config.corridor_speed_fraction,
            )
            reference[3:6, index] = position
            reference[9:12, index] = velocity
            position = position + velocity * self.config.dt_s
        return reference

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
        external_reference: ArrayLike | None = None,
        terminal_latched: bool | None = None,
    ) -> tuple[FloatArray, MPCStepDiagnostics]:
        state = np.asarray(relative_vector, dtype=np.float64)
        if state.shape != (12,) or not np.all(np.isfinite(state)):
            raise ValueError("MPC state must be finite and shape=(12,)")
        # Depends only on the current state, so build it once per control step.
        if self.config.precapture_task is not None and terminal_latched is None:
            raise ValueError("precapture MPC requires explicit terminal_latched")
        reference_trajectory = self._reference_trajectory(
            state,
            time_seconds,
            target_state=target_state,
            external_reference=external_reference,
            terminal_latched=terminal_latched,
        )
        self._reference.value = reference_trajectory
        use_exact = self.config.linearization_source == "exact"
        exact_refresh_time = 0.0
        if use_exact and (
            self._exact_linearization is None
            or self._control_step
            % self.config.exact_linearization_refresh_steps
            == 0
        ):
            started_exact = perf_counter()
            self._refresh_exact_linearization(state, target_state, time_seconds)
            exact_refresh_time = perf_counter() - started_exact
        elif not use_exact and self._control_step % self.config.drift_refresh_steps == 0:
            self._refresh_drift(state, target_state, time_seconds)
        controls = np.vstack((self._nominal_controls[1:], np.zeros((1, 6))))
        status = "not_solved"
        total_solve_time = 0.0
        constraint_linearization_time = 0.0
        model_linearization_time = 0.0
        rollout_time = 0.0
        solver_iterations = 0
        objective = float("nan")
        completed_outer = 0
        try:
            for outer in range(self.config.outer_iterations):
                started_rollout = perf_counter()
                nominal_states = self._rollout(state, controls)
                rollout_time += perf_counter() - started_rollout
                cached_linearization: tuple[FloatArray, FloatArray, FloatArray] | None = None
                for index in range(self.config.horizon_steps):
                    if use_exact:
                        assert self._exact_linearization is not None
                        cached_linearization = self._exact_linearization
                    elif index % self.config.linearization_stride == 0:
                        started_linearization = perf_counter()
                        if self.config.linearization_source == "analytic_local":
                            cached_linearization = (
                                self.local_model.analytic_kinematic_linearization(
                                    nominal_states[index], controls[index]
                                )
                            )
                            a_value, b_value, c_value = cached_linearization
                            cached_linearization = (
                                a_value,
                                b_value,
                                c_value + self._drift,
                            )
                        else:
                            cached_linearization = central_difference_linearization(
                                self._predict,
                                nominal_states[index],
                                controls[index],
                                state_scales=self.config.state_scales,
                                input_scales=self.config.input_scales,
                                relative_step=self.config.difference_relative_step,
                            )
                        model_linearization_time += (
                            perf_counter() - started_linearization
                        )
                    assert cached_linearization is not None
                    self._a[index].value, self._b[index].value, self._c[index].value = (
                        cached_linearization
                    )
                    if (
                        self.config.task is not None
                        or self.config.precapture_task is not None
                    ):
                        started_constraints = perf_counter()
                        if self.config.precapture_task is not None:
                            assert terminal_latched is not None
                            jacobian, offset = (
                                linearize_precapture_constraint_margins(
                                    nominal_states[index + 1],
                                    self.config.precapture_task,
                                    target_angular_velocity_rad_s=target_state.omega,
                                    terminal_latched=terminal_latched,
                                    corridor_facets=self.config.corridor_facets,
                                )
                            )
                        else:
                            assert self.config.task is not None
                            jacobian, offset = linearize_constraint_margins(
                                nominal_states[index + 1],
                                self.config.task,
                                corridor_facets=self.config.corridor_facets,
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
        if self._slack is not None and self._slack.value is not None:
            slack_values = np.asarray(self._slack.value, dtype=np.float64)
            maximum_slack = float(np.max(slack_values))
            total_slack = float(np.sum(slack_values))
        else:
            maximum_slack = total_slack = 0.0
        if self.config.runtime_diagnostics:
            started_rollout = perf_counter()
            predicted_states = self._rollout(state, self._nominal_controls)
            rollout_time += perf_counter() - started_rollout
            predicted_minimum_margin = (
                min(
                    float(
                        np.min(
                            normalized_precapture_truth_margins(
                                item,
                                self.config.precapture_task,
                                target_angular_velocity_rad_s=target_state.omega,
                                terminal_latched=bool(terminal_latched),
                            )
                            if self.config.precapture_task is not None
                            else normalized_truth_margins(item, self.config.task)
                        )
                    )
                    for item in predicted_states[1:]
                )
                if (
                    self.config.task is not None
                    or self.config.precapture_task is not None
                )
                else float("nan")
            )
            predicted_first_step_margins = (
                tuple(
                    float(value)
                    for value in (
                        normalized_precapture_truth_margins(
                            predicted_states[1],
                            self.config.precapture_task,
                            target_angular_velocity_rad_s=target_state.omega,
                            terminal_latched=bool(terminal_latched),
                        )
                        if self.config.precapture_task is not None
                        else normalized_truth_margins(
                            predicted_states[1], self.config.task
                        )
                    )
                )
                if (
                    self.config.task is not None
                    or self.config.precapture_task is not None
                )
                else ()
            )
            # Report the cost of the problem that was actually solved.
            learned_terminal = self.config.terminal_cost_source == "learned_convex"
            predicted_cost = nonlinear_rollout_cost(
                predicted_states,
                self._nominal_controls,
                state_scales=self.config.state_scales,
                input_scales=self.config.input_scales,
                state_weight=self.config.state_weight,
                input_weight=self.config.input_weight,
                terminal_weight=(
                    0.0 if learned_terminal else self.config.terminal_weight
                ),
                reference_state=reference_trajectory.T,
            )
            if learned_terminal:
                assert self.config.terminal_value is not None
                predicted_cost += self.config.terminal_value.value(
                    predicted_states[-1]
                )
        else:
            predicted_minimum_margin = float("nan")
            predicted_first_step_margins = ()
            predicted_cost = objective
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
            model_linearization_time_s=model_linearization_time,
            rollout_time_s=rollout_time,
            exact_refresh_time_s=exact_refresh_time,
            maximum_slack=maximum_slack,
            total_slack=total_slack,
            predicted_minimum_margin=predicted_minimum_margin,
            predicted_first_step_margins=predicted_first_step_margins,
        )
