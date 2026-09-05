"""Offline coast-then-match feasibility certificate for precapture planning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from dynamics.lie import make_transform, se3_log, so3_exp, so3_log
from env.phase2_env import precapture_planning_environment_config
from env.se3_rendezvous_env import SE3RendezvousEnv
from eval.metrics import summarize


# --- plan geometry -----------------------------------------------------------
# The manually fixed open-loop plan searches a deterministic nominal target
# forecast.  The search chooses the cheapest direction change; it is not an
# entry-timing policy and it is not an optimal-control oracle.
COAST_MIN_TIME_S = 120.0
COAST_MAX_TIME_S = 220.0
MATCH_TIME_S = 40.0
OUTER_STAGING_RADIUS_M = 7.5
OUTER_DESCENT_TIME_S = 80.0
FINAL_DESCENT_TIME_S = 80.0
FINAL_RADIUS_M = 3.0
ATTITUDE_BLEND_OUTER_M = 8.0
ATTITUDE_BLEND_INNER_M = 4.0

# --- tracking law ------------------------------------------------------------
# The feedforward carries the rotating-frame bias (centrifugal, Coriolis and
# Euler terms), so the feedback only has to remove the sampled initial error.
# Guaranteed single-axis authority is 5 N / 106 kg = 0.0472 m/s^2; the gains are
# sized so a 0.10 m/s initial velocity error stays inside that box.
POSITION_GAIN_PER_S2 = 0.015
VELOCITY_GAIN_PER_S = 0.25
FEEDBACK_ACCEL_LIMIT_M_S2 = 0.040
ATTITUDE_GAIN = 2.0
ANGULAR_RATE_GAIN = 8.0
DERIVATIVE_STEP_S = 0.2


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _slerp_direction(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    cosine = float(np.clip(start @ end, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= 1.0e-10:
        return start.copy()
    if np.pi - angle <= 1.0e-6:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(reference @ start)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(start, reference)
        tangent /= np.linalg.norm(tangent)
    else:
        tangent = (end - cosine * start) / np.sin(angle)
    result = np.cos(fraction * angle) * start + np.sin(fraction * angle) * tangent
    return result / np.linalg.norm(result)


def _los_rotation(relative_rotation: np.ndarray, position: np.ndarray, port: np.ndarray) -> np.ndarray:
    """Preserve current roll while aligning body +x with the port LOS."""

    line_of_sight = port - position
    line_of_sight /= np.linalg.norm(line_of_sight)
    current_y = relative_rotation[:, 1]
    desired_y = current_y - float(current_y @ line_of_sight) * line_of_sight
    if np.linalg.norm(desired_y) <= 1.0e-9:
        current_z = relative_rotation[:, 2]
        desired_y = np.cross(current_z, line_of_sight)
    desired_y /= np.linalg.norm(desired_y)
    desired_z = np.cross(line_of_sight, desired_y)
    desired_z /= np.linalg.norm(desired_z)
    desired_y = np.cross(desired_z, line_of_sight)
    return np.column_stack((line_of_sight, desired_y, desired_z))


class CoastThenMatchPlan:
    """One manually parameterised open-loop feasibility path.

    The target forecast is deterministic because the task has zero model
    mismatch.  The plan uses that nominal forecast to minimise the direction
    change into the match segment.  It therefore proves only that a cheap safe
    path exists; it does not demonstrate online entry-timing decisions and must
    not be described as an optimal oracle.

    The mission is a single twice-differentiable curve in *inertial* relative
    coordinates, written as a radius profile times a unit direction:

    * coast -- the direction slews slowly between two inertially fixed rays
      while the radius slides in, so the chaser pays almost nothing to hold it;
    * match -- the direction's inertial rate ramps from zero to the target's own
      ``omega``, which is where co-rotation is bought, once, at one radius;
    * descent -- the direction is the body-fixed approach axis and the radius
      slides to the desired pose, slow enough for the closing-speed limit.

    The match is integrated *backwards* from the moment the body axis is where
    the plan needs it. That is what makes it affordable: interpolating the
    direction forwards instead (a body-frame slerp between a frozen ray and the
    axis) has to sweep whatever angle the target has turned through during the
    window -- 1.6 rad over 40 s here -- and the reference then swings at nearly
    the tumble rate. Measured, that reference demanded 0.0545 m/s^2 against the
    0.0472 m/s^2 a single axis can guarantee, so no controller could have held
    it. Ramping the *rate* and solving for the entry direction instead keeps the
    same endpoint and costs 0.0116 m/s^2.
    """

    def __init__(
        self,
        env: SE3RendezvousEnv,
        *,
        coast_min_time_s: float = COAST_MIN_TIME_S,
        coast_max_time_s: float = COAST_MAX_TIME_S,
        outer_descent_time_s: float = OUTER_DESCENT_TIME_S,
    ) -> None:
        assert env.relative is not None and env.target_state is not None
        trajectory = env._target_trajectory
        if trajectory is None:
            raise RuntimeError("feasibility plan requires the cached nominal forecast")
        if min(coast_min_time_s, coast_max_time_s, outer_descent_time_s) <= 0.0:
            raise ValueError("plan timing constants must be positive")
        if coast_max_time_s <= coast_min_time_s:
            raise ValueError("coast_max_time_s must exceed coast_min_time_s")
        self._trajectory = trajectory
        self._dt_s = env.config.dt_s
        self.outer_descent_time_s = float(outer_descent_time_s)
        task = env.config.precapture_task
        self._axis = task.approach_axis
        initial = env.target_state.rotation @ env.relative.position
        self.initial_radius_m = float(np.linalg.norm(initial))
        self.initial_direction = initial / self.initial_radius_m

        match_steps = int(round(MATCH_TIME_S / self._dt_s))
        earliest_match_end_s = self.outer_descent_time_s + MATCH_TIME_S
        first = int(
            np.ceil(max(float(coast_min_time_s), earliest_match_end_s) / self._dt_s)
        )
        last = min(
            len(trajectory) - 1,
            int(np.floor(float(coast_max_time_s) / self._dt_s)),
        )
        stride = int(round(1.0 / self._dt_s))
        candidates = list(range(first, last + 1, stride))
        if not candidates:
            raise ValueError("coast search window contains no feasible match endpoint")
        entries = [self._match_entry_direction(index, match_steps) for index in candidates]
        scores = [float(self.initial_direction @ entry) for entry in entries]
        best = int(np.argmax(scores))
        self.coast_steps = candidates[best]
        self.coast_time_s = self.coast_steps * self._dt_s
        self.match_steps = match_steps
        self.match_start_steps = self.coast_steps - match_steps
        self.match_start_time_s = self.match_start_steps * self._dt_s
        self._match_directions = self._match_direction_track(self.coast_steps, match_steps)
        self.match_entry_direction = self._match_directions[0]
        self.match_direction = self._match_directions[-1]
        self.match_alignment_angle_rad = float(
            np.arccos(np.clip(scores[best], -1.0, 1.0))
        )

    # -- match construction ---------------------------------------------------
    def _blend(self, offset_steps: int, match_steps: int) -> float:
        return _smoothstep(offset_steps / match_steps)

    def _match_direction_track(self, end_index: int, match_steps: int) -> np.ndarray:
        """Unit directions over the match window, integrated backwards.

        ``e(t_end)`` is the body axis; going backwards the inertial rate is
        ``blend(t) * omega x e``, so the ramp starts inertially frozen and ends
        exactly co-rotating, with the endpoint fixed.
        """

        directions = np.zeros((match_steps + 1, 3), dtype=np.float64)
        state = self._trajectory[end_index]
        direction = state.rotation @ self._axis
        directions[match_steps] = direction
        for offset in range(match_steps, 0, -1):
            index = end_index - offset
            sample = self._trajectory[max(index, 0)]
            omega_inertial = sample.rotation @ sample.omega
            rate = self._blend(offset, match_steps) * omega_inertial
            direction = so3_exp(-rate * self._dt_s) @ direction
            direction /= np.linalg.norm(direction)
            directions[offset - 1] = direction
        return directions

    def _match_entry_direction(self, end_index: int, match_steps: int) -> np.ndarray:
        return self._match_direction_track(end_index, match_steps)[0]

    # -- trajectory -----------------------------------------------------------
    def _target_rotation(self, time_s: float) -> np.ndarray:
        index = int(round(time_s / self._dt_s))
        index = int(np.clip(index, 0, len(self._trajectory) - 1))
        return self._trajectory[index].rotation

    def phase(self, time_s: float) -> str:
        if time_s <= self.match_start_time_s:
            return "coast"
        if time_s <= self.coast_time_s:
            return "match"
        return "cross"

    def _radius(self, time_s: float) -> float:
        if time_s <= self.outer_descent_time_s:
            fraction = _smoothstep(time_s / self.outer_descent_time_s)
            return self.initial_radius_m + fraction * (
                OUTER_STAGING_RADIUS_M - self.initial_radius_m
            )
        if time_s <= self.coast_time_s:
            return OUTER_STAGING_RADIUS_M
        descent = _smoothstep(
            (time_s - self.coast_time_s) / FINAL_DESCENT_TIME_S
        )
        return OUTER_STAGING_RADIUS_M + descent * (
            FINAL_RADIUS_M - OUTER_STAGING_RADIUS_M
        )

    def _inertial_direction(self, time_s: float) -> np.ndarray:
        if time_s <= self.match_start_time_s:
            fraction = _smoothstep(time_s / max(self.match_start_time_s, self._dt_s))
            return _slerp_direction(
                self.initial_direction, self.match_entry_direction, fraction
            )
        if time_s >= self.coast_time_s:
            return self._target_rotation(time_s) @ self._axis
        offset = int(round((time_s - self.match_start_time_s) / self._dt_s))
        offset = int(np.clip(offset, 0, self.match_steps))
        return self._match_directions[offset]

    def body_position(self, time_s: float) -> np.ndarray:
        """Desired relative position in target body coordinates."""

        time_s = float(time_s)
        radius = self._radius(max(time_s, 0.0))
        direction = self._inertial_direction(max(time_s, 0.0))
        return radius * (self._target_rotation(max(time_s, 0.0)).T @ direction)

    def body_reference(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(p_d, dp_d/dt, d2p_d/dt2)`` in target body coordinates."""

        step = DERIVATIVE_STEP_S
        centre = self.body_position(time_s)
        if time_s < step:
            # The cached target trajectory does not extend before t = 0, so a
            # central difference there silently differentiates a constant and
            # reports an acceleration two orders too large.
            first = self.body_position(time_s + step)
            second = self.body_position(time_s + 2.0 * step)
            third = self.body_position(time_s + 3.0 * step)
            velocity = (-3.0 * centre + 4.0 * first - second) / (2.0 * step)
            acceleration = (
                2.0 * centre - 5.0 * first + 4.0 * second - third
            ) / (step * step)
            return centre, velocity, acceleration
        ahead = self.body_position(time_s + step)
        behind = self.body_position(time_s - step)
        velocity = (ahead - behind) / (2.0 * step)
        acceleration = (ahead - 2.0 * centre + behind) / (step * step)
        return centre, velocity, acceleration

    def target_angular_acceleration(self, time_s: float) -> np.ndarray:
        index = int(round(time_s / self._dt_s))
        last = len(self._trajectory) - 1
        ahead = int(np.clip(index + 1, 0, last))
        behind = int(np.clip(index - 1, 0, last))
        span = (ahead - behind) * self._dt_s
        if span <= 0.0:
            return np.zeros(3)
        return (self._trajectory[ahead].omega - self._trajectory[behind].omega) / span


def _desired_relative_rotation(
    relative_rotation: np.ndarray,
    position: np.ndarray,
    port: np.ndarray,
    range_m: float,
) -> np.ndarray:
    """LOS pointing far out, blended onto the docking attitude inside 8 m.

    Completion is scored against ``R_rel = I``, so the free roll the LOS law
    keeps has to be spent before the desired pose. Blending along the geodesic
    from ``I`` avoids the discontinuity a hard switch at the latch would inject
    into the attitude servo.
    """

    line_of_sight = _los_rotation(relative_rotation, position, port)
    blend = _smoothstep(
        (ATTITUDE_BLEND_OUTER_M - float(range_m))
        / (ATTITUDE_BLEND_OUTER_M - ATTITUDE_BLEND_INNER_M)
    )
    if blend <= 0.0:
        return line_of_sight
    if blend >= 1.0:
        return np.eye(3)
    return so3_exp((1.0 - blend) * so3_log(line_of_sight, project=True))


def _normalized_force(body_force: np.ndarray, max_force_per_axis_n: float) -> np.ndarray:
    """Scale into the actuator box without rotating the commanded direction.

    Per-axis clipping is what turned the previous oracle's saturated approach
    into an outward spiral: once the request exceeded the box the applied force
    no longer pointed where the law asked. Dividing by the largest normalised
    component keeps the direction exactly and still reaches the box corner.
    """

    normalized = body_force / max_force_per_axis_n
    peak = float(np.max(np.abs(normalized)))
    if peak > 1.0:
        normalized = normalized / peak
    return normalized


def _allocate_force(
    feedforward_n: np.ndarray,
    feedback_n: np.ndarray,
    max_force_per_axis_n: float,
) -> np.ndarray:
    """Spend the actuator box on the feedforward first, then on the feedback.

    The feedforward is what holds the chaser on a rotating orbit at all; giving
    up part of it to serve a transient tracking error is how a saturated step
    turns into a divergence. Where the sum does not fit, the feedback is scaled
    -- never rotated -- until it does.
    """

    if float(np.max(np.abs(feedforward_n))) >= max_force_per_axis_n:
        return _normalized_force(feedforward_n + feedback_n, max_force_per_axis_n)
    scale = 1.0
    for axis in range(3):
        demand = float(feedback_n[axis])
        if abs(demand) <= 1.0e-12:
            continue
        bound = max_force_per_axis_n if demand > 0.0 else -max_force_per_axis_n
        scale = min(scale, (bound - float(feedforward_n[axis])) / demand)
    scale = float(np.clip(scale, 0.0, 1.0))
    return (feedforward_n + scale * feedback_n) / max_force_per_axis_n


def _oracle_action(
    env: SE3RendezvousEnv, plan: CoastThenMatchPlan
) -> tuple[np.ndarray, str, float]:
    """Feedback-linearising tracker for ``plan`` in the rotating target frame.

    With ``p`` the relative position in target body coordinates and ``omega``
    the target body rate, the inertial acceleration the thrusters must produce
    is ``p_ddot + 2 omega x p_dot + omega_dot x p + omega x (omega x p)``. The
    three fictitious terms are the whole cost of co-rotation, so feeding them
    forward leaves the feedback loop with nothing but the sampled initial error
    -- and keeps the loop stable when it does saturate.
    """

    assert env.relative is not None and env.target_state is not None
    assert env.chaser_state is not None and env.chaser_parameters is not None
    task = env.config.precapture_task
    relative = env.relative
    position = relative.position
    velocity = relative.rotation @ relative.velocity
    range_m = float(np.linalg.norm(position))

    reference, reference_rate, reference_accel = plan.body_reference(env.time_seconds)
    feedback = POSITION_GAIN_PER_S2 * (reference - position) + VELOCITY_GAIN_PER_S * (
        reference_rate - velocity
    )
    magnitude = float(np.linalg.norm(feedback))
    if magnitude > FEEDBACK_ACCEL_LIMIT_M_S2:
        feedback *= FEEDBACK_ACCEL_LIMIT_M_S2 / magnitude
    omega = env.target_state.omega
    omega_rate = plan.target_angular_acceleration(env.time_seconds)
    feedforward = (
        reference_accel
        + 2.0 * np.cross(omega, velocity)
        + np.cross(omega_rate, position)
        + np.cross(omega, np.cross(omega, position))
    )
    mass = env.chaser_parameters.mass
    body_feedforward = relative.rotation.T @ (mass * feedforward)
    body_feedback = relative.rotation.T @ (mass * feedback)

    desired_relative_rotation = _desired_relative_rotation(
        relative.rotation, position, task.port_position, range_m
    )
    rotation_error = se3_log(
        make_transform(desired_relative_rotation.T @ relative.rotation, np.zeros(3))
    )[:3]
    body_torque = -ATTITUDE_GAIN * rotation_error - ANGULAR_RATE_GAIN * relative.omega
    action = np.concatenate(
        (
            np.clip(body_torque / env.config.max_torque_per_axis_nm, -1.0, 1.0),
            _allocate_force(
                body_feedforward, body_feedback, env.config.max_force_per_axis_n
            ),
        )
    )
    phase = "terminal" if env._terminal_region_entered else plan.phase(env.time_seconds)
    force_norm = float(
        np.linalg.norm(action[3:] * env.config.max_force_per_axis_n)
    )
    return action, phase, force_norm


def rollout(
    seed: int,
    *,
    coast_min_time_s: float = COAST_MIN_TIME_S,
    coast_max_time_s: float = COAST_MAX_TIME_S,
    outer_descent_time_s: float = OUTER_DESCENT_TIME_S,
) -> dict[str, Any]:
    env = SE3RendezvousEnv(precapture_planning_environment_config())
    try:
        _, info = env.reset(seed=seed)
        plan = CoastThenMatchPlan(
            env,
            coast_min_time_s=coast_min_time_s,
            coast_max_time_s=coast_max_time_s,
            outer_descent_time_s=outer_descent_time_s,
        )
        force_impulse = 0.0
        torque_impulse = 0.0
        force_norms: list[float] = []
        controller_times: list[float] = []
        saturation_steps = 0
        minimum_fov_margin = float(info["fov_margin_rad"])
        minimum_normalized_margin = float("inf")
        phase_impulse = {name: 0.0 for name in ("coast", "match", "cross", "terminal")}
        phase_time = {name: 0.0 for name in phase_impulse}
        trace: list[dict[str, Any]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            started = perf_counter()
            action, phase, force_norm = _oracle_action(env, plan)
            controller_times.append(perf_counter() - started)
            force = action[3:] * env.config.max_force_per_axis_n
            torque = action[:3] * env.config.max_torque_per_axis_nm
            force_impulse += float(np.linalg.norm(force)) * env.config.dt_s
            torque_impulse += float(np.linalg.norm(torque)) * env.config.dt_s
            force_norms.append(force_norm)
            saturation_steps += int(np.any(np.abs(action[3:]) >= 1.0 - 1.0e-12))
            phase_impulse[phase] += force_norm * env.config.dt_s
            phase_time[phase] += env.config.dt_s
            if env.step_count % 10 == 0:
                trace.append(
                    {
                        "time_s": env.time_seconds,
                        "phase": phase,
                        "target_position_m": env.relative.position.tolist(),
                        "range_m": float(info["target_center_distance_m"]),
                        "fov_margin_rad": float(info["fov_margin_rad"]),
                        "corridor_margin_m": min(
                            float(info["corridor_axial_margin_m"]),
                            float(info["corridor_lateral_margin_m"]),
                        ),
                        "target_frame_speed_m_s": float(
                            info["target_frame_speed_m_s"]
                        ),
                        "outer_radial_margin_m_s": float(
                            info["outer_radial_margin_m_s"]
                        ),
                        "terminal_region_active": bool(info["terminal_region_active"]),
                        "force_norm_n": force_norm,
                    }
                )
            _, _, terminated, truncated, info = env.step(action)
            minimum_fov_margin = min(minimum_fov_margin, float(info["fov_margin_rad"]))
            active_margins = [
                float(info["keepout_margin_m"]) / env.config.precapture_task.keepout_radius_m,
                float(info["fov_margin_rad"]) / env.config.precapture_task.fov_half_angle_rad,
            ]
            if info["terminal_region_active"]:
                active_margins.extend(
                    (
                        float(info["corridor_axial_margin_m"]) / 3.0,
                        float(info["corridor_lateral_margin_m"]) / 3.0,
                        float(info["terminal_total_speed_margin_m_s"])
                        / env.config.precapture_task.terminal_total_speed_limit_m_s,
                        float(info["closing_speed_margin_m_s"])
                        / env.config.precapture_task.closing_speed_max_m_s,
                    )
                )
            else:
                active_margins.extend(
                    (
                        float(info["outer_inertial_speed_margin_m_s"])
                        / env.config.precapture_task.outer_inertial_speed_limit_m_s,
                        float(info["outer_radial_margin_m_s"])
                        / env.config.precapture_task.outer_inertial_speed_limit_m_s,
                    )
                )
            minimum_normalized_margin = min(
                minimum_normalized_margin, min(active_margins)
            )
        zero_violation = all(
            int(info[f"{name}_violation_steps"]) == 0
            for name in (
                "keepout",
                "fov",
                "outer_speed",
                "outer_radial",
                "corridor",
                "total_speed",
                "closing_speed",
            )
        )
        mass = env.chaser_parameters.mass
        return {
            "seed": seed,
            "completed": bool(info["completed"]),
            "zero_truth_violation_completed": bool(info["completed"] and zero_violation),
            "zero_truth_violation": zero_violation,
            "time_s": float(info["time_seconds"]),
            "termination": {
                key: bool(info.get(key, False))
                for key in (
                    "keepout_failure",
                    "fov_failure",
                    "outer_speed_failure",
                    "outer_radial_failure",
                    "terminal_constraint_failure",
                    "distance_failure",
                    "time_failure",
                )
            },
            "equivalent_delta_v_m_s": force_impulse / mass,
            "force_impulse_n_s": force_impulse,
            "torque_impulse_nm_s": torque_impulse,
            "force_norm_n": summarize(force_norms),
            "force_channel_saturation_step_fraction": saturation_steps / len(force_norms),
            "minimum_fov_margin_rad": minimum_fov_margin,
            "minimum_normalized_margin": minimum_normalized_margin,
            "controller_time_s": summarize(controller_times),
            "phase_force_impulse_n_s": phase_impulse,
            "phase_delta_v_m_s": {
                name: value / mass for name, value in phase_impulse.items()
            },
            "phase_time_s": phase_time,
            "plan": {
                "coast_time_s": plan.coast_time_s,
                "outer_descent_time_s": plan.outer_descent_time_s,
                "initial_radius_m": plan.initial_radius_m,
                "match_alignment_angle_rad": plan.match_alignment_angle_rad,
                "selection_objective": "minimise_initial_to_match_direction_change",
            },
            "trace": trace,
        }
    finally:
        env.close()


def evaluate(
    episodes: int,
    seed: int,
    *,
    coast_min_time_s: float = COAST_MIN_TIME_S,
    coast_max_time_s: float = COAST_MAX_TIME_S,
    outer_descent_time_s: float = OUTER_DESCENT_TIME_S,
) -> dict[str, Any]:
    records = [
        rollout(
            seed + episode,
            coast_min_time_s=coast_min_time_s,
            coast_max_time_s=coast_max_time_s,
            outer_descent_time_s=outer_descent_time_s,
        )
        for episode in range(episodes)
    ]
    accepted = [row for row in records if row["zero_truth_violation_completed"]]
    checks = {
        "at_least_five_episodes_evaluated": episodes >= 5,
        "all_evaluated_episodes_zero_violation_completed": len(accepted) == episodes,
        "zero_force_channel_saturation_steps": all(
            row["force_channel_saturation_step_fraction"] == 0.0 for row in records
        ),
    }
    return {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "precapture_coast_then_match_open_loop_feasibility",
        "interpretation": {
            "plan_type": "manually_parameterised_open_loop_plan",
            "forecast": "deterministic_nominal_model_from_t0",
            "selection_objective": "minimise_direction_change_not_entry_timing",
            "not_claimed": ["optimal_control", "online_timing_policy", "future_truth_access"],
        },
        "episodes": episodes,
        "base_seed": seed,
        "plan_timing": {
            "coast_min_time_s": float(coast_min_time_s),
            "coast_max_time_s": float(coast_max_time_s),
            "outer_descent_time_s": float(outer_descent_time_s),
            "match_time_s": MATCH_TIME_S,
            "final_descent_time_s": FINAL_DESCENT_TIME_S,
        },
        "environment": asdict(precapture_planning_environment_config()),
        "records": records,
        "aggregate": {
            "completion_rate": mean(row["completed"] for row in records),
            "zero_violation_completion_rate": mean(
                row["zero_truth_violation_completed"] for row in records
            ),
            "equivalent_delta_v_m_s": summarize(
                row["equivalent_delta_v_m_s"] for row in records
            ),
            "force_norm_mean_n": summarize(
                row["force_norm_n"]["mean"] for row in records
            ),
            "minimum_fov_margin_rad": min(
                row["minimum_fov_margin_rad"] for row in records
            ),
        },
        "acceptance": {"checks": checks, "passed": all(checks.values())},
    }


def timing_scan(
    episodes: int,
    seed: int,
    *,
    coast_min_times_s: list[float],
    outer_descent_times_s: list[float],
    coast_max_time_s: float = COAST_MAX_TIME_S,
) -> dict[str, Any]:
    """Scan fixed plan timings; results are reference data, not a denominator."""

    rows: list[dict[str, Any]] = []
    for coast_min_time_s in coast_min_times_s:
        for outer_descent_time_s in outer_descent_times_s:
            result = evaluate(
                episodes,
                seed,
                coast_min_time_s=coast_min_time_s,
                coast_max_time_s=coast_max_time_s,
                outer_descent_time_s=outer_descent_time_s,
            )
            records = result["records"]
            all_clean = all(
                bool(record["zero_truth_violation_completed"]) for record in records
            )
            worst_margin = min(
                float(record["minimum_normalized_margin"]) for record in records
            )
            completion_times = [
                float(record["time_s"]) for record in records if record["completed"]
            ]
            rows.append(
                {
                    "coast_min_time_s": float(coast_min_time_s),
                    "outer_descent_time_s": float(outer_descent_time_s),
                    "all_zero_violation_completed": all_clean,
                    "worst_normalized_margin": worst_margin,
                    "mean_completion_time_s": (
                        mean(completion_times) if len(completion_times) == episodes else None
                    ),
                    "max_completion_time_s": (
                        max(completion_times) if len(completion_times) == episodes else None
                    ),
                    "zero_force_channel_saturation_steps": all(
                        record["force_channel_saturation_step_fraction"] == 0.0
                        for record in records
                    ),
                    "per_seed": [
                        {
                            key: record[key]
                            for key in (
                                "seed",
                                "completed",
                                "zero_truth_violation_completed",
                                "time_s",
                                "equivalent_delta_v_m_s",
                                "minimum_normalized_margin",
                                "force_channel_saturation_step_fraction",
                            )
                        }
                        for record in records
                    ],
                }
            )
    qualifying = [
        row
        for row in rows
        if row["all_zero_violation_completed"]
        and row["worst_normalized_margin"] >= 0.14
    ]
    selected = min(
        qualifying,
        key=lambda row: (row["mean_completion_time_s"], row["max_completion_time_s"]),
        default=None,
    )
    return {
        "schema_version": 1,
        "experiment": "precapture_open_loop_timing_scan",
        "episodes_per_point": episodes,
        "base_seed": seed,
        "selection_rule": (
            "all episodes zero-violation complete and worst normalized margin >= 0.14; "
            "minimise mean completion time"
        ),
        "performance_denominator": False,
        "rows": rows,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--coast-min-time", type=float, default=COAST_MIN_TIME_S)
    parser.add_argument("--coast-max-time", type=float, default=COAST_MAX_TIME_S)
    parser.add_argument(
        "--outer-descent-time", type=float, default=OUTER_DESCENT_TIME_S
    )
    parser.add_argument("--scan-coast-min-times", type=float, nargs="+")
    parser.add_argument("--scan-outer-descent-times", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scan_requested = bool(
        args.scan_coast_min_times or args.scan_outer_descent_times
    )
    if scan_requested:
        if not args.scan_coast_min_times or not args.scan_outer_descent_times:
            parser.error("both scan timing lists are required")
        result = timing_scan(
            args.episodes,
            args.seed,
            coast_min_times_s=args.scan_coast_min_times,
            outer_descent_times_s=args.scan_outer_descent_times,
            coast_max_time_s=args.coast_max_time,
        )
    else:
        result = evaluate(
            args.episodes,
            args.seed,
            coast_min_time_s=args.coast_min_time,
            coast_max_time_s=args.coast_max_time,
            outer_descent_time_s=args.outer_descent_time,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = (
        {"selected": result["selected"], "points": len(result["rows"])}
        if scan_requested
        else {"aggregate": result["aggregate"], "acceptance": result["acceptance"]}
    )
    print(json.dumps(summary, indent=2))
    print(f"Precapture feasibility-plan result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
