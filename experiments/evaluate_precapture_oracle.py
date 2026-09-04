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

from dynamics.lie import make_transform, se3_log
from env.phase2_env import precapture_planning_environment_config
from env.se3_rendezvous_env import SE3RendezvousEnv
from eval.metrics import summarize


COAST_MIN_TIME_S = 100.0
COAST_MAX_TIME_S = 190.0
OUTER_STAGING_RADIUS_M = 7.5
MATCH_HOLD_TIME_S = 20.0
MATCH_CROSS_TIME_S = 10.0
TERMINAL_RADIUS_M = 5.8
OUTER_POSITION_GAIN_PER_S = 0.12
OUTER_VELOCITY_GAIN_PER_S = 0.80
TERMINAL_VELOCITY_GAIN_PER_S = 1.5
TERMINAL_LATERAL_GAIN_PER_S = 0.4
TERMINAL_AXIAL_GAIN_PER_S = 0.5
ATTITUDE_GAIN = 2.0
ANGULAR_RATE_GAIN = 8.0


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
    """One offline, truth-aware path; deliberately not a reusable planner."""

    def __init__(self, env: SE3RendezvousEnv) -> None:
        assert env.relative is not None and env.target_state is not None
        trajectory = env._target_trajectory
        if trajectory is None:
            raise RuntimeError("oracle requires the cached truth target trajectory")
        task = env.config.precapture_task
        initial = env.target_state.rotation @ env.relative.position
        self.initial_radius_m = float(np.linalg.norm(initial))
        self.initial_direction = initial / self.initial_radius_m
        first = int(np.ceil(COAST_MIN_TIME_S / env.config.dt_s))
        last = min(
            len(trajectory) - 1 - int(0.5 * MATCH_HOLD_TIME_S / env.config.dt_s),
            int(np.floor(COAST_MAX_TIME_S / env.config.dt_s)),
        )
        lead_steps = int(0.5 * MATCH_HOLD_TIME_S / env.config.dt_s)
        scores = np.array(
            [
                self.initial_direction
                @ (trajectory[index + lead_steps].rotation @ task.approach_axis)
                for index in range(first, last + 1)
            ]
        )
        self.coast_steps = first + int(np.argmax(scores))
        self.coast_time_s = self.coast_steps * env.config.dt_s
        target_at_match = trajectory[self.coast_steps + lead_steps]
        self.match_direction = target_at_match.rotation @ task.approach_axis
        self.match_alignment_angle_rad = float(
            np.arccos(np.clip(np.max(scores), -1.0, 1.0))
        )

    def desired_position(
        self,
        time_s: float,
        target_rotation: np.ndarray,
        approach_axis: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        if time_s <= self.coast_time_s:
            fraction = _smoothstep(time_s / self.coast_time_s)
            radius = self.initial_radius_m + fraction * (
                OUTER_STAGING_RADIUS_M - self.initial_radius_m
            )
            direction = _slerp_direction(
                self.initial_direction, self.match_direction, fraction
            )
            return radius * direction, "coast"
        match_elapsed = time_s - self.coast_time_s
        rotating_axis = target_rotation @ approach_axis
        if match_elapsed <= MATCH_HOLD_TIME_S:
            return OUTER_STAGING_RADIUS_M * rotating_axis, "match"
        cross_elapsed = match_elapsed - MATCH_HOLD_TIME_S
        fraction = _smoothstep(cross_elapsed / MATCH_CROSS_TIME_S)
        radius = OUTER_STAGING_RADIUS_M + fraction * (
            TERMINAL_RADIUS_M - OUTER_STAGING_RADIUS_M
        )
        return radius * rotating_axis, "cross"


def _terminal_target_velocity(env: SE3RendezvousEnv) -> np.ndarray:
    relative = env.relative
    assert relative is not None
    task = env.config.precapture_task
    error = relative.position - task.desired_position
    axial_remaining = float(task.approach_axis @ error)
    lateral = error - axial_remaining * task.approach_axis
    axial_speed = min(
        0.5 * task.closing_speed_limit(axial_remaining),
        TERMINAL_AXIAL_GAIN_PER_S * max(axial_remaining, 0.0),
    )
    desired = (
        -axial_speed * task.approach_axis
        - TERMINAL_LATERAL_GAIN_PER_S * lateral
    )
    cap = 0.6 * task.terminal_total_speed_limit_m_s
    norm = float(np.linalg.norm(desired))
    return desired if norm <= cap else desired * (cap / norm)


def _oracle_action(
    env: SE3RendezvousEnv, plan: CoastThenMatchPlan
) -> tuple[np.ndarray, str, float]:
    assert env.relative is not None and env.target_state is not None
    assert env.chaser_state is not None and env.chaser_parameters is not None
    task = env.config.precapture_task
    relative = env.relative
    if env._terminal_region_entered:
        phase = "terminal"
        desired_target_velocity = _terminal_target_velocity(env)
        target_velocity = relative.rotation @ relative.velocity
        target_force = env.chaser_parameters.mass * TERMINAL_VELOCITY_GAIN_PER_S * (
            desired_target_velocity - target_velocity
        )
        body_force = relative.rotation.T @ target_force
        desired_relative_rotation = np.eye(3)
    else:
        desired_position, phase = plan.desired_position(
            env.time_seconds,
            env.target_state.rotation,
            task.approach_axis,
        )
        desired_relative_rotation = _los_rotation(
            relative.rotation, relative.position, task.port_position
        )
        if phase == "cross":
            staging_position = TERMINAL_RADIUS_M * task.approach_axis
            error = relative.position - staging_position
            axial_remaining = float(task.approach_axis @ error)
            lateral = error - axial_remaining * task.approach_axis
            axial_speed = min(0.10, 0.15 * max(axial_remaining, 0.0))
            desired_target_velocity = (
                -axial_speed * task.approach_axis - 0.25 * lateral
            )
            speed = float(np.linalg.norm(desired_target_velocity))
            if speed > 0.18:
                desired_target_velocity *= 0.18 / speed
            target_velocity = relative.rotation @ relative.velocity
            target_force = env.chaser_parameters.mass * 0.8 * (
                desired_target_velocity - target_velocity
            )
            body_force = relative.rotation.T @ target_force
        else:
            next_position, _ = plan.desired_position(
                env.time_seconds + env.config.dt_s,
                env._target_trajectory[
                    min(env.step_count + 1, len(env._target_trajectory) - 1)
                ].rotation,
                task.approach_axis,
            )
            desired_velocity = (next_position - desired_position) / env.config.dt_s
            actual_position = env.chaser_state.position - env.target_state.position
            actual_velocity = (
                env.chaser_state.rotation @ env.chaser_state.velocity
                - env.target_state.rotation @ env.target_state.velocity
            )
            position_gain = OUTER_POSITION_GAIN_PER_S
            velocity_gain = OUTER_VELOCITY_GAIN_PER_S
            if phase == "match":
                elapsed = env.time_seconds - plan.coast_time_s
                match_fraction = _smoothstep(elapsed / MATCH_HOLD_TIME_S)
                target_omega_inertial = (
                    env.target_state.rotation @ env.target_state.omega
                )
                desired_velocity = match_fraction * np.cross(
                    target_omega_inertial, actual_position
                )
                position_gain = 0.0
                velocity_gain = 0.55
            commanded_velocity = desired_velocity + position_gain * (
                desired_position - actual_position
            )
            radial_direction = actual_position / np.linalg.norm(actual_position)
            radial_closing = float(-radial_direction @ commanded_velocity)
            radial_limit = 0.75 * task.outer_radial_closing_speed_limit(
                np.linalg.norm(actual_position)
            )
            if radial_closing > radial_limit:
                commanded_velocity += (
                    radial_closing - radial_limit
                ) * radial_direction
            speed = float(np.linalg.norm(commanded_velocity))
            speed_cap = 0.75 * task.outer_inertial_speed_limit_m_s
            if speed > speed_cap:
                commanded_velocity *= speed_cap / speed
            inertial_force = env.chaser_parameters.mass * velocity_gain * (
                commanded_velocity - actual_velocity
            )
            body_force = env.chaser_state.rotation.T @ inertial_force
    rotation_error = se3_log(
        make_transform(desired_relative_rotation.T @ relative.rotation, np.zeros(3))
    )[:3]
    body_torque = -ATTITUDE_GAIN * rotation_error - ANGULAR_RATE_GAIN * relative.omega
    action = np.clip(
        np.concatenate(
            (
                body_torque / env.config.max_torque_per_axis_nm,
                body_force / env.config.max_force_per_axis_n,
            )
        ),
        -1.0,
        1.0,
    )
    force_norm = float(
        np.linalg.norm(action[3:] * env.config.max_force_per_axis_n)
    )
    return action, phase, force_norm


def rollout(seed: int) -> dict[str, Any]:
    env = SE3RendezvousEnv(precapture_planning_environment_config())
    try:
        _, info = env.reset(seed=seed)
        plan = CoastThenMatchPlan(env)
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
                "initial_radius_m": plan.initial_radius_m,
                "match_alignment_angle_rad": plan.match_alignment_angle_rad,
            },
            "trace": trace,
        }
    finally:
        env.close()


def evaluate(episodes: int, seed: int) -> dict[str, Any]:
    records = [rollout(seed + episode) for episode in range(episodes)]
    accepted = [row for row in records if row["zero_truth_violation_completed"]]
    checks = {
        "at_least_3_of_5_zero_violation_completed": len(accepted) >= 3,
        "accepted_delta_v_at_most_3_m_s": bool(accepted) and all(
            row["equivalent_delta_v_m_s"] <= 3.0 for row in accepted
        ),
        "accepted_mean_force_at_most_2p5_n": bool(accepted) and all(
            row["force_norm_n"]["mean"] <= 2.5 for row in accepted
        ),
        "accepted_fov_margin_strictly_positive": bool(accepted) and all(
            row["minimum_fov_margin_rad"] > 0.0 for row in accepted
        ),
        "accepted_time_at_most_300_s": bool(accepted) and all(
            row["time_s"] <= 300.0 for row in accepted
        ),
    }
    return {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "precapture_coast_then_match_offline_oracle",
        "episodes": episodes,
        "base_seed": seed,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": result["aggregate"], "acceptance": result["acceptance"]}, indent=2))
    print(f"Precapture oracle result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
