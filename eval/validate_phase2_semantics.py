"""End-to-end reachability and constraint validation for the full mission.

Phase-I already has a scripted ground truth (``validate_phase1_semantics``):
a saturated target-frame PD reaches the Gate 20/20, which is what licenses the
claim that the Phase-I task itself is clean. Phase-II never had the equivalent,
and unlike Phase-I it has no guidance law at all -- ``Phase2MissionReward.
active_desired_velocity`` returns a zero desired velocity once the terminal
constraints are active, while the closing-speed constraint simultaneously
requires the chaser to keep approaching. Nothing in the repository establishes
that the terminal task is reachable, or that a controller can hold every
Phase-II constraint from the Gate all the way to the completion set.

This module supplies that ground truth with a deliberately simple two-stage
scripted controller flown in ``full_mission`` mode, so the Gate transition is
exercised in the same episode:

  * Phase-I leg -- the Phase-I guidance law of ``phase1_desired_velocity``
    (cruise/braking velocity field) read from the mission config rather than
    hard-coded, tracked by a target-frame velocity PD.
  * Phase-II leg -- corridor-aware approach: axial speed commanded strictly
    below the distance-dependent closing-speed limit, lateral offset regulated
    toward the corridor axis, both capped below the total-speed limit.
  * Attitude -- one PD on the relative rotation and relative angular rate for
    both legs, which also drives the camera boresight onto the port.

Read-only: no training, no writes outside --output, and --output refuses to
overwrite, matching the convention of the other eval entry points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from dynamics.lie import so3_log
from env.phase2_env import make_phase2_env
from train.configs import PURE_SAC


GAMMA = PURE_SAC.gamma

# Guidance law default that Phase2MissionReward itself uses; the cruise speed
# is read from the mission config so this tracks the canonical task.
PHASE1_BRAKING_GAIN_PER_S = 0.08

# Fractions of the hard constraint limits the scripted controller aims for, so
# that a validation failure means the task is infeasible rather than that the
# reference controller was flown at the edge of the admissible set.
CLOSING_SPEED_FRACTION = 0.5
TOTAL_SPEED_FRACTION = 0.6

ATTITUDE_GAIN = 2.0
ANGULAR_RATE_GAIN = 8.0
PHASE1_VELOCITY_GAIN_PER_S = 0.8
PHASE2_VELOCITY_GAIN_PER_S = 1.5
PHASE2_LATERAL_GAIN_PER_S = 0.4
PHASE2_AXIAL_GAIN_PER_S = 0.5


def _body_action(env, target_frame_force: np.ndarray) -> np.ndarray:
    """Convert a target-frame force into the normalized 6-DoF action."""

    relative = env.relative
    body_force = relative.rotation.T @ target_frame_force
    rotation_error = so3_log(relative.rotation, project=True)
    body_torque = -ATTITUDE_GAIN * rotation_error - ANGULAR_RATE_GAIN * relative.omega
    limits = env.config
    return np.clip(
        np.concatenate(
            (
                body_torque / limits.max_torque_per_axis_nm,
                body_force / limits.max_force_per_axis_n,
            )
        ),
        -1.0,
        1.0,
    )


def phase1_desired_velocity(env) -> np.ndarray:
    """Cruise/braking velocity field toward the Gate, in the target frame."""

    relative = env.relative
    mission = env.config.phase2_mission
    error = relative.position - mission.gate_position
    distance = float(np.linalg.norm(error))
    if distance <= np.finfo(float).eps:
        return np.zeros(3, dtype=np.float64)
    speed = min(
        mission.phase1_cruise_speed_m_s,
        PHASE1_BRAKING_GAIN_PER_S * distance,
    )
    return -speed * error / distance


def phase2_desired_velocity(env) -> np.ndarray:
    """Corridor-aware approach velocity, in the target frame.

    The axial command is held at a fraction of the distance-dependent closing
    speed limit and the lateral command regulates the offset from the corridor
    axis; their norm is capped below the total speed limit.
    """

    relative = env.relative
    task = env.config.phase2_task
    axis = task.approach_axis
    error = relative.position - task.desired_position
    axial_remaining = float(axis @ error)
    lateral = error - axial_remaining * axis

    closing_limit = task.closing_speed_limit(axial_remaining)
    axial_speed = min(
        CLOSING_SPEED_FRACTION * closing_limit,
        PHASE2_AXIAL_GAIN_PER_S * max(axial_remaining, 0.0),
    )
    desired = -axial_speed * axis - PHASE2_LATERAL_GAIN_PER_S * lateral

    speed_cap = TOTAL_SPEED_FRACTION * task.total_speed_limit_m_s
    speed = float(np.linalg.norm(desired))
    if speed > speed_cap:
        desired *= speed_cap / speed
    return desired


def scripted_action(env, mission_phase: int) -> np.ndarray:
    relative = env.relative
    if mission_phase == 1:
        desired = phase2_desired_velocity(env)
        gain = PHASE2_VELOCITY_GAIN_PER_S
    else:
        desired = phase1_desired_velocity(env)
        gain = PHASE1_VELOCITY_GAIN_PER_S
    velocity = relative.rotation @ relative.velocity
    force = env.chaser_parameters.mass * gain * (desired - velocity)
    return _body_action(env, force)


def zero_action(env, mission_phase: int) -> np.ndarray:  # noqa: ARG001
    return np.zeros(6, dtype=np.float64)


POLICIES = {"scripted": scripted_action, "zero": zero_action}


def rollout(seed: int, policy: str, max_steps: int = 2001) -> dict[str, Any]:
    env = make_phase2_env("full_mission")
    action_fn = POLICIES[policy]
    try:
        _, info = env.reset(seed=seed)
        discounted_return = 0.0
        discount = 1.0
        saturation_sum = 0.0
        mission_phase = 0
        gate_entry: dict[str, float] | None = None
        worst = {
            "corridor_axial_margin_m": np.inf,
            "corridor_lateral_margin_m": np.inf,
            "fov_margin_rad": np.inf,
            "total_speed_margin_m_s": np.inf,
            "closing_speed_margin_m_s": np.inf,
        }
        best_completion_streak = 0
        minimum_terminal_position_error_m = np.inf
        step = 0
        for step in range(1, max_steps):
            action = action_fn(env, mission_phase)
            saturation_sum += float(np.mean(np.abs(action) >= 1.0 - 1.0e-12))
            _, reward, terminated, truncated, info = env.step(action)
            discounted_return += discount * float(reward)
            discount *= GAMMA
            if info.get("gate_transition", False) and gate_entry is None:
                gate_entry = {
                    "time_s": float(info["time_seconds"]),
                    "gate_position_error_m": float(info["gate_position_error_m"]),
                    "total_speed_m_s": float(info["total_speed_m_s"]),
                    "closing_speed_m_s": float(info["closing_speed_m_s"]),
                    "closing_speed_limit_m_s": float(
                        info["closing_speed_limit_m_s"]
                    ),
                    "closing_speed_margin_m_s": float(
                        info["closing_speed_margin_m_s"]
                    ),
                    "fov_angle_rad": float(info["fov_angle_rad"]),
                }
            mission_phase = int(info["mission_phase"])
            if mission_phase == 1:
                for key in worst:
                    worst[key] = min(worst[key], float(info[key]))
                best_completion_streak = max(
                    best_completion_streak, int(info["completion_streak"])
                )
                minimum_terminal_position_error_m = min(
                    minimum_terminal_position_error_m,
                    float(info["position_error_m"]),
                )
            if terminated or truncated:
                break
        reached_phase2 = gate_entry is not None
        return {
            "seed": seed,
            "policy": policy,
            "steps": step,
            "survival_s": step * env.config.dt_s,
            "discounted_return": discounted_return,
            "gate_reached": bool(info["gate_reached"]),
            "gate_entry": gate_entry,
            "final_completed": bool(info.get("final_completed", False)),
            "terminal_constraint_failure": bool(
                info.get("terminal_constraint_failure", False)
            ),
            "phase1_speed_failure": bool(info["phase1_speed_failure"]),
            "premature_entry_failure": bool(info["premature_entry_failure"]),
            "distance_failure": bool(info["distance_failure"]),
            "time_failure": bool(info["time_failure"]),
            "action_saturation_fraction": saturation_sum / step,
            "phase2_worst_margins": (
                {key: float(value) for key, value in worst.items()}
                if reached_phase2
                else None
            ),
            "phase2_best_completion_streak": best_completion_streak,
            "phase2_minimum_position_error_m": (
                float(minimum_terminal_position_error_m)
                if np.isfinite(minimum_terminal_position_error_m)
                else None
            ),
        }
    finally:
        env.close()


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    gate = [item for item in records if item["gate_entry"] is not None]
    done = [item for item in records if item["final_completed"]]
    summary: dict[str, Any] = {
        "episodes": len(records),
        "gate_transition_rate": len(gate) / len(records),
        "mission_completion_rate": len(done) / len(records),
        "terminal_constraint_failure_rate": mean(
            float(item["terminal_constraint_failure"]) for item in records
        ),
        "median_survival_s": median(float(item["survival_s"]) for item in records),
        "mean_discounted_return": mean(
            float(item["discounted_return"]) for item in records
        ),
        "mean_action_saturation_fraction": mean(
            float(item["action_saturation_fraction"]) for item in records
        ),
    }
    if gate:
        summary["mean_gate_time_s"] = mean(
            float(item["gate_entry"]["time_s"]) for item in gate
        )
        summary["minimum_gate_closing_speed_margin_m_s"] = min(
            float(item["gate_entry"]["closing_speed_margin_m_s"]) for item in gate
        )
        summary["phase2_worst_margins"] = {
            key: min(float(item["phase2_worst_margins"][key]) for item in gate)
            for key in gate[0]["phase2_worst_margins"]
        }
        summary["phase2_minimum_position_error_m"] = min(
            float(item["phase2_minimum_position_error_m"]) for item in gate
        )
    if done:
        summary["mean_completion_time_s"] = mean(
            float(item["survival_s"]) for item in done
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted end-to-end validation of the two-phase mission"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--preference-episodes", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    scripted = [
        rollout(args.seed + index, "scripted") for index in range(args.episodes)
    ]
    passive = [
        rollout(args.seed + index, "zero")
        for index in range(args.preference_episodes)
    ]
    margins = [
        scripted[index]["discounted_return"] - passive[index]["discounted_return"]
        for index in range(args.preference_episodes)
    ]
    result = {
        "schema_version": 1,
        "probe": "full_mission_scripted_reachability",
        "gamma": GAMMA,
        "seed": args.seed,
        "reachability": _summarize(scripted),
        "reward_preference": {
            "episodes": args.preference_episodes,
            "scripted_minus_zero_discounted_return_margins": margins,
            "minimum_margin": min(margins),
            "passes": all(margin > 1.0 for margin in margins),
        },
        "scripted_records": scripted,
        "zero_records": passive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reach = result["reachability"]
    print(
        json.dumps(
            {
                "gate_transition_rate": reach["gate_transition_rate"],
                "mission_completion_rate": reach["mission_completion_rate"],
                "terminal_constraint_failure_rate": reach[
                    "terminal_constraint_failure_rate"
                ],
                "median_survival_s": reach["median_survival_s"],
                "phase2_worst_margins": reach.get("phase2_worst_margins"),
                "phase2_minimum_position_error_m": reach.get(
                    "phase2_minimum_position_error_m"
                ),
                "minimum_reward_preference_margin": result["reward_preference"][
                    "minimum_margin"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
