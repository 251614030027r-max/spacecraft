"""Reachability and constraint validation for the single-phase (gate-free) task.

The two-phase mission postpones the terminal constraints until the Gate at 8 m,
where an inertially frozen chaser already moves at 0.330 m/s in the tumbling
target frame against a 0.35 m/s limit. Removing the Gate makes those
constraints live from the first step, at 10-14 m, where the same quantity is
0.41-0.58 m/s -- above the limit. Station-keeping is therefore inadmissible for
most of the task and the chaser must co-rotate continuously, which is what makes
the single-phase task worth posing.

Nothing established that the task is reachable under those conditions, so this
module supplies the ground truth, exactly as ``validate_phase2_semantics`` does
for the two-phase mission. It flies the corridor-aware control law that module
uses for its terminal leg -- one law for the whole mission, no phase switch, no
Gate waypoint -- so a failure here means the task is infeasible rather than
that the reference controller was rebuilt to pass.

That law and the reward's `terminal_desired_velocity` agree pointwise while
`axial_remaining >= 0` and diverge only past the desired pose, where the
reward's version commands a retreat and this one commands a hold. The scripted
controller never overshoots, so this certificate covers the approach and **not**
the overshoot region -- which is exactly where seeds 260850-260852 died before
the reward gained its axial restoring term. The 20/20 reachability result is
unaffected; the two laws are simply no longer identical.

Read-only: no training, no writes outside --output, and --output refuses to
overwrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from env.phase2_env import make_phase2_env
from eval.validate_phase2_semantics import (
    PHASE2_VELOCITY_GAIN_PER_S,
    _body_action,
    phase2_desired_velocity,
)
from train.configs import PURE_SAC


GAMMA = PURE_SAC.gamma

MARGIN_KEYS = (
    "corridor_axial_margin_m",
    "corridor_lateral_margin_m",
    "fov_margin_rad",
    "total_speed_margin_m_s",
    "closing_speed_margin_m_s",
)


def scripted_action(env) -> np.ndarray:
    """One corridor-aware law for the whole mission, in the target frame."""

    desired = phase2_desired_velocity(env)
    velocity = env.relative.rotation @ env.relative.velocity
    force = (
        env.chaser_parameters.mass
        * PHASE2_VELOCITY_GAIN_PER_S
        * (desired - velocity)
    )
    return _body_action(env, force)


def zero_action(env) -> np.ndarray:  # noqa: ARG001
    return np.zeros(6, dtype=np.float64)


POLICIES = {"scripted": scripted_action, "zero": zero_action}


def rollout(seed: int, policy: str, max_steps: int = 2001) -> dict[str, Any]:
    env = make_phase2_env("gate_free")
    action_fn = POLICIES[policy]
    try:
        _, info = env.reset(seed=seed)
        initial_distance = float(info["target_center_distance_m"])
        discounted_return = 0.0
        discount = 1.0
        saturation_sum = 0.0
        forces: list[float] = []
        torques: list[float] = []
        worst = {key: np.inf for key in MARGIN_KEYS}
        first_violation: dict[str, Any] | None = None
        best_completion_streak = 0
        minimum_position_error_m = np.inf
        maximum_total_speed_m_s = 0.0
        step = 0
        for step in range(1, max_steps):
            action = action_fn(env)
            saturation_sum += float(np.mean(np.abs(action) >= 1.0 - 1.0e-12))
            forces.append(
                float(np.linalg.norm(action[3:]) * env.config.max_force_per_axis_n)
            )
            torques.append(
                float(np.linalg.norm(action[:3]) * env.config.max_torque_per_axis_nm)
            )
            _, reward, terminated, truncated, info = env.step(action)
            discounted_return += discount * float(reward)
            discount *= GAMMA
            for key in MARGIN_KEYS:
                value = float(info[key])
                worst[key] = min(worst[key], value)
                if value < 0.0 and first_violation is None:
                    first_violation = {
                        "type": key,
                        "time_s": float(info["time_seconds"]),
                        "margin": value,
                    }
            best_completion_streak = max(
                best_completion_streak, int(info["completion_streak"])
            )
            minimum_position_error_m = min(
                minimum_position_error_m, float(info["position_error_m"])
            )
            maximum_total_speed_m_s = max(
                maximum_total_speed_m_s, float(info["total_speed_m_s"])
            )
            if terminated or truncated:
                break
        return {
            "seed": seed,
            "policy": policy,
            "steps": step,
            "initial_distance_m": initial_distance,
            "survival_s": step * env.config.dt_s,
            "discounted_return": discounted_return,
            "final_completed": bool(info.get("final_completed", False)),
            "terminal_constraint_failure": bool(
                info.get("terminal_constraint_failure", False)
            ),
            "distance_failure": bool(info["distance_failure"]),
            "time_failure": bool(info["time_failure"]),
            "constraint_success": bool(info["constraint_success"]),
            "first_violation": first_violation,
            "worst_margins": {key: float(value) for key, value in worst.items()},
            "best_completion_streak": best_completion_streak,
            "minimum_position_error_m": float(minimum_position_error_m),
            "maximum_total_speed_m_s": maximum_total_speed_m_s,
            "force_n": {
                "mean": float(np.mean(forces)),
                "p95": float(np.percentile(forces, 95)),
                "max": float(np.max(forces)),
            },
            "torque_nm": {
                "mean": float(np.mean(torques)),
                "max": float(np.max(torques)),
            },
            "action_saturation_fraction": saturation_sum / step,
        }
    finally:
        env.close()


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    done = [item for item in records if item["final_completed"]]
    summary: dict[str, Any] = {
        "episodes": len(records),
        "mission_completion_rate": len(done) / len(records),
        "terminal_constraint_failure_rate": mean(
            float(item["terminal_constraint_failure"]) for item in records
        ),
        "time_failure_rate": mean(float(item["time_failure"]) for item in records),
        "distance_failure_rate": mean(
            float(item["distance_failure"]) for item in records
        ),
        "median_survival_s": median(float(item["survival_s"]) for item in records),
        "mean_discounted_return": mean(
            float(item["discounted_return"]) for item in records
        ),
        "worst_margins": {
            key: min(float(item["worst_margins"][key]) for item in records)
            for key in MARGIN_KEYS
        },
        "maximum_total_speed_m_s": max(
            float(item["maximum_total_speed_m_s"]) for item in records
        ),
        "minimum_position_error_m": min(
            float(item["minimum_position_error_m"]) for item in records
        ),
        "mean_action_saturation_fraction": mean(
            float(item["action_saturation_fraction"]) for item in records
        ),
        "force_n": {
            "mean": mean(float(item["force_n"]["mean"]) for item in records),
            "p95": max(float(item["force_n"]["p95"]) for item in records),
            "max": max(float(item["force_n"]["max"]) for item in records),
        },
    }
    if done:
        summary["mean_completion_time_s"] = mean(
            float(item["survival_s"]) for item in done
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted validation of the single-phase gate-free task"
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
    reachability = _summarize(scripted)
    result = {
        "schema_version": 1,
        "probe": "gate_free_scripted_reachability",
        "gamma": GAMMA,
        "seed": args.seed,
        "reachability": reachability,
        "passive": _summarize(passive),
        "reward_preference": {
            "episodes": args.preference_episodes,
            "scripted_minus_zero_discounted_return_margins": margins,
            "minimum_margin": min(margins),
            "passes": all(margin > 1.0 for margin in margins),
        },
        "acceptance": {
            "completion_rate_is_one": reachability["mission_completion_rate"] == 1.0,
            "no_terminal_constraint_failure": (
                reachability["terminal_constraint_failure_rate"] == 0.0
            ),
            "all_margins_positive": all(
                value > 0.0 for value in reachability["worst_margins"].values()
            ),
            "reward_prefers_scripted": all(margin > 1.0 for margin in margins),
        },
        "scripted_records": scripted,
        "zero_records": passive,
    }
    result["acceptance"]["passed"] = all(result["acceptance"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "reachability": reachability,
                "acceptance": result["acceptance"],
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
