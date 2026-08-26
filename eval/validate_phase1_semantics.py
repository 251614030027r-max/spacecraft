"""Low-cost exact-truth reachability and Phase-I reward preference validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from dynamics.lie import so3_log
from env.phase2_env import make_phase2_env
from train.configs import PURE_SAC


GAMMA = PURE_SAC.gamma


def waypoint_pd_action(env) -> np.ndarray:
    """A deliberately simple saturated target-frame position/attitude PD."""

    relative = env.relative
    assert relative is not None
    assert env.chaser_parameters is not None
    error = relative.position - env.config.phase2_mission.waypoint_position
    distance = float(np.linalg.norm(error))
    direction = error / max(distance, np.finfo(float).eps)
    target_velocity = -min(0.18, 0.08 * distance) * direction
    current_velocity = relative.rotation @ relative.velocity
    target_force = env.chaser_parameters.mass * 0.8 * (
        target_velocity - current_velocity
    )
    body_force = relative.rotation.T @ target_force
    rotation_error = so3_log(relative.rotation, project=True)
    body_torque = -2.0 * rotation_error - 8.0 * relative.omega
    return np.clip(
        np.concatenate((body_torque / 0.6, body_force / 5.0)), -1.0, 1.0
    )


def high_speed_failure_action(env) -> np.ndarray:
    relative = env.relative
    assert relative is not None
    error = relative.position - env.config.phase2_mission.waypoint_position
    direction = error / max(float(np.linalg.norm(error)), np.finfo(float).eps)
    body_force = relative.rotation.T @ (5.0 * direction)
    rotation_error = so3_log(relative.rotation, project=True)
    body_torque = -2.0 * rotation_error - 8.0 * relative.omega
    return np.clip(
        np.concatenate((body_torque / 0.6, body_force / 5.0)), -1.0, 1.0
    )


def rollout(seed: int, policy: str) -> dict[str, object]:
    env = make_phase2_env("phase1_pretrain")
    try:
        _, info = env.reset(seed=seed)
        initial_waypoint_distance = float(info["waypoint_position_error_m"])
        discounted_return = 0.0
        undiscounted_return = 0.0
        discount = 1.0
        maximum_speed = float(info["total_speed_m_s"])
        minimum_waypoint_distance = initial_waypoint_distance
        saturation_sum = 0.0
        waypoint_entry = None
        components = {
            "progress": 0.0,
            "state": 0.0,
            "actuation": 0.0,
            "speed": 0.0,
            "event": 0.0,
        }
        for step in range(1, 2001):
            if policy == "waypoint_pd":
                action = waypoint_pd_action(env)
            elif policy == "high_speed_failure":
                action = high_speed_failure_action(env)
            elif policy == "zero":
                action = np.zeros(6, dtype=np.float64)
            else:
                raise ValueError(f"unsupported policy: {policy}")
            saturation_sum += float(np.mean(np.abs(action) >= 1.0 - 1.0e-12))
            _, reward, terminated, truncated, info = env.step(action)
            discounted_return += discount * float(reward)
            undiscounted_return += float(reward)
            discount *= GAMMA
            components["progress"] += float(info["reward_progress"])
            components["state"] += float(info["reward_state_penalty"])
            components["actuation"] += float(info["reward_actuation_penalty"])
            components["speed"] += float(info["reward_constraint_warning"])
            components["event"] += float(info["reward_event"])
            maximum_speed = max(maximum_speed, float(info["total_speed_m_s"]))
            minimum_waypoint_distance = min(
                minimum_waypoint_distance, float(info["waypoint_position_error_m"])
            )
            if info.get("waypoint_transition", False) and waypoint_entry is None:
                waypoint_entry = {
                    "time_s": float(info["time_seconds"]),
                    "position_error_m": float(info["waypoint_position_error_m"]),
                    "attitude_error_rad": float(info["attitude_error_rad"]),
                    "speed_m_s": float(info["total_speed_m_s"]),
                    "angular_velocity_rad_s": float(
                        info["angular_velocity_error_rad_s"]
                    ),
                }
            if terminated or truncated:
                break
        return {
            "seed": seed,
            "policy": policy,
            "waypoint_reached": bool(info["waypoint_reached"]),
            "steps": step,
            "discounted_return": discounted_return,
            "undiscounted_return": undiscounted_return,
            "initial_waypoint_distance_m": initial_waypoint_distance,
            "minimum_waypoint_distance_m": minimum_waypoint_distance,
            "final_waypoint_distance_m": float(info["waypoint_position_error_m"]),
            "maximum_speed_m_s": maximum_speed,
            "action_saturation_fraction": saturation_sum / step,
            "phase1_speed_failure": bool(info["phase1_speed_failure"]),
            "premature_entry_failure": bool(info["premature_entry_failure"]),
            "distance_failure": bool(info["distance_failure"]),
            "time_failure": bool(info["time_failure"]),
            "waypoint_entry": waypoint_entry,
            "reward_components_undiscounted": components,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--reachability-episodes", type=int, default=20)
    parser.add_argument("--preference-episodes", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reachability = [
        rollout(args.seed + index, "waypoint_pd")
        for index in range(args.reachability_episodes)
    ]
    preference = []
    for index in range(args.preference_episodes):
        seed = args.seed + index
        preference.append(
            {
                policy: rollout(seed, policy)
                for policy in ("waypoint_pd", "zero", "high_speed_failure")
            }
        )
    margins = [
        item["waypoint_pd"]["discounted_return"]
        - item["high_speed_failure"]["discounted_return"]
        for item in preference
    ]
    arrival_times = [
        float(item["waypoint_entry"]["time_s"])
        for item in reachability
        if item["waypoint_entry"] is not None
    ]
    result = {
        "schema_version": 1,
        "gamma": GAMMA,
        "seed": args.seed,
        "reachability": {
            "episodes": args.reachability_episodes,
            "waypoint_acquisition_rate": mean(
                float(item["waypoint_reached"]) for item in reachability
            ),
            "mean_arrival_time_s": mean(arrival_times) if arrival_times else None,
            "maximum_speed_m_s": max(
                float(item["maximum_speed_m_s"]) for item in reachability
            ),
            "mean_action_saturation_fraction": mean(
                float(item["action_saturation_fraction"])
                for item in reachability
            ),
            "records": reachability,
        },
        "reward_preference": {
            "episodes": args.preference_episodes,
            "waypoint_minus_high_speed_discounted_return_margins": margins,
            "minimum_margin": min(margins),
            "passes": all(margin > 1.0 for margin in margins),
            "records": preference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "waypoint_acquisition_rate": result["reachability"]["waypoint_acquisition_rate"],
        "mean_arrival_time_s": result["reachability"]["mean_arrival_time_s"],
        "maximum_speed_m_s": result["reachability"]["maximum_speed_m_s"],
        "minimum_reward_preference_margin": result["reward_preference"]["minimum_margin"],
        "reward_preference_passes": result["reward_preference"]["passes"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
