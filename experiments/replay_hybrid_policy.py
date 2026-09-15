"""Per-decision replay of a trained coupling policy, for diagnosis only.

T7 produced two seeds per version that complete nothing while their return
still improves, and the monitor only keeps one row per episode -- enough to
see *that* they run the clock out, not *how*. This records what the upper
layer actually commanded, decision by decision, so a dead seed and a live one
can be put side by side.

It is a read-only diagnostic. It runs the policy deterministically, changes
no configuration, and writes one JSON file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config


TERMINATION_KEYS = (
    "time_failure",
    "distance_failure",
    "keepout_failure",
    "fov_failure",
    "outer_speed_failure",
    "outer_radial_failure",
    "terminal_constraint_failure",
    "transition_speed_failure",
)

MARGIN_KEYS = (
    "fov_margin_rad",
    "keepout_margin_m",
    "outer_radial_margin_m_s",
    "outer_inertial_speed_margin_m_s",
    "terminal_total_speed_margin_m_s",
    "closing_speed_margin_m_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--parametrization",
        choices=["absolute", "radial_local", "arrival_condition"],
        default="arrival_condition",
    )
    parser.add_argument("--phase-time-observation", action="store_true")
    parser.add_argument("--execution-feedback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from stable_baselines3 import SAC

    policy = SAC.load(args.model, device="cpu")
    environment_config = replace(
        precapture_planning_environment_config(), cache_target_trajectory=False
    )

    episodes: list[dict[str, Any]] = []
    for index in range(args.episodes):
        seed = args.seed + index
        env = PrecaptureHybridEnv(
            environment_config=environment_config,
            hybrid_config=PrecaptureHybridConfig(
                horizon_steps=args.horizon,
                waypoint_parametrization=args.parametrization,
                runtime_diagnostics=False,
                include_target_phase_and_time_observation=(
                    args.phase_time_observation
                ),
                include_execution_feedback_observation=args.execution_feedback,
            ),
        )
        expected = int(np.prod(policy.observation_space.shape))
        actual = int(np.prod(env.observation_space.shape))
        if expected != actual:
            raise ValueError(
                f"the checkpoint expects a {expected}D observation and this "
                f"configuration builds a {actual}D one; match the run manifest"
            )
        observation, info = env.reset(seed=seed)
        decisions: list[dict[str, Any]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            observation, reward, terminated, truncated, info = env.step(action)
            row = {
                "t_s": round(float(env.env.time_seconds), 3),
                "action": [round(float(v), 5) for v in action],
                "reward": round(float(reward), 5),
                "true_range_m": round(
                    float(np.linalg.norm(env._current_position())), 4
                ),
                "waypoint_radius_m": round(
                    float(info["hybrid_waypoint_radius_m"]), 4
                ),
                "qp_zero_fallbacks": int(info["hybrid_qp_zero_fallbacks"]),
                "terminal_region_active": bool(info["terminal_region_active"]),
                "entry_crossed": bool(info.get("terminal_entry_crossed", False)),
                "entry_legal": bool(info.get("terminal_entry_legal", False)),
                "illegal_entries": float(info["illegal_terminal_entry_count"]),
            }
            for key in MARGIN_KEYS:
                if key in info:
                    row[key] = round(float(info[key]), 5)
            for key in (
                "hybrid_feedback_fallback_fraction",
                "hybrid_feedback_solved_peak_slack",
                "hybrid_feedback_mean_actuator_usage",
            ):
                if key in info:
                    row[key] = round(float(info[key]), 5)
            decisions.append(row)
        ended_by = [key for key in TERMINATION_KEYS if bool(info.get(key, False))]
        actions = np.array([row["action"] for row in decisions])
        episodes.append(
            {
                "seed": seed,
                "completed": bool(info.get("completed", False)),
                "decisions": len(decisions),
                "end_time_s": round(float(env.env.time_seconds), 3),
                "ended_by": ended_by or ["none_flagged"],
                "illegal_entries": float(info["illegal_terminal_entry_count"]),
                "terminal_region_active_at_end": bool(
                    info["terminal_region_active"]
                ),
                "action_mean": [round(float(v), 5) for v in actions.mean(axis=0)],
                "action_std": [round(float(v), 5) for v in actions.std(axis=0)],
                "action_min": [round(float(v), 5) for v in actions.min(axis=0)],
                "action_max": [round(float(v), 5) for v in actions.max(axis=0)],
                # The hypothesis the T7 verdict wants tested: a seed that never
                # completes may simply be holding. Under this parametrisation
                # holding is the negative half of the first channel.
                "fraction_commit_negative": round(
                    float(np.mean(actions[:, 0] < 0.0)), 4
                ),
                "trace": decisions,
            }
        )
        summary = episodes[-1]
        print(
            f"seed {seed}: completed={summary['completed']} "
            f"decisions={summary['decisions']} "
            f"ended_by={'/'.join(summary['ended_by'])} "
            f"a_commit mean={summary['action_mean'][0]:+.3f} "
            f"(neg {summary['fraction_commit_negative']:.0%}) "
            f"a_radius mean={summary['action_mean'][1]:+.3f}",
            flush=True,
        )
        env.close()

    args.output.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "parametrization": args.parametrization,
                "phase_time_observation": args.phase_time_observation,
                "execution_feedback": args.execution_feedback,
                "episodes": episodes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
