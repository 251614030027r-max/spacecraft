"""Generate the outer-divergence trajectory used for precapture curve A."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dynamics.lie import make_transform
from dynamics.relative import RelativeState, reconstruct_chaser_state
from env.phase2_env import precapture_planning_environment_config
from env.scenarios import target_initial_state
from env.se3_rendezvous_env import SE3RendezvousEnv


INITIAL_RADIUS_M = 16.0
COMMANDED_RATE_RAD_S = 0.08


def evaluate(seed: int) -> dict[str, object]:
    """Run the fixed zero-command diagnostic without changing task physics."""

    config = replace(
        precapture_planning_environment_config(),
        max_time_s=60.0,
        cache_target_trajectory=False,
        phase2_target_phase_sampling=False,
        precapture_fov_violation_terminates=False,
        include_j2=False,
    )
    target = target_initial_state(tumble_scale=0.0)
    relative = RelativeState(
        make_transform(np.eye(3), np.array([-INITIAL_RADIUS_M, 0.0, 0.0])),
        np.array(
            [
                0.0,
                0.0,
                COMMANDED_RATE_RAD_S,
                0.0,
                -INITIAL_RADIUS_M * COMMANDED_RATE_RAD_S,
                0.0,
            ]
        ),
    )
    env = SE3RendezvousEnv(config)
    trajectory: list[dict[str, object]] = []
    try:
        env.reset(
            seed=seed,
            options={
                "target_state": target,
                "chaser_state": reconstruct_chaser_state(target, relative),
            },
        )
        trajectory.append(
            {
                "time_s": 0.0,
                "target_frame_position_m": env.relative.transform[:3, 3].tolist(),
                "target_center_distance_m": INITIAL_RADIUS_M,
            }
        )
        action = np.zeros(6, dtype=np.float32)
        terminated = truncated = False
        info: dict[str, object] = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(action)
            trajectory.append(
                {
                    "time_s": float(info["time_seconds"]),
                    "target_frame_position_m": env.relative.transform[:3, 3].tolist(),
                    "target_center_distance_m": float(
                        info["target_center_distance_m"]
                    ),
                }
            )
    finally:
        env.close()

    checks = {
        "distance_failure": bool(info["distance_failure"]),
        "not_outer_speed_failure": not bool(info["outer_speed_failure"]),
        "termination_between_15_and_30_s": (
            15.0 <= float(info["time_seconds"]) <= 30.0
        ),
    }
    return {
        "schema_version": "precapture-divergence-v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "curve_A_outer_divergence_diagnostic",
        "interpretation": (
            "A fixed 16 m, 0.08 rad/s outer-motion diagnostic with zero control; "
            "it exposes force-limited radial divergence and is not an MPC result."
        ),
        "seed": seed,
        "initial_radius_m": INITIAL_RADIUS_M,
        "commanded_rate_rad_s": COMMANDED_RATE_RAD_S,
        "initial_tangential_speed_m_s": (
            INITIAL_RADIUS_M * COMMANDED_RATE_RAD_S
        ),
        "applied_action": [0.0] * 6,
        "final": {
            "time_s": float(info["time_seconds"]),
            "target_center_distance_m": float(info["target_center_distance_m"]),
            "distance_failure": bool(info["distance_failure"]),
            "outer_speed_failure": bool(info["outer_speed_failure"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        },
        "acceptance": {"checks": checks, "passed": all(checks.values())},
        "trajectory_target_frame": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/precapture_planning_v2/curve_a_divergence_seed99.json"),
    )
    args = parser.parse_args()
    result = evaluate(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["final"]}, indent=2))
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
