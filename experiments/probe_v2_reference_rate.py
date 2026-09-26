"""What does the V2 rate limit actually permit the reference to do?

V2 caps the two task axes separately: 0.5 m per decision on distance, 0.05 per
decision on commitment. The distance cap is in metres and binds. The commitment
cap is dimensionless, but its effect on the commanded point is a metric
displacement -- the direction slerps, so the reference sweeps an arc of about
``radius * angle(hold, capture) * 0.05``. At the 12-19 m staging radii that is
1-3 m per decision, and nothing bounds it.

The reason that matters is recorded in env/hybrid_env.py: every V1 episode that
completed held the commanded waypoint to a median 0.000-0.015 m per decision at
0.20-0.28 actuator usage, and every episode that timed out moved it 0.27-0.62 m
at 0.84-0.95. So a cap that permits 1-3 m per decision sits well above the band
this task associates with running the clock out.

This probe drives both axes to their far end and measures the reference
displacement the limiter actually allows. It is a limiter measurement, not a
performance arm: an absolute proposal pinned at maximum is adversarial by
construction and its completion count means nothing.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


SEEDS = (262000, 262001, 262005, 262011)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-reference-step-m", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_seed(seed: int, cap_m: float) -> dict[str, Any]:
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="task_state_v2",
            runtime_diagnostics=False,
            include_staging_direction_observation=True,
            v2_reference_step_max_m=cap_m,
        ),
    )
    try:
        env.reset(seed=seed)
        assert env._task_progress_m is not None and env._task_commitment is not None
        previous = env.reference_for_task_state(
            env._task_progress_m, env._task_commitment
        )
        motions: list[float] = []
        target_frame_motions: list[float] = []
        usages: list[float] = []
        moving_usages: list[float] = []
        terminated = truncated = False
        decisions = 0
        action = np.array([1.0, 1.0])
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(action)
            waypoint = np.asarray(
                info["hybrid_waypoint_target_frame"], dtype=np.float64
            )
            target_frame_motion = float(np.linalg.norm(waypoint - previous))
            motion = float(info["hybrid_v2_reference_step_m"])
            usage = float(info["hybrid_feedback_mean_actuator_usage"])
            motions.append(motion)
            target_frame_motions.append(target_frame_motion)
            usages.append(usage)
            if motion > 1.0e-9:
                moving_usages.append(usage)
            previous = waypoint
            decisions += 1
        motion_array = np.asarray(motions)
        active_motion = motion_array[motion_array > 1.0e-9]
        return {
            "seed": seed,
            "completed": bool(info.get("completed", False)),
            "decisions": decisions,
            "time_s": float(env.env.time_seconds),
            "reference_move_median_all_m": float(np.median(motion_array)),
            "reference_move_median_active_m": (
                float(np.median(active_motion)) if active_motion.size else 0.0
            ),
            "reference_move_max_m": float(np.max(motion_array)),
            "target_frame_waypoint_move_max_m": float(
                np.max(target_frame_motions)
            ),
            "active_reference_decisions": int(active_motion.size),
            "actuator_usage_mean_all": float(np.mean(usages)),
            "actuator_usage_mean_active": (
                float(np.mean(moving_usages)) if moving_usages else 0.0
            ),
            "maximum_cap_excess_m": float(np.max(motion_array) - cap_m),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    cap_m = float(args.max_reference_step_m)
    if cap_m <= 0.0:
        raise ValueError("--max-reference-step-m must be positive")
    rows = [run_seed(seed, cap_m) for seed in SEEDS]
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "limiter_measurement_not_performance",
        "v2_reference_step_max_m": cap_m,
        "seeds": list(SEEDS),
        "rows": rows,
        "summary": {
            "reference_move_max_m": max(row["reference_move_max_m"] for row in rows),
            "actuator_usage_mean_all_seeds": float(
                np.mean([row["actuator_usage_mean_all"] for row in rows])
            ),
            "actuator_usage_max_seed": max(
                row["actuator_usage_mean_all"] for row in rows
            ),
            "active_move_median_fraction_of_cap": float(
                np.median(
                    [
                        row["reference_move_median_active_m"] / cap_m
                        for row in rows
                    ]
                )
            ),
        },
    }
    print(
        f'{"seed":<8}{"ended":>8}{"decisions":>11}{"time_s":>9}'
        f'{"move_active":>14}{"move_max":>12}{"act_all":>10}{"act_active":>12}'
    )
    for row in rows:
        print(
            f'{row["seed"]:<8}{str(row["completed"]):>8}'
            f'{row["decisions"]:>11}{row["time_s"]:>9.1f}'
            f'{row["reference_move_median_active_m"]:>14.3f}'
            f'{row["reference_move_max_m"]:>12.3f}'
            f'{row["actuator_usage_mean_all"]:>10.3f}'
            f'{row["actuator_usage_mean_active"]:>12.3f}'
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
