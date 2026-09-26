"""Matched T0 vs T0b QP-fallback probe with a random task-level policy.

Runs the same seeded uniform-random action sequence (the behaviour SAC starts
from) through whichever V3 env the current checkout provides, always on the
learned branch, and records per-decision QP zero fallbacks, the applied
reference and the chaser state. Run it once from a T0 checkout and once from
a T0b checkout with identical arguments, then compare the JSON files.

No model is loaded, so the comparison isolates the interface: same initial
state, same actions, same MPC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


def run(seed: int, action_seed: int, bias: float = 0.0) -> dict:
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="task_state_v3",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_execution_feedback_observation=True,
            include_staging_direction_observation=True,
        ),
    )
    rng = np.random.default_rng(action_seed)
    rows = []
    try:
        env.reset(seed=seed)
        while True:
            # bias > 0: approach-biased exploration (advance and commit on average)
            action = np.clip(rng.normal(bias, 0.5, size=2), -1.0, 1.0) if bias else rng.uniform(-1.0, 1.0, size=2)
            _, reward, terminated, truncated, info = env.step_with_branch(action, branch="learned")
            position = env._current_position()
            rows.append(
                {
                    "t_s": float(env.env.time_seconds),
                    "fallbacks": int(info.get("hybrid_qp_zero_fallbacks", 0)),
                    "reference": [float(v) for v in info.get("hybrid_waypoint_target_frame", [np.nan] * 3)]
                    if "hybrid_waypoint_target_frame" in info
                    else None,
                    "jump_m": float(info.get("reference_jump_target_m", np.nan)),
                    "task_state": [float(env._task_progress_m), float(env._task_commitment)],
                    "range_m": float(np.linalg.norm(position)),
                    "terminal_active": bool(info.get("terminal_region_active", False)),
                    "lag_rad": float(info.get("hybrid_v3_reference_direction_lag_rad", np.nan)),
                }
            )
            if terminated or truncated:
                break
        return {
            "seed": seed,
            "action_seed": action_seed,
            "bias": bias,
            "decisions": len(rows),
            "control_steps": 20 * len(rows),
            "fallbacks": int(sum(r["fallbacks"] for r in rows)),
            "completed": bool(info.get("completed", False)),
            "distance_failure": bool(info.get("distance_failure", False)),
            "max_jump_m": float(np.nanmax([r["jump_m"] for r in rows])) if rows else None,
            "jump_bound_violations": int(info.get("hybrid_v3_episode_reference_jump_violations", -1)),
            "rows": rows,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--action-seed", type=int, required=True)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.action_seed, args.bias)
    args.output.write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
