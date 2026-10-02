"""Does a near-field, not-fully-committed V3 reference make the MPC infeasible?

Scripted learned-branch policy: advance progress at full rate every decision
and raise commitment only until it reaches ``--commit``, then hold it. With
commit < 1 the reference ends inside the entry region but off the corridor
axis. Every control step is instrumented: QP status, zero fallback, range,
whether the MPC's predicted-terminal rows are active, terminal latch, and the
angle between the applied reference and the approach axis.

No model, no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controllers.mpc.constraints import _predicted_terminal_active
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


def run(seed: int, commit: float) -> dict:
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
    task = env.environment_config.precapture_task
    axis = np.asarray(task.approach_axis, dtype=np.float64)
    fallback_steps: list[dict] = []
    original = env.controller.command

    def instrumented(*args, **kwargs):
        wrench, diagnostics = original(*args, **kwargs)
        if diagnostics.used_zero_fallback:
            position = env._current_position()
            reference = np.asarray(kwargs["external_reference"], dtype=np.float64)
            fallback_steps.append(
                {
                    "t_s": float(env.env.time_seconds),
                    "status": str(diagnostics.status),
                    "range_m": float(np.linalg.norm(position)),
                    "predicted_terminal_active": bool(
                        _predicted_terminal_active(position, task, terminal_latched=bool(kwargs["terminal_latched"]))
                    ),
                    "terminal_latched": bool(kwargs["terminal_latched"]),
                    "reference_radius_m": float(np.linalg.norm(reference)),
                    "reference_off_axis_deg": float(
                        np.degrees(np.arccos(np.clip(axis @ reference / np.linalg.norm(reference), -1, 1)))
                    ),
                }
            )
        return wrench, diagnostics

    env.controller.command = instrumented  # type: ignore[method-assign]
    try:
        env.reset(seed=seed)
        decisions = 0
        while True:
            a1 = 1.0 if env._task_commitment < commit - 1e-9 else 0.0
            _, _, terminated, truncated, info = env.step_with_branch(np.array([1.0, a1]), branch="learned")
            decisions += 1
            if terminated or truncated:
                break
        return {
            "seed": seed,
            "commit": commit,
            "decisions": decisions,
            "control_steps": 20 * decisions,
            "fallback_steps": len(fallback_steps),
            "completed": bool(info.get("completed", False)),
            "distance_failure": bool(info.get("distance_failure", False)),
            "final_range_m": float(np.linalg.norm(env._current_position())),
            "fallback_share_predicted_terminal_active": (
                float(np.mean([f["predicted_terminal_active"] for f in fallback_steps])) if fallback_steps else None
            ),
            "fallback_share_latched": (
                float(np.mean([f["terminal_latched"] for f in fallback_steps])) if fallback_steps else None
            ),
            "fallback_off_axis_deg_median": (
                float(np.median([f["reference_off_axis_deg"] for f in fallback_steps])) if fallback_steps else None
            ),
            "fallback_range_m_median": (
                float(np.median([f["range_m"] for f in fallback_steps])) if fallback_steps else None
            ),
            "first_fallbacks": fallback_steps[:5],
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--commit", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.commit)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "first_fallbacks"}))


if __name__ == "__main__":
    main()
