"""Instrumented rollouts of a V3 checkpoint to locate QP zero-fallback bursts.

Runs a trained ``task_state_v3`` checkpoint (stochastic by default, as in
training) and records, for every control step whose QP fell back to zero
wrench, the geometry and task state at that moment: range, position relative
to the entry plane, whether the MPC's predicted-terminal rows are active,
terminal latch, reference radius and off-axis angle, commitment, direction
lag, and the exact truth margins. Entry-plane crossings (legal or not) are
recorded per episode so bursts can be aligned with them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from controllers.mpc.constraints import _predicted_terminal_active, normalized_precapture_truth_margins
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config

MARGIN_NAMES = ("keepout", "fov", "slot2", "slot3", "corridor", "slot5", "slot6")


def run(model_path: Path, seed: int, deterministic: bool) -> dict:
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
    model = SAC.load(model_path, device="cpu")
    task = env.environment_config.precapture_task
    axis = np.asarray(task.approach_axis, dtype=np.float64)
    port = np.asarray(task.port_position, dtype=np.float64)
    events: list[dict] = []
    original = env.controller.command

    def instrumented(state, **kwargs):
        wrench, diagnostics = original(state, **kwargs)
        if diagnostics.used_zero_fallback:
            position = env._current_position()
            reference = np.asarray(kwargs["external_reference"], dtype=np.float64)
            margins = normalized_precapture_truth_margins(
                state,
                task,
                target_angular_velocity_rad_s=np.asarray(kwargs["target_state"].omega),
                terminal_latched=bool(kwargs["terminal_latched"]),
            )
            events.append(
                {
                    "t_s": float(env.env.time_seconds),
                    "status": str(diagnostics.status),
                    "range_m": float(np.linalg.norm(position)),
                    "port_axial_m": float(axis @ (position - port)),
                    "predicted_terminal_active": bool(
                        _predicted_terminal_active(position, task, terminal_latched=bool(kwargs["terminal_latched"]))
                    ),
                    "terminal_latched": bool(kwargs["terminal_latched"]),
                    "reference_radius_m": float(np.linalg.norm(reference)),
                    "reference_off_axis_deg": float(
                        np.degrees(np.arccos(np.clip(axis @ reference / np.linalg.norm(reference), -1, 1)))
                    ),
                    "chaser_off_axis_deg": float(
                        np.degrees(np.arccos(np.clip(axis @ position / np.linalg.norm(position), -1, 1)))
                    ),
                    "commitment": float(env._task_commitment),
                    "progress_m": float(env._task_progress_m),
                    "direction_lag_rad": float(env._last_v3_direction_lag_rad),
                    "truth_margins": [float(v) for v in margins],
                }
            )
        return wrench, diagnostics

    env.controller.command = instrumented  # type: ignore[method-assign]
    try:
        observation, _ = env.reset(seed=seed)
        crossings = []
        decisions = 0
        illegal_prev = 0
        while True:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, _, terminated, truncated, info = env.step_with_branch(np.asarray(action), branch="learned")
            decisions += 1
            illegal = int(info.get("illegal_terminal_entry_count", 0))
            if illegal > illegal_prev:
                position = env._current_position()
                crossings.append(
                    {
                        "t_s": float(env.env.time_seconds),
                        "range_m": float(np.linalg.norm(position)),
                        "commitment": float(env._task_commitment),
                    }
                )
            illegal_prev = illegal
            if terminated or truncated:
                break
        failure = [k for k in ("keepout_failure", "fov_failure", "outer_speed_failure", "outer_radial_failure", "transition_speed_failure", "terminal_constraint_failure", "distance_failure", "time_failure") if info.get(k)]
        return {
            "seed": seed,
            "deterministic": deterministic,
            "decisions": decisions,
            "completed": bool(info.get("completed", False)),
            "failure": failure,
            "fallback_steps": len(events),
            "illegal_entries": crossings,
            "fallback_events": events,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.seed, args.deterministic)
    args.output.write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "fallback_events"}))


if __name__ == "__main__":
    main()
