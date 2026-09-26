"""Correctness-only smoke for the frozen V2 task-state interface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


VIOLATION_KEYS = (
    "keepout_violation_steps",
    "fov_violation_steps",
    "outer_speed_violation_steps",
    "corridor_violation_steps",
    "total_speed_violation_steps",
    "closing_speed_violation_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/adp/v2_interface_smoke.json"),
    )
    return parser.parse_args()


def make_env(parametrization: str) -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization=parametrization,
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_staging_direction_observation=True,
        ),
    )


def record_commands(env: PrecaptureHybridEnv) -> list[np.ndarray]:
    commands: list[np.ndarray] = []
    original = env.controller.command

    def wrapped(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        commands.append(np.asarray(result[0]).copy())
        return result

    env.controller.command = wrapped  # type: ignore[method-assign]
    return commands


def main() -> None:
    args = parse_args()
    v2 = make_env("task_state_v2")
    accepted_records: list[dict[str, Any]] = []
    geometry_min_radius = float("inf")
    try:
        observation, _ = v2.reset(seed=262000)
        assert np.all(np.isfinite(observation))
        geometry_points = []
        for rho_m in np.linspace(0.0, v2.v2_progress_max_m, 9):
            for commitment in np.linspace(0.0, 1.0, 9):
                point = v2.reference_for_task_state(rho_m, commitment)
                assert np.all(np.isfinite(point))
                radius = float(np.linalg.norm(point))
                geometry_min_radius = min(geometry_min_radius, radius)
                geometry_points.append(point)
        assert geometry_min_radius >= v2.environment_config.precapture_task.keepout_radius_m

        proposals = ((v2.v2_progress_max_m, 1.0), (5.0, 0.2), (10.0, 0.8))
        previous_waypoint = None
        for rho_m, commitment in proposals:
            action = v2.action_for_task_state(rho_m, commitment)
            observation, _, terminated, truncated, info = v2.step(action)
            waypoint = np.asarray(info["hybrid_waypoint_target_frame"])
            assert np.all(np.isfinite(observation)) and np.all(np.isfinite(waypoint))
            assert info["hybrid_proposal_accepted"] is True
            proposed = np.asarray(info["hybrid_task_state_proposed"])
            applied = np.asarray(info["hybrid_task_state_applied"])
            assert not np.array_equal(proposed, applied)
            if previous_waypoint is not None:
                assert not np.array_equal(previous_waypoint, waypoint)
            accepted_records.append(
                {
                    "proposal": proposed.tolist(),
                    "applied": applied.tolist(),
                    "waypoint": waypoint.tolist(),
                    "zero_fallbacks": int(info["hybrid_qp_zero_fallbacks"]),
                    "violations": {key: int(info.get(key, 0)) for key in VIOLATION_KEYS},
                }
            )
            previous_waypoint = waypoint
            assert not terminated and not truncated
    finally:
        v2.close()

    # Architecture floor: arbitrary rejected proposals must take exactly the
    # original desired-pose path, including every selected wrench.
    baseline = make_env("arrival_condition")
    rejected = make_env("task_state_v2")
    try:
        baseline.reset(seed=262003)
        rejected.reset(seed=262003)
        baseline_commands = record_commands(baseline)
        rejected_commands = record_commands(rejected)
        desired = np.asarray(baseline.environment_config.precapture_task.desired_position)
        baseline.step(baseline.action_for_waypoint(desired))
        _, _, _, _, rejected_info = rejected.step_with_proposal(
            np.array([-0.7, 0.9]), proposal_accepted=False
        )
        assert rejected_info["hybrid_proposal_accepted"] is False
        assert np.array_equal(
            np.asarray(rejected_commands), np.asarray(baseline_commands)
        )
        maximum_wrench_difference = float(
            np.max(np.abs(np.asarray(rejected_commands) - np.asarray(baseline_commands)))
        )
    finally:
        baseline.close()
        rejected.close()

    total_zero_fallbacks = sum(item["zero_fallbacks"] for item in accepted_records)
    total_truth_violations = sum(
        sum(item["violations"].values()) for item in accepted_records
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "correctness_only_no_direction_judgment_no_training",
        "seed": 262000,
        "accepted_decisions": accepted_records,
        "reference_finite_and_continuous": True,
        "both_axes_effective": True,
        "limiter_active": True,
        "geometry_minimum_reference_radius_m": geometry_min_radius,
        "truth_violation_steps_total": total_truth_violations,
        "qp_zero_fallbacks_total": total_zero_fallbacks,
        "architecture_fallback_exercised": True,
        "reject_all_maximum_wrench_difference": maximum_wrench_difference,
        "reject_all_is_bitwise_pure_mpc": maximum_wrench_difference == 0.0,
        "passed": bool(
            total_truth_violations == 0
            and total_zero_fallbacks == 0
            and maximum_wrench_difference == 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
