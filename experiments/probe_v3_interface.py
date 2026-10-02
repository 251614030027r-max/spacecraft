"""Training-free acceptance probes for the V3 task-state interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


CONTINUITY_CASES = (
    (262410, 262004),
    (262412, 262004),
    (262411, 262034),
    (262410, 262010),
)


def _recover_hold_directions(
    references: np.ndarray,
    commitments: np.ndarray,
    desired_direction: np.ndarray,
) -> np.ndarray:
    holds: list[np.ndarray] = []
    for reference, commitment in zip(references, commitments, strict=True):
        direction = reference / np.linalg.norm(reference)
        beta = float(
            np.arccos(np.clip(direction @ desired_direction, -1.0, 1.0))
        )
        alpha = beta / (1.0 - float(commitment))
        tangent = direction - float(direction @ desired_direction) * desired_direction
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-10:
            basis = np.zeros(3)
            basis[int(np.argmin(np.abs(desired_direction)))] = 1.0
            tangent = basis - float(basis @ desired_direction) * desired_direction
            tangent_norm = float(np.linalg.norm(tangent))
        tangent /= tangent_norm
        holds.append(
            np.cos(alpha) * desired_direction + np.sin(alpha) * tangent
        )
    return np.asarray(holds)


def continuity_probe(raw_dir: Path) -> list[dict[str, Any]]:
    environment = precapture_adaptive_capture_environment_config()
    desired = np.asarray(environment.precapture_task.desired_position, dtype=float)
    desired_radius = float(np.linalg.norm(desired))
    desired_direction = desired / desired_radius
    env = PrecaptureHybridEnv(
        environment_config=environment,
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="task_state_v3", runtime_diagnostics=False
        ),
    )
    env.reset(seed=262004)
    rows: list[dict[str, Any]] = []
    try:
        for model_seed, episode_seed in CONTINUITY_CASES:
            payload = json.loads(
                (raw_dir / f"final_v2_nominal_{model_seed}.json").read_text()
            )
            record = next(
                row for row in payload["records"] if row["seed"] == episode_seed
            )
            states = np.asarray(record["task_state_applied"], dtype=float)
            v2_references = np.asarray(record["waypoints_target_frame"], dtype=float)
            holds = _recover_hold_directions(
                v2_references, states[:, 1], desired_direction
            )
            hold_radius = float(
                np.median(np.linalg.norm(v2_references, axis=1) + states[:, 0])
            )
            v3_references: list[np.ndarray] = []
            applied_direction: np.ndarray | None = None
            previous_hold: np.ndarray | None = None
            previous_radius: float | None = None
            for hold, (progress_m, commitment) in zip(holds, states, strict=True):
                radius = max(hold_radius - float(progress_m), desired_radius)
                weight = float(np.clip(commitment, 0.0, 1.0))
                cosine = float(np.clip(hold @ desired_direction, -1.0, 1.0))
                if cosine < -1.0 + 1.0e-10:
                    axis = env._fallback_blend_axis(hold)
                    angle = np.pi * weight
                    goal = np.cos(angle) * hold + np.sin(angle) * np.cross(axis, hold)
                else:
                    angle = float(np.arccos(cosine))
                    if angle < 1.0e-8:
                        goal = desired_direction if weight >= 1.0 else hold
                    else:
                        goal = (
                            np.sin((1.0 - weight) * angle) * hold
                            + np.sin(weight * angle) * desired_direction
                        ) / np.sin(angle)
                goal /= np.linalg.norm(goal)
                if applied_direction is None:
                    applied_direction = goal
                else:
                    assert previous_hold is not None and previous_radius is not None
                    natural_angle = 1.2 * float(
                        np.arccos(np.clip(previous_hold @ hold, -1.0, 1.0))
                    )
                    radial_change = abs(radius - previous_radius)
                    maximum_angle = natural_angle + max(
                        0.0, 0.40 - radial_change
                    ) / radius
                    applied_direction = env._direction_step(
                        applied_direction, goal, maximum_angle
                    )
                v3_references.append(radius * applied_direction)
                previous_hold = hold
                previous_radius = radius
            v3_array = np.asarray(v3_references)
            rows.append(
                {
                    "model_seed": model_seed,
                    "episode_seed": episode_seed,
                    "v2_max_jump_m": float(
                        np.max(np.linalg.norm(np.diff(v2_references, axis=0), axis=1))
                    ),
                    "v3_max_jump_m": float(
                        np.max(np.linalg.norm(np.diff(v3_array, axis=0), axis=1))
                    ),
                    "natural_hold_max_jump_m": float(
                        np.max(
                            np.linalg.norm(
                                np.diff(hold_radius * holds, axis=0), axis=1
                            )
                        )
                    ),
                }
            )
    finally:
        env.close()
    return rows


def scripted_probe(
    episodes: int, horizon: int, seed_start: int = 262000
) -> list[dict[str, Any]]:
    environment = replace(
        precapture_adaptive_capture_environment_config(),
        cache_target_trajectory=False,
    )
    rows: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + episodes):
        env = PrecaptureHybridEnv(
            environment_config=environment,
            hybrid_config=PrecaptureHybridConfig(
                horizon_steps=horizon,
                waypoint_parametrization="task_state_v3",
                runtime_diagnostics=False,
                include_target_phase_and_time_observation=True,
                include_execution_feedback_observation=True,
                include_staging_direction_observation=True,
            ),
        )
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        decisions = fallbacks = 0
        maximum_jump = 0.0
        violations = 0
        finite = True
        try:
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(
                    np.array([0.5, 0.5], dtype=np.float64)
                )
                finite &= bool(np.all(np.isfinite(observation)) and np.isfinite(reward))
                finite &= bool(
                    np.all(np.isfinite(info["hybrid_waypoint_target_frame"]))
                )
                maximum_jump = max(
                    maximum_jump, float(info["reference_jump_target_m"])
                )
                violations += int(info["reference_jump_target_violation"])
                fallbacks += int(info["hybrid_qp_zero_fallbacks"])
                decisions += 1
        finally:
            env.close()
        rows.append(
            {
                "seed": seed,
                "decisions": decisions,
                "completed": bool(info["completed"]),
                "finite": finite,
                "qp_zero_fallbacks": fallbacks,
                "reference_jump_target_max_m": maximum_jump,
                "reference_jump_target_violations": violations,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--scripted-episodes", type=int, default=8)
    parser.add_argument("--scripted-seed-start", type=int, default=262000)
    parser.add_argument("--horizon", type=int, default=35)
    parser.add_argument("--skip-scripted", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {}
    if args.raw_dir is not None:
        result["continuity"] = continuity_probe(args.raw_dir)
    if not args.skip_scripted:
        result["scripted"] = scripted_probe(
            args.scripted_episodes, args.horizon, args.scripted_seed_start
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))

    if any(
        not row["finite"] or row["reference_jump_target_violations"]
        for row in result.get("scripted", [])
    ):
        raise SystemExit("V3 scripted-arm acceptance failed")


if __name__ == "__main__":
    main()
