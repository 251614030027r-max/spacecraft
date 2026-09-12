"""Scripted upper layers inside the frozen `arrival_condition` action space.

This is the headroom question, asked without training: **on a lower layer that
is semantically correct, does the choice of when to commit still decide the
outcome?**

The T6 action is two-dimensional, so the useful scripts are actions, not
inverted waypoints:

``commit_now``
    ``a = (+1, 0)`` on every decision. The wrapper broadcasts the desired pose
    unchanged, so this is the fixed-setpoint Pure MPC controller, bitwise. It
    is the capability floor and the control row.

``commit_at``
    ``a = (-1, hold_radius_action)`` until ``--commit-time-s``, then
    ``a = (+1, 0)``. The hold end of the blend is the *inertially frozen*
    point, which is what waiting has to mean here -- a fixed target-frame point
    co-rotates with the body-fixed approach axis and no entry window ever
    opens.

Sweeping ``--commit-time-s`` over one seed answers whether *some* commit time
completes an episode the fixed setpoint fails. Sweeping it over the seed block
answers whether that is a property of the state, which is what an upper layer
would have to learn. Neither needs a trained policy.

Nothing here is a reportable main-table row except ``commit_now``, which is the
Pure MPC control.

Reproduce::

    python -B -m experiments.evaluate_arrival_condition_script --horizon 35 \\
        --policy commit_at --commit-time-s 40 \\
        --seeds 262000 262001 262002 --output logs/headroom/t40.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from controllers.mpc.prediction import relative_to_vector
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config


def scripted_action(
    policy: str,
    time_seconds: float,
    commit_time_s: float,
    hold_radius_action: float,
    action_mean: float = 0.0,
    action_noise_std: float = 0.0,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    if policy == "commit_now":
        return np.array([1.0, 0.0], dtype=np.float64)
    if policy == "commit_at":
        if time_seconds < commit_time_s:
            return np.array([-1.0, hold_radius_action], dtype=np.float64)
        return np.array([1.0, 0.0], dtype=np.float64)
    if policy == "noisy_commit":
        # The dead-seed signature, reproduced without training: a_commit sits at
        # an interior mean with the variation a deterministic policy actually
        # showed (mean 0.78-0.86, std 0.09). Under the raw blend that walks the
        # setpoint 0.6 m per decision against 0.094 m of authority.
        assert generator is not None
        noise = float(generator.normal(0.0, action_noise_std)) if action_noise_std else 0.0
        return np.array(
            [float(np.clip(action_mean + noise, -1.0, 1.0)), hold_radius_action],
            dtype=np.float64,
        )
    raise ValueError(f"unknown policy {policy!r}")


def run_episode(
    seed: int,
    horizon: int,
    policy: str,
    commit_time_s: float,
    hold_radius_action: float,
    action_mean: float = 0.0,
    action_noise_std: float = 0.0,
    monotone_commit: bool = True,
) -> dict[str, Any]:
    env = PrecaptureHybridEnv(
        environment_config=precapture_planning_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=horizon,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
            monotone_commit=monotone_commit,
        ),
    )
    generator = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    infeasible_steps = 0
    first_infeasible_time_s: float | None = None
    consecutive_zero = 0
    max_consecutive_zero = 0
    force_impulse = 0.0
    torque_impulse = 0.0
    steps = 0
    waypoint_radii: list[float] = []
    waypoint_steps: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = scripted_action(
            policy,
            float(env.env.time_seconds),
            commit_time_s,
            hold_radius_action,
            action_mean,
            action_noise_std,
            generator,
        )
        waypoint = env.waypoint_from_action(action)
        radius = float(np.linalg.norm(waypoint))
        if waypoint_radii:
            waypoint_steps.append(abs(radius - waypoint_radii[-1]))
        waypoint_radii.append(round(radius, 4))
        for _ in range(env.hybrid_config.decision_period_steps):
            assert env.env.relative is not None
            assert env.env.target_state is not None
            wrench, diagnostics = env.controller.command(
                relative_to_vector(env.env.relative),
                target_state=env.env.target_state,
                time_seconds=env.env.time_seconds,
                terminal_latched=bool(info["terminal_region_active"]),
                external_reference=waypoint,
            )
            if str(diagnostics.status).startswith("infeasible"):
                infeasible_steps += 1
                if first_infeasible_time_s is None:
                    first_infeasible_time_s = round(float(env.env.time_seconds), 1)
            if np.count_nonzero(wrench) == 0:
                consecutive_zero += 1
                max_consecutive_zero = max(max_consecutive_zero, consecutive_zero)
            else:
                consecutive_zero = 0
            force_impulse += (
                float(np.linalg.norm(wrench[3:])) * env.environment_config.dt_s
            )
            torque_impulse += (
                float(np.linalg.norm(wrench[:3])) * env.environment_config.dt_s
            )
            _, _, terminated, truncated, info = env.env.step(
                wrench_to_normalized(
                    GeneralizedForce.from_vector(wrench),
                    max_torque_per_axis_nm=(
                        env.environment_config.max_torque_per_axis_nm
                    ),
                    max_force_per_axis_n=env.environment_config.max_force_per_axis_n,
                )
            )
            steps += 1
            if terminated or truncated:
                break
    return {
        "seed": seed,
        "horizon": horizon,
        "policy": policy,
        "monotone_commit": monotone_commit,
        "action_mean": action_mean,
        "action_noise_std": action_noise_std,
        "commit_time_s": commit_time_s,
        "hold_radius_action": hold_radius_action,
        "completed": bool(info["completed"]),
        "end_time_s": round(float(env.env.time_seconds), 2),
        "steps": steps,
        "qp_infeasible_steps": infeasible_steps,
        "first_infeasible_time_s": first_infeasible_time_s,
        "max_consecutive_zero_wrench_steps": max_consecutive_zero,
        "force_impulse_n_s": round(force_impulse, 3),
        "torque_impulse_nm_s": round(torque_impulse, 4),
        "illegal_terminal_entry_count": int(info["illegal_terminal_entry_count"]),
        "terminal_region_active": bool(info["terminal_region_active"]),
        "final_range_m": round(float(info["target_center_distance_m"]), 3),
        "final_fov_margin_rad": round(float(info["fov_margin_rad"]), 4),
        "time_failure": bool(info["time_failure"]),
        "distance_failure": bool(info["distance_failure"]),
        "median_waypoint_step_m": (
            round(float(np.median(waypoint_steps)), 4) if waypoint_steps else None
        ),
        "max_waypoint_step_m": (
            round(float(np.max(waypoint_steps)), 4) if waypoint_steps else None
        ),
        "first_waypoint_radius_m": waypoint_radii[0] if waypoint_radii else None,
        "last_waypoint_radius_m": waypoint_radii[-1] if waypoint_radii else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=35)
    parser.add_argument(
        "--policy",
        choices=["commit_now", "commit_at", "noisy_commit"],
        required=True,
    )
    parser.add_argument("--action-mean", type=float, default=0.0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--no-monotone-commit",
        dest="monotone_commit",
        action="store_false",
        help="disable the commit ratchet, reproducing the pre-calibration interface",
    )
    parser.set_defaults(monotone_commit=True)
    parser.add_argument(
        "--commit-time-s",
        type=float,
        default=0.0,
        help="commit_at only: hold the inertially frozen point until this time",
    )
    parser.add_argument(
        "--hold-radius-action",
        type=float,
        default=0.0,
        help="a[1] during the hold, in [-1, 1]; scales the frozen hold radius",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not -1.0 <= args.hold_radius_action <= 1.0:
        raise ValueError("hold radius action must lie in [-1, 1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        record = run_episode(
            seed,
            args.horizon,
            args.policy,
            args.commit_time_s,
            args.hold_radius_action,
            args.action_mean,
            args.action_noise_std,
            args.monotone_commit,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps(records, indent=2))
    completed = sum(record["completed"] for record in records)
    print(
        f"{args.policy} t={args.commit_time_s} h{args.horizon}: "
        f"{completed}/{len(records)} completed",
        flush=True,
    )


if __name__ == "__main__":
    main()
