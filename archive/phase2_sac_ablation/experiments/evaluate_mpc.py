"""Evaluate the independent MPC-only controller on the final task envelope."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc import (
    LocalRelativePredictionModel,
    MPCConfig,
    MPCController,
    RelativePredictionModel,
    phase2_mpc_config,
)
from controllers.mpc.constraints import normalized_truth_margins
from controllers.mpc.prediction import relative_to_vector
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from env.scenarios import chaser_parameters, target_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MPC-only")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--outer-iterations", type=int)
    parser.add_argument("--input-weight", type=float)
    parser.add_argument("--terminal-weight", type=float)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--max-time", type=float)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values),
        "p95": ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def evaluate(
    config: MPCConfig,
    *,
    episodes: int,
    seed: int,
    environment_config: SE3RendezvousConfig | None = None,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env_config = environment_config or SE3RendezvousConfig(curriculum_enabled=False)
    if env_config.curriculum_enabled:
        raise ValueError("MPC evaluation must use a fixed, non-curriculum task")
    env = SE3RendezvousEnv(env_config)
    reference = RelativePredictionModel(
        target_parameters=target_parameters(),
        chaser_parameters=chaser_parameters(),
        dt_s=env_config.dt_s,
        gravity_options=GravityOptions(include_j2=env_config.include_j2),
        solver_settings=RK45Settings(
            rtol=env_config.solver_rtol,
            atol=env_config.solver_atol,
            max_step=env_config.dt_s,
        ),
    )
    local = LocalRelativePredictionModel(reference.chaser_parameters, env_config.dt_s)
    controller = MPCController(config, local, reference)
    records: list[dict[str, Any]] = []
    all_solve_times: list[float] = []
    all_command_times: list[float] = []
    total_fallbacks = 0
    predicted_safe_truth_violation_count = 0
    compared_constraint_steps = 0
    trajectories: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            _, info = env.reset(seed=seed + episode)
            controller.reset()
            force_impulse = 0.0
            torque_impulse = 0.0
            saturation_counts = np.zeros(6, dtype=np.int64)
            episode_solve_times: list[float] = []
            episode_command_times: list[float] = []
            episode_fallbacks = 0
            trace: list[dict[str, Any]] = []
            terminated = truncated = False
            while not (terminated or truncated):
                assert env.relative is not None and env.target_state is not None
                x = relative_to_vector(env.relative)
                command_started = perf_counter()
                wrench_vector, diagnostics = controller.command(
                    x,
                    target_state=env.target_state,
                    time_seconds=env.time_seconds,
                )
                episode_command_times.append(perf_counter() - command_started)
                action = wrench_to_normalized(
                    GeneralizedForce.from_vector(wrench_vector),
                    max_torque_per_axis_nm=env_config.max_torque_per_axis_nm,
                    max_force_per_axis_n=env_config.max_force_per_axis_n,
                )
                saturation_counts += np.abs(wrench_vector) >= config.input_scales - 1.0e-7
                torque_impulse += float(np.linalg.norm(wrench_vector[:3])) * env_config.dt_s
                force_impulse += float(np.linalg.norm(wrench_vector[3:])) * env_config.dt_s
                episode_solve_times.append(diagnostics.solve_time_s)
                episode_fallbacks += int(diagnostics.used_zero_fallback)
                if episode < 5 and env.step_count % 10 == 0:
                    trace.append(
                        {
                            "time_seconds": env.time_seconds,
                            "state": x.tolist(),
                            "control": wrench_vector.tolist(),
                            "predicted_cost": diagnostics.predicted_cost,
                            "solver_status": diagnostics.status,
                        }
                    )
                _, _, terminated, truncated, info = env.step(action)
                if config.task is not None:
                    assert env.relative is not None
                    actual_margins = normalized_truth_margins(
                        relative_to_vector(env.relative), config.task
                    )
                    predicted_margins = np.asarray(
                        diagnostics.predicted_first_step_margins
                    )
                    predicted_safe_truth_violation_count += int(
                        np.any((predicted_margins >= 0.0) & (actual_margins < 0.0))
                    )
                    compared_constraint_steps += 1
            all_solve_times.extend(episode_solve_times)
            all_command_times.extend(episode_command_times)
            total_fallbacks += episode_fallbacks
            records.append(
                {
                    "episode": episode,
                    "seed": seed + episode,
                    "completed": bool(info["completed"]),
                    "joint_success": bool(info["joint_success"]),
                    "attitude_success": bool(info["attitude_success"]),
                    "position_success": bool(info["position_success"]),
                    "distance_failure": bool(info["distance_failure"]),
                    "time_failure": bool(info["time_failure"]),
                    "steps": env.step_count,
                    "time_seconds": env.time_seconds,
                    "attitude_error_rad": float(info["attitude_error_rad"]),
                    "angular_velocity_error_rad_s": float(info["angular_velocity_error_rad_s"]),
                    "position_error_m": float(info["position_error_m"]),
                    "translational_velocity_error_m_s": float(info["translational_velocity_error_m_s"]),
                    "force_impulse_n_s": force_impulse,
                    "torque_impulse_nm_s": torque_impulse,
                    "saturation_fraction_per_axis": (
                        saturation_counts / max(1, env.step_count)
                    ).tolist(),
                    "qp_fallback_count": episode_fallbacks,
                    "solve_time_s": _summary(episode_solve_times),
                    "command_time_s": _summary(episode_command_times),
                    **(
                        {
                            "constraint_success": bool(info["constraint_success"]),
                            "corridor_violation_steps": int(
                                info["corridor_violation_steps"]
                            ),
                            "fov_violation_steps": int(info["fov_violation_steps"]),
                            "total_speed_violation_steps": int(
                                info["total_speed_violation_steps"]
                            ),
                            "closing_speed_violation_steps": int(
                                info["closing_speed_violation_steps"]
                            ),
                        }
                        if config.task is not None
                        else {}
                    ),
                }
            )
            if episode < 5:
                trajectories.append({"episode": episode, "seed": seed + episode, "samples": trace})
    finally:
        env.close()
    rate_keys = (
        "completed",
        "joint_success",
        "attitude_success",
        "position_success",
        "distance_failure",
        "time_failure",
    )
    rates = {key: mean(float(row[key]) for row in records) for key in rate_keys}
    if config.task is not None:
        rates["constraint_success"] = mean(
            float(row["constraint_success"]) for row in records
        )
    total_steps = sum(int(row["steps"]) for row in records)
    fallback_rate = total_fallbacks / max(1, total_steps)
    checks = {
        "at_least_100_episodes": episodes >= 100,
        "completed_rate_at_least_0p60": rates["completed"] >= 0.60,
        "distance_failure_rate_at_most_0p05": rates["distance_failure"] <= 0.05,
        "qp_fallback_rate_at_most_0p001": fallback_rate <= 0.001,
    }
    if config.task is not None:
        checks = {
            "development_seed_set": episodes >= 5,
            "completed_rate_at_least_0p80": rates["completed"] >= 0.80,
            "constraint_success_rate_at_least_0p80": (
                rates["constraint_success"] >= 0.80
            ),
            "qp_fallback_rate_at_most_0p02": fallback_rate <= 0.02,
            "no_predicted_safe_truth_violation": (
                predicted_safe_truth_violation_count == 0
            ),
        }
    return {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "mpc_only",
        "episodes": episodes,
        "base_seed": seed,
        "environment": asdict(env_config),
        "mpc_config": asdict(config),
        "rates": rates,
        "qp": {
            "fallback_count": total_fallbacks,
            "fallback_rate": fallback_rate,
            "solve_time_s": _summary(all_solve_times),
            "command_time_s": _summary(all_command_times),
            "predicted_safe_truth_violation_count": (
                predicted_safe_truth_violation_count
            ),
            "compared_constraint_steps": compared_constraint_steps,
        },
        "acceptance": {"checks": checks, "passed": all(checks.values())},
        "episode_records": records,
        "representative_trajectories": trajectories,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base_config = phase2_mpc_config() if args.phase2 else MPCConfig()
    config = replace(
        base_config,
        horizon_steps=(
            args.horizon if args.horizon is not None else (50 if args.phase2 else 20)
        ),
        outer_iterations=(
            args.outer_iterations
            if args.outer_iterations is not None
            else (1 if args.phase2 else 2)
        ),
        input_weight=(
            args.input_weight
            if args.input_weight is not None
            else (0.01 if args.phase2 else 0.02)
        ),
        terminal_weight=(
            args.terminal_weight
            if args.terminal_weight is not None
            else (100.0 if args.phase2 else 10.0)
        ),
    )
    environment_config = (
        SE3RendezvousConfig(
            phase2_enabled=True,
            curriculum_enabled=False,
            target_tumble_scale=0.25,
            max_time_s=(args.max_time if args.max_time is not None else 200.0),
        )
        if args.phase2
        else (
            SE3RendezvousConfig(
                curriculum_enabled=False,
                max_time_s=args.max_time,
            )
            if args.max_time is not None
            else None
        )
    )
    result = evaluate(
        config,
        episodes=args.episodes,
        seed=args.seed,
        environment_config=environment_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
    )
    args.output.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered)
    print(f"MPC evaluation result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
