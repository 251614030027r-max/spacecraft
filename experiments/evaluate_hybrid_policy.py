"""Evaluate a trained coupled policy on the same block, through the same metrics.

This is the row the paper's table takes from. It emits the shared
``eval.metrics.main_table_metrics`` block, so the coupled row is assembled by
the same code as the Pure MPC and scripted rows and cannot drift into a private
convention.

Compute accounting, which is the column most easily got wrong here: the MPC
still runs once per 0.1 s control step, and the learned layer runs once per
decision. So the per-control-step controller cost is the MPC solve, plus the
policy's forward pass charged to the first control step of its decision. That
is what a flight computer would actually pay.

**Run this serially, single process.** A parallel run's timings are not a
real-time claim and must not be reported as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc.prediction import relative_to_vector
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from eval.metrics import PRECAPTURE_MARGIN_KEYS, main_table_metrics
from train.hybrid_configs import SAC_MPC_HYBRID

VIOLATION_STEP_KEYS = (
    "keepout_violation_steps",
    "fov_violation_steps",
    "outer_speed_violation_steps",
    "outer_radial_violation_steps",
    "corridor_violation_steps",
    "total_speed_violation_steps",
    "closing_speed_violation_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--parametrization",
        choices=["absolute", "radial_local"],
        default="radial_local",
        help="Action parametrisation used by the trained policy or control.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Trained SAC checkpoint. Omit to run one of the controls.",
    )
    parser.add_argument(
        "--control",
        choices=["desired_pose", "random"],
        help=(
            "Run without a model. 'desired_pose' is the fixed-setpoint lower "
            "layer delivered through the wrapper -- the row the coupled policy "
            "has to beat. 'random' is the floor."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.model is None) == (args.control is None):
        raise ValueError("give exactly one of --model or --control")

    policy = None
    if args.model is not None:
        from stable_baselines3 import SAC

        policy = SAC.load(args.model, device="cpu")

    records: list[dict[str, Any]] = []
    controller_times_s: list[float] = []
    environment_times_s: list[float] = []
    generator = np.random.default_rng(args.seed)

    for episode in range(args.episodes):
        seed = args.seed + episode
        env = PrecaptureHybridEnv(
            hybrid_config=PrecaptureHybridConfig(
                horizon_steps=args.horizon,
                waypoint_parametrization=args.parametrization,
            )
        )
        observation, info = env.reset(seed=seed)
        desired = env.environment_config.precapture_task.desired_position
        minimum_margins = {
            key: float(info[key]) for key in PRECAPTURE_MARGIN_KEYS if key in info
        }
        force_impulse = torque_impulse = 0.0
        waypoints: list[list[float]] = []
        qp_infeasible_steps_total = 0
        first_infeasible_time_s: float | None = None
        consecutive_zero_wrench_steps = 0
        max_consecutive_zero_wrench_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if policy is not None:
                started = perf_counter()
                action, _ = policy.predict(observation, deterministic=True)
                inference_s = perf_counter() - started
            elif args.control == "desired_pose":
                started = perf_counter()
                action = env.action_for_waypoint(desired)
                inference_s = perf_counter() - started
            else:
                started = perf_counter()
                action = generator.uniform(-1.0, 1.0, size=env.action_space.shape)
                inference_s = perf_counter() - started

            waypoint = env.waypoint_from_action(action)
            waypoints.append([float(v) for v in waypoint])
            # Inline the decision so each control step's cost is separable.
            for index in range(env.hybrid_config.decision_period_steps):
                assert env.env.relative is not None
                assert env.env.target_state is not None
                started = perf_counter()
                wrench, diagnostics = env.controller.command(
                    relative_to_vector(env.env.relative),
                    target_state=env.env.target_state,
                    time_seconds=env.env.time_seconds,
                    terminal_latched=bool(info["terminal_region_active"]),
                    external_reference=waypoint,
                )
                controller_s = perf_counter() - started
                if str(diagnostics.status).startswith("infeasible"):
                    qp_infeasible_steps_total += 1
                    if first_infeasible_time_s is None:
                        first_infeasible_time_s = float(env.env.time_seconds)
                if np.count_nonzero(wrench) == 0:
                    consecutive_zero_wrench_steps += 1
                    max_consecutive_zero_wrench_steps = max(
                        max_consecutive_zero_wrench_steps,
                        consecutive_zero_wrench_steps,
                    )
                else:
                    consecutive_zero_wrench_steps = 0
                # The policy is charged to the first control step of its
                # decision, which is when a flight computer would pay it.
                controller_times_s.append(
                    controller_s + (inference_s if index == 0 else 0.0)
                )
                force_impulse += (
                    float(np.linalg.norm(wrench[3:])) * env.environment_config.dt_s
                )
                torque_impulse += (
                    float(np.linalg.norm(wrench[:3])) * env.environment_config.dt_s
                )
                started = perf_counter()
                observation, _, terminated, truncated, info = env.env.step(
                    wrench_to_normalized(
                        GeneralizedForce.from_vector(wrench),
                        max_torque_per_axis_nm=(
                            env.environment_config.max_torque_per_axis_nm
                        ),
                        max_force_per_axis_n=(
                            env.environment_config.max_force_per_axis_n
                        ),
                    )
                )
                environment_times_s.append(perf_counter() - started)
                for key in PRECAPTURE_MARGIN_KEYS:
                    if key in info:
                        minimum_margins[key] = min(
                            minimum_margins.get(key, float(info[key])),
                            float(info[key]),
                        )
                if terminated or truncated:
                    break

        zero_violation = all(
            int(info[key]) == 0 for key in VIOLATION_STEP_KEYS if key in info
        )
        records.append(
            {
                "episode": episode,
                "seed": seed,
                "completed": bool(info["completed"]),
                "truth_geometry_zero_violation_completed": bool(
                    info["completed"] and zero_violation
                ),
                "constraint_violated": not zero_violation,
                "survival_s": float(env.env.time_seconds),
                "steps": int(env.env.step_count),
                "decisions": len(waypoints),
                "force_impulse_n_s": force_impulse,
                "torque_impulse_nm_s": torque_impulse,
                "equivalent_delta_v_m_s": (
                    force_impulse / env.env.chaser_parameters.mass
                ),
                "minimum_margins": minimum_margins,
                "illegal_terminal_entry_count": int(
                    info.get("illegal_terminal_entry_count", 0)
                ),
                "terminal_region_active": float(info["terminal_region_active"]),
                "time_failure": bool(info.get("time_failure", False)),
                "distance_failure": bool(info.get("distance_failure", False)),
                "qp_infeasible_steps_total": qp_infeasible_steps_total,
                "first_infeasible_time_s": first_infeasible_time_s,
                "max_consecutive_zero_wrench_steps": (
                    max_consecutive_zero_wrench_steps
                ),
                "waypoints_target_frame": waypoints,
            }
        )
        env.close()
        last = records[-1]
        print(
            f"episode {episode + 1}/{args.episodes} seed={seed} "
            f"completed={last['completed']} t={last['survival_s']:.1f}s "
            f"dv={last['equivalent_delta_v_m_s']:.3f} "
            f"illegal={last['illegal_terminal_entry_count']}"
        )

    payload = {
        "source": str(args.model) if args.model is not None else args.control,
        "episodes": args.episodes,
        "seed_block": args.seed,
        "horizon": args.horizon,
        "waypoint_parametrization": args.parametrization,
        "hyperparameters": {"gamma": SAC_MPC_HYBRID.gamma},
        "compute_note": (
            "valid only if this ran serially in a single process; the MPC is "
            "charged every control step and the policy once per decision"
        ),
        "records": records,
        "main_table": main_table_metrics(
            records,
            controller_times_s=controller_times_s,
            environment_step_times_s=environment_times_s,
            control_period_s=0.1,
        ),
    }
    args.output.write_text(json.dumps(payload, indent=1))
    table = payload["main_table"]
    print(
        f"\ncompleted {table['completed_episodes']}/{table['episodes']}  "
        f"controller p95/budget "
        f"{table['per_step_compute_s']['controller_p95_over_budget']:.2f}x"
    )


if __name__ == "__main__":
    main()
