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
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc.prediction import relative_to_vector
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config
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


def summarize_entry_channel(
    records: list[dict[str, Any]], entry_limits: dict[str, float]
) -> dict[str, Any]:
    """Aggregate the pre-registered entry/retry questions from episode records."""

    crossings = [event for record in records for event in record["entry_crossings"]]
    illegal = [event for event in crossings if not event["legal"]]
    retries = [event for event in crossings if event["retry_after_illegal"]]

    def fraction_with_positive_excess(key: str) -> float | None:
        if not illegal:
            return None
        return sum(event["violation_excess"][key] > 0.0 for event in illegal) / len(
            illegal
        )

    violation_fractions = {
        "radial_distance": fraction_with_positive_excess("radial_distance_m"),
        "target_frame_speed": fraction_with_positive_excess(
            "target_frame_speed_m_s"
        ),
        "closing_speed": fraction_with_positive_excess("closing_speed_m_s"),
    }
    available = {
        name: value for name, value in violation_fractions.items() if value is not None
    }
    primary = "no_illegal_crossing"
    if available:
        candidate = max(available, key=available.__getitem__)
        primary = candidate if available[candidate] > 0.60 else "mixed_no_cause_above_60pct"

    episodes_with_illegal = [
        record for record in records if record["illegal_entry_crossing_count"] > 0
    ]
    episodes_with_exit = [
        record for record in records if record["illegal_exit_count"] > 0
    ]
    episodes_with_retry = [
        record for record in records if record["retry_crossing_count"] > 0
    ]
    return {
        "entry_limits": entry_limits,
        "crossings_total": len(crossings),
        "legal_crossings": sum(event["legal"] for event in crossings),
        "illegal_crossings": len(illegal),
        "illegal_violation_fractions": violation_fractions,
        "primary_illegal_cause_by_preregistered_60pct_rule": primary,
        "episodes_with_illegal_crossing": len(episodes_with_illegal),
        "episodes_with_exit_after_illegal": len(episodes_with_exit),
        "episodes_with_exit_and_retry": len(episodes_with_retry),
        "exit_and_retry_fraction_all_episodes": len(episodes_with_retry)
        / max(1, len(records)),
        "exit_and_retry_fraction_illegal_episodes": len(episodes_with_retry)
        / max(1, len(episodes_with_illegal)),
        "retry_crossings": len(retries),
        "legal_retry_crossings": sum(event["legal"] for event in retries),
        "legal_retry_fraction": (
            sum(event["legal"] for event in retries) / len(retries) if retries else None
        ),
        "retry_remaining_time_s": [
            event["remaining_time_s"] for event in retries
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--parametrization",
        choices=["absolute", "radial_local", "arrival_condition"],
        default="radial_local",
        help="Action parametrisation used by the trained policy or control.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Trained SAC checkpoint. Omit to run one of the controls.",
    )
    parser.add_argument(
        "--phase-time-observation",
        action="store_true",
        help="Use the V4 31D observation expected by phase/time-trained policies.",
    )
    parser.add_argument(
        "--execution-feedback",
        action="store_true",
        help=(
            "Append the lower layer's 3D execution summary, as the coupled "
            "policies trained with the reverse channel expect. A policy is "
            "loaded against the observation it was trained on, so this has to "
            "match the run's manifest rather than be guessed."
        ),
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
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
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
    environment_config = replace(
        precapture_planning_environment_config(), cache_target_trajectory=False
    )
    task = environment_config.precapture_task
    entry_position = (
        task.port_position
        + task.entry_port_axial_distance_m * task.approach_axis
    )
    entry_axial_remaining = float(
        task.approach_axis @ (entry_position - task.desired_position)
    )
    entry_limits = {
        "radial_distance_m": float(task.entry_disc_radius_m),
        "target_frame_speed_m_s": float(task.terminal_total_speed_limit_m_s),
        "closing_speed_m_s": float(task.closing_speed_limit(entry_axial_remaining)),
    }

    for episode in range(args.episodes):
        seed = args.seed + episode
        env = PrecaptureHybridEnv(
            environment_config=environment_config,
            hybrid_config=PrecaptureHybridConfig(
                horizon_steps=args.horizon,
                waypoint_parametrization=args.parametrization,
                runtime_diagnostics=False,
                include_target_phase_and_time_observation=(
                    args.phase_time_observation
                ),
                include_execution_feedback_observation=args.execution_feedback,
            )
        )
        if policy is not None and episode == 0:
            expected = int(np.prod(policy.observation_space.shape))
            actual = int(np.prod(env.observation_space.shape))
            if expected != actual:
                raise ValueError(
                    f"the checkpoint expects a {expected}D observation and this "
                    f"configuration builds a {actual}D one. Set "
                    "--phase-time-observation / --execution-feedback to match "
                    "the run's manifest instead of loading a policy against an "
                    "observation it never saw."
                )
            expected_action = int(np.prod(policy.action_space.shape))
            actual_action = int(np.prod(env.action_space.shape))
            if expected_action != actual_action:
                raise ValueError(
                    f"the checkpoint expects a {expected_action}D action and "
                    f"this parametrisation gives {actual_action}D"
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
        entry_crossings: list[dict[str, Any]] = []
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
                observation = env._policy_observation(observation)
                environment_times_s.append(perf_counter() - started)
                # An illegal crossing does not latch.  Once the chaser returns
                # outside the entry plane, the next outside-to-inside crossing
                # is a directly observed retry rather than an inferred intent.
                for prior in entry_crossings:
                    if (
                        not prior["legal"]
                        and prior["exit_time_s"] is None
                        and float(info["port_axial_distance_m"])
                        > task.entry_port_axial_distance_m
                    ):
                        prior["exit_time_s"] = float(env.env.time_seconds)
                if bool(info.get("terminal_entry_crossed", False)):
                    radial = float(info["entry_crossing_radial_distance_m"])
                    target_speed = float(
                        info["entry_crossing_target_frame_speed_m_s"]
                    )
                    closing_speed = float(info["entry_crossing_closing_speed_m_s"])
                    retry_of = next(
                        (
                            prior
                            for prior in reversed(entry_crossings)
                            if not prior["legal"]
                            and prior["exit_time_s"] is not None
                            and prior["next_crossing_attempt"] is None
                        ),
                        None,
                    )
                    event = {
                        "attempt": len(entry_crossings) + 1,
                        "time_s": float(env.env.time_seconds),
                        "remaining_time_s": float(
                            environment_config.max_time_s - env.env.time_seconds
                        ),
                        "legal": bool(info["terminal_entry_legal"]),
                        "radial_distance_m": radial,
                        "target_frame_speed_m_s": target_speed,
                        "closing_speed_m_s": closing_speed,
                        "violation_excess": {
                            "radial_distance_m": radial
                            - entry_limits["radial_distance_m"],
                            "target_frame_speed_m_s": target_speed
                            - entry_limits["target_frame_speed_m_s"],
                            "closing_speed_m_s": closing_speed
                            - entry_limits["closing_speed_m_s"],
                        },
                        "retry_after_illegal": retry_of is not None,
                        "retry_of_attempt": (
                            retry_of["attempt"] if retry_of is not None else None
                        ),
                        "exit_time_s": None,
                        "next_crossing_attempt": None,
                        "next_crossing_legal": None,
                    }
                    if retry_of is not None:
                        retry_of["next_crossing_attempt"] = event["attempt"]
                        retry_of["next_crossing_legal"] = event["legal"]
                    entry_crossings.append(event)
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
                "entry_crossings": entry_crossings,
                "entry_crossing_count": len(entry_crossings),
                "legal_entry_crossing_count": sum(
                    event["legal"] for event in entry_crossings
                ),
                "illegal_entry_crossing_count": sum(
                    not event["legal"] for event in entry_crossings
                ),
                "illegal_exit_count": sum(
                    not event["legal"] and event["exit_time_s"] is not None
                    for event in entry_crossings
                ),
                "retry_crossing_count": sum(
                    event["retry_after_illegal"] for event in entry_crossings
                ),
                "retry_legal_count": sum(
                    event["retry_after_illegal"] and event["legal"]
                    for event in entry_crossings
                ),
                "final_remaining_time_s": float(
                    environment_config.max_time_s - env.env.time_seconds
                ),
                "final_target_center_distance_m": float(
                    info["target_center_distance_m"]
                ),
                "final_position_error_m": float(info["position_error_m"]),
                "waypoints_target_frame": waypoints,
            }
        )
        partial_output.parent.mkdir(parents=True, exist_ok=True)
        partial_output.write_text(
            json.dumps(
                {
                    "source": str(args.model) if args.model is not None else args.control,
                    "episodes_finished": len(records),
                    "entry_limits": entry_limits,
                    "entry_channel": summarize_entry_channel(records, entry_limits),
                    "records": records,
                },
                indent=1,
            )
        )
        env.close()
        last = records[-1]
        print(
            f"episode {episode + 1}/{args.episodes} seed={seed} "
            f"completed={last['completed']} t={last['survival_s']:.1f}s "
            f"dv={last['equivalent_delta_v_m_s']:.3f} "
            f"illegal={last['illegal_terminal_entry_count']} "
            f"crossings={last['entry_crossing_count']} "
            f"retries={last['retry_crossing_count']}"
        )

    payload = {
        "source": str(args.model) if args.model is not None else args.control,
        "episodes": args.episodes,
        "seed_block": args.seed,
        "horizon": args.horizon,
        "waypoint_parametrization": args.parametrization,
        "phase_time_observation": args.phase_time_observation,
        "execution_feedback": args.execution_feedback,
        "hyperparameters": {"gamma": SAC_MPC_HYBRID.gamma},
        "compute_note": (
            "valid only if this ran serially in a single process; the MPC is "
            "charged every control step and the policy once per decision; "
            "post-solve runtime diagnostics are disabled"
        ),
        "records": records,
        "entry_channel": summarize_entry_channel(records, entry_limits),
        "main_table": main_table_metrics(
            records,
            controller_times_s=controller_times_s,
            environment_step_times_s=environment_times_s,
            control_period_s=0.1,
        ),
    }
    args.output.write_text(json.dumps(payload, indent=1))
    partial_output.unlink(missing_ok=True)
    table = payload["main_table"]
    print(
        f"\ncompleted {table['completed_episodes']}/{table['episodes']}  "
        f"controller p95/budget "
        f"{table['per_step_compute_s']['controller_p95_over_budget']:.2f}x"
    )


if __name__ == "__main__":
    main()
