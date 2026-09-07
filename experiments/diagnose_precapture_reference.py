"""Measure how well the precapture MPC tracks a 3D waypoint, and why it does not.

The declared learned-policy interface hands the MPC one target-centred,
inertially oriented 3D waypoint held for two seconds. Whether a coupled upper
layer can steer anything through that interface depends on a quantity nothing
in the evaluation path reported: how far the chaser actually sits from the
waypoint it was given. This records that lag per step, together with the
attitude quantities the field-of-view margin is a function of, so a reference
formulation can be judged on measurements rather than on argument.

Two settings are exposed because both were suspected of the same symptom:

``--attitude-reference``
    ``frozen`` (the default everywhere) aims one attitude from the current
    state and holds it across the horizon with a zero relative-rate reference.
    ``swept`` and ``aimed`` instead let the reference attitude follow the
    sightline as the reference position sweeps the target frame.
``--position-scale``
    The position entry of ``state_scales``, which normalises the tracking cost.
    The default 20 m is coarse against a task that runs from 17 m to 3 m.

Truth still drives the environment and every margin below; this changes only
what the controller is asked to track.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

from controllers.mpc import MPCController
from controllers.mpc.config import precapture_mpc_config
from controllers.mpc.prediction import (
    LocalRelativePredictionModel,
    RelativePredictionModel,
    relative_to_vector,
)
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.lie import se3_exp
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.phase2_env import precapture_planning_environment_config
from env.scenarios import chaser_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousEnv
from experiments.evaluate_precapture_oracle import CoastThenMatchPlan


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.arctan2(
            float(np.linalg.norm(np.cross(first, second))),
            float(np.clip(first @ second, -1.0, 1.0)),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--terminal-weight", type=float, default=1000.0)
    parser.add_argument("--input-weight", type=float, default=0.01)
    parser.add_argument(
        "--guidance",
        choices=["fixed", "oracle_plan", "desired_pose"],
        default="oracle_plan",
        help=(
            "'fixed' tracks the desired pose directly, with no upper layer. "
            "'desired_pose' sends that same point through the declared 3D "
            "waypoint channel instead, so the two differ only by the channel "
            "-- the 2 s hold, the inertial-to-target-frame mapping and the "
            "differenced velocity feed-forward. 'oracle_plan' sends the "
            "coast-then-match plan through the same channel."
        ),
    )
    parser.add_argument(
        "--attitude-reference",
        choices=["frozen", "swept", "aimed"],
        default="frozen",
    )
    parser.add_argument(
        "--waypoint-frame", choices=["inertial", "target"], default="inertial"
    )
    parser.add_argument("--position-scale", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    env_config = precapture_planning_environment_config()
    task = env_config.precapture_task
    config = replace(
        precapture_mpc_config(),
        horizon_steps=args.horizon,
        outer_iterations=1,
        input_weight=args.input_weight,
        terminal_weight=args.terminal_weight,
        reference_source=(
            "fixed" if args.guidance == "fixed" else "external_local"
        ),
        precapture_attitude_reference=args.attitude_reference,
        external_reference_frame=args.waypoint_frame,
        state_scales=np.array(
            [
                *([np.deg2rad(75.0)] * 3),
                *([args.position_scale] * 3),
                *([0.05] * 3),
                *([1.10] * 3),
            ],
            dtype=np.float64,
        ),
    )

    env = SE3RendezvousEnv(env_config)
    _, info = env.reset(seed=args.seed)
    reference_model = RelativePredictionModel(
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
    controller = MPCController(
        config,
        LocalRelativePredictionModel(
            reference_model.chaser_parameters, env_config.dt_s
        ),
        reference_model,
    )
    controller.reset()
    oracle_plan = (
        CoastThenMatchPlan(env) if args.guidance == "oracle_plan" else None
    )
    desired_position = task.desired_position

    port = task.port_position
    boresight_body = task.camera_boresight
    trace: list[dict[str, object]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        state = relative_to_vector(env.relative)
        latched = bool(info["terminal_region_active"])
        transform = se3_exp(state[:6])
        rotation, position = transform[:3, :3], transform[:3, 3]
        if oracle_plan is not None:
            # ``body_position`` is already a target-body-frame point; the
            # inertial contract needs it rotated out and the channel then
            # rotates it back, which is where the frame error enters.
            body_waypoint = oracle_plan.body_position(env.time_seconds)
            external_reference = (
                body_waypoint
                if args.waypoint_frame == "target"
                else env.target_state.rotation @ body_waypoint
            )
        elif args.guidance == "desired_pose":
            external_reference = (
                desired_position
                if args.waypoint_frame == "target"
                else env.target_state.rotation @ desired_position
            )
        else:
            external_reference = None
        reference = controller._reference_trajectory(
            state,
            env.time_seconds,
            target_state=env.target_state,
            external_reference=external_reference,
            terminal_latched=latched,
        )
        reference_position = se3_exp(reference[:6, 0])[:3, 3]
        reference_rotation = se3_exp(reference[:6, 0])[:3, :3]
        lag = reference_position - position
        radial = position / max(float(np.linalg.norm(position)), 1.0e-12)
        radial_lag = float(lag @ radial)

        wrench, diagnostics = controller.command(
            state,
            target_state=env.target_state,
            time_seconds=env.time_seconds,
            terminal_latched=latched,
            external_reference=external_reference,
        )
        trace.append(
            {
                "t": float(env.time_seconds),
                "range_m": float(np.linalg.norm(position)),
                "fov_angle_deg": float(np.rad2deg(info["fov_angle_rad"])),
                "reference_position_lag_m": float(np.linalg.norm(lag)),
                "reference_radial_lag_m": radial_lag,
                "reference_tangential_lag_m": float(
                    np.linalg.norm(lag - radial_lag * radial)
                ),
                "reference_attitude_error_deg": float(
                    np.rad2deg(
                        _angle_between(
                            rotation @ boresight_body,
                            reference_rotation @ boresight_body,
                        )
                    )
                ),
                "reference_angular_rate_rad_s": float(
                    np.linalg.norm(reference[6:9, -1])
                ),
                "relative_angular_rate_rad_s": float(np.linalg.norm(state[6:9])),
                "target_angular_rate_rad_s": float(
                    np.linalg.norm(env.target_state.omega)
                ),
                "torque_norm_nm": float(np.linalg.norm(wrench[:3])),
                "force_norm_n": float(np.linalg.norm(wrench[3:])),
                "maximum_slack": float(diagnostics.maximum_slack),
                "status": str(diagnostics.status),
            }
        )
        action = wrench_to_normalized(
            GeneralizedForce.from_vector(wrench),
            max_torque_per_axis_nm=env_config.max_torque_per_axis_nm,
            max_force_per_axis_n=env_config.max_force_per_axis_n,
        )
        _, _, terminated, truncated, info = env.step(action)

    def column(name: str) -> list[float]:
        return [float(sample[name]) for sample in trace]

    summary = {
        "seed": args.seed,
        "horizon": args.horizon,
        "guidance": args.guidance,
        "attitude_reference": args.attitude_reference,
        "position_scale_m": args.position_scale,
        "waypoint_frame": args.waypoint_frame,
        "steps": len(trace),
        "final_time_s": trace[-1]["t"],
        "final_range_m": trace[-1]["range_m"],
        "reference_position_lag_median_m": median(column("reference_position_lag_m")),
        "reference_position_lag_max_m": max(column("reference_position_lag_m")),
        "reference_tangential_lag_median_m": median(
            column("reference_tangential_lag_m")
        ),
        "fov_angle_max_deg": max(column("fov_angle_deg")),
        "torque_mean_nm": float(np.mean(column("torque_norm_nm"))),
        "force_mean_n": float(np.mean(column("force_norm_n"))),
        "completed": bool(info["completed"]),
        "terminal_region_active": float(info["terminal_region_active"]),
        "illegal_terminal_entry_count": float(info["illegal_terminal_entry_count"]),
        "violation_steps": {
            key: float(value)
            for key, value in info.items()
            if key.endswith("violation_steps") and float(value) > 0.0
        },
        "trace": trace,
    }
    args.output.write_text(json.dumps(summary, indent=1))
    print(
        f"seed={args.seed} mode={args.attitude_reference} frame={args.waypoint_frame} "
        f"scale={args.position_scale} "
        f"completed={summary['completed']} "
        f"t={summary['final_time_s']:.1f}s r={summary['final_range_m']:.2f}m "
        f"lag_med={summary['reference_position_lag_median_m']:.2f}m "
        f"fov_max={summary['fov_angle_max_deg']:.2f}deg"
    )


if __name__ == "__main__":
    main()
