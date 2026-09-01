"""Evaluate the independent MPC-only controller on the final task envelope."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc import (
    ConvexQuadraticTerminalValue,
    LocalRelativePredictionModel,
    MPCConfig,
    MPCController,
    RelativePredictionModel,
    constrained_mpc_nominal_config,
)
from controllers.mpc.constraints import normalized_truth_margins
from controllers.mpc.prediction import relative_to_vector
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.relative import relative_state
from dynamics.types import GeneralizedForce, SpacecraftParameters
from env.observation_error import TargetStateEstimator
from env.action import wrench_to_normalized
from env.phase2_env import (
    phase2_environment_config,
    terminal_phase_environment_config,
)
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from env.scenarios import chaser_parameters, target_parameters
from eval.metrics import main_table_metrics, summarize
from train.configs import PURE_SAC


# One discount for every reported return, shared with the learned rows and the
# scripted validators, so the main table is on a single scale.
GAMMA = PURE_SAC.gamma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MPC-only")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument(
        "--linearization-source",
        choices=("exact", "local"),
        default=None,
        help=(
            "exact refreshes an RK45-truth Jacobian every --exact-refresh-steps "
            "control steps (the ~250 ms refresh lump that sets the compute p95); "
            "local re-linearises the fast prediction model along the horizon with "
            "no truth refresh. This is the single factor of the refresh compute "
            "experiment: does dropping/lengthening the refresh bring p95 into "
            "budget without losing constraint margin?"
        ),
    )
    parser.add_argument(
        "--exact-refresh-steps",
        type=int,
        default=None,
        help="control steps between exact-linearisation refreshes (default 10); "
        "larger fires the refresh lump less often, trading model staleness for a "
        "lower compute p95",
    )
    parser.add_argument("--outer-iterations", type=int)
    parser.add_argument("--input-weight", type=float)
    parser.add_argument("--terminal-weight", type=float)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-time", type=float)
    parser.add_argument(
        "--task",
        choices=("terminal", "single_phase", "single_phase_phase_sampled"),
        default="terminal",
        help=(
            "terminal is the 2-10 m terminal-only shell the fixed-setpoint "
            "evidence was measured on; single_phase is the benchmark the SAC "
            "and scripted rows use (10-14 m, constrained from step one); "
            "single_phase_phase_sampled is that benchmark with the target's "
            "initial attitude and tumble direction sampled per episode (rate "
            "magnitude frozen) -- the route-A harder regime"
        ),
    )
    parser.add_argument(
        "--reference-source",
        choices=("fixed", "corridor_guidance"),
        default=None,
        help=(
            "fixed holds the reference at the desired pose; corridor_guidance "
            "tracks the shared guidance path. Left unset it defaults per task: "
            "corridor_guidance for single_phase (the fair benchmark reference) "
            "and fixed for terminal (the terminal-only record). Pass it only to "
            "override that default."
        ),
    )
    parser.add_argument(
        "--terminal-cost-source",
        choices=("fixed_quadratic", "learned_convex"),
        default="fixed_quadratic",
        help=(
            "fixed_quadratic keeps the diagonal terminal penalty (the Pure MPC "
            "baseline and the matched short-horizon control row); learned_convex "
            "uses the fitted convex terminal value from --terminal-value-file. "
            "This is the single interpretable factor of the coupling experiment."
        ),
    )
    parser.add_argument(
        "--terminal-value-file",
        type=Path,
        default=None,
        help="JSON terminal value from experiments.fit_terminal_value; required "
        "when --terminal-cost-source learned_convex",
    )
    parser.add_argument(
        "--target-model-mismatch",
        type=float,
        default=0.0,
        help="fractional half-range of the per-episode truth-vs-nominal target "
        "inertia/mass mismatch (0 = perfect model, 0.2 = +-20%%). The truth "
        "tumble uses the sampled parameters; the controller predicts with "
        "nominal unless --controller-model truth. The single factor of the "
        "model-mismatch probe.",
    )
    parser.add_argument(
        "--controller-model",
        choices=("nominal", "truth"),
        default="nominal",
        help="nominal: the controller predicts with the fixed nominal target "
        "parameters (the realistic non-cooperative case). truth: rebuild the "
        "controller each episode with the sampled truth parameters (feasibility "
        "control -- confirms the task is solvable with a perfect model).",
    )
    parser.add_argument(
        "--obs-bias-deg",
        type=float,
        default=0.0,
        help="partial-observability probe: per-episode constant target attitude "
        "estimation bias (degrees) fed to the controller. Truth is unchanged; "
        "violations are still judged on real geometry.",
    )
    parser.add_argument(
        "--obs-delay-steps",
        type=int,
        default=0,
        help="control steps the controller's target-pose estimate lags the truth",
    )
    parser.add_argument(
        "--obs-update-every",
        type=int,
        default=1,
        help="the controller's target-pose estimate refreshes every N control "
        "steps and is held in between (1 = every step)",
    )
    parser.add_argument(
        "--target-estimator",
        choices=("raw", "ekf"),
        default="raw",
        help="raw: the controller flies the corrupted sensor fix directly. ekf: "
        "an output-feedback baseline forward-propagates the delayed/held fix with "
        "a nominal target model (predict step), de-lagging it -- compensates "
        "delay/low-rate but not a constant attitude bias.",
    )
    parser.add_argument(
        "--corridor-speed-fraction",
        type=float,
        default=0.6,
        help="fraction of the total-speed limit the MPC's corridor reference "
        "aims for (default 0.6). Lower flies co-rotation more conservatively for "
        "more total_speed margin -- the observation-error margin control.",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float] | None:
    """One summary shape across every path that fills a main-table row."""

    return summarize(values)


def evaluate(
    config: MPCConfig,
    *,
    episodes: int,
    seed: int,
    environment_config: SE3RendezvousConfig | None = None,
    controller_model_source: str = "nominal",
    observation_bias_rad: float = 0.0,
    observation_delay_steps: int = 0,
    observation_update_every: int = 1,
    observation_filter: str = "hold",
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if controller_model_source not in {"nominal", "truth"}:
        raise ValueError("controller_model_source must be 'nominal' or 'truth'")
    observed_target = (
        observation_bias_rad > 0.0
        or observation_delay_steps > 0
        or observation_update_every > 1
    )
    env_config = environment_config or terminal_phase_environment_config()
    if env_config.curriculum_enabled or not env_config.phase2_enabled:
        raise ValueError("MPC evaluation requires canonical fixed Phase-2")
    env = SE3RendezvousEnv(env_config)

    def build_controller(target_params: SpacecraftParameters) -> MPCController:
        reference = RelativePredictionModel(
            target_parameters=target_params,
            chaser_parameters=chaser_parameters(),
            dt_s=env_config.dt_s,
            gravity_options=GravityOptions(include_j2=env_config.include_j2),
            solver_settings=RK45Settings(
                rtol=env_config.solver_rtol,
                atol=env_config.solver_atol,
                max_step=env_config.dt_s,
            ),
        )
        local = LocalRelativePredictionModel(
            reference.chaser_parameters, env_config.dt_s
        )
        return MPCController(config, local, reference)

    if observation_filter not in {"hold", "ekf"}:
        raise ValueError("observation_filter must be 'hold' or 'ekf'")
    # The EKF-style output-feedback baseline forward-propagates the delayed/held
    # sensor fix with a nominal target model (predict step), de-lagging the
    # estimate the controller flies on.
    estimator_reference = RelativePredictionModel(
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

    # "nominal": one controller that always predicts with the nominal target
    # parameters -- the realistic non-cooperative case, built once. "truth":
    # rebuild the controller each episode from the env's sampled truth
    # parameters -- the feasibility control that confirms the task is solvable
    # when the model is exactly right, isolating the mismatch as the cause.
    controller = build_controller(target_parameters())
    records: list[dict[str, Any]] = []
    all_solve_times: list[float] = []
    all_command_times: list[float] = []
    # The controller half is `command_time_s`, which this path has always timed
    # alone; the environment half is recorded beside it so the compute column
    # can be read in the same convention as the learned rows.
    all_environment_times: list[float] = []
    total_fallbacks = 0
    predicted_safe_truth_violation_count = 0
    compared_constraint_steps = 0
    trajectories: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            _, info = env.reset(seed=seed + episode)
            if controller_model_source == "truth":
                # Give the controller the exact truth parameters this episode
                # sampled, so its prediction model matches the truth tumble.
                controller = build_controller(env.target_parameters)
            controller.reset()
            estimator = (
                TargetStateEstimator(
                    bias_rad=observation_bias_rad,
                    bias_seed=seed + episode,
                    delay_steps=observation_delay_steps,
                    update_every=observation_update_every,
                    filter_mode="propagate" if observation_filter == "ekf" else "hold",
                    propagate_step=estimator_reference.propagate_target,
                    dt_s=env_config.dt_s,
                )
                if observed_target
                else None
            )
            control_step = 0
            force_impulse = 0.0
            torque_impulse = 0.0
            saturation_counts = np.zeros(6, dtype=np.int64)
            episode_solve_times: list[float] = []
            episode_command_times: list[float] = []
            episode_environment_times: list[float] = []
            discounted_return = 0.0
            discount = 1.0
            episode_fallbacks = 0
            min_position_error_m = float(info["position_error_m"])
            minimum_margins = {
                name: float(info[name])
                for name in (
                    "corridor_axial_margin_m",
                    "corridor_lateral_margin_m",
                    "fov_margin_rad",
                    "total_speed_margin_m_s",
                    "closing_speed_margin_m_s",
                )
            }
            first_violation: dict[str, Any] | None = None
            trace: list[dict[str, Any]] = []
            terminated = truncated = False
            while not (terminated or truncated):
                assert env.relative is not None and env.target_state is not None
                assert env.chaser_state is not None
                # What the controller sees. Without an estimator it is the truth
                # relative state; with one, the target pose is the corrupted
                # estimate and the perceived relative state is recomputed against
                # it (own chaser state is known). Truth still drives env.step and
                # every constraint/metric below, so violations are real-geometry.
                if estimator is not None:
                    target_estimate = estimator.estimate(
                        env.target_state, control_step, env.time_seconds
                    )
                    command_state = relative_to_vector(
                        relative_state(target_estimate, env.chaser_state)
                    )
                    command_target = target_estimate
                else:
                    command_state = relative_to_vector(env.relative)
                    command_target = env.target_state
                x = relative_to_vector(env.relative)
                command_started = perf_counter()
                wrench_vector, diagnostics = controller.command(
                    command_state,
                    target_state=command_target,
                    time_seconds=env.time_seconds,
                )
                control_step += 1
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
                environment_started = perf_counter()
                _, reward, terminated, truncated, info = env.step(action)
                episode_environment_times.append(perf_counter() - environment_started)
                discounted_return += discount * float(reward)
                discount *= GAMMA
                min_position_error_m = min(
                    min_position_error_m, float(info["position_error_m"])
                )
                for name in minimum_margins:
                    margin = float(info[name])
                    minimum_margins[name] = min(minimum_margins[name], margin)
                    if margin < 0.0 and first_violation is None:
                        first_violation = {
                            "time_s": float(info["time_seconds"]),
                            "type": name,
                            "margin": margin,
                        }
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
            all_environment_times.extend(episode_environment_times)
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
                    # `survival_s` and `discounted_return` under the names the
                    # shared metric path reads; for a completed episode the
                    # first is the completion time.
                    "survival_s": float(env.time_seconds),
                    "discounted_return": discounted_return,
                    "first_violation": first_violation,
                    "minimum_margins": minimum_margins,
                    "minimum_position_error_m": min_position_error_m,
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
                    # `controller_time_s` is `command_time_s` under the shared
                    # name; the QP alone is `solve_time_s`, and the gap between
                    # them is this implementation's finite-difference overhead.
                    "controller_time_s": _summary(episode_command_times),
                    "environment_step_time_s": _summary(episode_environment_times),
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
        "schema_version": 2,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "mpc_only",
        "controller_model_source": controller_model_source,
        "observation_error": {
            "bias_rad": observation_bias_rad,
            "delay_steps": observation_delay_steps,
            "update_every": observation_update_every,
            "filter": observation_filter,
        },
        "corridor_speed_fraction": config.corridor_speed_fraction,
        "episodes": episodes,
        "base_seed": seed,
        "environment": asdict(env_config),
        "mpc_config": asdict(config),
        "gamma": GAMMA,
        "main_table": main_table_metrics(
            records,
            controller_times_s=all_command_times,
            environment_step_times_s=all_environment_times,
            control_period_s=env_config.dt_s,
        ),
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


def resolve_reference_source(task: str, explicit: str | None) -> str:
    """Pick the fair reference for a task unless the user overrode it.

    A fixed setpoint is a myopic regulator on the 95 s single_phase task, so
    that task defaults to the shared corridor-guidance path; the terminal shell
    keeps the fixed setpoint its historical evidence was measured on. An
    explicit choice always wins.
    """

    if explicit is not None:
        return explicit
    single_phase_family = task in {"single_phase", "single_phase_phase_sampled"}
    return "corridor_guidance" if single_phase_family else "fixed"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference_source = resolve_reference_source(args.task, args.reference_source)
    if args.terminal_cost_source == "learned_convex":
        if args.terminal_value_file is None:
            raise ValueError(
                "--terminal-cost-source learned_convex requires --terminal-value-file"
            )
        terminal_value = ConvexQuadraticTerminalValue.load(args.terminal_value_file)
    else:
        terminal_value = None
    base_config = constrained_mpc_nominal_config()
    config = replace(
        base_config,
        horizon_steps=(
            args.horizon if args.horizon is not None else 50
        ),
        outer_iterations=(
            args.outer_iterations
            if args.outer_iterations is not None
            else 1
        ),
        input_weight=(
            args.input_weight
            if args.input_weight is not None
            else 0.01
        ),
        terminal_weight=(
            args.terminal_weight
            if args.terminal_weight is not None
            else 100.0
        ),
        reference_source=reference_source,
        corridor_speed_fraction=args.corridor_speed_fraction,
        terminal_cost_source=args.terminal_cost_source,
        terminal_value=terminal_value,
        linearization_source=(
            args.linearization_source
            if args.linearization_source is not None
            else base_config.linearization_source
        ),
        exact_linearization_refresh_steps=(
            args.exact_refresh_steps
            if args.exact_refresh_steps is not None
            else base_config.exact_linearization_refresh_steps
        ),
    )
    environment_config = (
        phase2_environment_config(args.task)
        if args.task in {"single_phase", "single_phase_phase_sampled"}
        else terminal_phase_environment_config()
    )
    if args.max_time is not None:
        environment_config = replace(environment_config, max_time_s=args.max_time)
    if args.target_model_mismatch > 0.0:
        environment_config = replace(
            environment_config,
            phase2_target_model_mismatch=args.target_model_mismatch,
        )
    result = evaluate(
        config,
        episodes=args.episodes,
        seed=args.seed,
        environment_config=environment_config,
        controller_model_source=args.controller_model,
        observation_bias_rad=float(np.deg2rad(args.obs_bias_deg)),
        observation_delay_steps=args.obs_delay_steps,
        observation_update_every=args.obs_update_every,
        observation_filter="ekf" if args.target_estimator == "ekf" else "hold",
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
