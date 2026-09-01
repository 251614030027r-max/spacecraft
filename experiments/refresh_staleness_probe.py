"""Does a cheap staleness signal predict where relaxed-refresh MPC loses safety?

Route A, stage 0, point 3. If dropping the exact-linearisation refresh (running
``local``) costs constraint margin under phase sampling, an adaptive-refresh
layer needs a *cheap* trigger that says "refresh now". This probe runs the
short-horizon MPC in ``local`` mode on the phase-sampled task and, at every
control step, logs two candidate triggers next to the live worst truth margin:

* ``model_error`` -- the normalised one-step deviation between the fast local
  prediction the QP linearises and the RK45-truth prediction it would get from a
  refresh. This is exactly the error a refresh would erase, computed without
  paying for a refresh.
* ``turn_per_horizon`` -- ``|omega| * horizon * dt``, the angle the target
  rotates within one horizon; a near-free proxy for how stale a held
  linearisation becomes.

It then reports, per episode, whether the steps that lost margin were preceded
by an elevated signal (the precondition for the signal to be usable as a gate).

Usage::

    python -B -m experiments.refresh_staleness_probe --episodes 6 --seed 990000
"""

from __future__ import annotations

import argparse
from statistics import mean

import numpy as np

from controllers.mpc import MPCController, corridor_tracking_mpc_config
from controllers.mpc.constraints import normalized_truth_margins
from controllers.mpc.prediction import (
    LocalRelativePredictionModel,
    RelativePredictionModel,
    relative_to_vector,
)
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.phase2_env import phase2_environment_config
from env.scenarios import chaser_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh staleness vs safety")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=990000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--lead-steps",
        type=int,
        default=10,
        help="how many steps before a margin loss to inspect the signal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = phase2_environment_config("single_phase_phase_sampled")
    config = corridor_tracking_mpc_config(
        horizon_steps=args.horizon, linearization_source="local"
    )
    reference = RelativePredictionModel(
        target_parameters=target_parameters(),
        chaser_parameters=chaser_parameters(),
        dt_s=env_config.dt_s,
        gravity_options=GravityOptions(include_j2=env_config.include_j2),
        solver_settings=RK45Settings(
            rtol=env_config.solver_rtol, atol=env_config.solver_atol,
            max_step=env_config.dt_s,
        ),
    )
    local = LocalRelativePredictionModel(reference.chaser_parameters, env_config.dt_s)
    controller = MPCController(config, local, reference)
    scales = np.asarray(config.state_scales, dtype=np.float64)

    env = SE3RendezvousEnv(env_config)
    per_episode: list[dict] = []
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed + episode)
            controller.reset()
            errors: list[float] = []
            margins: list[float] = []
            terminated = truncated = False
            while not (terminated or truncated):
                assert env.relative is not None and env.target_state is not None
                x = relative_to_vector(env.relative)
                target = env.target_state
                t = env.time_seconds
                wrench, _ = controller.command(x, target_state=target, time_seconds=t)
                # Staleness = normalised deviation between the local prediction the
                # QP used and the RK45-truth prediction a refresh would give.
                target_next = reference.propagate_target(target, t)
                exact_next = reference.predict_given_target_next(
                    x, wrench, target, target_next, t
                )
                local_next = local.predict(x, wrench)
                model_error = float(np.linalg.norm((exact_next - local_next) / scales))
                margin = float(np.min(normalized_truth_margins(x, config.task)))
                errors.append(model_error)
                margins.append(margin)
                action = wrench_to_normalized(
                    GeneralizedForce.from_vector(wrench),
                    max_torque_per_axis_nm=env_config.max_torque_per_axis_nm,
                    max_force_per_axis_n=env_config.max_force_per_axis_n,
                )
                _, _, terminated, truncated, info = env.step(action)

            errors_arr = np.asarray(errors)
            margins_arr = np.asarray(margins)
            worst = float(margins_arr.min())
            # Correlation between staleness and tightness (-margin): a positive
            # value means high staleness coincides with a tighter margin.
            if errors_arr.std() > 0 and margins_arr.std() > 0:
                corr = float(np.corrcoef(errors_arr, -margins_arr)[0, 1])
            else:
                corr = float("nan")
            lead_signal = None
            if worst < 0.0:
                first = int(np.argmax(margins_arr < 0.0))
                window = errors_arr[max(0, first - args.lead_steps):first]
                if window.size:
                    lead_signal = float(window.max() / (errors_arr.mean() + 1e-12))
            per_episode.append(
                {
                    "phase_seed": info.get("episode_target_phase_seed"),
                    "worst_margin": worst,
                    "error_mean": float(errors_arr.mean()),
                    "error_p95": float(np.percentile(errors_arr, 95)),
                    "turn_per_horizon_rad": float(
                        np.linalg.norm(target.omega) * args.horizon * env_config.dt_s
                    ),
                    "corr_error_vs_tightness": corr,
                    "pre_violation_error_ratio": lead_signal,
                }
            )
            print(
                f"ep {episode} seed={per_episode[-1]['phase_seed']} "
                f"worst={worst:+.4f} err_mean={errors_arr.mean():.4f} "
                f"err_p95={np.percentile(errors_arr,95):.4f} "
                f"corr(err,tightness)={corr:+.2f} "
                f"pre_viol_ratio={lead_signal}"
            )
    finally:
        env.close()

    violations = [e for e in per_episode if e["worst_margin"] < 0.0]
    print("\nsummary:")
    print(f"  episodes                 : {len(per_episode)}")
    print(f"  episodes losing margin   : {len(violations)}")
    corrs = [e["corr_error_vs_tightness"] for e in per_episode if e["corr_error_vs_tightness"] == e["corr_error_vs_tightness"]]
    if corrs:
        print(f"  mean corr(error,tightness): {mean(corrs):+.2f}")
    if violations:
        ratios = [e["pre_violation_error_ratio"] for e in violations if e["pre_violation_error_ratio"]]
        if ratios:
            print(f"  pre-violation error ratio : mean {mean(ratios):.2f} (>1 = signal rises before the loss)")
    else:
        print("  no margin loss under local refresh on this block -- relaxed refresh held safe")


if __name__ == "__main__":
    main()
