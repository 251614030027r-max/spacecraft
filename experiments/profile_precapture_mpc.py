"""Serial single-process per-step cost of the precapture MPC.

Real time is a worst-case property and it is only meaningful when nothing else
shares the machine, so this script runs one episode, one process, and times the
controller call alone. ``env.step`` -- the RK45 truth propagation -- is timed
separately and is *not* part of the controller budget; merging the two is the
mistake ``eval/metrics.py`` exists to prevent.

Absolute milliseconds are a property of this machine. The ratio to the 0.1 s
control period is the reportable figure, and even that must be stated with the
implementation beside it (Python, CVXPY, CLARABEL, one core).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

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
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.phase2_env import precapture_planning_environment_config
from env.scenarios import chaser_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env_config = precapture_planning_environment_config()
    config = replace(
        precapture_mpc_config(),
        horizon_steps=args.horizon,
        outer_iterations=1,
        input_weight=0.01,
        terminal_weight=1000.0,
        reference_source="fixed",
        state_scales=np.array(
            [
                *([np.deg2rad(75.0)] * 3),
                *([20.0] * 3),
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
        LocalRelativePredictionModel(reference_model.chaser_parameters, env_config.dt_s),
        reference_model,
    )
    controller.reset()

    controller_ms: list[float] = []
    environment_ms: list[float] = []
    statuses: list[str] = []
    terminated = truncated = False
    while not (terminated or truncated) and len(controller_ms) < args.steps:
        state = relative_to_vector(env.relative)
        started = perf_counter()
        wrench, diagnostics = controller.command(
            state,
            target_state=env.target_state,
            time_seconds=env.time_seconds,
            terminal_latched=bool(info["terminal_region_active"]),
        )
        controller_ms.append(1000.0 * (perf_counter() - started))
        statuses.append(str(diagnostics.status))
        action = wrench_to_normalized(
            GeneralizedForce.from_vector(wrench),
            max_torque_per_axis_nm=env_config.max_torque_per_axis_nm,
            max_force_per_axis_n=env_config.max_force_per_axis_n,
        )
        started = perf_counter()
        _, _, terminated, truncated, info = env.step(action)
        environment_ms.append(1000.0 * (perf_counter() - started))

    times = np.asarray(controller_ms)
    budget_ms = 1000.0 * env_config.dt_s
    summary = {
        "horizon": args.horizon,
        "seed": args.seed,
        "steps": int(times.size),
        "serial_single_process": True,
        "control_period_ms": budget_ms,
        "controller_ms": {
            "mean": float(times.mean()),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "max": float(times.max()),
        },
        "controller_over_budget": {
            "mean": float(times.mean() / budget_ms),
            "p95": float(np.percentile(times, 95) / budget_ms),
            "max": float(times.max() / budget_ms),
        },
        "environment_step_ms_mean": float(np.mean(environment_ms)),
        "worst_steps": [
            {"index": int(i), "ms": float(times[i])}
            for i in np.argsort(times)[-5:][::-1]
        ],
        "steps_over_budget": int(np.count_nonzero(times > budget_ms)),
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
    }
    args.output.write_text(json.dumps(summary, indent=1))
    print(
        f"h{args.horizon} seed={args.seed} n={times.size} "
        f"mean={summary['controller_ms']['mean']:.1f}ms "
        f"p95={summary['controller_ms']['p95']:.1f}ms "
        f"max={summary['controller_ms']['max']:.1f}ms "
        f"p95_budget={summary['controller_over_budget']['p95']:.2f}x"
    )


if __name__ == "__main__":
    main()
