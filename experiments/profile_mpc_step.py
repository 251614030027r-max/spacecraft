"""Where a Pure MPC control step spends its 0.1 s budget.

Whether per-step compute is a property of MPC or an artefact of this
implementation is a claim the three-way table rests on, and it is
hardware-dependent, so it has to be re-measurable rather than quoted. This
prints the breakdown of ``MPCController.command`` -- the QP, the constraint
linearisation, the nominal rollout, and the exact-linearisation refresh -- with
the refresh steps separated out, because that term fires once every
``exact_linearization_refresh_steps`` and dominates the p95 while contributing
little to the mean.

    python -B -m experiments.profile_mpc_step
    python -B -m experiments.profile_mpc_step --horizon 15 --steps 120

Read-only: prints, writes nothing, and holds the state fixed so the numbers are
a cost measurement rather than a trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from time import perf_counter

import numpy as np

from controllers.mpc import (
    LocalRelativePredictionModel,
    MPCController,
    RelativePredictionModel,
    constrained_mpc_nominal_config,
)
from controllers.mpc.prediction import relative_to_vector
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from env.phase2_env import terminal_phase_environment_config
from env.scenarios import chaser_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousEnv


def _line(name: str, samples: list[float], period_s: float) -> str:
    if not samples:
        return f"  {name:<34} --"
    array = np.asarray(samples, dtype=np.float64)
    return (
        f"  {name:<34} mean {array.mean() * 1e3:7.1f} ms"
        f"   p95 {np.percentile(array, 95) * 1e3:7.1f} ms"
        f"   {array.mean() / period_s:5.2f}x budget   n={array.size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost breakdown of one MPC step")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=260812)
    args = parser.parse_args()

    environment_config = terminal_phase_environment_config()
    config = constrained_mpc_nominal_config()
    if args.horizon is not None:
        config = replace(config, horizon_steps=args.horizon)
    period = environment_config.dt_s

    environment = SE3RendezvousEnv(environment_config)
    try:
        environment.reset(seed=args.seed)
        assert environment.relative is not None
        assert environment.target_state is not None
        state = relative_to_vector(environment.relative)
        target_state = environment.target_state
    finally:
        environment.close()

    reference = RelativePredictionModel(
        target_parameters(),
        chaser_parameters(),
        environment_config.dt_s,
        GravityOptions(include_j2=environment_config.include_j2),
        RK45Settings(
            rtol=environment_config.solver_rtol,
            atol=environment_config.solver_atol,
            max_step=environment_config.dt_s,
        ),
    )
    local = LocalRelativePredictionModel(
        reference.chaser_parameters, environment_config.dt_s
    )
    controller = MPCController(config, local, reference)

    for _ in range(3):  # warm the CVXPY canonicalisation and the solver
        controller.command(state, target_state=target_state, time_seconds=0.0)
    controller.reset()

    period_steps = config.exact_linearization_refresh_steps
    plain: list[float] = []
    refreshed: list[float] = []
    qp: list[float] = []
    constraints: list[float] = []
    for step in range(args.steps):
        started = perf_counter()
        _, diagnostics = controller.command(
            state, target_state=target_state, time_seconds=0.0
        )
        elapsed = perf_counter() - started
        (refreshed if step % period_steps == 0 else plain).append(elapsed)
        qp.append(diagnostics.solve_time_s)
        constraints.append(diagnostics.constraint_linearization_time_s)

    total = plain + refreshed
    print(
        f"horizon {config.horizon_steps} steps ({config.horizon_steps * period:.1f} s)"
        f"   solver {config.solver}"
        f"   linearisation {config.linearization_source},"
        f" refreshed every {period_steps} control steps"
        f"   control period {period * 1e3:.0f} ms"
    )
    print(_line("full command, all steps", total, period))
    print(_line("full command, no refresh due", plain, period))
    print(_line("full command, refresh step", refreshed, period))
    print(_line("of which: QP solve", qp, period))
    print(_line("of which: constraint linearisation", constraints, period))
    if plain and refreshed:
        cost = float(np.mean(refreshed) - np.mean(plain))
        print(
            f"\n  one exact linearisation point costs {cost * 1e3:.0f} ms"
            f" ({cost / period:.1f}x the control period)."
        )
        print(
            "  Re-linearising against the truth at every horizon index, which is"
            f" what\n  References/南航.pdf lists as future work, is"
            f" {config.horizon_steps} of those per step:"
            f" about {cost * config.horizon_steps:.0f} s."
        )


if __name__ == "__main__":
    main()
