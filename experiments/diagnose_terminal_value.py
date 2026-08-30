"""Diagnose why the learned convex-quadratic terminal value fails on single_phase.

The 20-episode main table shows the learned-terminal short-horizon MPC at 0/20
with a negative worst margin, while the matched fixed-terminal row is 20/20.
This probe pins the mechanism with two cheap sweeps (a few episodes each), so
the negative result is reproducible rather than asserted:

1. ``--mode weight-sweep`` (corridor reference): scale the learned terminal by
   lambda in {1, 0.3, 0.1, 0.03, 0.01, 0} and evaluate. If some lambda recovered
   completion the failure would be mis-scaling; instead every positive weight
   degrades monotonically and lambda=0 (no terminal) is best -- so it is not
   scaling.
2. ``--mode fixed-setpoint`` (myopic reference, no corridor path): give the
   terminal value the only source of beyond-horizon foresight. If it could
   substitute for a path plan it would rescue the myopic case; instead it blocks
   the approach and violates, while plain fixed-terminal completes.

Both point to the function class: a global convex quadratic's gradient is radial
toward its centre x*, but the admissible path is a curved co-rotating corridor
inside an FOV cone, so the terminal drags the chaser straight at the goal, off
the corridor. A high value-fit R^2 does not imply a usable control gradient.

Usage::

    python -B -m experiments.diagnose_terminal_value \
        --terminal-value-file models/terminal_value_scripted_seed990000.json \
        --mode weight-sweep --episodes 3 --seed 262000
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from controllers.mpc import (
    ConvexQuadraticTerminalValue,
    constrained_mpc_nominal_config,
    corridor_tracking_mpc_config,
    learned_terminal_mpc_config,
)
from env.phase2_env import phase2_environment_config
from experiments.evaluate_mpc import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose learned terminal value")
    parser.add_argument("--terminal-value-file", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("weight-sweep", "fixed-setpoint"),
        default="weight-sweep",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--horizon", type=int, default=10)
    return parser.parse_args()


def _scaled(
    value: ConvexQuadraticTerminalValue, lam: float
) -> ConvexQuadraticTerminalValue:
    return ConvexQuadraticTerminalValue(
        center=value.center,
        weight_matrix=lam * value.weight_matrix,
        linear=lam * value.linear,
        constant=lam * value.constant,
    )


def _summarize(result: dict) -> str:
    records = result["episode_records"]
    completed = sum(r["completed"] for r in records)
    worst = min(min(r["minimum_margins"].values()) for r in records)
    closest = min(r["minimum_position_error_m"] for r in records)
    violations = Counter(
        r["first_violation"]["type"] for r in records if r["first_violation"]
    )
    return (
        f"completed={completed}/{len(records)} "
        f"worst_margin={worst:+.4f} closest={closest:.2f}m "
        f"violations={dict(violations)}"
    )


def main() -> None:
    args = parse_args()
    terminal_value = ConvexQuadraticTerminalValue.load(args.terminal_value_file)
    env_config = phase2_environment_config("single_phase")

    def run(config) -> str:
        return _summarize(
            evaluate(
                config,
                episodes=args.episodes,
                seed=args.seed,
                environment_config=env_config,
            )
        )

    if args.mode == "weight-sweep":
        print("corridor reference, learned terminal scaled by lambda:")
        print(
            f"  fixed-terminal        : "
            f"{run(corridor_tracking_mpc_config(horizon_steps=args.horizon))}"
        )
        for lam in (1.0, 0.3, 0.1, 0.03, 0.01, 0.0):
            config = learned_terminal_mpc_config(
                _scaled(terminal_value, lam), horizon_steps=args.horizon
            )
            print(f"  learned lambda={lam:<4}  : {run(config)}")
    else:
        print("fixed-setpoint (myopic) reference, terminal is the only foresight:")
        print(
            f"  h50 fixed-terminal    : {run(constrained_mpc_nominal_config())}"
        )
        print(
            f"  h10 fixed-terminal    : "
            f"{run(constrained_mpc_nominal_config(horizon_steps=args.horizon))}"
        )
        for lam in (1.0, 0.1):
            config = constrained_mpc_nominal_config(
                horizon_steps=args.horizon,
                terminal_cost_source="learned_convex",
                terminal_value=_scaled(terminal_value, lam),
            )
            print(f"  h10 learned lambda={lam:<3}: {run(config)}")


if __name__ == "__main__":
    main()
