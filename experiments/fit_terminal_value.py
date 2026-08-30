"""Fit a convex-quadratic terminal value from reference closed-loop cost-to-go.

This is the offline, training-free first version of the SAC-MPC coupling
terminal value (handoff section 4). It flies the scripted corridor law on the
``single_phase`` task, measures the *cost-to-go* of every visited 12-d MPC
relative state under the MPC stage-cost convention, and fits a convex quadratic
``V(x) = (x-c)^T H (x-c) + g^T (x-c) + const`` with ``H`` PSD. The result drops
into the short-horizon MPC as its terminal cost.

The stage cost matches ``controllers/mpc``:
``state_weight * ||S^-1 (x - x*)||^2 + input_weight * ||a||^2`` where ``x*`` is
the desired pose (the anchor of "cost remaining to the goal"), ``S`` the state
scales, and ``a`` the normalised action (which is exactly ``S_u^-1 u``, so the
input term is layout-free). Cost-to-go is the discounted tail sum from each
step to the end of the episode.

**Fit on a seed block disjoint from the 262000 evaluation block.** The default
``--seed 990000`` is chosen for that separation; never fit on 262000, or the
terminal value would be tuned on the evaluation episodes.

Usage::

    python -B -m experiments.fit_terminal_value \
        --episodes 12 --seed 990000 \
        --output models/terminal_value_scripted_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from controllers.mpc import MPCConfig, fit_convex_quadratic
from controllers.mpc.prediction import relative_to_vector
from dynamics.lie import se3_log
from env.phase2_env import make_phase2_env
from env.task import Phase2TaskConfig
from eval.validate_single_phase_semantics import scripted_action
from train.configs import PURE_SAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit convex-quadratic terminal value")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument(
        "--seed",
        type=int,
        default=990_000,
        help="fit seed block; keep disjoint from the 262000 evaluation block",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="discount for the cost-to-go tail sum; 1.0 = undiscounted, matching "
        "the finite-horizon MPC objective which carries no discount",
    )
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--max-steps", type=int, default=2001)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _desired_pose_state(task: Phase2TaskConfig) -> np.ndarray:
    """12-d anchor x* = [se3_log(desired_transform), 0] -- the mission goal."""

    return np.concatenate((se3_log(task.desired_transform), np.zeros(6)))


def collect_cost_to_go(
    *,
    episodes: int,
    seed: int,
    gamma: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (states[N,12], costs_to_go[N], center[12]) from scripted rollouts."""

    reference = MPCConfig()
    state_scales = np.asarray(reference.state_scales, dtype=np.float64)
    state_weight = reference.state_weight
    input_weight = reference.input_weight

    task = Phase2TaskConfig()
    center = _desired_pose_state(task)

    all_states: list[np.ndarray] = []
    all_costs: list[np.ndarray] = []
    for episode in range(episodes):
        env = make_phase2_env("single_phase")
        try:
            env.reset(seed=seed + episode)
            episode_states: list[np.ndarray] = []
            stage_costs: list[float] = []
            terminated = truncated = False
            step = 0
            while not (terminated or truncated) and step < max_steps:
                assert env.relative is not None
                x = relative_to_vector(env.relative)
                action = scripted_action(env)
                state_term = state_weight * float(
                    np.sum(np.square((x - center) / state_scales))
                )
                input_term = input_weight * float(np.sum(np.square(action)))
                episode_states.append(x)
                stage_costs.append(state_term + input_term)
                _, _, terminated, truncated, _ = env.step(action)
                step += 1
        finally:
            env.close()

        # Discounted tail sum: cost_to_go[t] = stage[t] + gamma * cost_to_go[t+1].
        costs = np.zeros(len(stage_costs), dtype=np.float64)
        running = 0.0
        for t in range(len(stage_costs) - 1, -1, -1):
            running = stage_costs[t] + gamma * running
            costs[t] = running
        all_states.append(np.asarray(episode_states, dtype=np.float64))
        all_costs.append(costs)

    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_costs, axis=0),
        center,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    states, costs, center = collect_cost_to_go(
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        max_steps=args.max_steps,
    )
    terminal_value = fit_convex_quadratic(
        states, costs, center=center, ridge=args.ridge
    )

    predictions = np.array([terminal_value.value(x) for x in states])
    residual = costs - predictions
    denominator = float(np.sum(np.square(costs - costs.mean())))
    r_squared = 1.0 - float(np.sum(np.square(residual))) / denominator
    eigenvalues = np.linalg.eigvalsh(terminal_value.weight_matrix)

    diagnostics: dict[str, Any] = {
        "episodes": args.episodes,
        "seed": args.seed,
        "gamma": args.gamma,
        "ridge": args.ridge,
        "eval_gamma": PURE_SAC.gamma,
        "samples": int(states.shape[0]),
        "cost_to_go_mean": float(costs.mean()),
        "cost_to_go_max": float(costs.max()),
        "fit_r_squared": r_squared,
        "fit_rms_error": float(np.sqrt(np.mean(np.square(residual)))),
        "weight_eigenvalue_min": float(eigenvalues.min()),
        "weight_eigenvalue_max": float(eigenvalues.max()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = terminal_value.to_dict()
    payload["fit_diagnostics"] = diagnostics
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.quiet:
        print(json.dumps(diagnostics, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
