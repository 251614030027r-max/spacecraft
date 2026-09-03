"""Evaluate one pre-registered A2 guidance-free method."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import cvxpy as cp

from controllers.mpc import (
    corridor_tracking_mpc_config,
    planning_tracking_mpc_config,
    receding_plan_mpc_config,
    terminal_short_mpc_config,
)
from env.phase2_env import (
    perception_guidance_free_environment_config,
    phase2_perception_environment_config,
)
from experiments.evaluate_mpc import evaluate


METHODS = (
    "guided_a1_control",
    "terminal_short",
    "receding_plan_long",
    "planning_tracking",
    "planning_tracking_oracle",
)


def method_spec(name: str):
    """Return (controller, environment, state source) for one frozen row."""

    if name == "guided_a1_control":
        return (
            corridor_tracking_mpc_config(),
            phase2_perception_environment_config(),
            "estimate",
        )
    environment = perception_guidance_free_environment_config()
    if name == "terminal_short":
        return terminal_short_mpc_config(), environment, "estimate"
    if name == "receding_plan_long":
        return receding_plan_mpc_config(), environment, "estimate"
    if name == "planning_tracking":
        return planning_tracking_mpc_config(), environment, "estimate"
    if name == "planning_tracking_oracle":
        return planning_tracking_mpc_config(), environment, "oracle"
    raise ValueError(f"unsupported A2 method: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one A2 method")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--serial-compute-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    controller, environment, state_source = method_spec(args.method)
    if args.max_time is not None:
        environment = replace(environment, max_time_s=args.max_time)
    result = evaluate(
        controller,
        episodes=args.episodes,
        seed=args.seed,
        environment_config=environment,
        control_state_source=state_source,
        progress=args.progress,
    )
    result["a2_method"] = args.method
    result["compute_profile_valid"] = bool(args.serial_compute_profile)
    result["compute_profile_note"] = (
        "declared one-process serial profile; valid only if run alone"
        if args.serial_compute_profile
        else "trajectory timing is not accepted; use the separate serial profile"
    )
    result["runtime"] = {
        "python": sys.version,
        "cvxpy": cp.__version__,
        "solver": controller.solver,
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.tolist() if isinstance(value, np.ndarray) else value
            ),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"method": args.method, "output": str(args.output)}))


if __name__ == "__main__":
    main()
