"""Evaluate one A3 deployable-planning P1 method."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path

import cvxpy as cp
import numpy as np

from controllers.mpc import online_endpoint_mpc_config, planning_tracking_mpc_config
from env.phase2_env import deployable_planning_environment_config
from experiments.evaluate_mpc import evaluate


METHODS = (
    "deployable_online",
    "offline_estimate_bound",
    "offline_truth_bound",
    "long_exact_h50",
)


def method_spec(name: str, *, deployable_horizon: int):
    environment = deployable_planning_environment_config()
    if name == "deployable_online":
        return online_endpoint_mpc_config(
            horizon_steps=deployable_horizon
        ), environment, "estimate"
    if name == "offline_estimate_bound":
        return replace(
            planning_tracking_mpc_config(), runtime_diagnostics=False
        ), environment, "estimate"
    if name == "offline_truth_bound":
        return replace(
            planning_tracking_mpc_config(), runtime_diagnostics=False
        ), environment, "oracle"
    if name == "long_exact_h50":
        return online_endpoint_mpc_config(
            horizon_steps=50, exact=True
        ), environment, "estimate"
    raise ValueError(f"unsupported A3 method: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one A3 P1 method")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--deployable-horizon", type=int, default=10)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--serial-compute-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.deployable_horizon <= 0:
        raise ValueError("deployable horizon must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    controller, environment, state_source = method_spec(
        args.method, deployable_horizon=args.deployable_horizon
    )
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
    result["a3_method"] = args.method
    result["deployable_horizon"] = args.deployable_horizon
    result["compute_profile_valid"] = bool(args.serial_compute_profile)
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
