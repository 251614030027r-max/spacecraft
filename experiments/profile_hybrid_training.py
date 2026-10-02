"""Profile hybrid training without turning throughput into a science result.

Run the one-process case first.  Only that mode reports per-step/module latency.
The three-process mode deliberately emits aggregate throughput only: concurrent
latencies are contention measurements and are not real-time controller claims.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from stable_baselines3 import SAC

import controllers.mpc.prediction as prediction_module
import env.se3_rendezvous_env as rendezvous_module
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from train.hybrid_configs import SAC_MPC_HYBRID, hybrid_model_kwargs
from train.train_hybrid import accelerated_training_configs


def _summary(samples: list[float]) -> dict[str, float | int] | None:
    if not samples:
        return None
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_s": float(values.mean()),
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
        "max_s": float(values.max()),
    }


class TimedHybridEnv(PrecaptureHybridEnv):
    """The production environment with wall timers around existing calls."""

    _BREAKDOWN_KEYS = (
        "mpc_command",
        "qp_problem_solve_wall",
        "solver_stats_solve_time",
        "pre_solve_setup_wall",
        "model_linearization",
        "constraint_linearization",
        "reported_rollout",
        "truth_step_wall",
        "truth_rk45",
        "mpc_drift_rk45",
    )

    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.decision_breakdowns: dict[str, list[float]] = defaultdict(list)

        original_truth_rk45 = rendezvous_module.propagate_rk45
        original_mpc_rk45 = prediction_module.propagate_rk45

        def timed_truth_rk45(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            result = original_truth_rk45(*args, **kwargs)
            self.timings["truth_rk45"].append(perf_counter() - started)
            return result

        def timed_mpc_rk45(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            result = original_mpc_rk45(*args, **kwargs)
            self.timings["mpc_drift_rk45"].append(perf_counter() - started)
            return result

        rendezvous_module.propagate_rk45 = timed_truth_rk45
        prediction_module.propagate_rk45 = timed_mpc_rk45

        environment, hybrid = accelerated_training_configs(
            horizon_steps=35,
            waypoint_parametrization="task_state_v2",
            execution_feedback=True,
            monotone_commit=True,
            baseline_anchored_residual=False,
            opportunity_task=False,
            adaptive_task=True,
        )
        super().__init__(environment_config=environment, hybrid_config=hybrid)

        original_command = self.controller.command

        def timed_command(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            command, diagnostics = original_command(*args, **kwargs)
            self.timings["mpc_command"].append(perf_counter() - started)
            self.timings["qp_problem_solve_wall"].append(diagnostics.solve_time_s)
            self.timings["solver_stats_solve_time"].append(
                diagnostics.solver_stats_solve_time_s
            )
            self.timings["pre_solve_setup_wall"].append(
                diagnostics.pre_solve_setup_wall_s
            )
            self.timings["model_linearization"].append(
                diagnostics.model_linearization_time_s
            )
            self.timings["constraint_linearization"].append(
                diagnostics.constraint_linearization_time_s
            )
            self.timings["reported_rollout"].append(diagnostics.rollout_time_s)
            return command, diagnostics

        self.controller.command = timed_command  # type: ignore[method-assign]
        original_truth_step = self.env.step

        def timed_truth_step(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            result = original_truth_step(*args, **kwargs)
            self.timings["truth_step_wall"].append(perf_counter() - started)
            return result

        self.env.step = timed_truth_step  # type: ignore[method-assign]

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        started = perf_counter()
        result = super().reset(*args, **kwargs)
        self.timings["reset"].append(perf_counter() - started)
        return result

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        starts = {name: len(self.timings[name]) for name in self._BREAKDOWN_KEYS}
        started = perf_counter()
        result = super().step(action)
        decision_time = perf_counter() - started
        self.timings["decision"].append(decision_time)
        self.decision_breakdowns["decision"].append(decision_time)
        for name, start_index in starts.items():
            self.decision_breakdowns[name].append(
                float(sum(self.timings[name][start_index:]))
            )
        self.decision_breakdowns["cvxpy_wrapper_overhead_approx"].append(
            max(
                0.0,
                self.decision_breakdowns["qp_problem_solve_wall"][-1]
                - self.decision_breakdowns["solver_stats_solve_time"][-1],
            )
        )
        self.decision_breakdowns["parameter_and_loop_overhead_approx"].append(
            max(
                0.0,
                self.decision_breakdowns["pre_solve_setup_wall"][-1]
                - self.decision_breakdowns["model_linearization"][-1]
                - self.decision_breakdowns["constraint_linearization"][-1],
            )
        )
        return result


class TimedSAC(SAC):
    def __init__(self, *args: Any, timing_sink: dict[str, list[float]], **kwargs: Any):
        self._timing_sink = timing_sink
        super().__init__(*args, **kwargs)

    def _sample_action(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        result = super()._sample_action(*args, **kwargs)
        self._timing_sink["sac_action_sampling"].append(perf_counter() - started)
        return result

    def train(self, *args: Any, **kwargs: Any) -> None:
        started = perf_counter()
        super().train(*args, **kwargs)
        self._timing_sink["sac_gradient_update"].append(perf_counter() - started)


def _summaries(
    samples: dict[str, list[float]], *, steady_label: str
) -> dict[str, Any]:
    return {
        name: {
            "all": _summary(values),
            steady_label: _summary(values[1:]),
        }
        for name, values in sorted(samples.items())
    }


def _worker(spec: tuple[int, int, str]) -> dict[str, Any]:
    seed, decisions, mode = spec
    env = TimedHybridEnv()
    kwargs = hybrid_model_kwargs(SAC_MPC_HYBRID)
    if mode == "steady-update":
        # Profiler-only override: expose steady SAC update cost without first
        # paying 2,000 expensive production environment decisions.
        kwargs["learning_starts"] = 0
    model = TimedSAC(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        timing_sink=env.timings,
        **kwargs,
    )
    started = perf_counter()
    try:
        model.learn(total_timesteps=decisions, progress_bar=False)
    finally:
        wall_time_s = perf_counter() - started
        env.close()
    return {
        "seed": seed,
        "decisions": decisions,
        "wall_time_s": wall_time_s,
        "module_timings": _summaries(
            env.timings, steady_label="after_first_call"
        ),
        "per_decision_breakdown": _summaries(
            env.decision_breakdowns,
            steady_label="steady_after_first_decision",
        ),
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=30)
    parser.add_argument("--processes", type=int, choices=[1, 3], required=True)
    parser.add_argument(
        "--mode", choices=["rollout-only", "steady-update"], required=True
    )
    parser.add_argument("--seed", type=int, default=264000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decisions < 1:
        raise ValueError("decisions must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    specs = [
        (args.seed + index, args.decisions, args.mode)
        for index in range(args.processes)
    ]
    started = perf_counter()
    if args.processes == 1:
        workers = [_worker(specs[0])]
    else:
        with mp.get_context("spawn").Pool(args.processes) as pool:
            workers = pool.map(_worker, specs)
    aggregate_wall_time_s = perf_counter() - started
    total_decisions = args.decisions * args.processes
    training_wall_time_s = max(worker["wall_time_s"] for worker in workers)
    training_decisions_per_s = total_decisions / training_wall_time_s
    payload: dict[str, Any] = {
        "git": _git_state(),
        "mode": args.mode,
        "processes": args.processes,
        "decisions_per_process": args.decisions,
        "total_decisions": total_decisions,
        "configuration": {
            "parametrization": "task_state_v2",
            "adaptive_task": True,
            "execution_feedback": True,
            "monotone_commit": True,
            "baseline_anchored_residual": False,
            "opportunity_task": False,
            "runtime_diagnostics": False,
            "cache_target_propagation": False,
            "solver": "CLARABEL",
        },
        "horizon_steps": 35,
        "decision_period_steps": 20,
        "sac_learning_starts": 0 if args.mode == "steady-update" else 2000,
        "sac_timing_override": (
            "learning_starts=0 only to expose steady-state update cost"
            if args.mode == "steady-update"
            else None
        ),
        "aggregate_wall_time_s": aggregate_wall_time_s,
        "process_launch_and_training_decisions_per_s": (
            total_decisions / aggregate_wall_time_s
        ),
        "training_wall_time_s": training_wall_time_s,
        "aggregate_training_decisions_per_s": training_decisions_per_s,
        "projected_180k_wall_hours_at_observed_throughput": (
            180000.0 / training_decisions_per_s / 3600.0
        ),
    }
    if args.processes == 1:
        payload["reporting_scope"] = "serial_single_process_module_latency"
        payload["seconds_per_decision"] = workers[0]["wall_time_s"] / args.decisions
        payload["module_timings"] = workers[0]["module_timings"]
        payload["per_decision_breakdown"] = workers[0]["per_decision_breakdown"]
    else:
        payload["reporting_scope"] = "aggregate_throughput_only"
        payload["latency_omitted_by_design"] = (
            "concurrent per-step latency is not a real-time controller claim"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
