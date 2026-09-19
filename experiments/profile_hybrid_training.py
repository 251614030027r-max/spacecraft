"""Profile hybrid training without turning throughput into a science result.

Run the one-process case first.  Only that mode reports per-step/module latency.
The three-process mode deliberately emits aggregate throughput only: concurrent
latencies are contention measurements and are not real-time controller claims.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config
from train.hybrid_configs import SAC_MPC_HYBRID, hybrid_model_kwargs


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

    def __init__(
        self,
        *,
        eager_target_cache: bool,
        runtime_diagnostics: bool,
    ) -> None:
        environment = replace(
            precapture_planning_environment_config(),
            cache_target_trajectory=eager_target_cache,
        )
        hybrid = PrecaptureHybridConfig(
            horizon_steps=20,
            waypoint_parametrization="radial_local",
            decision_discount_factor=SAC_MPC_HYBRID.gamma,
            runtime_diagnostics=runtime_diagnostics,
            include_target_phase_and_time_observation=True,
        )
        super().__init__(environment_config=environment, hybrid_config=hybrid)
        self.timings: dict[str, list[float]] = defaultdict(list)

        original_command = self.controller.command

        def timed_command(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            command, diagnostics = original_command(*args, **kwargs)
            self.timings["mpc_command"].append(perf_counter() - started)
            self.timings["qp_solve"].append(diagnostics.solve_time_s)
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
            self.timings["truth_step"].append(perf_counter() - started)
            return result

        self.env.step = timed_truth_step  # type: ignore[method-assign]

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        started = perf_counter()
        result = super().reset(*args, **kwargs)
        self.timings["reset"].append(perf_counter() - started)
        return result

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        started = perf_counter()
        result = super().step(action)
        self.timings["decision"].append(perf_counter() - started)
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


def _worker(spec: tuple[int, int, bool, bool]) -> dict[str, Any]:
    seed, decisions, eager_target_cache, runtime_diagnostics = spec
    env = TimedHybridEnv(
        eager_target_cache=eager_target_cache,
        runtime_diagnostics=runtime_diagnostics,
    )
    kwargs = hybrid_model_kwargs(SAC_MPC_HYBRID)
    # Timing-only override: exercise one representative steady-state SAC update
    # per decision without first paying 2,000 expensive environment decisions.
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
        "module_timings": {
            name: _summary(samples) for name, samples in sorted(env.timings.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=30)
    parser.add_argument("--processes", type=int, choices=[1, 3], required=True)
    parser.add_argument("--seed", type=int, default=264000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eager-target-cache", action="store_true")
    parser.add_argument("--runtime-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decisions < 1:
        raise ValueError("decisions must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    specs = [
        (
            args.seed + index,
            args.decisions,
            args.eager_target_cache,
            args.runtime_diagnostics,
        )
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
    payload: dict[str, Any] = {
        "processes": args.processes,
        "decisions_per_process": args.decisions,
        "total_decisions": total_decisions,
        "eager_target_cache": args.eager_target_cache,
        "runtime_diagnostics": args.runtime_diagnostics,
        "horizon_steps": 20,
        "decision_period_steps": 20,
        "sac_timing_override": (
            "learning_starts=0 only to measure steady-state update cost; all other "
            "SAC hyperparameters match SAC_MPC_HYBRID"
        ),
        "aggregate_wall_time_s": aggregate_wall_time_s,
        "aggregate_decisions_per_s": total_decisions / aggregate_wall_time_s,
    }
    if args.processes == 1:
        payload["reporting_scope"] = "serial_single_process_module_latency"
        payload["seconds_per_decision"] = workers[0]["wall_time_s"] / args.decisions
        payload["module_timings"] = workers[0]["module_timings"]
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
