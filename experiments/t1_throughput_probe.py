"""Machine check before the M0/M1 round: real throughput and memory at N parallel workers.

Each worker builds the training environment (M0: T1GStarEnv, M1: T1CounterfactualEnv)
and flies ``--decisions`` decisions with a fixed, reproducible action sequence on probe
openings 9_000_000 + worker (outside every research block). No learning, no network.
Reports seconds per decision per worker, aggregate decisions per hour and peak RSS.

    python -B -m experiments.t1_throughput_probe --workers 6 --method m1 --decisions 40 --output probe_6_m1.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 2**20
    except Exception:  # noqa: BLE001
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:  # noqa: BLE001
            return None


def worker(args: tuple[int, str, int]) -> dict:
    index, method, decisions = args
    import numpy as np
    import torch

    torch.set_num_threads(1)
    if method == "m1":
        from env.t1_counterfactual import T1CounterfactualEnv as Env
    else:
        from env.t1_gym import T1GStarEnv as Env
    env = Env(openings=[9_000_000 + index])
    rng = np.random.default_rng(index)
    env.reset()
    started, done = time.perf_counter(), 0
    while done < decisions:
        action = np.array([0.0, 0.0]) if rng.uniform() < 0.6 else rng.uniform(-1, 1, 2)
        _, _, terminated, truncated, _ = env.step(action)
        done += 1
        if terminated or truncated:
            env.reset()
    wall = time.perf_counter() - started
    return {"worker": index, "decisions": done, "wall_s": wall, "s_per_decision": wall / done,
            "peak_rss_mb": _peak_rss_mb()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--method", choices=("m0", "m1"), required=True)
    parser.add_argument("--decisions", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(worker, [(i, args.method, args.decisions) for i in range(args.workers)])
    wall = time.perf_counter() - started
    result = {
        "workers": args.workers, "method": args.method, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "rows": rows,
        "s_per_decision_mean": sum(r["s_per_decision"] for r in rows) / len(rows),
        "s_per_decision_max": max(r["s_per_decision"] for r in rows),
        "aggregate_decisions_per_hour": sum(r["decisions"] for r in rows) / wall * 3600,
        "peak_rss_mb_max": max((r["peak_rss_mb"] or 0.0) for r in rows),
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
