"""P1 policy interventions on frozen checkpoints (no training; G*, MPC, reward and physics untouched).

    python -B -m diagnostics.t1_rootcause.interventions --checkpoint <m1_30000_decisions.zip> --label 30k \
        --modes A,B,C,D --openings 295001,295002,... --output-dir <dir> --workers 6

Modes (diagnostics/t1_rootcause/common.py): A actor; B keep a live plan (a0 = 0, a legal
gym action); C earliest feasible arrival when there is no live plan; D = B + C; E full
earliest-feasible rule with kappa 0.7 (checkpoint-independent, equals rule:earliest).
Deterministic actor, serial episodes inside each worker, one file per (mode, opening).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics.t1_rootcause.common import actor_policy, load_model, rule_policy, run_from
from env.t1_gym import T1GStarEnv


def episode(checkpoint: str | None, label: str, mode: str, opening: int, output_dir: str) -> None:
    path = Path(output_dir) / f"{label}_{mode}_{opening}.json"
    if path.exists():
        return
    from time import perf_counter

    started = perf_counter()
    env = T1GStarEnv()
    obs, _ = env.reset(options={"opening": opening})
    policy = rule_policy() if mode == "E" else actor_policy(load_model(checkpoint), mode)
    result = run_from(env, obs, policy)
    result.update({"checkpoint": checkpoint, "label": label, "mode": mode, "opening": opening,
                   "wall_s": perf_counter() - started})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    print(label, mode, opening, result["event"], round(result["discounted_return_from_k"], 2),
          result["time_s"], "new_plans", result["new_plans"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--label", required=True)
    parser.add_argument("--modes", default="A,B,C,D")
    parser.add_argument("--openings", required=True, help="comma-separated development openings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.checkpoint, args.label, m, int(o), str(args.output_dir))
            for m in args.modes.split(",") for o in args.openings.split(",")]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(episode, jobs)
    else:
        for job in jobs:
            episode(*job)


if __name__ == "__main__":
    main()
