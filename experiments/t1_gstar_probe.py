"""R0 / R1 runner for G* rows (no learning).

    python -B -m experiments.t1_gstar_probe run --row tau:60:0.7 --seeds 290000-290003 --output-dir eval/t1/r0
    python -B -m experiments.t1_gstar_probe run --row pure_fast:0.7 ...
    python -B -m experiments.t1_gstar_probe run --row geo:0.7 ...
    python -B -m experiments.t1_gstar_probe run --row pure ...

Rows:
  pure            Pure MPC (fixed terminal setpoint), T1Env pure step.
  pure_fast:K     Pure approach; inside the mouth region the G* descent at pace K.
  tau:T:K         G* with initial arrival time T s (decremented 2 s per decision), pace K.
  geo:K           G* with T from the next-pass geometric rule at reset, pace K.
  (the four rows above use the v0.1/v0.2 linear generator, superseded in R0)

G* v0.3 rows (rendezvous planner, env/t1_gstar.py GStarRendezvous):
  pf:K            Pure approach (fixed setpoint), G* descent at pace K: the
                  no-learning terminal fast-descent rule (not a G* action).
  rv:T:K          arrival time T s at reset; each later decision asks for the
                  remaining planned time (slips carried, so the plan is kept), pace K.
                  rv:0:K is "earliest feasible arrival" at every decision.
  rvgeo:K         T from the next-pass geometric rule at reset (one more pass if infeasible).
  rvbest:K        T at reset minimising the planned phase-A cost 0.05 T + 5 dv over
                  T = 20..200 s (step 10): a no-learning timing rule that uses the plan.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from env.t1_env import T1Env, T1EnvConfig
import numpy as np

from env.t1_gstar import GStarPlan, free_body_rotations, make_rendezvous, next_pass_tau
from experiments.t1_probe import parse_seeds


def run_row(env: T1Env, seed: int, row: str) -> dict[str, Any]:
    info = env.reset(seed)
    initial = dict(info["features"])
    parts = row.split(":")
    kind = parts[0]
    tau0 = None
    rendezvous = None
    if kind == "pf":
        kappa = float(parts[1])
        rendezvous = make_rendezvous(env, kappa, pure_approach=True)
        tau0 = 0.0
    elif kind.startswith("rv"):
        kappa = float(parts[-1])
        rendezvous = make_rendezvous(env, kappa)
        if kind == "rv":
            tau0 = float(parts[1])
        else:
            tau0 = _initial_tau(env, rendezvous, kind)
    elif kind == "tau":
        tau0, kappa = float(parts[1]), float(parts[2])
    elif kind == "geo":
        kappa = float(parts[1])
        tau0 = next_pass_tau(env)
    elif kind == "pure_fast":
        kappa = float(parts[1])
    elif kind != "pure":
        raise ValueError(row)
    gamma = env.config.reward.gamma
    discounted, k, trace = 0.0, 0, []
    while True:
        if kind == "pure":
            reward, terminated, truncated, step_info = env.step(pure=True)
        elif rendezvous is not None:
            tau_request = tau0 if k == 0 else rendezvous.remaining_s(env)
            rendezvous.decide(env, tau_request)
            reward, terminated, truncated, step_info = env.step(reference=rendezvous)
        else:
            plan = GStarPlan(
                tau_s=max((tau0 or 0.0) - 2.0 * k, 0.0),
                kappa=kappa,
                approach="pure" if kind == "pure_fast" else "gstar",
            )
            plan.begin(env)
            reward, terminated, truncated, step_info = env.step(reference=plan)
        f = env.features()
        trace.append({"k": k, "range_m": f["range_m"], "port_los_angle_deg": f["port_los_angle_deg"], "reward": reward,
                      **({"plan_status": rendezvous.last_status, "tau_effective_s": rendezvous.tau_effective_s,
                          "remaining_s": rendezvous.remaining_s(env)}
                         if rendezvous is not None else {})})
        discounted += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    return {
        "seed": int(seed),
        "row": row,
        "tau0_s": tau0,
        "event": step_info["event"],
        "clean_capture": step_info["event"] == "success",
        **{key: step_info[key] for key in ("violations", "numerical_error") if key in step_info},
        "discounted_return": discounted,
        "initial_features": initial,
        **({"plan_wall_s_total": float(np.sum(rendezvous.plan_wall_s)),
            "plan_wall_s_max": float(np.max(rendezvous.plan_wall_s, initial=0.0))} if rendezvous is not None else {}),
        **env.summary(),
        "trace": trace,
    }


def _initial_tau(env: T1Env, rendezvous, kind: str) -> float:
    target, chaser = env.target_state, env.chaser_state
    q0 = chaser.position - target.position
    v0 = chaser.rotation @ chaser.velocity - target.rotation @ target.velocity
    history = free_body_rotations(target.rotation, target.omega, env.target_parameters.inertia, 400.0)
    planner = rendezvous.planner
    if kind == "rvgeo":
        tau = next_pass_tau(env)
        period = 360.0 / env.features()["omega_deg_s"]
        for _ in range(3):
            if planner.solve(q0, v0, tau, history)["feasible"]:
                return float(tau)
            tau += period
        return float(tau)
    if kind == "rvbest":
        best, best_cost = 0.0, np.inf
        for tau in np.arange(20.0, 201.0, 10.0):
            plan = planner.solve(q0, v0, float(tau), history)
            if plan["feasible"]:
                cost = 0.05 * tau + 5.0 * plan["delta_v_m_s"]
                if cost < best_cost:
                    best, best_cost = float(tau), cost
        return best
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--row", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="t1", choices=("t1", "t1l"), help="t1l: initial range 12-18 m")
    args = parser.parse_args()
    from env.t1_l_task import task_env_config

    env = T1Env(task_env_config(args.task))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.row.replace(":", "_")
    for seed in parse_seeds(args.seeds):
        path = out / f"{tag}__{seed}.json"
        lock = path.with_suffix(".lock")
        if path.exists():
            continue
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        started = perf_counter()
        result = run_row(env, seed, args.row)
        result["wall_clock_s"] = perf_counter() - started
        path.write_text(json.dumps(result))
        lock.unlink()
        print(args.row, seed, result["event"], round(result["time_s"], 1), round(result["delta_v_m_s"], 2),
              round(result["discounted_return"], 1), result.get("violations", ""), flush=True)


if __name__ == "__main__":
    main()
