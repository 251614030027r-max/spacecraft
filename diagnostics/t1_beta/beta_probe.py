"""B0/B1/B2 of the reference-shaping candidate beta (prepared; running needs the user's approval).

beta adds a force-smoothness term to the G* rendezvous planner only:

    J_G(beta) = J_G(0) + c * sum_{i=1}^{N-1} |f_i - f_{i-1}|^2,   c = beta * dt / (m * F_ref)

F_ref = the planner's force bound (0.8 * 2 N), dt = tau / nodes; the term is in m/s.
beta = 0 keeps the planner's original problem object (no term is added), so the frozen
behaviour is the same code path. The MPC, the recovery planner (full force), the task,
reward and truth are unchanged. The task intent is the earliest-feasible rule (keep the
plan, kappa 0.7); no SAC model is involved.

    python -B -m diagnostics.t1_beta.beta_probe --beta 1 --openings 295001,... --output-dir <dir> --workers 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np

from diagnostics.t1_rootcause.common import mode_action
from env.t1_gstar import RendezvousPlanner
from env.t1_gym import T1GStarEnv


class SmoothRendezvousPlanner(RendezvousPlanner):
    """The G* planner with the force-smoothness term (beta > 0 only)."""

    def __init__(self, task, mass: float, max_force_n: float, *, beta: float, **kwargs: Any) -> None:
        assert beta > 0.0
        super().__init__(task, mass, max_force_n, **kwargs)
        self.beta = float(beta)
        # one combined parameter: a product of two parameters would not be DPP
        self.smooth_coefficient = cp.Parameter(nonneg=True)
        self.problem = cp.Problem(
            cp.Minimize(self.problem.objective.expr
                        + self.smooth_coefficient * cp.sum_squares(self.f[1:] - self.f[:-1])),
            self.problem.constraints)

    def solve(self, q0, v0, tau_s, history, guess=None, iterations: int = 4) -> dict[str, Any]:
        self.smooth_coefficient.value = self.beta * (tau_s / self.nodes) / (self.mass * self.max_force_n)
        return super().solve(q0, v0, tau_s, history, guess, iterations)


class RecordingPlanner:
    """Delegates to a planner (the frozen one at beta = 0) and records every solve."""

    def __init__(self, inner: RendezvousPlanner) -> None:
        self.inner, self.records = inner, []

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def solve(self, q0, v0, tau_s, history, guess=None, iterations: int = 4) -> dict[str, Any]:
        inner = self.inner
        started = perf_counter()
        result = inner.solve(q0, v0, tau_s, history, guess, iterations)
        record = {"tau_s": tau_s, "feasible": result["feasible"], "status": result["status"],
                  "objective": result["delta_v_m_s"], "wall_s": perf_counter() - started}
        if result["feasible"]:
            dt, f = result["dt"], np.asarray(result["f"])
            coefficient = getattr(inner, "smooth_coefficient", None)
            record["delta_v_plan_m_s"] = float(dt / inner.mass * np.sum(np.linalg.norm(f, axis=1)))
            record["smooth_sum_n2"] = float(np.sum(np.linalg.norm(f[1:] - f[:-1], axis=1) ** 2))
            record["j_smooth_m_s"] = 0.0 if coefficient is None else float(coefficient.value) * record["smooth_sum_n2"]
            record["j_range_m_s"] = float(result["delta_v_m_s"] - record["delta_v_plan_m_s"] - record["j_smooth_m_s"])
            if not self.records:  # the first plan's nodes, for the reference-difference criterion
                record["q_nodes"] = np.asarray(result["q"]).tolist()
        self.records.append(record)
        return result


def install(env: T1GStarEnv, beta: float) -> None:
    """Wrap the reset-built G* planner (beta = 0: the frozen planner itself) to record solves."""

    g = env.generator
    assert g is not None
    old = g.planner
    inner = old if beta == 0.0 else SmoothRendezvousPlanner(old.task, old.mass, old.max_force_n, beta=beta)
    g.planner = RecordingPlanner(inner)


def episode(beta: float, opening: int, output_dir: str) -> None:
    path = Path(output_dir) / f"beta{beta}_{opening}.json"
    if path.exists():
        return
    started = perf_counter()
    env = T1GStarEnv()
    env.reset(options={"opening": opening})
    install(env, beta)
    g = env.generator
    gamma, ret, k, log = env.env.config.reward.gamma, 0.0, 0, []
    while True:
        action = mode_action("E", env, None)
        _, reward, terminated, truncated, info = env.step(action)
        x = info["execution"]
        log.append({"k": k, "status": x["plan_status"], "slip_s": x["slip_s"], "reward": reward,
                    "plan_deviation_end_m": x["plan_deviation_end_m"], "force_saturated_fraction": x["force_saturated_fraction"],
                    "max_slack_solved": x["max_slack_solved"]})
        ret += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    s = info["episode_summary"]
    far = [d for d in log if d["status"] != "descent"]
    result = {
        "beta": beta, "opening": opening, "event": s["event"], "discounted_return": ret, "time_s": s["time_s"],
        "delta_v_m_s": s["delta_v_m_s"], "min_margins": s["min_margins"], "violations": s["violations"],
        "end_phase": s["end_phase"],
        "far_field_new_plans": sum(d["status"].startswith(("optimal", "recovery")) for d in far),
        "far_field_delta_v_m_s": float(sum((-d["reward"] - 0.1) / 5.0 for d in far)),
        "slips": sum(d["slip_s"] > 0 for d in log), "holds": sum(d["status"].startswith("hold") for d in log),
        "planner_records": g.planner.records, "decision_log": log, "wall_s": perf_counter() - started,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    print("beta", beta, opening, result["event"], round(ret, 2), "far-field plans", result["far_field_new_plans"],
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.beta, int(o), str(args.output_dir)) for o in args.openings.split(",")]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(episode, jobs)
    else:
        for job in jobs:
            episode(*job)


if __name__ == "__main__":
    main()
