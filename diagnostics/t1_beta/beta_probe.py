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
from env.t1_gstar import RendezvousPlanner, _plan_position
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
        record = {"kind": "attempted", "tau_s": tau_s, "feasible": result["feasible"], "status": result["status"],
                  "objective": result["delta_v_m_s"], "wall_s": perf_counter() - started}
        if result["feasible"]:
            dt, f = result["dt"], np.asarray(result["f"])
            coefficient = getattr(inner, "smooth_coefficient", None)
            record["delta_v_plan_m_s"] = float(dt / inner.mass * np.sum(np.linalg.norm(f, axis=1)))
            record["smooth_sum_n2"] = float(np.sum(np.linalg.norm(f[1:] - f[:-1], axis=1) ** 2))
            record["j_smooth_m_s"] = 0.0 if coefficient is None else float(coefficient.value) * record["smooth_sum_n2"]
            record["j_range_m_s"] = float(result["delta_v_m_s"] - record["delta_v_plan_m_s"] - record["j_smooth_m_s"])
        self.records.append(record)
        return result


def install(env: T1GStarEnv, beta: float) -> None:
    """Wrap the reset-built G* planner (beta = 0: the frozen planner itself) to record solves."""

    g = env.generator
    assert g is not None
    old = g.planner
    inner = old if beta == 0.0 else SmoothRendezvousPlanner(old.task, old.mass, old.max_force_n, beta=beta)
    g.planner = RecordingPlanner(inner)


GRID_S = 1.0  # time grid of the accepted first plan (time-parameterised comparison across beta)
NEW_PLAN = ("optimal", "recovery")


def _accepted(g, env) -> dict:
    """The plan G* adopted in this decision (not a candidate it tried)."""

    plan, mass = g.plan, g.planner.mass
    times = np.arange(0.0, float(plan["tau_s"]) + 1e-9, GRID_S)
    return {"kind": "accepted", "plan_start_s": float(g.plan_start_s), "tau_s": float(plan["tau_s"]),
            "arrival_abs_s": float(g.plan_start_s + plan["tau_s"]), "dt": float(plan["dt"]),
            "delta_v_plan_m_s": float(plan["dt"] / mass * np.sum(np.linalg.norm(plan["f"], axis=1))),
            "status": g.last_status,
            "position_on_grid_m": [_plan_position(plan, float(t), mass).tolist() for t in times]}


def episode(beta: float, opening: int, output_dir: str) -> None:
    """Rule intent (earliest feasible, keep, kappa 0.7) with the beta planner.

    Per decision, before the step: time, remaining time, deviation from the live plan, its
    usability (deviation <= replan_deviation_m and time remaining), the request; after: the
    G* status and, for a new plan, its trigger: "first" (k = 0), "expired" (no time remained),
    "deviation" (live plan more than replan_deviation_m off), "other" (anything else, e.g. a
    recovery or a request that did not agree). Delta-v per decision is the env's exact
    accumulator (no reward inversion); far field = decisions whose status is not "descent".
    """

    path = Path(output_dir) / f"beta{beta}_{opening}.json"
    if path.exists():
        return
    started = perf_counter()
    env = T1GStarEnv()
    obs0, _ = env.reset(options={"opening": opening})
    install(env, beta)
    g = env.generator
    gamma, ret, k, log, accepted = env.env.config.reward.gamma, 0.0, 0, [], []
    while True:
        e = env.env
        remaining = g.remaining_s(e)
        deviation = None
        if g.plan is not None:
            q = e.chaser_state.position - e.target_state.position
            deviation = float(np.linalg.norm(_plan_position(g.plan, e.time_s - g.plan_start_s, g.planner.mass) - q))
        usable = deviation is not None and deviation <= g.replan_deviation_m and remaining > 0.0
        action = mode_action("E", env, None)
        tau, kappa = env.intent(action)
        dv_before = e._delta_v
        _, reward, terminated, truncated, info = env.step(action)
        x = info["execution"]
        status = x["plan_status"]
        trigger = None
        if status.startswith(NEW_PLAN):
            trigger = ("first" if k == 0 else "expired" if remaining <= 0.0
                       else "deviation" if not usable else "other")
            accepted.append(_accepted(g, env) | {"k": k, "trigger": trigger})
        log.append({"k": k, "t_s": float(e.time_s) - e.config.dt_s * e.config.decision_steps,
                    "remaining_before_s": remaining, "deviation_before_m": deviation, "usable_before": usable,
                    "tau_request_s": tau, "status": status, "trigger": trigger, "slip_s": x["slip_s"],
                    "delta_v_m_s": e._delta_v - dv_before, "reward": reward,
                    "terminal": bool(terminated or truncated), "event": info["event"],
                    "force_saturated_fraction": x["force_saturated_fraction"],
                    "max_slack_solved": x["max_slack_solved"], "plan_deviation_end_m": x["plan_deviation_end_m"]})
        ret += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    s = info["episode_summary"]
    far = [d for d in log if d["status"] != "descent"]
    triggers = {t: sum(a["trigger"] == t for a in accepted) for t in ("first", "deviation", "expired", "other")}
    result = {
        "beta": beta, "opening": opening, "initial_observation": obs0.tolist(),
        "event": s["event"], "discounted_return": ret, "time_s": s["time_s"], "delta_v_m_s": s["delta_v_m_s"],
        "min_margins": s["min_margins"], "violations": s["violations"], "end_phase": s["end_phase"],
        "terminal_decision_far_field": bool(far and far[-1]["terminal"]),
        "far_field_delta_v_m_s": float(sum(d["delta_v_m_s"] for d in far)),
        "far_field_duration_s": 2.0 * len(far), "new_plan_triggers": triggers,
        "slips": sum(d["slip_s"] > 0 for d in log), "holds": sum(d["status"].startswith("hold") for d in log),
        "numerical_failure": info["event"] == "numerical_failure",
        "planner_failures": sum(not r["feasible"] for r in g.planner.records),
        "accepted_plans": accepted, "attempted_solves": g.planner.records, "decision_log": log,
        "wall_s": perf_counter() - started,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    print("beta", beta, opening, result["event"], round(ret, 2), "triggers", triggers, flush=True)


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
