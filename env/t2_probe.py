"""ECC mode prediction: the executor's own model of what it would do next under a stage intent (``docs/paper/METHOD.md``).

For a probe intent the frozen executor's stage layer flies the intent for at most three 16 s stages (48 s) and checks,
at its own 2 s cadence, whether it would leave the stage layer for the near-field modes -- the window-committed
conditional G* accepting an entry, or the descent. The result is (event, time, delta-v) and the planned entry delta-v;
``T2StageEnv._opportunity_eval`` evaluates the nine intents of PROBE_GRID and the certification map uses them.

What it is not: no return, value or ranking is computed; no calibration from the run's history is used (planners and
models are fixed; same state -> same answer); a predicted event is the executor model's mode switch, not a capture, and
a non-event within the horizon is not infeasibility.

Executor model (the executor's own rules and inputs, fixed):
* stage layer: ``stage_goal`` (sweep-cone target velocity, stage authority, safety projection) and the stage planner
  from the predicted state, the target attitude from the constant-rate forecast of the decision's attitude and body
  rate; a stage the planner rejects is a hold at the stage's start position (the executor holds);
* the nominal plan is the flown trajectory (no residual model, no calibration);
* the entry rule of ``T2StageEnv._advance`` / ``_try_bridge`` at every 2 s node: descent region first (mouth region,
  inside the operation sphere, or in the cone within the descent radius); otherwise range <= 12 m and radial
  envelope margin >= 0.02 m/s, then for each arrival time of the grid with a forecast port angle <= 35 deg and an
  arrival before the deadline less 45 s, the entry planner (same configuration, the executor's estimated isotropic
  inertia) -- the first feasible plan is the entry event;
* the prediction ends at the horizon (3 stages), at the mission deadline, or when the planned delta-v reaches the
  remaining budget.

The planners are separate instances of the executor's planners, so the prediction never touches the solver state of
the executor that flies the episode (the executed trajectory is the same with or without certification's evaluation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynamics.lie import so3_exp

PROBE_GRID = tuple((float(p), float(z)) for p in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0))
PROBE_STAGES = 3


def executor_gate(q: np.ndarray, v: np.ndarray, rotation: np.ndarray, task, descent_radius_m: float) -> dict[str, Any]:
    """The executor's mode-relevant metrics of a relative state (``compute_t1_metrics`` and ``_descending``): range,
    cone / mouth region, inward radial closing speed against the outer braking envelope, descent region.
    ``q``, ``v``: inertial relative position and velocity; ``rotation``: target attitude."""

    from env.t1_task import cone_quantities
    r = float(np.linalg.norm(q))
    p = rotation.T @ q
    _, _, cone_margin = cone_quantities(p, task)
    in_cone = cone_margin >= 0.0
    mouth = bool(in_cone and r < task.operation_sphere_radius_m + task.mouth_band_m)
    closing = float(-(q / r) @ v) if r > np.finfo(np.float64).eps else float("inf")
    margin = float("inf") if mouth else float(task.outer_radial_closing_speed_limit(r) - closing)
    descending = bool(mouth or r < task.operation_sphere_radius_m or (in_cone and r < descent_radius_m))
    return {"range_m": r, "in_cone": bool(in_cone), "mouth": mouth, "radial_margin_m_s": margin, "descending": descending}


class ExecutorProbe:
    """Fixed executor model of one episode (its own planner instances). ``probe(env)`` at a decision state."""

    def __init__(self, env) -> None:
        from env.t1_p_stage import LocalStagePlanner
        from env.t2_stage import (BRIDGE_DEADLINE_MARGIN_S, PLAN_FORCE_N, STAGE_NODES, STAGE_S, make_window_gstar)
        e = env.env
        self.task, self.mass = e.config.task, float(e.chaser_parameters.mass)
        self.stage_planner = LocalStagePlanner(self.task, self.mass, PLAN_FORCE_N, horizon_s=STAGE_S, nodes=STAGE_NODES)
        bridge = make_window_gstar(env.info_env, elastic=env.window, deadline_margin_s=BRIDGE_DEADLINE_MARGIN_S)
        self.entry_planner, self.entry_speed = bridge.planner, bridge.entry_speed_m_s
        self.descent_radius_m = float(bridge.descent_radius_m)
        self.inertia = np.asarray(env.info_env.target_parameters.inertia, dtype=np.float64)   # the executor's estimate

    # ------------------------------------------------------------------ one node of the entry rule
    def entry_check(self, q, v, rotation, omega, t_left_s: float) -> dict | None:
        """The executor's mode switch at one 2 s check of a predicted state; ``t_left_s``: time to the mission deadline.
        Returns {'kind': 'descent'} / {'kind': 'entry', 'tau': ...} / None, with the entry-planner solves used."""

        from env.t1_gstar import free_body_rotations
        from env.t2_stage import (BRIDGE_ANGLE_DEG, BRIDGE_DEADLINE_MARGIN_S, BRIDGE_ENVELOPE_MARGIN_M_S, BRIDGE_RANGE_M,
                                  BRIDGE_TAUS_S, angle_deg)
        g = executor_gate(q, v, rotation, self.task, self.descent_radius_m)
        if g["descending"]:
            return {"kind": "descent", "solves": 0}
        if g["range_m"] > BRIDGE_RANGE_M or g["radial_margin_m_s"] < BRIDGE_ENVELOPE_MARGIN_M_S:
            return None
        horizon = float(t_left_s) - BRIDGE_DEADLINE_MARGIN_S
        solves = 0
        for tau in BRIDGE_TAUS_S:
            if angle_deg(rotation @ so3_exp(tau * omega) @ self.task.approach_axis, q) > BRIDGE_ANGLE_DEG:
                continue
            if tau > horizon + 1e-6:                     # ConditionalGStar.decide: beyond the deadline, not planned
                continue
            history = free_body_rotations(rotation, omega, self.inertia, tau + 1.0)
            res = self.entry_planner.solve(np.asarray(q, float), np.asarray(v, float), tau, history, None, entry_speed_m_s=self.entry_speed)
            solves += 1
            if res["feasible"]:
                return {"kind": "entry", "tau": float(tau), "dv_plan": float(res.get("delta_v_m_s", 0.0)), "solves": solves}
        return {"kind": None, "solves": solves} if solves else None

    # ------------------------------------------------------------------ one probe intent
    def probe_intent(self, a, q0, v0, rot0, omega, t_abs0: float, dv0: float, mission: dict, geometry: bool) -> dict:
        """Mode prediction of intent ``a`` from the relative state (q0, v0) at absolute time t_abs0 with dv0 spent."""
        from env.t1_p_stage import plan_state
        from env.t2_stage import ENTRY_NODES_S, STAGE_S, spin_frame, stage_goal
        q, v = np.asarray(q0, float).copy(), np.asarray(v0, float).copy()
        deadline, budget = float(mission["deadline_s"]), float(mission["budget_m_s"])
        dv_used, solves, holds = 0.0, 0, 0
        for k in range(PROBE_STAGES):
            off = k * STAGE_S
            rot_k = rot0 @ so3_exp(off * omega)
            frame = spin_frame(rot_k, omega, q, self.task.approach_axis)
            goal = stage_goal(q, v, frame, np.asarray(a, float), self.mass, geometry=geometry, task=self.task)
            plan = self.stage_planner.solve(q, v, goal["q_goal"], goal["v_goal"])
            if not plan["feasible"]:
                holds += 1
            cum = 0.0
            for s in ENTRY_NODES_S:
                t = off + s
                if plan["feasible"]:
                    qs, vs = plan_state(plan, s, self.mass)
                    n = int(round(s / plan["dt"]))
                    cum = float(np.sum(np.linalg.norm(plan["f"][:n], axis=1)) * plan["dt"] / self.mass)
                else:
                    qs, vs = q, np.zeros(3)
                if t_abs0 + t >= deadline - 1e-9:
                    return {"event": 0, "kind": None, "t": max(deadline - t_abs0, 0.0), "dv": dv_used + cum, "end": "deadline", "solves": solves, "holds": holds}
                if dv0 + dv_used + cum >= budget:
                    return {"event": 0, "kind": None, "t": t, "dv": dv_used + cum, "end": "budget", "solves": solves, "holds": holds}
                hit = self.entry_check(qs, vs, rot0 @ so3_exp(t * omega), omega, deadline - (t_abs0 + t))
                if hit is not None:
                    solves += hit["solves"]
                    if hit["kind"] is not None:
                        out = {"event": 1, "kind": hit["kind"], "t": t, "dv": dv_used + cum, "tau": hit.get("tau"), "end": "event",
                               "dv_entry": float(hit.get("dv_plan", 0.0)), "solves": solves, "holds": holds}
                        return out
            if plan["feasible"]:
                q, v = plan_state(plan, STAGE_S, self.mass)
                dv_used += cum
            else:
                v = np.zeros(3)
        return {"event": 0, "kind": None, "t": PROBE_STAGES * STAGE_S, "dv": dv_used, "end": "horizon", "solves": solves, "holds": holds}


__all__ = ["PROBE_GRID", "PROBE_STAGES", "ExecutorProbe", "executor_gate"]
