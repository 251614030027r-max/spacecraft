"""T1-P stage layer (research plan T1_P_FINAL_REDESIGN_FOR_FEASIBILITY_20261018, sec. 2-3).

SAC chooses, at each review, a three-dimensional 12 s stage goal around the zero-thrust coast of
the current *inertial* relative state; a local G* (12 s, 12 nodes) plans to it; the shared MPC flies
the plan; near the target a short G* bridge (the T1-N conditional planner, exactly one of the
fixed arrival times {20, 30, 40} s) connects to the 6.3 m entry, then the T1 descent and capture.

Stage action (``stage_goal``): with the inertial relative state (q, v) at the review and the
planner's own model (double integrator, differential gravity neglected as in G*),
    q_coast = q + H v,  v_coast = v,  H = 12 s;
    e_r = q / |q|;  e_t = the tangential part of the port axis predicted (torque-free) at t + H,
          normalised (fallback: the orbit normal's tangential part, then a fixed axis);  e_b = e_r x e_t;
    u = (-a_r e_r + a_t e_t + a_b e_b) / max(1, |a|),   d_max = 0.70 (1.6 N / 106 kg) H^2 / 2 = 0.761 m;
    q_goal = q_coast + d_max u,   v_goal = v_coast + (2 d_max / H) u.
a = 0 is the coast (the physical drift), not a pre-selected rendezvous.

Local planner (``LocalStagePlanner``): minimum of
    10 |q_H - q_goal|^2 / (1 m)^2 + 2 |v_H - v_goal|^2 / (0.1 m/s)^2 + 0.1 dv / (1 m/s) (+ range penalty)
under hard constraints: |F| <= 1.6 N (2-norm), the operation sphere kept out (half-spaces linearised at
the previous iterate), the radial braking envelope (every chord), the range cap -- imposed at the
nodes **and at four interior points of every 1 s node** (exact double-integrator positions and
velocities, linear in the decision variables). Every accepted plan is then checked at 0.1 s
(range >= 6.05 m, closing speed under the envelope, range <= 29.5 m); a plan that fails is re-solved
with the keep-out radius raised by 0.2 m and 0.01 m/s more closing-speed margin (at most twice), else the
request is not executable. The truth rules are never relaxed.

Potential (training only, ``potential``): Phi(s) = -min(25 m, |q - p_mouth(t + 24 s)|) in reward units
per metre, p_mouth the 6.3 m entry point predicted with the torque-free model from the current state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np

from dynamics.lie import so3_exp
from env.t1_gstar import GStarPlan, _plan_position, _rotation_at, free_body_rotations
from env.t1_n_task import T1N_ENTRY_SPEED_M_S, T1N_KAPPA, make_conditional_gstar
from env.t1_task import lvlh_basis

T1P_VERSION = "t1p-0.1"
STAGE_S = 12.0
STAGE_NODES = 12
PLAN_FORCE_N = 1.6
D_MAX_FRACTION = 0.70
SUBSTEPS = (0.25, 0.5, 0.75, 1.0)
KEEP_OUT_M = 6.3
MOUTH_LOOKAHEAD_S = 24.0
POTENTIAL_CAP_M = 25.0
#: entry bridge (sec. 2.3): readiness and the fixed candidate arrival times
BRIDGE_RANGE_M = 9.0
BRIDGE_ENVELOPE_MARGIN_M_S = 0.02
BRIDGE_ANGLE_DEG = 25.0
BRIDGE_TAUS_S = (20.0, 30.0, 40.0)


def d_max(mass: float, force_n: float = PLAN_FORCE_N, fraction: float = D_MAX_FRACTION, h: float = STAGE_S) -> float:
    return float(fraction * (force_n / mass) * h * h / 2.0)


def relative_inertial(env) -> tuple[np.ndarray, np.ndarray]:
    t, c = env.target_state, env.chaser_state
    return c.position - t.position, c.rotation @ c.velocity - t.rotation @ t.velocity


def stage_basis(env, h: float = STAGE_S) -> np.ndarray:
    """Rows e_r, e_t, e_b (ECI) of the stage at the current state (deterministic, orthonormal)."""

    q, _ = relative_inertial(env)
    e_r = q / np.linalg.norm(q)
    t = env.target_state
    history = free_body_rotations(t.rotation, t.omega, env.target_parameters.inertia, h + 1.0)
    rotation, _ = _rotation_at(history, h)
    candidates = (rotation @ env.config.task.approach_axis, lvlh_basis(t)[:, 2], np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0]))
    for c in candidates:
        tangential = c - float(c @ e_r) * e_r
        n = float(np.linalg.norm(tangential))
        if n > 1e-3:
            e_t = tangential / n
            break
    e_b = np.cross(e_r, e_t)
    return np.vstack([e_r, e_t, e_b])


def stage_goal(env, action: np.ndarray, h: float = STAGE_S) -> dict[str, Any]:
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
    q, v = relative_inertial(env)
    basis = stage_basis(env, h)
    u = (-a[0] * basis[0] + a[1] * basis[1] + a[2] * basis[2]) / max(1.0, float(np.linalg.norm(a)))
    dm = d_max(env.chaser_parameters.mass, h=h)
    q_coast, v_coast = q + h * v, v.copy()
    return {"a": a, "basis": basis, "u": u, "d_max": dm, "q_coast": q_coast, "v_coast": v_coast,
            "q_goal": q_coast + dm * u, "v_goal": v_coast + (2.0 * dm / h) * u}


def plan_state(plan: dict[str, Any], s: float, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of a plan s seconds after its start (exact; held after its end)."""

    q, v, f, dt = plan["q"], plan["v"], plan["f"], plan["dt"]
    n = len(f)
    if s >= n * dt:
        return q[-1] + v[-1] * (s - n * dt), v[-1].copy()
    i = int(np.clip(np.floor(max(s, 0.0) / dt), 0, n - 1))
    h = max(s, 0.0) - i * dt
    return q[i] + v[i] * h + 0.5 * f[i] / mass * h * h, v[i] + f[i] / mass * h


class LocalStagePlanner:
    """12 s / 12 node SCP to a soft stage goal under hard physical constraints (module docstring)."""

    def __init__(self, task, mass: float, max_force_n: float = PLAN_FORCE_N, *, horizon_s: float = STAGE_S,
                 nodes: int = STAGE_NODES, max_range_m: float = 20.0) -> None:
        self.task, self.mass, self.max_force_n = task, float(mass), float(max_force_n)
        self.horizon_s, self.nodes, self.max_range_m = float(horizon_s), int(nodes), float(max_range_m)
        self.hard_range_m = 28.0
        self.range_penalty_per_m_s = 0.01
        n, dt = self.nodes, self.horizon_s / self.nodes
        self.dt = dt
        self.q, self.v, self.f = cp.Variable((n + 1, 3)), cp.Variable((n + 1, 3)), cp.Variable((n, 3))
        self.q0, self.v0 = cp.Parameter(3), cp.Parameter(3)
        self.qg, self.vg = cp.Parameter(3), cp.Parameter(3)
        self.normals = cp.Parameter((n * len(SUBSTEPS), 3))
        self.range_cap = cp.Parameter(nonneg=True)
        self.keep_out = cp.Parameter(nonneg=True)
        #: closing-speed margin under the envelope (m/s), raised on a re-solve
        self.envelope_margin = cp.Parameter(nonneg=True)
        q, v, f = self.q, self.v, self.f
        cons = [q[0] == self.q0, v[0] == self.v0,
                q[1:] == q[:-1] + dt * v[:-1] + 0.5 * dt * dt / self.mass * f,
                v[1:] == v[:-1] + dt / self.mass * f,
                cp.norm(f, axis=1) <= self.max_force_n]
        sphere = float(task.operation_sphere_radius_m)
        knots = task.envelope_knots_m
        rows = []
        for i in range(n):
            for s_frac in SUBSTEPS:
                s = s_frac * dt
                rows.append((q[i] + v[i] * s + 0.5 * s * s / self.mass * f[i], v[i] + s / self.mass * f[i]))
        positions = cp.vstack([p for p, _ in rows])
        velocities = cp.vstack([w for _, w in rows])
        radius = cp.sum(cp.multiply(self.normals, positions), axis=1)
        closing = -cp.sum(cp.multiply(self.normals, velocities), axis=1)
        cons.append(radius >= self.keep_out)
        cons.append(cp.norm(positions, axis=1) <= self.range_cap)
        for a, b in zip(knots[:-1], knots[1:]):
            value = task.braking_speed_limit(a)
            slope = (task.braking_speed_limit(b) - value) / (b - a)
            cons.append(closing <= value + slope * (radius - sphere - a) - self.envelope_margin)
        excess = cp.sum(cp.pos(cp.norm(q[1:], axis=1) - self.max_range_m))
        dv = dt / self.mass * cp.sum(cp.norm(f, axis=1))
        objective = (10.0 * cp.sum_squares(q[n] - self.qg) + 2.0 / 0.01 * cp.sum_squares(v[n] - self.vg)
                     + 0.1 * dv + self.range_penalty_per_m_s * dt * excess)
        self.problem = cp.Problem(cp.Minimize(objective), cons)

    def _guess(self, q0: np.ndarray, v0: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        n, dt, h = self.nodes, self.dt, self.horizon_s
        offset = q_goal - (q0 + h * v0)
        return np.array([q0 + v0 * (k * dt) + offset * (k * dt / h) ** 2 for k in range(n + 1)])

    def _normals(self, guess_q: np.ndarray, guess_v: np.ndarray, guess_f: np.ndarray | None) -> np.ndarray:
        out = []
        for i in range(self.nodes):
            for s_frac in SUBSTEPS:
                s = s_frac * self.dt
                f = np.zeros(3) if guess_f is None else guess_f[i]
                p = guess_q[i] + guess_v[i] * s + 0.5 * s * s / self.mass * f
                out.append(p / max(np.linalg.norm(p), 1e-9))
        return np.array(out)

    def verify(self, plan: dict[str, Any], step_s: float = 0.1) -> list[str]:
        """0.1 s check of the planned path against the far-field rules (never the truth judge)."""

        task, sphere = self.task, float(self.task.operation_sphere_radius_m)
        problems = []
        for k in range(int(round(self.horizon_s / step_s)) + 1):
            p, w = plan_state(plan, k * step_s, self.mass)
            r = float(np.linalg.norm(p))
            if r < sphere + 0.05:
                problems.append("sphere")
            if r > 29.5:
                problems.append("range")
            closing = -float(p @ w) / max(r, 1e-9)
            if closing > task.braking_speed_limit(r - sphere) + 1e-6:
                problems.append("envelope")
            if problems:
                break
        return problems

    def solve(self, q0: np.ndarray, v0: np.ndarray, q_goal: np.ndarray, v_goal: np.ndarray,
              iterations: int = 4) -> dict[str, Any]:
        r0 = float(np.linalg.norm(q0))
        # room to stop the *whole* relative velocity at the plan force (a tangential drift also
        # grows the range; the radial part alone made far-field stages spuriously infeasible)
        stopping = float(v0 @ v0) / (2.0 * self.max_force_n / self.mass)
        result: dict[str, Any] = {"feasible": False, "status": "not_solved"}
        for attempt in range(3):
            cache = getattr(self.problem, "_solver_cache", None)
            if cache is not None:
                cache.clear()
            self.q0.value, self.v0.value, self.qg.value, self.vg.value = q0, v0, q_goal, v_goal
            self.range_cap.value = min(max(self.max_range_m, r0 + stopping), self.hard_range_m)
            self.keep_out.value = max(min(KEEP_OUT_M + 0.2 * attempt, r0 - 0.05), 6.05)
            self.envelope_margin.value = 0.01 * attempt
            guess_q = self._guess(q0, v0, q_goal)
            guess_v = np.vstack([v0] * (self.nodes + 1))
            guess_f = None
            status, ok = "not_solved", False
            for _ in range(iterations):
                self.normals.value = self._normals(guess_q, guess_v, guess_f)
                try:
                    self.problem.solve(solver="CLARABEL")
                except cp.error.SolverError as error:
                    status = f"solver_error:{error}"
                    break
                status = self.problem.status
                if status not in ("optimal", "optimal_inaccurate"):
                    break
                new_q = np.asarray(self.q.value)
                change = float(np.max(np.linalg.norm(new_q - guess_q, axis=1)))
                guess_q, guess_v, guess_f = new_q, np.asarray(self.v.value), np.asarray(self.f.value)
                ok = True
                if change < 1e-2:
                    break
            if not ok:
                result = {"feasible": False, "status": status, "attempt": attempt}
                break
            plan = {"q": guess_q, "v": guess_v, "f": guess_f, "dt": self.dt, "tau_s": self.horizon_s}
            problems = self.verify(plan)
            if not problems:
                q_end, v_end = guess_q[-1], guess_v[-1]
                result = {"feasible": True, "status": str(status), "attempt": attempt, **plan,
                          "q_goal": np.asarray(q_goal).copy(), "v_goal": np.asarray(v_goal).copy(),
                          "terminal_position_error_m": float(np.linalg.norm(q_end - q_goal)),
                          "terminal_velocity_error_m_s": float(np.linalg.norm(v_end - v_goal)),
                          "delta_v_m_s": float(np.sum(np.linalg.norm(guess_f, axis=1)) * self.dt / self.mass)}
                break
            result = {"feasible": False, "status": f"unsafe_between_nodes:{'+'.join(problems)}", "attempt": attempt}
        return result


def mouth_point(env, lookahead_s: float = MOUTH_LOOKAHEAD_S, entry_radius_m: float = 6.3) -> np.ndarray:
    t = env.target_state
    history = free_body_rotations(t.rotation, t.omega, env.target_parameters.inertia, lookahead_s + 1.0)
    rotation, _ = _rotation_at(history, lookahead_s)
    return rotation @ (entry_radius_m * env.config.task.approach_axis)


def potential(env) -> float:
    q, _ = relative_inertial(env)
    return -min(POTENTIAL_CAP_M, float(np.linalg.norm(q - mouth_point(env))))


@dataclass
class T1PGenerator:
    """Reference generator of T1-P: far-field stages, the entry bridge, the T1 descent.

    Modes: ``stage`` (a local plan is flown), ``hold`` (no executable stage: brake and hold the
    current inertial position; SAC is asked again at the next 2 s step), ``bridge`` (the conditional
    G* plan to the 6.3 m entry), ``descent`` (the T1 descent carrot, whenever the truth state is in the
    mouth region, the sphere or the cone within 7 m)."""

    planner: LocalStagePlanner
    bridge: Any
    mass: float
    replan_deviation_m: float = 1.0
    mode: str = "hold"
    plan: dict[str, Any] | None = None
    plan_start_s: float = 0.0
    hold_position: np.ndarray | None = None
    request: dict[str, Any] | None = None
    last_status: str = "none"
    last_outcome: str = "none"
    event: str | None = None
    bridge_tau_s: float | None = None
    plan_wall_s: list[float] = field(default_factory=list)
    _descent: GStarPlan = field(default_factory=lambda: GStarPlan(tau_s=0.0, kappa=T1N_KAPPA))

    # ---------------------------------------------------------------- #
    def descending(self, env, metrics=None) -> bool:
        return bool(self.bridge._descending(env, metrics or env.metrics()))

    def stage_remaining_s(self, env) -> float:
        if self.mode != "stage" or self.plan is None:
            return 0.0
        return max(float(self.plan["tau_s"]) - (env.time_s - self.plan_start_s), 0.0)

    def stage_deviation(self, env) -> float:
        if self.plan is None:
            return 0.0
        q, _ = relative_inertial(env)
        p, _ = plan_state(self.plan, env.time_s - self.plan_start_s, self.mass)
        return float(np.linalg.norm(p - q))

    def _enter_hold(self, env, status: str) -> None:
        self.mode, self.plan, self.last_status = "hold", None, status
        q, _ = relative_inertial(env)
        if self.hold_position is None:
            self.hold_position = q.copy()

    def decide_stage(self, env, action: np.ndarray) -> None:
        """Review decision: plan the requested stage (or keep a usable stage plan, or hold)."""

        started = perf_counter()
        goal = stage_goal(env, action, self.planner.horizon_s)
        q, v = relative_inertial(env)
        result = self.planner.solve(q, v, goal["q_goal"], goal["v_goal"])
        self.plan_wall_s.append(perf_counter() - started)
        self.request = {**goal, "result_status": result["status"]}
        if result["feasible"]:
            self.mode, self.plan, self.plan_start_s = "stage", result, float(env.time_s)
            self.hold_position, self.last_status, self.last_outcome = None, "stage:" + result["status"], "accepted"
            self.request["terminal_position_error_m"] = result["terminal_position_error_m"]
            self.request["terminal_velocity_error_m_s"] = result["terminal_velocity_error_m_s"]
            return
        self.last_outcome = "rejected"
        if self.mode == "stage" and self.plan is not None and self.stage_remaining_s(env) > 0.0 \
                and self.stage_deviation(env) <= self.replan_deviation_m:
            self.last_status = "stage:kept_infeasible_request"
            return
        self._enter_hold(env, f"hold:rejected:{result['status']}")

    def continue_step(self, env) -> None:
        """Continue decision at a non-review step: a stage plan flies on (re-planned only through a
        > 1 m deviation event); the bridge keeps or re-plans the same arrival (conditional G*)."""

        if self.mode == "bridge":
            self.bridge.decide(env, self.bridge.remaining_s(env))
            if self.bridge.plan is None and not self.bridge._descending(env, env.metrics()):
                self.mode, self.event = "hold", "bridge_lost"
                self._enter_hold(env, "hold:bridge_lost")

    def after_step(self, env) -> None:
        """Mode bookkeeping at the end of a 2 s step: descent, deviation, bridge readiness."""

        self.event = None
        m = env.metrics()
        if self.descending(env, m):
            if self.mode != "descent":
                self.mode = "descent"
            self.bridge.plan, self.bridge.last_status = None, "descent"
            return
        if self.mode == "descent":  # left the descent region: back to the far field
            self.mode, self.event = "hold", "left_descent"
            self.hold_position = None
            self._enter_hold(env, "hold:left_descent")
            return
        if self.mode == "stage" and self.stage_deviation(env) > self.replan_deviation_m:
            self.event = "stage_deviation"
            self.hold_position = None
            self._enter_hold(env, "hold:stage_deviation")
        if self.mode in ("stage", "hold") and self.try_bridge(env, m):
            self.event = "bridge_start"

    def try_bridge(self, env, metrics) -> bool:
        if metrics.range_m > BRIDGE_RANGE_M or metrics.radial_envelope_margin_m_s < BRIDGE_ENVELOPE_MARGIN_M_S:
            return False
        q, _ = relative_inertial(env)
        los = q / np.linalg.norm(q)
        t = env.target_state
        history = free_body_rotations(t.rotation, t.omega, env.target_parameters.inertia, max(BRIDGE_TAUS_S) + 1.0)
        for tau in BRIDGE_TAUS_S:
            rotation, _ = _rotation_at(history, tau)
            port = rotation @ env.config.task.approach_axis
            if np.degrees(np.arccos(np.clip(port @ los, -1.0, 1.0))) > BRIDGE_ANGLE_DEG:
                continue
            self.bridge.plan, self.bridge.hold_position = None, None
            self.bridge.decide(env, tau)
            if self.bridge.plan is not None:
                self.mode, self.plan, self.bridge_tau_s = "bridge", None, tau
                self.last_status = "bridge:" + self.bridge.last_status
                return True
        return False

    # ---------------------------------------------------------------- #
    def __call__(self, env, metrics):
        if self.descending(env, metrics):
            return self._descent(env, metrics)
        if self.mode == "bridge":
            return self.bridge(env, metrics)
        count = env.mpc_config.horizon_steps + 1
        dt = env.config.dt_s
        path = np.zeros((3, count))
        for j in range(count):
            rotation = env.target_state.rotation @ so3_exp(j * dt * env.target_state.omega)
            if self.mode == "stage" and self.plan is not None:
                p, _ = plan_state(self.plan, env.time_s + j * dt - self.plan_start_s, self.mass)
            else:
                if self.hold_position is None:
                    self.hold_position, _ = relative_inertial(env)
                p = self.hold_position
            path[:, j] = rotation.T @ p
        return path[:, 0].copy(), path


def make_t1p_generator(env) -> T1PGenerator:
    planner = LocalStagePlanner(env.config.task, env.chaser_parameters.mass)
    bridge = make_conditional_gstar(env, kappa=T1N_KAPPA, entry_speed_m_s=T1N_ENTRY_SPEED_M_S)
    return T1PGenerator(planner=planner, bridge=bridge, mass=env.chaser_parameters.mass)


__all__ = [
    "BRIDGE_ANGLE_DEG", "BRIDGE_RANGE_M", "BRIDGE_TAUS_S", "LocalStagePlanner", "STAGE_S", "T1PGenerator",
    "T1P_VERSION", "d_max", "make_t1p_generator", "mouth_point", "plan_state", "potential", "relative_inertial",
    "stage_basis", "stage_goal",
]
