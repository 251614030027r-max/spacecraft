"""G*: arrival-time / descent-pace reference generator for T1 (versioned, R0 prototype).

The task intent is two numbers chosen every 2 s decision:

``tau_s``  remaining time until the chaser is planned to sit at the mouth
           entry point (on the port axis, just outside the operation sphere)
           and co-rotate with it;
``kappa``  descent pace inside the cone, as a fraction of the braking envelope.

Approach (outside the cone): the reference moves, at a constant fraction of the
remaining time per second, from the chaser's current inertial line of sight and
range to the moving entry point: direction ``slerp(u0, R(t) n_p, h)`` and
radius ``r0 + (r_e - r0) h`` with ``h = (t - t0) / tau`` clipped to [0, 1].
Because the endpoint is the *moving* entry point, choosing tau chooses where
on its sweep the port is met. The radius never drops below ``r_e`` before the
direction reaches the axis, so the reference does not cut through the sphere.
A linear ``h`` keeps the plan consistent when it is re-anchored every 2 s with
the remaining time.

Descent (state inside the cone, or inside the sphere): a target-frame carrot
on the port axis that advances toward the precapture point at ``kappa`` times
the braking envelope speed.

``tau <= 0`` means "go now": the approach reference is Pure's fixed terminal
setpoint (so the row with tau = 0 everywhere is Pure up to the descent pace).

v0.2 (R0): v0.1 kept the approach reference on the moving entry point until
the state reached the mouth band. At 2.7 deg/s, co-rotating at 7-8 m needs
1.7-1.9 N of the 2 N per-axis force for the centripetal term alone; the h35
tracking MPC then lost the tangential race and was flung out to 30 m
(290001, tau 60 and geo rows). The carrot now starts as soon as the state is
in the cone, and an expired tau hands the approach to the Pure setpoint.

Nothing here touches the MPC: the reference goes through the same
``external_reference`` / ``external_reference_path`` channel as G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np

from dynamics.lie import so3_exp
from env.t1_task import T1Metrics, _slerp_deterministic

#: Version of the G* reference generator used by M0/M1 training (frozen):
#: v0.3 (R1 pass 2, e80b476) + v0.3.1 range cap (83f035a) + v0.3.2 fresh solves (acc3366)
#: + v0.4 plan stability (an unmeetable earlier request or an infeasible request keeps a
#: usable plan; no usable plan -> full-force earliest-arrival recovery before the hold).
GSTAR_VERSION = "0.5"


@dataclass
class GStarPlan:
    tau_s: float
    kappa: float
    #: "gstar" approach, or "pure" (fixed terminal setpoint until the mouth)
    approach: str = "gstar"
    entry_radius_m: float = 6.3
    coriolis_force_fraction: float = 0.5
    #: set by begin(): decision start time, line of sight and range there
    t0: float = 0.0
    u0: np.ndarray | None = None
    r0: float = 0.0

    def begin(self, env) -> None:
        los = env.chaser_state.position - env.target_state.position
        self.r0 = float(np.linalg.norm(los))
        self.u0 = los / self.r0
        self.t0 = float(env.time_s)

    # ------------------------------------------------------------------ #
    def _predicted_rotations(self, env, count: int) -> list[np.ndarray]:
        dt = env.config.dt_s
        return [
            env.target_state.rotation @ so3_exp(j * dt * env.target_state.omega)
            for j in range(count)
        ]

    def _approach_point(self, env, rotation: np.ndarray, t: float) -> np.ndarray:
        n_p = env.config.task.approach_axis
        h = 1.0 if self.tau_s <= 1e-9 else float(np.clip((t - self.t0) / self.tau_s, 0.0, 1.0))
        port_inertial = rotation @ n_p
        direction = _slerp_deterministic(self.u0, port_inertial, h)
        radius = self.r0 + (self.entry_radius_m - self.r0) * h
        return rotation.T @ (radius * direction)

    def _descent_path(self, env, metrics: T1Metrics, count: int) -> np.ndarray:
        task = env.config.task
        n_p = task.approach_axis
        desired = np.asarray(task.desired_position, dtype=np.float64)
        d = max(float(n_p @ (metrics.position_target_m - desired)), 0.0)
        # Legal pace: the braking envelope, and what the force bound can
        # carry in the rotating frame. Moving along the spinning axis at v
        # costs 2 omega v m of Coriolis force; half the per-axis bound is
        # allowed for it (the rest holds the centripetal term and attitude).
        # Without this cap, kappa 0.7 at 2.7 deg/s saturated every axis at
        # 5.6 m and the FOV was lost (R1 pass 1, 291000).
        omega = float(np.linalg.norm(env.target_state.omega))
        coriolis_speed = (
            self.coriolis_force_fraction * env.config.max_force_per_axis_n
            / (2.0 * max(omega, 1e-9) * env.chaser_parameters.mass)
        )
        path = np.zeros((3, count))
        for j in range(count):
            path[:, j] = desired + d * n_p
            pace = min(task.closing_speed_limit(d), coriolis_speed)
            d = max(d - self.kappa * pace * env.config.dt_s, 0.0)
        return path

    def __call__(self, env, metrics: T1Metrics) -> tuple[np.ndarray, np.ndarray | None]:
        count = env.mpc_config.horizon_steps + 1
        if (
            metrics.in_cone
            or metrics.mouth_region
            or metrics.range_m < env.config.task.operation_sphere_radius_m
        ):
            path = self._descent_path(env, metrics, count)
            return path[:, 0].copy(), path
        if self.approach == "pure" or self.tau_s <= 1e-9:
            return np.asarray(env.config.task.desired_position, dtype=np.float64), None
        rotations = self._predicted_rotations(env, count)
        dt = env.config.dt_s
        path = np.column_stack(
            [self._approach_point(env, rotations[j], env.time_s + j * dt) for j in range(count)]
        )
        return path[:, 0].copy(), path


def next_pass_tau(env, transfer_speed_m_s: float = 0.15) -> float:
    """Geometric rule without learning: meet the port on its next pass.

    The port normal sweeps about the spin axis; ``phase`` (deg) is how far it
    still has to rotate to reach the chaser's line of sight. If the chaser can
    cover the radial distance to the entry point (at ``transfer_speed``) before
    that, wait for the pass; otherwise take the following one.
    """

    f = env.features()
    omega = np.deg2rad(f["omega_deg_s"])
    wait = np.deg2rad(f["phase_deg"]) / omega
    transfer = max(f["range_m"] - 6.3, 0.0) / transfer_speed_m_s
    period = 2 * np.pi / omega
    return float(wait if wait >= transfer else wait + period)


# ---------------------------------------------------------------------- #
# G* v0.3: tau is the arrival time of a planned rendezvous with the moving
# mouth entry point.
# ---------------------------------------------------------------------- #


def free_body_rotations(
    rotation: np.ndarray, omega_body: np.ndarray, inertia: np.ndarray, horizon_s: float, step_s: float = 0.5
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Torque-free rigid-body prediction (Euler equations, RK4) from the current estimate.

    Returns times, rotations and body rates. This is the onboard model's
    forecast of the target attitude; gravity-gradient torque is neglected.
    """

    inverse = np.linalg.inv(inertia)

    def rate(w: np.ndarray) -> np.ndarray:
        return inverse @ (-np.cross(w, inertia @ w))

    times, rotations, rates = [0.0], [rotation.copy()], [omega_body.copy()]
    r, w, t = rotation.copy(), omega_body.copy(), 0.0
    while t < horizon_s - 1e-9:
        k1 = rate(w)
        k2 = rate(w + 0.5 * step_s * k1)
        k3 = rate(w + 0.5 * step_s * k2)
        k4 = rate(w + step_s * k3)
        w_next = w + step_s / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        r = r @ so3_exp(0.5 * step_s * (w + w_next))
        w, t = w_next, t + step_s
        times.append(t)
        rotations.append(r.copy())
        rates.append(w.copy())
    return np.array(times), rotations, rates


def _rotation_at(history, t: float) -> tuple[np.ndarray, np.ndarray]:
    times, rotations, rates = history
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))
    rotation = rotations[i] @ so3_exp((t - times[i]) * rates[i])
    return rotation, rotation @ rates[i]


class RendezvousPlanner:
    """Minimum delta-v transfer, in a fixed time, to the moving mouth entry point.

    Relative ECI double integrator (differential gravity neglected: ~1e-5 m/s^2
    at 12 m), ``|F| <= F_plan`` (2-norm; ``make_rendezvous`` plans with 80 %
    of the per-axis bound so that the MPC keeps authority to track: a
    minimum-delta-v plan is bang-coast-bang and a plan at the full bound
    turns infeasible at the first tracking error), within ``max_range_m``
    (a free coast otherwise drifts out towards the 30 m limit; beyond it the
    cap only leaves the stopping distance and the excess range is penalised),
    from the
    current relative state to ``E(t+tau) = R(t+tau) r_e n_p`` with the
    co-rotating velocity there minus a small inward speed. The operation
    sphere is kept out by half-spaces linearised at the previous iterate and
    the radial braking envelope is imposed through all of its chords. Solved by
    sequential convex programming on a DPP-compiled problem with ``nodes``
    uniform intervals. The plan only shapes the reference: the shared MPC and
    the truth check are unchanged.
    """

    def __init__(self, task, mass: float, max_force_n: float, *, nodes: int = 30,
                 entry_radius_m: float = 6.3, entry_inward_speed_m_s: float = 0.06,
                 max_range_m: float = 20.0) -> None:
        self.task, self.mass, self.max_force_n, self.nodes = task, float(mass), float(max_force_n), nodes
        self.entry_radius_m, self.entry_inward_speed_m_s = entry_radius_m, entry_inward_speed_m_s
        self.max_range_m = max_range_m
        #: delta-v equivalent (m/s) of one metre beyond max_range_m held for one second
        self.range_penalty_per_m_s = 0.01
        #: the plan never extends past this (30 m is the truth limit)
        self.hard_range_m = 28.0
        n = nodes
        self.q, self.v, self.f = cp.Variable((n + 1, 3)), cp.Variable((n + 1, 3)), cp.Variable((n, 3))
        self.q0, self.v0 = cp.Parameter(3), cp.Parameter(3)
        self.qe, self.ve = cp.Parameter(3), cp.Parameter(3)
        self.dt = cp.Parameter(nonneg=True)
        self.half_dt2_m = cp.Parameter(nonneg=True)
        self.dt_m = cp.Parameter(nonneg=True)
        self.normals = cp.Parameter((n - 1, 3))
        self.range_cap = cp.Parameter(nonneg=True)
        q, v, f = self.q, self.v, self.f
        cons = [q[0] == self.q0, v[0] == self.v0, q[n] == self.qe, v[n] == self.ve,
                q[1:] == q[:-1] + self.dt * v[:-1] + self.half_dt2_m * f,
                v[1:] == v[:-1] + self.dt_m * f,
                cp.norm(f, axis=1) <= self.max_force_n]
        radius = cp.sum(cp.multiply(self.normals, q[1:n]), axis=1)
        closing = -cp.sum(cp.multiply(self.normals, v[1:n]), axis=1)
        sphere = float(task.operation_sphere_radius_m)
        cons.append(radius >= entry_radius_m)
        cons.append(cp.norm(q[1:n], axis=1) <= self.range_cap)
        knots = task.envelope_knots_m
        for a, b in zip(knots[:-1], knots[1:]):
            value = task.braking_speed_limit(a)
            slope = (task.braking_speed_limit(b) - value) / (b - a)
            cons.append(closing <= value + slope * (radius - sphere - a))
        # Range beyond ``max_range_m`` is penalised (convex), so a plan that
        # starts outside it comes back instead of coasting along its cap.
        excess = cp.sum(cp.pos(cp.norm(q[1:n], axis=1) - max_range_m))
        self.problem = cp.Problem(
            cp.Minimize(self.dt_m * cp.sum(cp.norm(f, axis=1)) + self.range_penalty_per_m_s * self.dt * excess), cons
        )

    def solve(self, q0: np.ndarray, v0: np.ndarray, tau_s: float, history,
              guess: np.ndarray | None = None, iterations: int = 4) -> dict[str, Any]:
        # Every plan is solved from a freshly built solver: CVXPY otherwise
        # updates the cached CLARABEL object in place and the result depends
        # on earlier solves at ~1e-5 (v0.3.2). A plan must be a function of
        # its inputs only -- for replay, for the M1 shadow and for the policy.
        cache = getattr(self.problem, "_solver_cache", None)
        if cache is not None:
            cache.clear()
        n = self.nodes
        dt = tau_s / n
        rotation, omega_inertial = _rotation_at(history, tau_s)
        entry = rotation @ (self.entry_radius_m * self.task.approach_axis)
        entry_velocity = np.cross(omega_inertial, entry) - self.entry_inward_speed_m_s * entry / np.linalg.norm(entry)
        if guess is None:
            u0, ue = q0 / np.linalg.norm(q0), entry / np.linalg.norm(entry)
            r0 = float(np.linalg.norm(q0))
            guess = np.array([
                (r0 + (self.entry_radius_m - r0) * k / n) * _slerp_deterministic(u0, ue, k / n)
                for k in range(n + 1)
            ])
        self.q0.value, self.v0.value, self.qe.value, self.ve.value = q0, v0, entry, entry_velocity
        self.dt.value, self.half_dt2_m.value, self.dt_m.value = dt, 0.5 * dt * dt / self.mass, dt / self.mass
        # A state beyond the cap, or moving out near it, gets exactly the room
        # it needs to stop at the plan's force bound -- not a fixed +5 m: with
        # re-planning, a fixed allowance ratcheted out 5 m per plan (R1 pass 2
        # and random-action checks: 291047 walked from 13 m to 30 m).
        r0 = float(np.linalg.norm(q0))
        outward = max(float(q0 @ v0) / max(r0, 1e-9), 0.0)
        stopping = outward**2 / (2.0 * self.max_force_n / self.mass)
        self.range_cap.value = min(max(self.max_range_m, r0 + stopping), self.hard_range_m)
        status, cost = "not_solved", np.inf
        for _ in range(iterations):
            self.normals.value = guess[1:n] / np.linalg.norm(guess[1:n], axis=1, keepdims=True)
            try:
                self.problem.solve(solver="CLARABEL")
            except cp.error.SolverError as error:
                status = f"solver_error:{error}"
                break
            status = self.problem.status
            if status not in ("optimal", "optimal_inaccurate"):
                break
            new = np.asarray(self.q.value)
            change = float(np.max(np.linalg.norm(new - guess, axis=1)))
            guess, cost = new, float(self.problem.value)
            if change < 1e-2:
                break
        ok = status in ("optimal", "optimal_inaccurate")
        return {
            "status": status, "feasible": ok, "tau_s": tau_s, "dt": dt,
            "delta_v_m_s": cost if ok else None,
            "q": guess if ok else None,
            "v": np.asarray(self.v.value) if ok else None,
            "f": np.asarray(self.f.value) if ok else None,
        }


def _plan_position(plan: dict[str, Any], s: float, mass: float) -> np.ndarray:
    """Plan position ``s`` seconds after its start (exact double-integrator interpolation)."""

    q, v, f, dt = plan["q"], plan["v"], plan["f"], plan["dt"]
    n = len(f)
    if s >= n * dt:
        return q[-1] + v[-1] * (s - n * dt)
    i = int(np.clip(np.floor(max(s, 0.0) / dt), 0, n - 1))
    h = max(s, 0.0) - i * dt
    return q[i] + v[i] * h + 0.5 * f[i] / mass * h * h


@dataclass
class GStarRendezvous:
    """G* v0.3 reference: tau = arrival time at the moving entry point, kappa = descent pace.

    Call ``decide(env, tau)`` once per 2 s decision, then use the object as
    the per-control-step reference callable. The plan is kept while the
    request agrees with it (``|tau - remaining_s| <= replan_tau_tolerance_s``)
    and the state is within ``replan_deviation_m`` of it; otherwise it is
    re-planned from the current state. Re-planning a minimum-delta-v plan
    from the measured state every 2 s procrastinates: the shared MPC (input
    cost, frozen) lags a bang-coast-bang plan slightly, the remaining fixed
    time becomes infeasible, the arrival slips, and the slip repeats (R0,
    290003). A kept plan is what the offline OCP replay flew.

    * ``tau > tau_min``: the plan arrives at ``tau``; if the current state can
      no longer meet it, at the smallest feasible ``tau + slip`` (slips in
      ``slips_s``). ``tau_effective_s`` is the arrival time actually planned.
    * ``tau <= tau_min``: the earliest feasible arrival on ``earliest_grid_s``.
    * No feasible plan at all: hold the current inertial position (no
      co-rotation is demanded; see the module note on Pure's setpoint).
    * Inside the mouth region, inside the sphere, or in the cone within
      ``descent_radius_m``: the descent carrot at pace ``kappa``.
    * ``pure_approach=True`` (diagnostic row only, not part of G*): Pure's
      fixed terminal setpoint outside the descent region, the G* descent in it.

    Pure's setpoint is never used as a G* fallback: a constant target-frame
    setpoint asks for zero target-frame velocity, i.e. co-rotation at the
    current range, which at 9-15 m with the port facing away exceeds the 2 N
    bound and the shared MPC is flung out (R0, 290002/290003 after a late
    hand-over). Pure keeps that property; it is not changed.
    """

    kappa: float
    planner: RendezvousPlanner
    recovery_planner: RendezvousPlanner | None = None
    pure_approach: bool = False
    tau_min_s: float = 2.0
    descent_radius_m: float = 7.0
    replan_tau_tolerance_s: float = 1.0
    replan_deviation_m: float = 1.0
    slips_s: tuple[float, ...] = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    earliest_grid_s: tuple[float, ...] = (8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
                                          130.0, 160.0, 200.0)
    #: v0.5 task-deadline consistency: no plan may arrive later than
    #: ``max_time_s - deadline_margin_s`` (60 s ~ the 90th percentile of the time from the
    #: last planned arrival to capture in the rule runs of evidence/T1_BETA_B1 and
    #: evidence/T1_S1_SCREEN; no successful rule episode there planned an arrival past it)
    deadline_margin_s: float = 60.0
    tau_s: float = 0.0
    tau_effective_s: float = 0.0
    #: what the deadline did in the last decision: "none", "clamped" (a later request cut to
    #: the latest admissible arrival), "plan_beyond" (a live plan arriving too late was
    #: re-planned), "infeasible_within" (no plan within the deadline; the earliest feasible
    #: arrival beyond it was taken and is reported)
    last_deadline: str = "none"
    plan: dict[str, Any] | None = None
    plan_start_s: float = 0.0
    hold_position: np.ndarray | None = None
    last_status: str = "none"
    plan_wall_s: list[float] = field(default_factory=list)
    _descent: GStarPlan | None = None

    def remaining_s(self, env) -> float:
        """Planned time to arrival from now (0 without a plan)."""

        if self.plan is None:
            return 0.0
        return max(float(self.plan["tau_s"]) - (env.time_s - self.plan_start_s), 0.0)

    def _descending(self, env, metrics: T1Metrics) -> bool:
        return bool(
            metrics.mouth_region
            or metrics.range_m < env.config.task.operation_sphere_radius_m
            or (metrics.in_cone and metrics.range_m < self.descent_radius_m)
        )

    def decide(self, env, tau_s: float) -> None:
        from time import perf_counter

        self.tau_s = float(tau_s)
        self.last_deadline = "none"
        self._descent = GStarPlan(tau_s=0.0, kappa=self.kappa)
        if self._descending(env, env.metrics()):
            self.plan, self.last_status, self.tau_effective_s = None, "descent", 0.0
            return
        if self.pure_approach:
            self.plan, self.last_status, self.tau_effective_s = None, "pure", 0.0
            return
        started = perf_counter()
        target, chaser = env.target_state, env.chaser_state
        q0 = chaser.position - target.position
        v0 = chaser.rotation @ chaser.velocity - target.rotation @ target.velocity
        remaining = self.remaining_s(env)
        # v0.5: the latest admissible arrival, measured from now
        horizon = float(env.config.max_time_s) - self.deadline_margin_s - float(env.time_s)
        if self.tau_s > self.tau_min_s and self.tau_s > horizon:
            self.tau_s = max(horizon, 0.0)  # a later request is cut to the deadline (reported)
            self.last_deadline = "clamped"
        usable_plan = False
        if self.plan is not None:
            elapsed = env.time_s - self.plan_start_s
            deviation = float(np.linalg.norm(_plan_position(self.plan, elapsed, self.planner.mass) - q0))
            usable_plan = deviation <= self.replan_deviation_m and remaining > 0.0
            if usable_plan and remaining > horizon + self.replan_tau_tolerance_s:
                usable_plan = False  # a live plan arriving past the deadline is not kept
                self.last_deadline = "plan_beyond"
            agrees = (
                abs(self.tau_s - remaining) <= self.replan_tau_tolerance_s
                or (self.tau_s <= self.tau_min_s and remaining > 0.0)
            )
            if agrees and usable_plan:
                self.last_status = "kept"
                return
        # v0.5: a request (and its slips) is planned only up to the deadline. When the deadline
        # cut slips and no usable plan is live, the earliest feasible arrival follows (as an
        # earliest request would); with a usable plan an infeasible request keeps it (v0.4).
        # An arrival past the deadline is reported, never asked for. When the deadline cuts
        # nothing, and on the earliest-request path (the rule), the search is v0.4's.
        stages = [list(self.earliest_grid_s)]
        if self.tau_s > self.tau_min_s:
            slips = [self.tau_s + slip for slip in self.slips_s]
            requested = [tau for tau in slips if tau <= horizon + 1e-9]
            if len(requested) == len(slips) or (requested and usable_plan):
                stages = [requested]
            elif requested:
                stages = [requested, list(self.earliest_grid_s)]
        history = free_body_rotations(
            target.rotation, target.omega, env.target_parameters.inertia,
            max(max(stage) for stage in stages) + 1.0,
        )
        result: dict[str, Any] = {"feasible": False, "status": "not_solved"}
        for candidates in stages:
            for tau in candidates:
                guess = None
                if self.plan is not None:
                    n, shift = self.planner.nodes, env.time_s - self.plan_start_s
                    guess = np.array([
                        _plan_position(self.plan, shift + k * tau / n, self.planner.mass) for k in range(n + 1)
                    ])
                    guess[0] = q0
                result = self.planner.solve(q0, v0, tau, history, guess)
                if not result["feasible"] and guess is not None:
                    result = self.planner.solve(q0, v0, tau, history, None)
                if result["feasible"]:
                    break
            if result["feasible"]:
                break
        if result["feasible"] and float(result["tau_s"]) > horizon + 1e-9:
            self.last_deadline = "infeasible_within"
        # v0.4 plan stability. A usable plan is not replaced by one that fails the request
        # it was asked for: an earlier arrival that is not achievable (the new plan would
        # arrive no earlier than the current one), or a request with no feasible plan at all.
        # Replacing it there only churns the plan (R1-round-1 seed 0: SAC requests at the
        # action bound re-planned every 2 s, with slips and holds, and failed far out).
        earlier_request = self.tau_s < remaining - self.replan_tau_tolerance_s
        if usable_plan and (
            not result["feasible"]
            or (earlier_request and float(result["tau_s"]) >= remaining - self.replan_tau_tolerance_s)
        ):
            self.plan_wall_s.append(perf_counter() - started)
            self.last_status = "kept:no_earlier" if result["feasible"] else "kept:infeasible_request"
            return
        if not result["feasible"] and self.recovery_planner is not None:
            # No usable plan: the earliest arrival at the full force bound before holding.
            for tau in self.earliest_grid_s:
                history = free_body_rotations(
                    target.rotation, target.omega, env.target_parameters.inertia, tau + 1.0)
                result = self.recovery_planner.solve(q0, v0, tau, history, None)
                if result["feasible"]:
                    result["status"] = f"recovery:{result['status']}"
                    break
        self.plan_wall_s.append(perf_counter() - started)
        self.plan_start_s = float(env.time_s)
        if result["feasible"]:
            self.plan, self.last_status = result, str(result["status"])
            self.tau_effective_s = float(result["tau_s"])
            self.hold_position = None
        else:
            # The hold point is fixed on entering the hold. Re-anchoring it
            # at the current position every decision let an outward drift
            # carry the reference along with it (R1 pass 1, 291047: 19 -> 30 m).
            self.plan, self.last_status, self.tau_effective_s = None, f"hold:{result['status']}", 0.0
            if self.hold_position is None:
                self.hold_position = q0.copy()

    def __call__(self, env, metrics: T1Metrics) -> tuple[np.ndarray, np.ndarray | None]:
        if self._descending(env, metrics):
            assert self._descent is not None
            return self._descent(env, metrics)
        if self.pure_approach:
            return np.asarray(env.config.task.desired_position, dtype=np.float64), None
        count = env.mpc_config.horizon_steps + 1
        dt = env.config.dt_s
        path = np.zeros((3, count))
        for j in range(count):
            rotation = env.target_state.rotation @ so3_exp(j * dt * env.target_state.omega)
            if self.plan is None:
                if self.hold_position is None:
                    self.hold_position = env.chaser_state.position - env.target_state.position
                path[:, j] = rotation.T @ self.hold_position
                continue
            s = env.time_s + j * dt - self.plan_start_s
            if s >= self.plan["tau_s"]:
                # after the planned arrival: the co-rotating entry point
                path[:, j] = self.planner.entry_radius_m * env.config.task.approach_axis
            else:
                path[:, j] = rotation.T @ _plan_position(self.plan, s, self.planner.mass)
        return path[:, 0].copy(), path


def make_rendezvous(env, kappa: float, plan_force_fraction: float = 0.8, **kwargs: Any) -> GStarRendezvous:
    planner = RendezvousPlanner(
        env.config.task, env.chaser_parameters.mass, plan_force_fraction * env.config.max_force_per_axis_n
    )
    recovery = RendezvousPlanner(env.config.task, env.chaser_parameters.mass, env.config.max_force_per_axis_n)
    return GStarRendezvous(kappa=kappa, planner=planner, recovery_planner=recovery, **kwargs)


__all__ = [
    "GSTAR_VERSION", "GStarPlan", "GStarRendezvous", "RendezvousPlanner", "free_body_rotations",
    "make_rendezvous", "next_pass_tau",
]
