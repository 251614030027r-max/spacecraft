"""R0: offline approach planner for T1 openings and replay through the shared MPC.

Diagnostic only (not a baseline, not a bound). Phase A is planned offline;
phase B (descent inside the cone) is left to the shared MPC with the G*
descent carrot, exactly as the learned rows would fly it.

Phase A, for a candidate arrival time t_e (convex, solved by SCP):
  relative ECI double integrator q'' = F/m (differential gravity neglected,
  ~1e-5 m/s^2 at 12 m), |F| <= F_max, from the opening state to the mouth
  entry point E(t_e) = R(t_e) (r_e n_p) moving with the target (co-rotating
  velocity plus a small inward speed); outside the sphere via half-spaces
  linearised at the previous iterate; radial braking envelope imposed as all
  of its chords (the envelope is the minimum of its chords); minimise delta-v.
The target attitude over time comes from propagating the same truth model.

    python -B -m experiments.t1_ocp plan --seed 290000 --output-dir eval/t1/r0_ocp
    python -B -m experiments.t1_ocp replay --seed 290000 --tau 60 --kappa 0.7 --output-dir ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np

from dynamics.disturbance import zero_disturbance
from dynamics.integrator import propagate_rk45
from dynamics.lie import so3_exp
from dynamics.types import GeneralizedForce
from env.t1_env import T1Env, T1EnvConfig
from env.t1_gstar import GStarPlan

ENTRY_RADIUS_M = 6.3
ENTRY_INWARD_SPEED_M_S = 0.06


def target_history(env: T1Env, horizon_s: float, step_s: float = 0.5) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Times, target rotations and inertial angular velocities from the truth model."""

    state = env.target_state.copy()
    times, rotations, omegas = [0.0], [state.rotation.copy()], [state.rotation @ state.omega]
    t = 0.0
    while t < horizon_s - 1e-9:
        state, _ = propagate_rk45(
            state, env.target_parameters, env.time_s + t, step_s,
            control=GeneralizedForce.zeros(), disturbance=zero_disturbance,
            gravity_options=env._gravity, settings=env._solver,
        )
        t += step_s
        times.append(t)
        rotations.append(state.rotation.copy())
        omegas.append(state.rotation @ state.omega)
    return np.array(times), rotations, omegas


def _rotation_at(times, rotations, omegas, t):
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 1))
    dt = t - times[i]
    body_omega = rotations[i].T @ omegas[i]
    return rotations[i] @ so3_exp(dt * body_omega), omegas[i]


def plan_phase_a(env: T1Env, history, t_e: float, step_s: float = 1.0, iterations: int = 6) -> dict[str, Any]:
    times, rotations, omegas = history
    task = env.config.task
    m = env.chaser_parameters.mass
    f_max = env.config.max_force_per_axis_n
    n = max(int(round(t_e / step_s)), 2)
    dt = t_e / n
    q0 = env.chaser_state.position - env.target_state.position
    v0 = env.chaser_state.rotation @ env.chaser_state.velocity - env.target_state.rotation @ env.target_state.velocity
    r_end, w_end = _rotation_at(times, rotations, omegas, t_e)
    entry = r_end @ (ENTRY_RADIUS_M * task.approach_axis)
    entry_velocity = np.cross(w_end, entry) - ENTRY_INWARD_SPEED_M_S * entry / np.linalg.norm(entry)
    # initial guess: radius and direction interpolated linearly (outside the sphere)
    u0, ue = q0 / np.linalg.norm(q0), entry / np.linalg.norm(entry)
    guess = []
    for k in range(n + 1):
        h = k / n
        omega = np.arccos(np.clip(u0 @ ue, -1, 1))
        d = u0 if omega < 1e-6 else (np.sin((1 - h) * omega) * u0 + np.sin(h * omega) * ue) / np.sin(omega)
        guess.append((np.linalg.norm(q0) * (1 - h) + ENTRY_RADIUS_M * h) * d / np.linalg.norm(d))
    guess = np.array(guess)
    knots = task.envelope_knots_m
    chords = [
        (task.braking_speed_limit(a), (task.braking_speed_limit(b) - task.braking_speed_limit(a)) / (b - a), a)
        for a, b in zip(knots[:-1], knots[1:])
    ]
    q = cp.Variable((n + 1, 3))
    v = cp.Variable((n + 1, 3))
    f = cp.Variable((n, 3))
    status, dv_value = "not_solved", np.inf
    for _ in range(iterations):
        normals = guess / np.linalg.norm(guess, axis=1, keepdims=True)
        cons = [q[0] == q0, v[0] == v0, q[n] == entry, v[n] == entry_velocity]
        for k in range(n):
            cons += [
                q[k + 1] == q[k] + dt * v[k] + 0.5 * dt * dt * f[k] / m,
                v[k + 1] == v[k] + dt * f[k] / m,
                cp.norm(f[k]) <= f_max,
            ]
        for k in range(1, n):
            radius = normals[k] @ q[k]
            cons.append(radius >= ENTRY_RADIUS_M)
            closing = -(normals[k] @ v[k])
            for value, slope, left in chords:
                cons.append(closing <= value + slope * (radius - task.operation_sphere_radius_m - left))
        problem = cp.Problem(cp.Minimize(cp.sum(cp.norm(f, axis=1)) * dt / m), cons)
        try:
            problem.solve(solver="CLARABEL")
        except Exception as error:  # noqa: BLE001
            status = f"solver_error:{error!r}"
            break
        status = problem.status
        if status not in ("optimal", "optimal_inaccurate"):
            break
        new = np.asarray(q.value)
        change = float(np.max(np.linalg.norm(new - guess, axis=1)))
        guess = new
        dv_value = float(problem.value)
        if change < 1e-3:
            break
    ok = status in ("optimal", "optimal_inaccurate")
    min_range = float(np.min(np.linalg.norm(guess, axis=1))) if ok else None
    return {
        "t_e": t_e,
        "status": status,
        "delta_v_m_s": dv_value if ok else None,
        "phase_a_cost": (0.05 * t_e + 5.0 * dv_value) if ok else None,
        "min_range_m": min_range,
        "positions": guess.tolist() if ok else None,
        "dt": dt,
    }


class OCPReference:
    """Phase A: track the planned inertial path; phase B: G* descent carrot."""

    def __init__(self, plan: dict[str, Any], kappa: float, t_start: float) -> None:
        self.positions = np.asarray(plan["positions"])
        self.dt = float(plan["dt"])
        self.t_e = float(plan["t_e"])
        self.t_start = t_start
        self.descent = GStarPlan(tau_s=0.0, kappa=kappa)

    def _inertial(self, t: float) -> np.ndarray:
        s = (t - self.t_start) / self.dt
        if s >= len(self.positions) - 1:
            return self.positions[-1]
        i = int(np.floor(max(s, 0.0)))
        w = max(s, 0.0) - i
        return (1 - w) * self.positions[i] + w * self.positions[i + 1]

    def __call__(self, env, metrics):
        count = env.mpc_config.horizon_steps + 1
        if (
            metrics.in_cone
            or metrics.mouth_region
            or metrics.range_m < env.config.task.operation_sphere_radius_m
        ):
            return self.descent(env, metrics)
        dt = env.config.dt_s
        path = np.zeros((3, count))
        for j in range(count):
            t = env.time_s + j * dt
            rotation = env.target_state.rotation @ so3_exp(j * dt * env.target_state.omega)
            if t - self.t_start >= self.t_e:
                path[:, j] = ENTRY_RADIUS_M * env.config.task.approach_axis
            else:
                path[:, j] = rotation.T @ self._inertial(t)
        return path[:, 0].copy(), path


def replay(env: T1Env, seed: int, plan: dict[str, Any], kappa: float) -> dict[str, Any]:
    env.reset(seed)
    reference = OCPReference(plan, kappa, env.time_s)
    gamma, discounted, k = env.config.reward.gamma, 0.0, 0
    while True:
        reward, terminated, truncated, info = env.step(reference=reference)
        discounted += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    return {
        "seed": seed, "t_e": plan["t_e"], "kappa": kappa, "event": info["event"],
        **{key: info[key] for key in ("violations", "numerical_error") if key in info},
        "discounted_return": discounted, **env.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "replay"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tau", type=float, nargs="*", default=None)
    parser.add_argument("--kappa", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = T1Env(T1EnvConfig())
    env.reset(args.seed)
    history = target_history(env, 300.0)
    taus = args.tau or list(np.arange(20.0, 201.0, 10.0))
    plans = []
    for t_e in taus:
        started = perf_counter()
        plan = plan_phase_a(env, history, float(t_e))
        plan["wall_s"] = perf_counter() - started
        plans.append(plan)
        print(args.seed, "t_e", t_e, plan["status"], plan["delta_v_m_s"], plan["phase_a_cost"], plan["min_range_m"], flush=True)
    (args.output_dir / f"plans_{args.seed}.json").write_text(json.dumps(plans))
    if args.command == "replay":
        for plan in plans:
            if plan["positions"] is None:
                continue
            started = perf_counter()
            result = replay(env, args.seed, plan, args.kappa)
            result["wall_s"] = perf_counter() - started
            (args.output_dir / f"replay_{args.seed}_te{int(plan['t_e'])}_k{args.kappa}.json").write_text(json.dumps(result))
            print("replay", args.seed, plan["t_e"], result["event"], round(result["time_s"], 1),
                  round(result["delta_v_m_s"], 2), round(result["discounted_return"], 1), result.get("violations", ""), flush=True)


if __name__ == "__main__":
    main()
