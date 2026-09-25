"""Stress the V3 direction blend under training-like task-state sequences.

The hold direction in the target frame, h(t) = R(t)^T h_inertial, depends
only on the target's rotation and the chaser position at the first decision,
not on later chaser actions. Record it once per seed over a full 300 s
episode (task state held at z = 0 so the episode runs to the time limit),
then replay many synthetic (rho, c) sequences through candidate blends
offline and measure the largest adjacent-decision reference jump.

Record:  python -B -m experiments.probe_v3_blend_stress record --seed S --output h_S.npz
Analyse: python -B -m experiments.probe_v3_blend_stress analyse --inputs h_*.npz --output out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config

DECISION_S = 2.0
STEP_CAP_M = 0.40


def record(seed: int, output: Path) -> None:
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="task_state_v3",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_execution_feedback_observation=True,
            include_staging_direction_observation=True,
        ),
    )
    try:
        env.reset(seed=seed)
        hold, rotations = [], []
        while True:
            _, _, terminated, truncated, _ = env.step_with_branch(np.zeros(2), branch="learned")
            rotation = np.asarray(env.env.target_state.rotation, dtype=np.float64)
            rotations.append(rotation)
            hold.append(rotation.T @ env._hold_inertial)
            if terminated or truncated:
                break
        np.savez(
            output,
            seed=seed,
            hold=np.asarray(hold),
            hold_radius=env._hold_radius_m,
            desired=np.asarray(env.environment_config.precapture_task.desired_position),
        )
    finally:
        env.close()


# --- candidate blends ------------------------------------------------------


def _rodrigues(v, axis, angle):
    return v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * (axis @ v) * (1 - np.cos(angle))


def _geodesic(u, g, t):
    """Rotate unit u toward unit g by fraction t of the shortest arc."""
    angle = float(np.arccos(np.clip(u @ g, -1.0, 1.0)))
    if angle < 1e-12:
        return g.copy()
    axis = np.cross(u, g)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:  # antipodal: any perpendicular axis
        basis = np.zeros(3)
        basis[int(np.argmin(np.abs(u)))] = 1.0
        axis = np.cross(u, basis)
        n = float(np.linalg.norm(axis))
    return _rodrigues(u, axis / n, t * angle)


class MemoryAxis:
    """The blend shipped in T0 (unwrapped angle, projected axis)."""

    def __init__(self):
        self.axis = None
        self.angle = None

    def __call__(self, h, d, c):
        w = h - d
        if self.axis is None:
            axis = np.cross(h, d)
            if np.linalg.norm(axis) < 1e-10:
                basis = np.zeros(3); basis[int(np.argmin(np.abs(h)))] = 1.0
                axis = np.cross(h, basis)
        else:
            axis = self.axis - (self.axis @ w) / max(w @ w, 1e-14) * w if w @ w > 1e-14 else self.axis - (self.axis @ h) * h
            if np.linalg.norm(axis) < 1e-10:
                axis = np.cross(h, d)
        axis = axis / np.linalg.norm(axis)
        if self.axis is not None and axis @ self.axis < 0:
            axis = -axis
        hp, dp = h - (h @ axis) * axis, d - (d @ axis) * axis
        angle = float(np.arctan2(axis @ np.cross(hp, dp), hp @ dp))
        if self.angle is not None:
            angle += 2 * np.pi * round((self.angle - angle) / (2 * np.pi))
        self.axis, self.angle = axis, angle
        return _rodrigues(h, axis, c * angle)


class ShortestArc:
    """V2 blend: stateless shortest-arc slerp from h to d."""

    def __call__(self, h, d, c):
        return _geodesic(h, d, c)


class RateLimitedDirection:
    """Applied direction is a state that tracks a stateless blend target.

    Each decision the applied unit direction u moves along the shortest arc
    from *itself* toward the target g = slerp(h, d, c) by at most
    ``max_angle(radius)``. u never jumps, whatever g does; the arc from u is
    ill-defined only if g is exactly antipodal to u.
    """

    def __init__(self, omega_rad_s: float):
        self.u = None
        self.omega = omega_rad_s

    def __call__(self, h, d, c, radius):
        target = _geodesic(h, d, c)
        if self.u is None:
            self.u = target
            return target
        max_angle = 1.2 * self.omega * DECISION_S + STEP_CAP_M / max(radius, 1e-6)
        angle = float(np.arccos(np.clip(self.u @ target, -1.0, 1.0)))
        self.u = target if angle <= max_angle else _geodesic(self.u, target, max_angle / angle)
        return self.u


def _sequences(n_decisions: int, rng: np.random.Generator, progress_max: float):
    """Training-like task-state sequences: random walks at the V3 limits."""
    out = {}
    for c0 in (0.3, 0.5, 0.7):
        out[f"const_c{c0}"] = [(0.0, c0)] * n_decisions
    for k in range(6):
        rho, c, seq = 0.0, 0.0, []
        for _ in range(n_decisions):
            a = rng.uniform(-1, 1, size=2)
            rho = float(np.clip(rho + a[0] * 0.5, 0.0, progress_max))
            c = float(np.clip(c + (a[1] * 0.05 if a[1] >= 0 else a[1] * 0.02), 0.0, 1.0))
            seq.append((rho, c))
        out[f"random_walk_{k}"] = seq
    return out


def analyse(inputs: list[Path], output: Path) -> None:
    omega = 0.041231
    rng = np.random.default_rng(20260925)
    rows = []
    for path in inputs:
        data = np.load(path)
        hold, radius0, desired = data["hold"], float(data["hold_radius"]), data["desired"]
        d = desired / np.linalg.norm(desired)
        progress_max = max(radius0 - float(np.linalg.norm(desired)), 0.0)
        for name, seq in _sequences(len(hold), rng, progress_max).items():
            results = {}
            for label, make in (("v2_shortest_arc", ShortestArc), ("t0_memory_axis", MemoryAxis), ("rate_limited", lambda: RateLimitedDirection(omega))):
                blend = make()
                refs, worst_angle, worst = [], None, 0.0
                z = (0.0, 0.0)
                for i, (h, proposal) in enumerate(zip(hold, seq)):
                    def ref(state, commit_state):
                        r = max(radius0 - state[0], float(np.linalg.norm(desired)))
                        if label == "rate_limited":
                            # evaluate without committing the filter state
                            saved = None if blend.u is None else blend.u.copy()
                            out = r * blend(h, d, state[1], r)
                            if not commit_state:
                                blend.u = saved
                            return out
                        return r * blend(h, d, state[1])
                    # same-instant metric cap on the task-state step, as in the env
                    current = ref(z, False) if i > 0 else None
                    candidate = proposal
                    if current is not None and np.linalg.norm(ref(candidate, False) - current) > STEP_CAP_M:
                        lo, hi = 0.0, 1.0
                        for _ in range(8):
                            mid = 0.5 * (lo + hi)
                            trial = (z[0] + mid * (proposal[0] - z[0]), z[1] + mid * (proposal[1] - z[1]))
                            if np.linalg.norm(ref(trial, False) - current) <= STEP_CAP_M:
                                lo = mid
                            else:
                                hi = mid
                        candidate = (z[0] + lo * (proposal[0] - z[0]), z[1] + lo * (proposal[1] - z[1]))
                    z = candidate
                    refs.append(ref(z, True))
                    if i > 0:
                        jump = float(np.linalg.norm(refs[-1] - refs[-2]))
                        if jump > worst:
                            worst = jump
                            worst_angle = float(np.degrees(np.arccos(np.clip(h @ d, -1, 1))))
                results[label] = worst
                results[label + "_angle_at_worst_deg"] = worst_angle
            natural = float(np.max(np.linalg.norm(np.diff(hold, axis=0), axis=1)) * radius0)
            rows.append({"seed": int(data["seed"]), "sequence": name, "natural_hold_motion_m": natural, **results})
    summary = {}
    for label in ("v2_shortest_arc", "t0_memory_axis", "rate_limited"):
        vals = [r[label] for r in rows]
        summary[label] = {
            "max_jump_m": max(vals),
            "cases_over_3m": sum(v > 3.0 for v in vals),
            "cases": len(vals),
        }
    output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--seed", type=int, required=True); r.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("analyse"); a.add_argument("--inputs", type=Path, nargs="+", required=True); a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "record":
        record(args.seed, args.output)
    else:
        analyse(args.inputs, args.output)


if __name__ == "__main__":
    main()
