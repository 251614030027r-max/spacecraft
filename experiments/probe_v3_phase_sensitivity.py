"""Is the task sensitive to the target's tumble phase? (review-window Probe B)

For a base seed, the initial target attitude is advanced by phi about its own
spin axis -- same inertial spin, same inertial position and velocity for both
spacecraft; only how far the target has turned at reset changes -- and one fixed
controller is flown from each phase:

    pure     Pure MPC (fixed setpoint, baseline branch)
    direct   smooth nominal: the V3 learned branch advanced at full rate

No learning, no waiting rule. Reports completion (RK45 truth, zero-violation),
time, equivalent delta-v and QP zero fallbacks per (seed, phase, controller).

    python -B -m experiments.probe_v3_phase_sensitivity --seed 265000 --phase-deg 45 --controller direct --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamics.lie import so3_exp
from experiments.probe_v3_staging_headroom import VIOLATION_STEP_KEYS, _make_env


def _phase_shifted(env, seed: int, phase_rad: float):
    env.reset(seed=seed)
    target = env.env.target_state.copy()
    chaser = env.env.chaser_state.copy()
    omega = np.asarray(target.omega, dtype=np.float64)
    axis = omega / np.linalg.norm(omega)
    shifted = target.copy()
    old_rotation = np.asarray(target.rotation, dtype=np.float64)
    new_rotation = old_rotation @ so3_exp(phase_rad * axis)
    shifted.rotation = new_rotation
    # velocities are body-frame: keep the inertial velocity unchanged
    shifted.velocity = new_rotation.T @ (old_rotation @ np.asarray(target.velocity))
    return {"target_state": shifted, "chaser_state": chaser}


def run(seed: int, phase_deg: float, controller: str) -> dict:
    env = _make_env()
    impulse = [0.0]
    fallbacks = [0]
    original = env.controller.command
    dt = env.environment_config.dt_s

    def recorded(*args, **kwargs):
        wrench, diagnostics = original(*args, **kwargs)
        impulse[0] += float(np.linalg.norm(np.asarray(wrench)[3:])) * dt
        fallbacks[0] += int(diagnostics.used_zero_fallback)
        return wrench, diagnostics

    try:
        options = _phase_shifted(env, seed, np.radians(phase_deg))
        env.controller.command = recorded  # type: ignore[method-assign]
        env.reset(seed=seed, options=options)
        branch = "baseline" if controller == "pure" else "learned"
        action = np.zeros(2) if controller == "pure" else np.ones(2)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step_with_branch(action, branch=branch)
        zero_violation = all(int(info.get(key, 0)) == 0 for key in VIOLATION_STEP_KEYS)
        return {
            "seed": seed,
            "phase_deg": phase_deg,
            "controller": controller,
            "completed": bool(info.get("completed", False)),
            "zero_violation_completed": bool(info.get("completed", False) and zero_violation),
            "survival_s": float(env.env.time_seconds),
            "equivalent_delta_v_m_s": impulse[0] / env.env.chaser_parameters.mass,
            "qp_zero_fallbacks": fallbacks[0],
            "illegal_terminal_entry_count": int(info.get("illegal_terminal_entry_count", 0)),
            "failure": [
                k
                for k in ("keepout_failure", "fov_failure", "outer_speed_failure", "outer_radial_failure", "transition_speed_failure", "terminal_constraint_failure", "distance_failure", "time_failure")
                if info.get(k)
            ],
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase-deg", type=float, required=True)
    parser.add_argument("--controller", choices=("pure", "direct"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.phase_deg, args.controller)
    args.output.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
