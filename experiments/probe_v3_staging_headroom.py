"""How much can a staging decision gain on this task, through the V3 interface?

Scripted learned-branch strategies, all expressed through the same bounded
task-state actions the SAC policy uses (so anything they reach, the learned
layer can reach):

    direct          commit at full rate from the first decision
    s<R>_T<t>       close with commitment 0 (inertial hold direction) until the
                    reference and the chaser are at radius R, hold t seconds,
                    then commit at full rate
    s<R>_a<deg>     same, but commit when the port direction has turned to
                    within <deg> of the inertial hold direction (or after one
                    tumble period, whichever is first)

Per episode: completion, zero-violation completion (RK45 truth), time,
equivalent delta-v (force impulse / mass, as the evaluator computes it) and
QP zero fallbacks. Taking the best strategy per seed *in hindsight* is an
oracle over this family: a lower bound on what a state-dependent staging
decision can gain, not a result of the method.

    python -B -m experiments.probe_v3_staging_headroom --seed 262001 --variant s7.5_a60 --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from env.hybrid_env import PrecaptureHybridEnv
from train.train_hybrid import accelerated_training_configs

VIOLATION_STEP_KEYS = (
    "keepout_violation_steps",
    "fov_violation_steps",
    "outer_speed_violation_steps",
    "outer_radial_violation_steps",
    "outer_approach_violation_steps",
    "corridor_violation_steps",
    "total_speed_violation_steps",
    "closing_speed_violation_steps",
)
ARRIVAL_TOLERANCE_M = 0.5
MAX_WAIT_S = 152.4  # one tumble period


def _make_env() -> PrecaptureHybridEnv:
    environment_config, hybrid_config = accelerated_training_configs(
        horizon_steps=35,
        waypoint_parametrization="task_state_v3",
        execution_feedback=True,
        adaptive_task=True,
    )
    return PrecaptureHybridEnv(environment_config, hybrid_config)


def _parse(variant: str) -> tuple[float | None, str | None, float | None]:
    if variant == "direct":
        return None, None, None
    match = re.fullmatch(r"s([0-9.]+)_(T|a)([0-9.]+)", variant)
    if match is None:
        raise ValueError(f"unknown variant {variant}")
    return float(match.group(1)), match.group(2), float(match.group(3))


def _alignment_deg(env: PrecaptureHybridEnv) -> float:
    task = env.environment_config.precapture_task
    desired = np.asarray(task.desired_position, dtype=np.float64)
    port_inertial = np.asarray(env.env.target_state.rotation, dtype=np.float64) @ (
        desired / np.linalg.norm(desired)
    )
    cosine = float(np.clip(env._hold_inertial @ port_inertial, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def run(seed: int, variant: str) -> dict:
    stage_radius, trigger, value = _parse(variant)
    env = _make_env()
    impulse = [0.0]
    zero_fallbacks = [0]
    original = env.controller.command
    dt = env.environment_config.dt_s

    def recorded(*args, **kwargs):
        wrench, diagnostics = original(*args, **kwargs)
        impulse[0] += float(np.linalg.norm(np.asarray(wrench)[3:])) * dt
        zero_fallbacks[0] += int(diagnostics.used_zero_fallback)
        return wrench, diagnostics

    env.controller.command = recorded  # type: ignore[method-assign]
    try:
        env.reset(seed=seed)
        phase = "commit" if stage_radius is None else "approach"
        wait_started = commit_time = None
        info: dict = {}
        terminated = truncated = False
        while not (terminated or truncated):
            now = float(env.env.time_seconds)
            if phase == "approach":
                target = max(env._hold_radius_m - stage_radius, 0.0)
                arrived = (
                    env._task_progress_m >= target - 1e-6
                    and float(np.linalg.norm(env._current_position())) <= stage_radius + ARRIVAL_TOLERANCE_M
                )
                if arrived:
                    phase, wait_started = "wait", now
            if phase == "wait":
                waited = now - wait_started
                fire = (
                    waited >= value
                    if trigger == "T"
                    else _alignment_deg(env) <= value or waited >= MAX_WAIT_S
                )
                if fire:
                    phase, commit_time = "commit", now
            if phase == "approach":
                action = env.action_for_task_state(max(env._hold_radius_m - stage_radius, 0.0), 0.0)
            elif phase == "wait":
                action = np.zeros(2)
            else:
                action = np.ones(2)
            _, _, terminated, truncated, info = env.step_with_branch(action, branch="learned")
        zero_violation = all(int(info.get(key, 0)) == 0 for key in VIOLATION_STEP_KEYS)
        return {
            "seed": seed,
            "variant": variant,
            "completed": bool(info.get("completed", False)),
            "zero_violation_completed": bool(info.get("completed", False) and zero_violation),
            "survival_s": float(env.env.time_seconds),
            "equivalent_delta_v_m_s": impulse[0] / env.env.chaser_parameters.mass,
            "commit_time_s": commit_time,
            "qp_zero_fallbacks": zero_fallbacks[0],
            "illegal_terminal_entry_count": int(info.get("illegal_terminal_entry_count", 0)),
            "failure": [k for k in ("keepout_failure", "fov_failure", "outer_speed_failure", "outer_radial_failure", "transition_speed_failure", "terminal_constraint_failure", "distance_failure", "time_failure") if info.get(k)],
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.variant)
    args.output.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
