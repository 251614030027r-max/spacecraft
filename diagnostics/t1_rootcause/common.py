"""Shared pieces of the T1 root-cause diagnosis.

Bookkeeping fix used throughout: ``tau_effective_s`` of G* is the *total* duration of
the live plan from its start, not the time remaining. The absolute planned arrival is
``plan_start + tau_effective_s`` and changes only when a new plan is adopted.
(evidence/T1_V2_S0/review_readout.json added ``2 k + tau_effective_s`` on every
decision, which drifts by 2 s per kept decision; see the diagnosis report.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from env.t1_gym import T1GStarEnv

KAPPA_RULE = 0.7
NEW_PLAN_PREFIXES = ("optimal", "recovery")


def load_model(path: str | Path):
    """M1SAC for M1 checkpoints (the critic/actor are SB3 SAC either way)."""

    from stable_baselines3 import SAC

    from train.t1_m1 import M1SAC

    try:
        return M1SAC.load(str(path), device="cpu")
    except Exception:  # an M0 checkpoint
        return SAC.load(str(path), device="cpu")


def rule_a1(env: T1GStarEnv) -> float:
    c = env.config
    return 2.0 * (KAPPA_RULE - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0


def live_plan(env: T1GStarEnv) -> bool:
    assert env.generator is not None
    return env.generator.remaining_s(env.env) > 0.0


# Intervention modes (P1). The actor always sees the real (post-intervention) observation.
#   A actor | B keep a live plan | C earliest when no live plan | D C+B | E full rule (kappa 0.7)
# "No live plan" (the first decision, an expired plan, a hold) is pre-specified per mode:
# A and B use the actor's absolute request; C, D and E ask for the earliest feasible arrival.
MODES = ("A", "B", "C", "D", "E")


def mode_action(mode: str, env: T1GStarEnv, actor_action: np.ndarray | None) -> np.ndarray:
    plan = live_plan(env)
    if mode == "E":
        return np.array([0.0 if plan else -1.0, rule_a1(env)])
    assert actor_action is not None
    a = np.array(actor_action, dtype=np.float64)
    if mode == "A":
        return a
    if mode == "B":
        return np.array([0.0, a[1]]) if plan else a
    if mode == "C":
        return a if plan else np.array([-1.0, a[1]])
    if mode == "D":
        return np.array([0.0 if plan else -1.0, a[1]])
    raise ValueError(mode)


def request_kind(env: T1GStarEnv, action: np.ndarray) -> str:
    """What the action asks of G* in the current state (before the step)."""

    if not live_plan(env):
        return "no_plan"
    shift = env.shift_of(float(np.clip(action[0], -1.0, 1.0)))
    return "keep" if shift == 0.0 else ("later" if shift > 0.0 else "earlier")


def run_from(env: T1GStarEnv, obs: np.ndarray, policy: Callable[[T1GStarEnv, np.ndarray], tuple[np.ndarray, Any]],
             k0: int = 0, max_decisions: int = 10_000) -> dict:
    """Run ``env`` from its current state to termination. ``policy(env, obs) -> (executed, actor)``.

    Returns per-decision records with the absolute planned arrival (correct bookkeeping),
    and the discounted return from the starting decision (gamma per decision, from k0).
    """

    gamma = env.env.config.reward.gamma
    g = env.generator
    assert g is not None
    arrival = (g.plan_start_s + g.tau_effective_s) if g.plan is not None else None
    log, ret, k = [], 0.0, 0
    while k < max_decisions:
        t = float(env.env.time_s)
        executed, actor = policy(env, obs)
        kind = request_kind(env, executed)
        remaining = g.remaining_s(env.env)
        tau, kappa = env.intent(executed)
        obs, reward, terminated, truncated, info = env.step(executed)
        x = info["execution"]
        status = x["plan_status"]
        if status.startswith(NEW_PLAN_PREFIXES):
            arrival = g.plan_start_s + g.tau_effective_s
        elif status == "descent" or status.startswith("hold"):
            arrival = None
        log.append({
            "k": k0 + k, "t_s": t, "actor_a": None if actor is None else [float(v) for v in actor],
            "a": [float(v) for v in executed], "request": kind, "remaining_before_s": remaining,
            "tau_request_s": tau, "kappa": kappa, "status": status, "slip_s": x["slip_s"],
            "arrival_abs_s": arrival, "reward": float(reward),
        })
        ret += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    summary = info.get("episode_summary", {})
    statuses = [r["status"] for r in log]
    return {
        "event": info["event"], "discounted_return_from_k": ret, "time_s": float(env.env.time_s),
        "delta_v_m_s": summary.get("delta_v_m_s"), "end_phase": summary.get("end_phase"),
        "end_range_m": summary.get("end_range_m"), "violations": summary.get("violations"),
        "new_plans": sum(s.startswith(NEW_PLAN_PREFIXES) for s in statuses),
        "kept": sum(s.startswith("kept") for s in statuses), "holds": sum(s.startswith("hold") for s in statuses),
        "requests": {kind: sum(r["request"] == kind for r in log) for kind in ("keep", "earlier", "later", "no_plan")},
        "decisions": len(log), "log": log,
    }


def actor_policy(model, mode: str = "A", deterministic: bool = True):
    def policy(env: T1GStarEnv, obs: np.ndarray):
        actor = None if mode == "E" else np.asarray(model.predict(obs, deterministic=deterministic)[0], dtype=np.float64)
        return mode_action(mode, env, actor), actor
    return policy


def rule_policy():
    def policy(env: T1GStarEnv, obs: np.ndarray):
        return mode_action("E", env, None), None
    return policy
