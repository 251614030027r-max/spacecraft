"""T1-N cross-timescale ranking signal (Research Plan R2, research-layer ruling of 2026-10-18).

At a review state z outside G*'s descent mode, the executed action a and one alternative a~
from the same arrival-time action space (``alternative_action``, fixed before training) are each
decided in an isolated shadow of the review wrapper (its own ``T1Env``, MPC controller and
conditional G*, synchronised from the real one with fresh solvers). For each branch b:

* **G* long-horizon conditional plan.** The plan G* accepts for b (a new plan, or the kept live
  plan) gives the nominal reward from now to the planned entry, per real 2 s step j (time cost
  0.05 / s and fuel ``lambda_f * |F| dt / m`` of the planned force; the last step is partial):
  ``P_j(b)``. Nothing after the planned arrival is predicted: no planned arrival state is ever
  turned into an observation or bootstrapped (G* does not model the descent, the attitude
  tracking, re-planning or termination).
* **MPC short-horizon closed-loop correction.** The shared MPC flies the first 2 s of b in the
  shadow, solving every 0.1 s at the *predicted* state and advancing it with its own prediction
  model; its predicted reward ``r^_0(b)`` (the real metering) replaces the plan's first step.

The branch value and the pair signal (same definition for both branches, gamma per 2 s step):

    S(b) = r^_0(b) + sum_{j >= 1} gamma^j P_j(b),        D = S(a~) - S(a)
    plan only (diagnostic): S_plan(b) = sum_{j >= 0} gamma^j P_j(b)
    MPC correction of a branch: r^_0(b) - P_0(b)

D is a partial proxy of the long-horizon task value, never an unbiased Q label: the learner
(``train.t1_n_sac``) uses only its *sign*, as a confidence-weighted ranking constraint with a
finite margin on Q(z, a~) - Q(z, a). Confidence (fixed before training): zero unless both branches
have a plan accepted by G* (no hold), the two planned arrivals differ by >= 4 s, the 2 s MPC window
predicts no truth-rule violation, the executed action's own 2 s prediction matches what really
happened (|r^_0(a) - r_0| <= 0.25) and the review is not near the horizon; otherwise
    w = min(1, |D| / 1.0) * (1 - e_r1 / 0.25),   and w = 0 when |D| < 0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc.prediction import relative_to_vector
from dynamics.relative import relative_state
from env.t1_counterfactual import CounterfactualPredictor
from env.t1_env import T1EnvConfig
from env.t1_gym import T1GymConfig
from env.t1_n_review import T1NReviewEnv, action_for_tau, neutral_tau
from env.t1_task import compute_t1_metrics

#: request shift of the alternative without a live plan (s), and the shift tried against a kept plan
ALT_NO_PLAN_SHIFT_S = 30.0
ALT_LIVE_SHIFT_S = 15.0


@dataclass(frozen=True)
class T1NShadowConfig:
    """Fixed before training (research-layer ruling 2026-10-18)."""

    min_tau_difference_s: float = 4.0
    #: executed-action 2 s reward error bound at fuel weight 20 (~10 x the measured p95 0.026)
    max_first_reward_error: float = 0.25
    end_margin_s: float = 4.0
    #: |D| below this is not identifiable: no supervision
    min_signal: float = 0.1
    #: |D| at which the confidence saturates
    ref_signal: float = 1.0
    #: margin of the ranking constraint on Q(z, a~) - Q(z, a) (return units)
    rank_margin: float = 0.5


def alternative_action(env: T1NReviewEnv, action: np.ndarray) -> tuple[np.ndarray | None, str]:
    """The single alternative of a review, fixed before any training.

    * no live plan: the request moved by 30 s towards the inside of [tau_low, tau_high(t)];
    * live plan, the action keeps it: a 15 s shift (later on even reviews, earlier on odd ones,
      later when an earlier shift would leave < 15 s);
    * live plan, the action shifts: keep.
    """

    g, e = env.generator, env.env
    assert g is not None
    if g._descending(e, e.metrics()):
        return None, "descent_mode"
    a0 = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
    remaining = g.remaining_s(e)
    if remaining > 0.0:
        if abs(env.shift_of(a0)) > env.config.keep_tolerance_s:
            return np.zeros(1), "replan_vs_keep"
        sign = 1.0 if env._review_index % 2 == 0 else -1.0
        if remaining - ALT_LIVE_SHIFT_S <= 2.0:
            sign = 1.0
        return np.array([env.action_for_shift(sign * ALT_LIVE_SHIFT_S)]), "keep_vs_shift"
    tau = neutral_tau(a0, e.time_s, e.config.max_time_s, g.deadline_margin_s)
    later = tau + ALT_NO_PLAN_SHIFT_S
    alt = action_for_tau(later, e.time_s, e.config.max_time_s, g.deadline_margin_s)
    if alt <= a0 + 1e-9:  # no room above: go earlier
        alt = action_for_tau(tau - ALT_NO_PLAN_SHIFT_S, e.time_s, e.config.max_time_s, g.deadline_margin_s)
    return np.array([alt]), "request_shift"


def plan_step_rewards(plan: dict[str, Any], start_s: float, mass: float, time_cost: float, fuel_cost: float,
                      step_s: float = 2.0) -> list[float]:
    """Nominal reward of a G* plan per real step from plan time ``start_s`` to its arrival.

    Step j covers plan time [start + j step, start + (j + 1) step], cut at the arrival (the last
    step is partial). Per node the planned force is constant: reward rate -time_cost - fuel_cost |F| / m.
    """

    f, dt = np.asarray(plan["f"]), float(plan["dt"])
    arrival = float(plan["tau_s"])
    rates = -time_cost - fuel_cost * np.linalg.norm(f, axis=1) / mass
    out: list[float] = []
    a = float(start_s)
    while a < arrival - 1e-9:
        b = min(a + step_s, arrival)
        total = 0.0
        first, last = int(np.floor(a / dt)), int(np.ceil(b / dt))
        for i in range(max(first, 0), min(last, len(rates))):
            lo, hi = max(a, i * dt), min(b, (i + 1) * dt)
            if hi > lo:
                total += rates[i] * (hi - lo)
        out.append(float(total))
        a = b
    return out


class RollingPredictor(CounterfactualPredictor):
    """Shadow of the T1-N review wrapper: the review decision of an action and its first 2 s."""

    def __init__(self, gym_config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 config: T1NShadowConfig | None = None) -> None:
        super().__init__(gym_config, env_config,
                         shadow_factory=lambda: T1NReviewEnv(gym_config, env_config, earliest_summary=False))
        self.shadow_config = config or T1NShadowConfig()

    def rollout(self, action: np.ndarray, h: int = 1) -> dict[str, Any]:
        """The review decision of ``action`` in the synced shadow, then h predicted 2 s steps
        (the MPC solved every 0.1 s at the predicted state, advanced by its own model).

        Returns the predicted step rewards and the plan G* accepted at the decision (with its
        elapsed time then), read before any predicted motion."""

        started = perf_counter()
        sh = self.shadow
        e, g = sh.env, sh.generator
        assert g is not None
        c = e.config
        controller, model = e.controller, e.controller.reference_model
        limits = np.array([*([c.max_torque_per_axis_nm] * 3), *([c.max_force_per_axis_n] * 3)])
        rewards: list[float] = []
        statuses: list[str] = []
        outcome, tau_eff, stop, violation = None, None, None, []
        plan, plan_elapsed = None, None
        solves = 0
        for k in range(h):
            reviewed = k == 0
            sh._continuing = not reviewed
            tau, _, _ = sh.apply_intent(np.asarray(action) if reviewed else sh.continue_action())
            g.decide(e, tau)
            sh._last_request = tau
            statuses.append(g.last_status)
            if reviewed:
                outcome, tau_eff = g.last_outcome, g.tau_effective_s
                sh._last_outcome = outcome
                sh._review_request = tau
                if g.plan is not None:
                    plan, plan_elapsed = g.plan, float(e.time_s - g.plan_start_s)
            reward = 0.0
            for _ in range(c.decision_steps):
                m = e.metrics()
                waypoint, path = g(e, m)
                x = relative_to_vector(relative_state(e.target_state, e.chaser_state))
                wrench, _ = controller.command(
                    x, target_state=e.target_state, time_seconds=e.time_s,
                    external_reference=waypoint, external_reference_path=path, terminal_latched=bool(m.mouth_region))
                solves += 1
                u = np.clip(np.asarray(wrench, dtype=np.float64), -limits, limits)
                x_next = controller._predict(x, u)
                target_next = model.propagate_target(e.target_state, e.time_s)
                e.target_state, e.chaser_state = target_next, model.reconstruct_chaser(target_next, x_next)
                e.time_s = round(e.time_s + c.dt_s, 10)
                dv = float(np.linalg.norm(u[3:])) * c.dt_s / e.chaser_parameters.mass
                reward -= c.reward.time_cost_per_s * c.dt_s + c.reward.delta_v_cost_per_m_s * dv
                m_next = compute_t1_metrics(e.target_state, e.chaser_state, c.task)
                if m_next.hard_violation or m_next.range_m > c.task.max_distance_m or not np.all(np.isfinite(x_next)):
                    violation = sorted(m_next.violations) or ["distance"]
                    break
            rewards.append(reward)
            sh.advance_review(reviewed)
            if violation:
                stop = "violation"
                break
            if e.time_s > c.max_time_s - self.shadow_config.end_margin_s:
                stop = "near_horizon"
                break
        return {"rewards": rewards, "outcome": outcome, "tau_effective_s": tau_eff, "statuses": statuses,
                "stop": stop, "violation": violation, "solves": solves, "plan": plan, "plan_elapsed_s": plan_elapsed,
                "wall_s": perf_counter() - started}

    def truth_branch(self, action: np.ndarray, h: int | None, continuation=None) -> dict[str, Any]:
        """Diagnostics only: the real closed loop (RK45 truth) of ``action`` in the synced shadow, for
        h steps or to the end with ``continuation(obs, env)`` at later reviews (keep by default)."""

        sh = self.shadow
        rewards = []
        obs, reward, term, trunc, info = sh.step(np.asarray(action))
        rewards.append(reward)
        k = 1
        while not (term or trunc) and (h is None or k < h):
            if sh._due and continuation is not None:
                a = continuation(obs, sh)
            else:
                a = sh.continue_action() if not sh._due else np.zeros(1)
            obs, reward, term, trunc, info = sh.step(a)
            rewards.append(reward)
            k += 1
        gamma = sh.env.config.reward.gamma
        return {"rewards": rewards, "discounted_return": float(sum(gamma ** i * r for i, r in enumerate(rewards))),
                "event": info["event"], "observation": obs, "steps": k, "time_s": sh.env.time_s}


def branch_values(p: dict[str, Any], env_config: T1EnvConfig, mass: float) -> dict[str, float] | None:
    """S(b), S_plan(b) and the MPC correction of one branch (module docstring); None without a plan."""

    if p["plan"] is None or not p["rewards"]:
        return None
    c = env_config.reward
    steps = plan_step_rewards(p["plan"], p["plan_elapsed_s"], mass, c.time_cost_per_s, c.delta_v_cost_per_m_s)
    if not steps:
        return None
    gamma = c.gamma
    rest = float(sum(gamma ** j * r for j, r in enumerate(steps) if j >= 1))
    return {"combined": float(p["rewards"][0] + rest), "plan_only": float(steps[0] + rest),
            "mpc_correction": float(p["rewards"][0] - steps[0]), "plan_steps": len(steps)}


def rank_signal(cfg: T1NShadowConfig, p_real: dict, p_alt: dict, env_config: T1EnvConfig, mass: float,
                first_reward_error: float | None, near_horizon: bool) -> dict[str, Any]:
    """D, the plan-only D, both MPC corrections, the confidence w and the reasons it is reduced."""

    reasons: list[str] = []
    v_real, v_alt = branch_values(p_real, env_config, mass), branch_values(p_alt, env_config, mass)
    if p_alt["outcome"] != "accepted":
        reasons.append(f"alt_{p_alt['outcome']}")
    if p_real["outcome"] != "accepted":
        reasons.append(f"real_{p_real['outcome']}")
    if v_real is None or v_alt is None:
        reasons.append("no_plan")
    for name, p in (("real", p_real), ("alt", p_alt)):
        if p["violation"]:
            reasons.append(f"{name}_predicted_violation")
    if (p_real["tau_effective_s"] is not None and p_alt["tau_effective_s"] is not None
            and abs(p_alt["tau_effective_s"] - p_real["tau_effective_s"]) < cfg.min_tau_difference_s):
        reasons.append("same_effective_tau")
    if first_reward_error is None or first_reward_error > cfg.max_first_reward_error:
        reasons.append("e_r1")
    if near_horizon:
        reasons.append("near_horizon")
    signal = plan_only = None
    if v_real is not None and v_alt is not None:
        signal = v_alt["combined"] - v_real["combined"]
        plan_only = v_alt["plan_only"] - v_real["plan_only"]
        if abs(signal) < cfg.min_signal:
            reasons.append("small_signal")
    weight = 0.0
    if not reasons:
        weight = min(1.0, abs(signal) / cfg.ref_signal) * max(0.0, 1.0 - first_reward_error / cfg.max_first_reward_error)
    return {"signal": signal, "plan_only": plan_only, "weight": float(weight), "reasons": reasons,
            "mpc_correction_real": None if v_real is None else v_real["mpc_correction"],
            "mpc_correction_alt": None if v_alt is None else v_alt["mpc_correction"]}


class T1NReviewCFEnv(T1NReviewEnv):
    """The T1-N review wrapper (transitions bitwise those of ``T1NReviewEnv``) plus the ranking
    signal at review states: ``info["counterfactual"]`` with ``a_tilde``, ``signal`` (D), its
    confidence ``rank_weight`` and the diagnostics (plan-only D, MPC corrections, reasons)."""

    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, review_period_steps: int = 3,
                 shadow: T1NShadowConfig | None = None) -> None:
        super().__init__(config, env_config, openings, review_period_steps)
        self.shadow_config = shadow or T1NShadowConfig()
        self.predictor = RollingPredictor(config, env_config, self.shadow_config)

    def _empty(self, kind: str, started: float) -> dict[str, Any]:
        return {"kind": kind, "a_tilde": None, "w": 0, "rank_weight": 0.0, "signal": None, "r_hat": None,
                "z_hat": None, "reasons": [kind], "errors_real": None, "tau_effective_alt_s": None,
                "alt_status": None, "overhead_wall_s": perf_counter() - started}

    def step(self, action: np.ndarray):
        started = perf_counter()
        g = self.generator
        assert g is not None
        if not self._due:
            obs, reward, term, trunc, info = super().step(action)
            info["counterfactual"] = self._empty("continue", started)
            return obs, reward, term, trunc, info
        executed = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:1], -1.0, 1.0)
        alt, kind = alternative_action(self, executed)
        if alt is None:
            obs, reward, term, trunc, info = super().step(action)
            info["counterfactual"] = self._empty(kind, started)
            return obs, reward, term, trunc, info
        pred = self.predictor
        pred.sync(self, fresh_solvers=True)
        p_real = pred.rollout(executed, 1)
        pred.sync(self, fresh_solvers=True)
        p_alt = pred.rollout(alt, 1)
        overhead = perf_counter() - started
        obs, reward, term, trunc, info = super().step(action)
        error = abs(p_real["rewards"][0] - float(reward)) if p_real["rewards"] else None
        near = self.env.time_s > self.env.config.max_time_s - self.shadow_config.end_margin_s
        r = rank_signal(self.shadow_config, p_real, p_alt, self.env.config, self.env.chaser_parameters.mass, error, near)
        if term or trunc:
            r["reasons"].append("real_terminated")
            r["weight"] = 0.0
        valid = r["weight"] > 0.0
        info["counterfactual"] = {
            "kind": kind, "a_tilde": alt.astype(np.float32), "w": int(valid), "rank_weight": r["weight"],
            "signal": r["signal"], "signal_plan_only": r["plan_only"],
            "mpc_correction_real": r["mpc_correction_real"], "mpc_correction_alt": r["mpc_correction_alt"],
            # the base buffer stores (a~, r_hat, z_hat) of a valid sample: r_hat carries D; nothing
            # is bootstrapped from a predicted successor (z_hat is a placeholder, never read)
            "r_hat": r["signal"] if valid else None,
            "z_hat": np.zeros_like(obs) if valid else None,
            "reasons": r["reasons"], "first_reward_error": error,
            "errors_real": None if error is None else {"e_r1": error},
            "tau_effective_real_s": p_real["tau_effective_s"], "tau_effective_alt_s": p_alt["tau_effective_s"],
            "alt_status": p_alt["statuses"][0] if p_alt["statuses"] else None, "alt_outcome": p_alt["outcome"],
            "solves": p_real["solves"] + p_alt["solves"], "overhead_wall_s": overhead,
        }
        return obs, reward, term, trunc, info


__all__ = ["ALT_LIVE_SHIFT_S", "ALT_NO_PLAN_SHIFT_S", "RollingPredictor", "T1NReviewCFEnv", "T1NShadowConfig",
           "alternative_action", "branch_values", "plan_step_rewards", "rank_signal"]
