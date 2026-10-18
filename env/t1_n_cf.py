"""T1-N paired MPC shadow (Research Plan R2 sec. 6.2-6.3): same-state rolling predictions.

At a review state z (outside G*'s descent mode) the executed action a and one alternative a~
from the same arrival-time action space are each flown in an isolated shadow of the review
wrapper (its own ``T1Env``, MPC controller and conditional G*, synchronised from the real one
with fresh solvers) for ``h`` 2 s steps:

* the first step makes the review decision for the action (conditional G*); the next ones are
  continue steps (the accepted plan is kept, a > 1 m deviation is re-planned for the same
  remaining time, exactly as in the real wrapper);
* every 0.1 s the shadow MPC solves at the *predicted* state (``MPCController.command``, the
  same receding-horizon controller with its own warm start and drift refresh) and the state is
  advanced with the MPC's own prediction model (``controller._predict``: local linear model plus
  the drift); the target is advanced with the prediction model's torque-free propagation. The
  real truth integrator is never called and no real branch is ever read;
* the predicted reward uses the real metering (time and fuel at the configured weight); the
  predicted truth rules are checked at every 0.1 s step.

A prediction stops early (``h_eff < h``) when the shadow would hand the decision back to SAC
(a plan event: hold, lost plan) or ends (predicted violation, capture). The pair is usable only
when both predictions reach the same ``h_eff >= 1`` with no predicted violation, the alternative
was accepted by G*, the two differ physically (effective arrival >= 4 s apart, or keep vs a new
plan), and the executed action's own first-step prediction agrees with what really happened
(|r^_1 - r_1| <= ``max_first_reward_error``; the real step is what the training sample holds).

The learner's relative label (``train.t1_n_sac``) is the *paired* one,
    delta = (R^(a~) - R^(a)) + gamma^h_eff [ Vbar(z^'(a~)) - Vbar(z^'(a)) ],
both terms predicted by the same model from the same state, so model errors common to the two
branches cancel. It still bootstraps on the target critic and is not an unbiased long-run label.
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
from env.t1_n_review import N_HAS_PLAN_INDEX, T1NReviewEnv, action_for_tau, neutral_tau
from env.t1_task import compute_t1_metrics

#: request shift of the alternative without a live plan (s), and the shift tried against a kept plan
ALT_NO_PLAN_SHIFT_S = 30.0
ALT_LIVE_SHIFT_S = 15.0


@dataclass(frozen=True)
class T1NShadowConfig:
    horizon_steps: int = 2
    min_tau_difference_s: float = 4.0
    #: executed-action first-step reward error bound (fuel weight 20); provisional until calibrated
    max_first_reward_error: float = 0.25
    end_margin_s: float = 4.0


def alternative_action(env: T1NReviewEnv, action: np.ndarray) -> tuple[np.ndarray | None, str]:
    """The single alternative of a review, fixed before any training (R2 sec. 6.2).

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


class RollingPredictor(CounterfactualPredictor):
    """Shadow of the T1-N review wrapper; ``rollout`` is the rolling MPC prediction."""

    def __init__(self, gym_config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 config: T1NShadowConfig | None = None) -> None:
        super().__init__(gym_config, env_config,
                         shadow_factory=lambda: T1NReviewEnv(gym_config, env_config, earliest_summary=True))
        self.shadow_config = config or T1NShadowConfig()

    def rollout(self, action: np.ndarray, h: int) -> dict[str, Any]:
        """h predicted 2 s steps of ``action`` from the synced shadow (module docstring)."""

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
        first_controls = None
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
            reward = 0.0
            controls = []
            for _ in range(c.decision_steps):
                m = e.metrics()
                waypoint, path = g(e, m)
                x = relative_to_vector(relative_state(e.target_state, e.chaser_state))
                wrench, diagnostics = controller.command(
                    x, target_state=e.target_state, time_seconds=e.time_s,
                    external_reference=waypoint, external_reference_path=path, terminal_latched=bool(m.mouth_region))
                solves += 1
                u = np.clip(np.asarray(wrench, dtype=np.float64), -limits, limits)
                controls.append(u)
                x_next = controller._predict(x, u)
                target_next = model.propagate_target(e.target_state, e.time_s)
                chaser_next = model.reconstruct_chaser(target_next, x_next)
                e.target_state, e.chaser_state = target_next, chaser_next
                e.time_s = round(e.time_s + c.dt_s, 10)
                dv = float(np.linalg.norm(u[3:])) * c.dt_s / e.chaser_parameters.mass
                reward -= c.reward.time_cost_per_s * c.dt_s + c.reward.delta_v_cost_per_m_s * dv
                m_next = compute_t1_metrics(e.target_state, e.chaser_state, c.task)
                if m_next.hard_violation or m_next.range_m > c.task.max_distance_m or not np.all(np.isfinite(x_next)):
                    violation = sorted(m_next.violations) or ["distance"]
                    break
                if m_next.instantaneous_capture:
                    stop = "capture_region"
            if k == 0:
                first_controls = np.array(controls)
            rewards.append(reward)
            sh.advance_review(reviewed)
            if violation:
                stop = "violation"
                break
            if e.time_s > c.max_time_s - self.shadow_config.end_margin_s:
                stop = "near_horizon"
                break
            if sh._due and k < h - 1 and sh._plan_event():
                stop = "plan_event"
                break
        gamma = c.reward.gamma
        return {
            "rewards": rewards, "h_eff": len(rewards),
            "discounted_return": float(sum(gamma ** i * r for i, r in enumerate(rewards))),
            "observation": sh._observation(), "outcome": outcome, "tau_effective_s": tau_eff,
            "statuses": statuses, "stop": stop, "violation": violation, "solves": solves,
            "first_controls": first_controls, "wall_s": perf_counter() - started,
        }

    def truth_branch(self, action: np.ndarray, h: int | None, continuation=None) -> dict[str, Any]:
        """Validation only: the real closed loop (RK45 truth) of ``action`` in the synced shadow, for h
        steps or to the end with ``continuation(obs, env)`` at later reviews (keep by default)."""

        sh = self.shadow
        rewards, events = [], None
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
        events = info["event"]
        gamma = sh.env.config.reward.gamma
        return {"rewards": rewards, "discounted_return": float(sum(gamma ** i * r for i, r in enumerate(rewards))),
                "event": events, "observation": obs, "steps": k, "time_s": sh.env.time_s}


def pair_gate(cfg: T1NShadowConfig, p_real: dict, p_alt: dict, kind: str, first_reward_error: float | None) -> list[str]:
    reasons = []
    if p_alt["outcome"] != "accepted":
        reasons.append(f"alt_{p_alt['outcome']}")
    if p_real["outcome"] not in ("accepted",):
        reasons.append(f"real_{p_real['outcome']}")
    if p_real["h_eff"] != p_alt["h_eff"]:
        reasons.append("unequal_window")
    for name, p in (("real", p_real), ("alt", p_alt)):
        if p["violation"]:
            reasons.append(f"{name}_predicted_violation")
    if kind == "request_shift" or kind == "keep_vs_shift":
        if (p_real["tau_effective_s"] is not None and p_alt["tau_effective_s"] is not None
                and abs(p_alt["tau_effective_s"] - p_real["tau_effective_s"]) < cfg.min_tau_difference_s
                and kind == "request_shift"):
            reasons.append("same_effective_tau")
    if first_reward_error is not None and first_reward_error > cfg.max_first_reward_error:
        reasons.append("e_r1")
    return reasons


class T1NReviewCFEnv(T1NReviewEnv):
    """The T1-N review wrapper (transitions bitwise those of ``T1NReviewEnv``) plus the paired
    shadow at review states: ``info["counterfactual"]`` with ``a_tilde``, ``r_hat`` (alternative,
    discounted over h_eff steps), ``r_hat_real`` (executed action, predicted), ``z_hat``,
    ``z_hat_real``, ``h_eff``, ``w`` and the gate reasons."""

    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, review_period_steps: int = 3,
                 shadow: T1NShadowConfig | None = None) -> None:
        super().__init__(config, env_config, openings, review_period_steps)
        self.shadow_config = shadow or T1NShadowConfig()
        self.predictor = RollingPredictor(config, env_config, self.shadow_config)

    def _empty(self, kind: str, started: float) -> dict[str, Any]:
        return {"kind": kind, "a_tilde": None, "r_hat": None, "r_hat_real": None, "z_hat": None, "z_hat_real": None,
                "h_eff": 0, "w": 0, "reasons": [kind], "errors_real": None, "tau_effective_alt_s": None,
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
        h = self.shadow_config.horizon_steps
        pred = self.predictor
        pred.sync(self, fresh_solvers=True)
        p_real = pred.rollout(executed, h)
        pred.sync(self, fresh_solvers=True)
        p_alt = pred.rollout(alt, h)
        overhead = perf_counter() - started
        obs, reward, term, trunc, info = super().step(action)
        error = abs(p_real["rewards"][0] - float(reward)) if p_real["rewards"] else None
        reasons = pair_gate(self.shadow_config, p_real, p_alt, kind, error)
        if term or trunc:
            reasons.append("real_terminated")
        h_eff = min(p_real["h_eff"], p_alt["h_eff"])
        gamma = self.env.config.reward.gamma
        info["counterfactual"] = {
            "kind": kind, "a_tilde": alt.astype(np.float32),
            "r_hat": float(sum(gamma ** i * r for i, r in enumerate(p_alt["rewards"][:h_eff]))),
            "r_hat_real": float(sum(gamma ** i * r for i, r in enumerate(p_real["rewards"][:h_eff]))),
            "z_hat": p_alt["observation"], "z_hat_real": p_real["observation"], "h_eff": int(h_eff),
            "w": int(not reasons), "reasons": reasons, "first_reward_error": error,
            "errors_real": None if error is None else {"e_r1": error}, "alt_status": p_alt["statuses"][0],
            "tau_effective_real_s": p_real["tau_effective_s"], "tau_effective_alt_s": p_alt["tau_effective_s"],
            "alt_outcome": p_alt["outcome"], "solves": p_real["solves"] + p_alt["solves"],
            "overhead_wall_s": overhead,
        }
        return obs, reward, term, trunc, info


__all__ = ["ALT_LIVE_SHIFT_S", "ALT_NO_PLAN_SHIFT_S", "RollingPredictor", "T1NReviewCFEnv", "T1NShadowConfig",
           "alternative_action", "pair_gate"]
