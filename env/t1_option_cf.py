"""T1-L counterfactual (plan 2026-10-15, module D): one MPC-predicted alternative per review.

Same isolation, prediction and gates as ``env.t1_counterfactual`` (a shadow copy of the
*review* wrapper, synchronised from the real one; one G* decision and one MPC solve; the
whole predicted 2 s window checked against the truth rules; the real action's own
prediction error bounds e_r / e_p / e_o). New here:

* only review steps get an alternative; a continue step records ``kind = "continue"``, w = 0;
* the single alternative alternates by ``(episode + review index) % 2`` between
  - a tau change: without a live plan the delay moves by +-15 s (a[0] +- 0.5, towards the
    inside of [-1, 1]); with a live plan, keep vs a +-15 s shift (or keep, if the real action
    re-plans), as before;
  - an entry-speed change: a[2] +- 1 (0.04 m/s, towards the inside); without a live plan the
    alternative's feasible anchor is that of its own entry speed (the interface recomputes it);
* the alternative counts only when its accepted plan differs physically from the real one:
  effective arrival by >= 4 s (tau change) or both a new plan with entry speeds >= 0.02 m/s
  apart (entry-speed change); a candidate that fails is recorded as invalid, no other is tried.

The learner uses the cached ``(a~, r^, z^', w)`` in a *relative* target anchored on the real
transition (``train.t1_option_sac``); the successor ``z^'`` carries the shadow's own plan
memory and review clock.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from env.t1_counterfactual import CounterfactualConfig, CounterfactualPredictor, prediction_errors
from env.t1_env import T1EnvConfig
from env.t1_gym import T1GymConfig
from env.t1_task_review import DELAY_MAX_S, T1LReviewEnv

TAU_SHIFT_S = 15.0
#: T1-O near-field kappa counterfactual (research decision 2026-10-17): the alternative changes
#: the committed descent pace by 0.2, towards the middle of [0.3, 1] (down above 0.65, up
#: otherwise), fixed before any data; it is gated only when it changes the predicted controls
#: by more than numerical noise and the whole predicted window stays in the same mode
NEAR_KAPPA_SHIFT = 0.2
NEAR_KAPPA_PIVOT = 0.65
NEAR_MIN_CONTROL_DIFF = 1.0e-3


class ReviewCounterfactualPredictor(CounterfactualPredictor):
    def alternative(self, real: T1LReviewEnv, action: np.ndarray, real_prediction) -> tuple[np.ndarray | None, str]:
        g = real.generator
        assert g is not None
        if g._descending(real.env, real.env.metrics()):
            return None, "descent_mode"
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.size < 3:
            a = np.concatenate([a, np.zeros(3 - a.size)])
        alt = a.copy()
        index = real._review_index
        if (real.episode + index) % 2 == 1:
            alt[2] = a[2] + (1.0 if a[2] < 0.0 else -1.0)
            return alt, "entry_speed"
        remaining = g.remaining_s(real.env)
        if remaining > 0.0:
            if abs(real.shift_of(float(a[0]))) > real.config.keep_tolerance_s:
                alt[0] = 0.0
                return alt, "replan_vs_keep"
            sign = 1.0 if index % 4 < 2 else -1.0
            if remaining - TAU_SHIFT_S <= g.tau_min_s:
                sign = 1.0
            alt[0] = real.action_for_shift(sign * TAU_SHIFT_S)
            return alt, "keep_vs_shift"
        step = TAU_SHIFT_S / (0.5 * real.delay_max_s)  # delay = delay_max (a0 + 1) / 2: 15 s = 0.5 (T1-L), 1/3 (T1-O)
        alt[0] = a[0] + step if a[0] + step <= 1.0 else a[0] - step
        return alt, "delay_shift"


def near_kappa_alternative(real: T1LReviewEnv, action: np.ndarray) -> np.ndarray:
    """The near-field alternative: the same action with kappa moved by NEAR_KAPPA_SHIFT."""

    c = real.config
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if a.size < 3:
        a = np.concatenate([a, np.zeros(3 - a.size)])
    kappa = c.kappa_min + (c.kappa_max - c.kappa_min) * 0.5 * (a[1] + 1.0)
    alt_kappa = kappa - NEAR_KAPPA_SHIFT if kappa > NEAR_KAPPA_PIVOT else kappa + NEAR_KAPPA_SHIFT
    alt = a.copy()
    alt[1] = float(np.clip(2.0 * (alt_kappa - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0, -1.0, 1.0))
    return alt


def near_kappa_gate(real: T1LReviewEnv, cfg: CounterfactualConfig, p_real, p_alt, errors) -> tuple[int, list[str]]:
    """Gate of the near-field kappa alternative: the far-field gate with the descent mode as the
    tracking mode, the near-field mode switch, and a physical-difference requirement."""

    reasons = []
    if float(np.max(np.linalg.norm(p_alt.extra["controls"] - p_real.extra["controls"], axis=1))) < NEAR_MIN_CONTROL_DIFF:
        reasons.append("same_execution")
    for name, p in (("alt", p_alt), ("real", p_real)):
        if p.plan_status != "descent":
            reasons.append(f"{name}_not_descent:{p.plan_status.split(':')[0]}")
        if not p.solved:
            reasons.append(f"{name}_fallback")
        if p.solve_slack > cfg.max_solve_slack:
            reasons.append(f"{name}_slack")
        if p.extra.get("near_mode_switch"):
            reasons.append(f"{name}_mode_switch")
        if p.window_violation:
            reasons.append(f"{name}_predicted_violation:{'+'.join(p.window_violation_kinds)}")
    if p_alt.saturation > cfg.max_predicted_saturation:
        reasons.append("alt_saturation")
    if real.env.time_s > real.env.config.max_time_s - cfg.end_margin_s:
        reasons.append("near_horizon")
    if errors["e_r"] > cfg.max_reward_error:
        reasons.append("e_r")
    if errors["e_p"] > cfg.max_position_error_m:
        reasons.append("e_p")
    if errors["e_o"] > cfg.max_observation_error:
        reasons.append("e_o")
    return int(not reasons), reasons


class T1LReviewCFEnv(T1LReviewEnv):
    """The review wrapper (transitions bitwise those of ``T1LReviewEnv``) plus
    ``info["counterfactual"]`` at review steps. ``near_field_kappa`` (T1-O): reviews in G*'s
    descent mode get the kappa alternative (``near_kappa_alternative``) instead of none."""

    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, review_period_steps: int = 3,
                 counterfactual: CounterfactualConfig | None = None, delay_max_s: float = DELAY_MAX_S,
                 near_field_kappa: bool = False) -> None:
        super().__init__(config, env_config, openings, review_period_steps, delay_max_s=delay_max_s)
        cfg, env_cfg, period, delay = config, env_config, review_period_steps, delay_max_s
        self.near_field_kappa = bool(near_field_kappa)
        self.predictor = ReviewCounterfactualPredictor(
            config, env_config, counterfactual,
            shadow_factory=lambda: T1LReviewEnv(cfg, env_cfg, review_period_steps=period, delay_max_s=delay))

    @staticmethod
    def _empty(kind: str, started: float) -> dict:
        return {"kind": kind, "a_tilde": None, "r_hat": None, "z_hat": None, "w": 0, "reasons": [kind],
                "errors_real": None, "real_first_wrench": None, "tau_effective_real_pred_s": None,
                "tau_effective_alt_s": None, "alt_status": None, "alt_saturation": None,
                "entry_speed_real_plan_m_s": None, "entry_speed_alt_plan_m_s": None,
                "alt_first_wrench_diff": None, "alt_controls_diff_max": None, "alt_reward_diff": None,
                "alt_observation_diff": None, "overhead_wall_s": perf_counter() - started}

    def step(self, action: np.ndarray):
        started = perf_counter()
        assert self.generator is not None
        if not self._due:
            obs, reward, terminated, truncated, info = super().step(action)
            info["counterfactual"] = self._empty("continue", started)
            return obs, reward, terminated, truncated, info
        descending = self.generator._descending(self.env, self.env.metrics())
        if descending and not self.near_field_kappa:
            obs, reward, terminated, truncated, info = super().step(action)
            info["counterfactual"] = self._empty("descent_mode", started)
            return obs, reward, terminated, truncated, info
        pred = self.predictor
        executed = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        pred.sync(self)
        targets = pred.target_window()
        p_real = pred.predict(executed, targets)
        if descending:
            alt, kind = near_kappa_alternative(self, executed), "near_kappa"
        else:
            alt, kind = pred.alternative(self, executed, p_real)
        p_alt = None
        if alt is not None:
            pred.sync(self)
            p_alt = pred.predict(alt, targets)
        overhead = perf_counter() - started
        obs, reward, terminated, truncated, info = super().step(action)
        rel = self.env.chaser_state.position - self.env.target_state.position
        errors = prediction_errors(p_real, reward, obs, rel)
        if p_alt is None or terminated or truncated:
            w, reasons = 0, [kind if p_alt is None else "real_terminated"]
        elif kind == "near_kappa":
            w, reasons = near_kappa_gate(self, pred.config, p_real, p_alt, errors)
        else:
            w, reasons = pred.gate(self, p_real, p_alt, errors, "entry_speed" if kind == "entry_speed" else kind)
        info["counterfactual"] = {
            "kind": kind,
            "a_tilde": None if alt is None else alt.astype(np.float32),
            "r_hat": None if p_alt is None else p_alt.reward,
            "z_hat": None if p_alt is None else p_alt.observation,
            "w": w, "reasons": reasons, "errors_real": errors,
            "real_first_wrench": p_real.first_wrench,
            "tau_effective_real_pred_s": p_real.tau_effective_s,
            "tau_effective_alt_s": None if p_alt is None else p_alt.tau_effective_s,
            "alt_status": None if p_alt is None else p_alt.plan_status,
            "alt_saturation": None if p_alt is None else p_alt.saturation,
            "entry_speed_real_plan_m_s": p_real.extra.get("entry_speed_plan_m_s"),
            "entry_speed_alt_plan_m_s": None if p_alt is None else p_alt.extra.get("entry_speed_plan_m_s"),
            "alt_first_wrench_diff": None if p_alt is None else float(np.linalg.norm(p_alt.first_wrench - p_real.first_wrench)),
            "alt_controls_diff_max": None if p_alt is None else float(np.max(np.linalg.norm(
                p_alt.extra["controls"] - p_real.extra["controls"], axis=1))),
            "alt_reward_diff": None if p_alt is None else float(p_alt.reward - p_real.reward),
            "alt_observation_diff": None if p_alt is None else float(np.linalg.norm(p_alt.observation - p_real.observation)),
            "overhead_wall_s": overhead,
        }
        return obs, reward, terminated, truncated, info


__all__ = ["NEAR_KAPPA_PIVOT", "NEAR_KAPPA_SHIFT", "NEAR_MIN_CONTROL_DIFF", "ReviewCounterfactualPredictor",
           "T1LReviewCFEnv", "TAU_SHIFT_S", "near_kappa_alternative", "near_kappa_gate"]
