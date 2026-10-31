"""T1-P stage counterfactual (research plan T1-P sec. 4.2-4.3): same-state 12 s MPC pairs.

On a fixed quarter of the far-field reviews (hash of opening and review index, never of any
outcome) the executed stage action a and one alternative a~ (one axis, by the same hash, moved by
+-0.6 and clipped; the other sign if clipping leaves < 0.3) are each flown from the same snapshot in
an isolated shadow of the review wrapper (own ``T1Env``, MPC and T1-P generator, fresh solvers): the
stage decision, then six 2 s steps with the shared MPC solved every 0.1 s at the *predicted* state
and advanced with its own prediction model (never the truth integrator), the generator's continue and
mode logic after every step, the predicted truth rules at every 0.1 s, and the training reward with
the same potential (``r_train``). A prediction stops at a mode change (bridge, descent, hold), a
predicted violation or capture.

The real stage then runs. When it closes after exactly six steps with no event, the executed
action's own prediction is compared with what really happened: |q^ - q| <= 0.35 m, |v^ - v| <= 0.05
m/s, |R^ - R| <= 1.0; the pair also needs two complete six-step predictions, both stage plans
accepted, and endpoints >= 0.10 m apart. Then w = 1, else 0 (the reasons are logged). The learner's
label uses the *real* stage as the anchor (``train.t1_p_sac``):
    Y = [R^(a~) - R_real(a)] + gamma^6 [Vbar(z^_6(a~)) - Vbar(z_6(a))].
``info["counterfactual"]`` at the review carries (a~, R^(a~), z^(a~)) with ``w = 1`` meaning "pending";
``info["cf_final"]`` on the step that closes (or cancels) the stage carries the final weight.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc.prediction import relative_to_vector
from dynamics.relative import relative_state
from env.t1_counterfactual import _ENV_FIELDS, copy_controller_state, copy_generator_state
from env.t1_env import T1EnvConfig
from env.t1_gym import T1GymConfig
from env.t1_p_review import STAGE_STEPS, T1PReviewEnv
from env.t1_p_stage import potential, relative_inertial
from env.t1_task import compute_t1_metrics

_P_GENERATOR_FIELDS = ("mode", "plan", "plan_start_s", "hold_position", "request", "last_status", "last_outcome",
                       "event", "bridge_tau_s")


@dataclass(frozen=True)
class T1PCFConfig:
    sample_mod: int = 4
    shift: float = 0.6
    min_shift: float = 0.3
    max_position_error_m: float = 0.35
    max_velocity_error_m_s: float = 0.05
    max_return_error: float = 1.0
    min_endpoint_difference_m: float = 0.10


def review_hash(opening: int, review_index: int) -> int:
    return int(hashlib.sha256(f"t1p-cf:{opening}:{review_index}".encode()).hexdigest()[:12], 16)


def alternative(opening: int, review_index: int, action: np.ndarray, cfg: T1PCFConfig) -> tuple[np.ndarray | None, int, float]:
    h = review_hash(opening, review_index)
    if h % cfg.sample_mod != 0:
        return None, -1, 0.0
    axis, sign = (h // cfg.sample_mod) % 3, (1.0 if (h // (3 * cfg.sample_mod)) % 2 == 0 else -1.0)
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
    alt = a.copy()
    alt[axis] = np.clip(a[axis] + sign * cfg.shift, -1.0, 1.0)
    if abs(alt[axis] - a[axis]) < cfg.min_shift:
        sign = -sign
        alt[axis] = np.clip(a[axis] + sign * cfg.shift, -1.0, 1.0)
    return alt, int(axis), float(sign)


class StageShadow:
    def __init__(self, gym_config: T1GymConfig | None, env_config: T1EnvConfig | None) -> None:
        self.shadow = T1PReviewEnv(gym_config, env_config)

    def sync(self, real: T1PReviewEnv) -> None:
        s, r = self.shadow, real
        if s.generator is None:
            s.reset(options={"opening": r.current_opening if r.current_opening is not None else 0})
        s.env.target_state, s.env.chaser_state = r.env.target_state.copy(), r.env.chaser_state.copy()
        for name in _ENV_FIELDS:
            setattr(s.env, name, deepcopy(getattr(r.env, name)))
        copy_controller_state(r.env.controller, s.env.controller)
        cache = getattr(s.env.controller._problem, "_solver_cache", None)
        if cache is not None:
            cache.clear()
        for name in _P_GENERATOR_FIELDS:
            setattr(s.generator, name, deepcopy(getattr(r.generator, name)))
        copy_generator_state(r.generator.bridge, s.generator.bridge)
        s.generator.bridge.last_outcome = r.generator.bridge.last_outcome
        s.generator.plan_wall_s = []
        s.set_review_state(r.review_state())
        s.current_opening, s.episode = r.current_opening, r.episode

    def rollout(self, action: np.ndarray, steps: int = STAGE_STEPS) -> dict[str, Any]:
        """The stage of ``action`` predicted by the MPC on its own model (module docstring)."""

        started = perf_counter()
        sh = self.shadow
        e, g = sh.env, sh.generator
        c = e.config
        controller, model = e.controller, e.controller.reference_model
        limits = np.array([*([c.max_torque_per_axis_nm] * 3), *([c.max_force_per_axis_n] * 3)])
        gamma = c.reward.gamma
        rewards, stop = [], None
        outcome = None
        for k in range(steps):
            sh.decide(np.asarray(action), k == 0)
            if k == 0:
                outcome = g.last_outcome
                if g.mode != "stage":
                    stop = f"not_accepted:{g.last_status}"
                    break
            raw = 0.0
            for _ in range(c.decision_steps):
                m = e.metrics()
                waypoint, path = g(e, m)
                x = relative_to_vector(relative_state(e.target_state, e.chaser_state))
                wrench, _ = controller.command(x, target_state=e.target_state, time_seconds=e.time_s,
                                               external_reference=waypoint, external_reference_path=path,
                                               terminal_latched=bool(m.mouth_region))
                u = np.clip(np.asarray(wrench, dtype=np.float64), -limits, limits)
                x_next = controller._predict(x, u)
                target_next = model.propagate_target(e.target_state, e.time_s)
                e.target_state, e.chaser_state = target_next, model.reconstruct_chaser(target_next, x_next)
                e.time_s = round(e.time_s + c.dt_s, 10)
                dv = float(np.linalg.norm(u[3:])) * c.dt_s / e.chaser_parameters.mass
                raw -= c.reward.time_cost_per_s * c.dt_s + c.reward.delta_v_cost_per_m_s * dv
                m_next = compute_t1_metrics(e.target_state, e.chaser_state, c.task)
                if m_next.hard_violation or m_next.range_m > c.task.max_distance_m or not np.all(np.isfinite(x_next)):
                    stop = "predicted_violation"
                    break
                if m_next.instantaneous_capture:
                    stop = "capture_region"
            if stop == "predicted_violation":
                break
            g.after_step(e)
            phi_next = potential(e)
            rewards.append(raw + gamma * phi_next - sh._phi)
            sh._phi = phi_next
            sh._schedule(k == 0)
            if g.mode != "stage" or g.event is not None:
                stop = f"mode:{g.mode}:{g.event}"
                break
            if e.time_s > c.max_time_s - 4.0:
                stop = "near_horizon"
                break
        q, v = relative_inertial(e)
        return {"rewards": rewards, "complete": len(rewards) == steps and stop in (None, "capture_region"),
                "discounted": float(sum(gamma ** i * r for i, r in enumerate(rewards))), "stop": stop,
                "outcome": outcome, "q_end": q, "v_end": v, "observation": sh._observation(),
                "wall_s": perf_counter() - started}


class T1PReviewCFEnv(T1PReviewEnv):
    """``T1PReviewEnv`` (real transitions bitwise unchanged) plus the stage pairs of the module docstring."""

    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, cf: T1PCFConfig | None = None) -> None:
        super().__init__(config, env_config, openings)
        self.cf_config = cf or T1PCFConfig()
        self.shadow = StageShadow(config, env_config)
        self._pending: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._pending = None
        return super().reset(seed=seed, options=options)

    @staticmethod
    def _empty(kind: str, started: float) -> dict[str, Any]:
        return {"kind": kind, "a_tilde": None, "w": 0, "r_hat": None, "z_hat": None, "reasons": [kind],
                "errors_real": None, "tau_effective_alt_s": None, "alt_status": None,
                "overhead_wall_s": perf_counter() - started}

    def step(self, action: np.ndarray):
        started = perf_counter()
        reviewed, far = self._due, self.generator.mode in ("stage", "hold")
        cf_info = None
        pair = None
        if reviewed and far and not self.generator.descending(self.env):
            executed = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
            alt, axis, sign = alternative(int(self.current_opening), self._review_index, executed, self.cf_config)
            if alt is not None:
                self.shadow.sync(self)
                p_real = self.shadow.rollout(executed)
                self.shadow.sync(self)
                p_alt = self.shadow.rollout(alt)
                reasons = []
                for name, p in (("real", p_real), ("alt", p_alt)):
                    if not p["complete"]:
                        reasons.append(f"{name}_incomplete:{p['stop']}")
                if float(np.linalg.norm(p_alt["q_end"] - p_real["q_end"])) < self.cf_config.min_endpoint_difference_m:
                    reasons.append("same_endpoint")
                pair = {"a_tilde": alt, "axis": axis, "sign": sign, "p_real": p_real, "p_alt": p_alt, "reasons": reasons}
                cf_info = {"kind": "stage_pair", "a_tilde": alt.astype(np.float32), "w": int(not reasons),
                           "r_hat": p_alt["discounted"], "z_hat": p_alt["observation"], "reasons": list(reasons),
                           "pending": not reasons, "axis": axis, "sign": sign,
                           "r_hat_real_pred": p_real["discounted"], "errors_real": None,
                           "tau_effective_alt_s": None, "alt_status": p_alt["stop"],
                           "endpoint_difference_m": float(np.linalg.norm(p_alt["q_end"] - p_real["q_end"])),
                           "overhead_wall_s": perf_counter() - started}
            else:
                cf_info = self._empty("not_sampled", started)
        elif reviewed:
            cf_info = self._empty("not_far_field", started)
        else:
            cf_info = self._empty("continue", started)
        obs, reward, term, trunc, info = super().step(action)
        info["counterfactual"] = cf_info
        final = None
        if pair is not None and not pair["reasons"]:
            self._pending = {"pair": pair, "rewards": [reward], "steps": 1}
        elif self._pending is not None:
            self._pending["rewards"].append(reward)
            self._pending["steps"] += 1
        if self._pending is not None and (term or trunc or self._due):
            final = self._finalize(self._pending, term or trunc)
            self._pending = None
        if final is not None:
            info["cf_final"] = final
        return obs, reward, term, trunc, info

    def _finalize(self, pending: dict[str, Any], ended: bool) -> dict[str, Any]:
        cfg = self.cf_config
        pair = pending["pair"]
        reasons = []
        if ended:
            reasons.append("episode_ended")
        if pending["steps"] != STAGE_STEPS or not self._due:
            reasons.append(f"stage_length:{pending['steps']}")
        gamma = self.env.config.reward.gamma
        real_return = float(sum(gamma ** i * r for i, r in enumerate(pending["rewards"])))
        q, v = relative_inertial(self.env)
        p = pair["p_real"]
        eq, ev = float(np.linalg.norm(p["q_end"] - q)), float(np.linalg.norm(p["v_end"] - v))
        er = abs(p["discounted"] - real_return)
        if eq > cfg.max_position_error_m:
            reasons.append("e_q")
        if ev > cfg.max_velocity_error_m_s:
            reasons.append("e_v")
        if er > cfg.max_return_error:
            reasons.append("e_R")
        return {"weight": 0.0 if reasons else 1.0, "reasons": reasons, "e_q_m": eq, "e_v_m_s": ev, "e_R": er,
                "real_return": real_return, "alt_return_pred": pair["p_alt"]["discounted"],
                "real_return_pred": p["discounted"]}


__all__ = ["StageShadow", "T1PCFConfig", "T1PReviewCFEnv", "alternative", "review_hash"]
