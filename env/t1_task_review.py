"""T1-L task interface: feasible-anchor delay, plan reviews and the continue operation.

Plan 2026-10-15 "T1-L", modules B and C. The physics, G* (v0.6), the shared MPC and the
reward are those of ``T1GStarEnv`` (interface v4); this wrapper changes *when* the policy's
action is a decision and *how* a decision without a live plan is read.

Action ``a`` in [-1, 1]^3 at a review:
  no live plan (and not descending):
      v_in = 0.06 + 0.04 a[2];  tau_min = G*'s earliest feasible arrival for this v_in at the
      current state (``GStarRendezvous.earliest_feasible``: same planner, same ascending grid);
      delay = 30 (a[0] + 1) s in [0, 60];  tau_request = tau_min + delay.
      G* then plans tau_request with its bounded slips, deadline and recovery (v0.5/v0.6);
      every number (tau_min, delay, request, effective tau, status) is logged. a[0] = -1 is
      "earliest feasible"; the untrained policy (a[0] ~ 0) asks for a 30 s delay.
      No feasible anchor: an earliest request (G*'s recovery / hold), logged as such.
  live plan: interface v2/v4 -- |a[0]| <= 0.5 keeps the plan, beyond it shifts the arrival by
      up to +-30 s around the remaining time; a[2] only reaches a plan that is actually made.
  a[1] -> kappa (0.3..1.0), committed until the next review.

Reviews: the first step of an episode; then every ``review_period_steps`` 2 s steps (3 = 6 s,
the main method; 1 = every step, the B2 ablation); and early, at the step after a plan event
(G* hold or recovery, or no live plan outside the descent). Between reviews the step executes
the *continue* operation, whatever the policy proposed: G* is asked to keep the accepted
arrival (tau = remaining time) with the committed kappa and the live plan's own entry speed;
its deviation re-plan, deadline and recovery logic stay active. The continue action written
to the replay buffer is a deterministic function of the observation
(``continue_action_from_observation``): (0, kappa of the observation, plan entry speed of the
observation, or 0 without a plan), which reproduces the operation exactly.

Observation: interface v4's 25 dims, then ``review_due``, ``review_gap_3`` (2 s steps since
the last review / 3; at a review it is the gap the review closes), ``deadline_slack_160s``
((max_time - deadline margin - t - remaining) / 160) and the signed position and velocity
errors against the live plan in the target body frame (/1 m, /0.3 m/s; 0 without a plan).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from env.t1_env import T1EnvConfig
from env.t1_gstar import _plan_position
from env.t1_gym import OBSERVATION_NAMES, T1GStarEnv, T1GymConfig, _plan_velocity
from env.t1_l_task import t1l_env_config

REVIEW_OBSERVATION_NAMES = OBSERVATION_NAMES + (
    "review_due", "review_gap_3", "deadline_slack_160s",
    "plan_error_x_1m", "plan_error_y_1m", "plan_error_z_1m",
    "plan_velocity_error_x_0p3", "plan_velocity_error_y_0p3", "plan_velocity_error_z_0p3",
)
DUE_INDEX = REVIEW_OBSERVATION_NAMES.index("review_due")
GAP_INDEX = REVIEW_OBSERVATION_NAMES.index("review_gap_3")
KAPPA_INDEX = REVIEW_OBSERVATION_NAMES.index("kappa")
HAS_PLAN_INDEX = REVIEW_OBSERVATION_NAMES.index("has_plan")
ENTRY_INDEX = REVIEW_OBSERVATION_NAMES.index("plan_entry_speed")
DELAY_MAX_S = 60.0
INTERFACE_VERSION = "t1l-review-1"


def continue_action_from_observation(obs: np.ndarray, config: T1GymConfig | None = None) -> np.ndarray:
    """The action recorded for a continue step: (keep, committed kappa, live plan entry speed)."""

    c = config or T1GymConfig()
    o = np.asarray(obs, dtype=np.float32)
    a1 = 2.0 * (float(o[KAPPA_INDEX]) - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
    a2 = float(o[ENTRY_INDEX]) if o[HAS_PLAN_INDEX] > 0.5 else 0.0
    return np.clip(np.array([0.0, a1, a2], dtype=np.float32), -1.0, 1.0)


def gap_steps(obs: np.ndarray) -> int:
    return int(round(float(np.asarray(obs)[GAP_INDEX]) * 3.0))


class T1LReviewEnv(T1GStarEnv):
    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, review_period_steps: int = 3) -> None:
        super().__init__(config, env_config or t1l_env_config(), openings)
        self.review_period_steps = int(review_period_steps)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(REVIEW_OBSERVATION_NAMES),),
                                                dtype=np.float32)
        self._init_review_state()

    # ------------------------------------------------------------------ #
    def _init_review_state(self) -> None:
        c = self.config
        self._due = True
        self._since_review = self.review_period_steps
        self._review_index = 0
        self._committed_kappa = 0.5 * (c.kappa_min + c.kappa_max)
        self._committed_entry_speed = c.entry_speed_nominal_m_s
        self._continuing = False
        self._anchor: dict[str, Any] | None = None

    def review_state(self) -> dict[str, Any]:
        return {"due": self._due, "since_review": self._since_review, "review_index": self._review_index,
                "committed_kappa": self._committed_kappa, "committed_entry_speed": self._committed_entry_speed}

    def set_review_state(self, state: dict[str, Any]) -> None:
        self._due = bool(state["due"])
        self._since_review = int(state["since_review"])
        self._review_index = int(state["review_index"])
        self._committed_kappa = float(state["committed_kappa"])
        self._committed_entry_speed = float(state["committed_entry_speed"])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._init_review_state()
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------ #
    def apply_intent(self, action: np.ndarray) -> tuple[float, float, float]:
        """(tau, kappa, v_in) for this step, set on G*: a review reads ``action``; a continue
        step ignores it."""

        g, e = self.generator, self.env
        assert g is not None
        remaining = g.remaining_s(e)
        if self._continuing:
            kappa = self._committed_kappa
            v_in = self._plan_entry_speed()
            v_in = self._committed_entry_speed if v_in is None else v_in
            tau = remaining  # keep the accepted arrival (0 in the descent: ignored by G*)
            self._anchor = None
        else:
            a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
            if a.size < 3:
                a = np.concatenate([a, np.zeros(3 - a.size)])
            c = self.config
            kappa = c.kappa_min + (c.kappa_max - c.kappa_min) * 0.5 * (a[1] + 1.0)
            v_in = self.entry_speed_of(a)
            self._anchor = None
            if remaining > 0.0:
                tau = max(remaining + self.shift_of(float(a[0])), 0.0)
            elif g._descending(e, e.metrics()):
                tau = 0.0  # G* is in its descent mode; the arrival request has no effect
            else:
                tau_min, _ = g.earliest_feasible(e, v_in)
                delay = 0.5 * DELAY_MAX_S * (a[0] + 1.0)
                tau = 0.0 if tau_min is None else tau_min + delay
                self._anchor = {"tau_min_feasible_s": tau_min, "delay_s": float(delay),
                                "tau_request_s": float(tau), "no_feasible_anchor": tau_min is None}
        g.kappa = float(kappa)
        g.entry_speed_m_s = float(v_in)
        return float(tau), float(kappa), float(v_in)

    def continue_action(self) -> np.ndarray:
        return continue_action_from_observation(self._observation(), self.config)

    def _plan_event(self) -> bool:
        g, e = self.generator, self.env
        assert g is not None
        if g.last_status.startswith(("hold", "recovery")):
            return True
        return g.remaining_s(e) <= 0.0 and not g._descending(e, e.metrics())

    def advance_review(self, reviewed: bool, kappa: float, v_in: float) -> None:
        """Review bookkeeping after the decision of a step (used by ``step`` and the M2 shadow)."""

        if reviewed:
            self._committed_kappa, self._committed_entry_speed = kappa, v_in
            self._since_review = 1
            self._review_index += 1
        else:
            self._since_review += 1
        self._due = self._since_review >= self.review_period_steps or self._plan_event()

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray):
        reviewed = self._due
        gap = self._since_review
        review_index = self._review_index
        if reviewed:
            executed = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
            if executed.size < 3:
                executed = np.concatenate([executed, np.zeros(3 - executed.size, dtype=np.float32)])
            self._continuing = False
        else:
            executed = self.continue_action()
            self._continuing = True
        _, reward, terminated, truncated, info = super().step(executed)
        self._continuing = False
        x = info["execution"]
        anchor = self._anchor
        self.advance_review(reviewed, float(info["kappa"]), float(x["entry_speed_request_m_s"]))
        info["executed_action"] = executed.astype(np.float32)
        info["review"] = {
            "reviewed": reviewed, "gap_steps": gap, "review_index": review_index,
            "next_due": self._due, "plan_event": self._plan_event(), **(anchor or {}),
        }
        return self._observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _signed_plan_errors(self) -> tuple[np.ndarray, np.ndarray]:
        e, g = self.env, self.generator
        assert g is not None
        if g.plan is None or g.remaining_s(e) <= 0.0:
            return np.zeros(3), np.zeros(3)
        s = e.time_s - g.plan_start_s
        q = e.chaser_state.position - e.target_state.position
        v = e.chaser_state.rotation @ e.chaser_state.velocity - e.target_state.rotation @ e.target_state.velocity
        body = e.target_state.rotation.T
        return (body @ (_plan_position(g.plan, s, g.planner.mass) - q),
                body @ (_plan_velocity(g.plan, s, g.planner.mass) - v))

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        e, g = self.env, self.generator
        assert g is not None
        horizon = float(e.config.max_time_s) - g.deadline_margin_s - float(e.time_s)
        slack = (horizon - g.remaining_s(e)) / 160.0
        dq, dv = self._signed_plan_errors()
        extra = [float(self._due), self._since_review / 3.0, slack, *(dq / 1.0), *(dv / 0.3)]
        obs = np.concatenate([base, np.asarray(extra, dtype=np.float32)])
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError(f"non-finite observation {obs}")
        return obs


__all__ = [
    "DELAY_MAX_S", "DUE_INDEX", "ENTRY_INDEX", "GAP_INDEX", "HAS_PLAN_INDEX", "INTERFACE_VERSION", "KAPPA_INDEX",
    "REVIEW_OBSERVATION_NAMES", "T1LReviewEnv", "continue_action_from_observation", "gap_steps",
]
