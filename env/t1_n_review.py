"""T1-N review interface ``t1n-review-1`` (Research Plan R2 sec. 3 and 5).

One SAC action ``a`` in [-1, 1] (``Box(1)``), read only at review states (every 3 real 2 s steps,
and at the next step after a plan event: hold, plan lost, no plan outside the descent):

* **no live plan** (first decision, after a rejection or a lost plan): a neutral requested
  arrival time, not anchored on G*'s earliest feasible solution,
      tau_req = tau_low + (a + 1) / 2 * (tau_high(t) - tau_low),
      tau_low = 20 s, tau_high(t) = min(200 s, max_time - deadline_margin - t) = min(200, 240 - t);
* **live plan**: |a| <= 0.5 keeps it; outside the band the arrival is shifted linearly up to
  +-30 s (``T1GStarEnv.shift_of``, unchanged).

Between reviews the accepted plan continues (continue action ``[0.0]`` = keep). The executed
(raw) review action is what Replay stores; the request outcome (accepted / rejected), the
effective arrival and the plan memory enter the successor observation.

The descent pace (kappa 1.0) and entry speed (0.02 m/s) are fixed for every plan
(``env.t1_n_task``). G* is ``ConditionalGStar``: it plans exactly the requested time.

Observation (``T1N_OBSERVATION_NAMES``, 41 values) in the episode's angular-momentum frame
F_H (z = initial angular momentum direction of the target, x = initial line of sight projected
normal to it; inertial and fixed per episode): relative position and inertial relative velocity,
the target attitude as two body axes (approach axis, body z), the target's inertial angular
velocity, interpretable geometry (range, latitude, port phase, FOV margin), time and deadline
slack, the mode flags, the plan memory and the last request's outcome, the signed tracking error
against the live plan, a read-only earliest-feasible summary (computed before the decision with
the same planner on G*'s grid, only at review states without a live plan; never future truth)
and the review clock.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from env.t1_env import T1EnvConfig
from env.t1_gstar import _plan_position
from env.t1_gym import T1GStarEnv, T1GymConfig, _plan_velocity, end_phase
from env.t1_n_task import (
    T1N_ENTRY_SPEED_M_S, T1N_KAPPA, T1N_VERSION, ConditionalGStar, make_conditional_gstar, t1n_env_config,
)

INTERFACE_VERSION = "t1n-review-1"
TAU_LOW_S = 20.0
TAU_HIGH_S = 200.0

T1N_OBSERVATION_NAMES = (
    "pos_h_x_15m", "pos_h_y_15m", "pos_h_z_15m",
    "vel_h_x_0p2", "vel_h_y_0p2", "vel_h_z_0p2",
    "port_axis_h_x", "port_axis_h_y", "port_axis_h_z",
    "body_z_h_x", "body_z_h_y", "body_z_h_z",
    "omega_h_x_2p7", "omega_h_y_2p7", "omega_h_z_2p7",
    "range_15m", "sin_latitude", "cos_latitude", "sin_phase", "cos_phase",
    "fov_margin_50deg", "time_300s", "deadline_slack_200s",
    "in_cone", "mouth_region", "descent", "hold",
    "has_plan", "plan_remaining_200s", "last_request_200s", "last_request_accepted",
    "plan_error_x_1m", "plan_error_y_1m", "plan_error_z_1m",
    "plan_velocity_error_x_0p3", "plan_velocity_error_y_0p3", "plan_velocity_error_z_0p3",
    "earliest_feasible_200s", "earliest_feasible_known",
    "review_due", "review_gap_3",
)
N_DUE_INDEX = T1N_OBSERVATION_NAMES.index("review_due")
N_GAP_INDEX = T1N_OBSERVATION_NAMES.index("review_gap_3")
N_HAS_PLAN_INDEX = T1N_OBSERVATION_NAMES.index("has_plan")
N_PLAN_REMAINING_INDEX = T1N_OBSERVATION_NAMES.index("plan_remaining_200s")
N_DESCENT_INDEX = T1N_OBSERVATION_NAMES.index("descent")
OMEGA_SCALE_RAD_S = float(np.deg2rad(2.7))


def tau_high(time_s: float, max_time_s: float = 300.0, deadline_margin_s: float = 60.0) -> float:
    return float(min(TAU_HIGH_S, max_time_s - deadline_margin_s - time_s))


def neutral_tau(a0: float, time_s: float, max_time_s: float = 300.0, deadline_margin_s: float = 60.0) -> float:
    """Requested arrival (s from now) of a no-plan review; tau_high(t) < tau_low gives tau_high."""

    high = tau_high(time_s, max_time_s, deadline_margin_s)
    if high <= TAU_LOW_S:
        return max(high, 0.0)
    return float(TAU_LOW_S + 0.5 * (float(np.clip(a0, -1.0, 1.0)) + 1.0) * (high - TAU_LOW_S))


def action_for_tau(tau_s: float, time_s: float, max_time_s: float = 300.0, deadline_margin_s: float = 60.0) -> float:
    """Inverse of ``neutral_tau`` (clipped to [-1, 1])."""

    high = tau_high(time_s, max_time_s, deadline_margin_s)
    if high <= TAU_LOW_S:
        return 1.0
    return float(np.clip(2.0 * (tau_s - TAU_LOW_S) / (high - TAU_LOW_S) - 1.0, -1.0, 1.0))


class T1NReviewEnv(T1GStarEnv):
    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, review_period_steps: int = 3,
                 earliest_summary: bool = True) -> None:
        super().__init__(config, env_config or t1n_env_config(), openings)
        self.review_period_steps = int(review_period_steps)
        self.earliest_summary = bool(earliest_summary)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(T1N_OBSERVATION_NAMES),),
                                                dtype=np.float32)
        self._init_review_state()

    # ------------------------------------------------------------------ #
    def _init_review_state(self) -> None:
        self._due = True
        self._since_review = self.review_period_steps
        self._review_index = 0
        self._continuing = False
        self._last_outcome = "none"
        self._earliest_cache: tuple[float, float | None] | None = None
        self._request_record: dict[str, Any] | None = None
        #: the arrival time asked at the last review (s from that review), not overwritten by continues
        self._review_request = 0.0

    def review_state(self) -> dict[str, Any]:
        return {"due": self._due, "since_review": self._since_review, "review_index": self._review_index,
                "last_outcome": self._last_outcome, "last_request": self._last_request,
                "review_request": self._review_request,
                "earliest_cache": self._earliest_cache}

    def set_review_state(self, state: dict[str, Any]) -> None:
        self._due = bool(state["due"])
        self._since_review = int(state["since_review"])
        self._review_index = int(state["review_index"])
        self._last_outcome = str(state["last_outcome"])
        self._last_request = float(state["last_request"])
        self._review_request = float(state["review_request"])
        self._earliest_cache = state["earliest_cache"]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        gym.Env.reset(self, seed=seed)
        self._init_review_state()
        opening = int(options["opening"]) if options and "opening" in options else self.opening_seed()
        self.episode += 1
        self.current_opening = opening
        self.env.reset(opening)
        self.generator = make_conditional_gstar(self.env, replan_tau_tolerance_s=self.config.keep_tolerance_s)
        self._last_request, self._last_kappa = 0.0, T1N_KAPPA
        return self._observation(), {"opening": opening}

    # ------------------------------------------------------------------ #
    def apply_intent(self, action: np.ndarray) -> tuple[float, float, float]:
        """(tau, kappa, v_in) of this step: a review reads ``action``; a continue step keeps."""

        g, e = self.generator, self.env
        assert g is not None
        remaining = g.remaining_s(e)
        if self._continuing:
            tau = remaining
            self._request_record = None
        else:
            a0 = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
            if remaining > 0.0:
                shift = self.shift_of(a0)
                tau = max(remaining + shift, 0.0)
                self._request_record = {"mode": "live_plan", "a": a0, "shift_s": shift, "tau_request_s": tau}
            elif g._descending(e, e.metrics()):
                tau = 0.0
                self._request_record = {"mode": "descent", "a": a0, "tau_request_s": 0.0}
            else:
                tau = neutral_tau(a0, e.time_s, e.config.max_time_s, g.deadline_margin_s)
                self._request_record = {"mode": "no_plan", "a": a0, "tau_request_s": tau,
                                        "tau_high_s": tau_high(e.time_s, e.config.max_time_s, g.deadline_margin_s)}
        g.kappa = T1N_KAPPA
        g.entry_speed_m_s = T1N_ENTRY_SPEED_M_S
        return float(tau), T1N_KAPPA, T1N_ENTRY_SPEED_M_S

    def continue_action(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def _plan_event(self) -> bool:
        g, e = self.generator, self.env
        assert g is not None
        if g.last_status.startswith(("hold", "recovery")):
            return True
        return g.remaining_s(e) <= 0.0 and not g._descending(e, e.metrics())

    def advance_review(self, reviewed: bool) -> None:
        if reviewed:
            self._since_review = 1
            self._review_index += 1
        else:
            self._since_review += 1
        self._due = self._since_review >= self.review_period_steps or self._plan_event()

    def step(self, action: np.ndarray):
        reviewed = self._due
        gap = self._since_review
        review_index = self._review_index
        if reviewed:
            executed = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:1], -1.0, 1.0)
            self._continuing = False
        else:
            executed = self.continue_action()
            self._continuing = True
        _, reward, terminated, truncated, info = super().step(executed)
        self._continuing = False
        g = self.generator
        assert g is not None
        outcome = getattr(g, "last_outcome", "none")
        if reviewed:
            self._last_outcome = outcome
            self._review_request = float(info["execution"]["tau_request_s"])
        elif outcome == "replan_failed":
            self._last_outcome = outcome
        record = self._request_record
        self.advance_review(reviewed)
        info["executed_action"] = executed.astype(np.float32)
        info["review"] = {"reviewed": reviewed, "gap_steps": gap, "review_index": review_index,
                          "next_due": self._due, "plan_event": self._plan_event(), "outcome": outcome,
                          **(record or {})}
        return self._observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def frame(self) -> np.ndarray:
        """Rows: the F_H axes (x = e1, y = H x e1, z = H) in ECI."""

        a = self.env.anchor
        assert a is not None
        return np.vstack([a.e1, a.e2, a.h_axis])

    def earliest_feasible_summary(self) -> float | None:
        """Read-only earliest feasible arrival (s from now) on G*'s grid; cached per state."""

        e, g = self.env, self.generator
        assert g is not None
        key = round(float(e.time_s), 6)
        if self._earliest_cache is not None and self._earliest_cache[0] == key:
            return self._earliest_cache[1]
        tau, _ = g.earliest_feasible(e, T1N_ENTRY_SPEED_M_S)
        self._earliest_cache = (key, tau)
        return tau

    def _observation(self) -> np.ndarray:
        e, g = self.env, self.generator
        assert g is not None
        m = e.metrics()
        f = e.features(m)
        frame = self.frame()
        target, chaser = e.target_state, e.chaser_state
        q = chaser.position - target.position
        v = chaser.rotation @ chaser.velocity - target.rotation @ target.velocity
        rotation = target.rotation
        port_axis = frame @ (rotation @ e.config.task.approach_axis)
        body_z = frame @ rotation[:, 2]
        omega = frame @ (rotation @ target.omega)
        lat, ph = np.deg2rad(f["latitude_deg"]), np.deg2rad(f["phase_deg"])
        horizon = float(e.config.max_time_s) - g.deadline_margin_s - float(e.time_s)
        remaining = g.remaining_s(e)
        status = g.last_status
        dq, dv = np.zeros(3), np.zeros(3)
        if g.plan is not None and remaining > 0.0:
            s = e.time_s - g.plan_start_s
            dq = frame @ (_plan_position(g.plan, s, g.planner.mass) - q)
            dv = frame @ (_plan_velocity(g.plan, s, g.planner.mass) - v)
        earliest, known = 0.0, 0.0
        if (self.earliest_summary and self._due and g.plan is None and status != "descent"
                and not g._descending(e, m)):
            tau = self.earliest_feasible_summary()
            if tau is not None:
                earliest, known = tau / 200.0, 1.0
        values = [
            *(frame @ q / 15.0), *(frame @ v / 0.2), *port_axis, *body_z, *(omega / OMEGA_SCALE_RAD_S),
            f["range_m"] / 15.0, np.sin(lat), np.cos(lat), np.sin(ph), np.cos(ph),
            f["fov_margin_deg"] / 50.0, f["time_s"] / 300.0, (horizon - remaining) / 200.0,
            float(m.in_cone), float(m.mouth_region), float(status == "descent"), float(status.startswith("hold")),
            float(g.plan is not None), remaining / 200.0, self._review_request / 200.0,
            float(self._last_outcome in ("accepted", "descent")),
            *(dq / 1.0), *(dv / 0.3), earliest, known,
            float(self._due), self._since_review / 3.0,
        ]
        obs = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError(f"non-finite observation {obs}")
        return obs


def t1n_review_env(fuel_cost: float | None = None, review_period_steps: int = 3,
                   gym_config: T1GymConfig | None = None, openings: list[int] | None = None) -> T1NReviewEnv:
    from env.t1_n_task import T1N_FUEL_COST

    return T1NReviewEnv(gym_config, t1n_env_config(T1N_FUEL_COST if fuel_cost is None else fuel_cost), openings,
                        review_period_steps=review_period_steps)


__all__ = [
    "INTERFACE_VERSION", "N_DESCENT_INDEX", "N_DUE_INDEX", "N_GAP_INDEX", "N_HAS_PLAN_INDEX",
    "N_PLAN_REMAINING_INDEX", "T1NReviewEnv", "T1N_OBSERVATION_NAMES", "T1N_VERSION", "TAU_HIGH_S", "TAU_LOW_S",
    "action_for_tau", "end_phase", "neutral_tau", "t1n_review_env", "tau_high", "ConditionalGStar",
]
