"""T1-P review interface ``t1p-review-1``: 3-D 12 s stage goals, local G*, entry bridge, PBRS.

One SAC action ``a = (a_r, a_t, a_b)`` in [-1, 1]^3, read only at review states: every 6 real 2 s steps
(one 12 s stage) in the far field, and at the next step after an event (no executable stage = hold,
stage deviation > 1 m, bridge lost, left the descent region). In the bridge and the descent no review
is due (G* bridge / T1 descent fly; the stored action of those continue steps is ``[0, 0, 0]``).
The raw review action is what Replay stores; the request outcome, the stage goal and the planned
terminal errors enter the successor observation.

Reward: ``step`` returns the **training** reward ``r_train = r_raw + gamma Phi(s') - Phi(s)``
(``env.t1_p_stage.potential``; Phi(s') = 0 at success, unsafe and timeout; a numerical-failure
truncation keeps Phi(s')); ``info["reward_raw"]`` is the task reward used by every evaluation.

Observation ``T1P_OBSERVATION_NAMES`` (57 values) in the episode's angular-momentum frame F_H.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from env.t1_env import T1Env, T1EnvConfig
from env.t1_gstar import _plan_position
from env.t1_gym import T1GymConfig, _plan_velocity, end_phase
from env.t1_n_task import t1n_env_config
from env.t1_p_stage import STAGE_S, T1P_VERSION, make_t1p_generator, mouth_point, plan_state, potential, relative_inertial

INTERFACE_VERSION = "t1p-review-1"
STAGE_STEPS = 6
OMEGA_SCALE_RAD_S = float(np.deg2rad(2.7))
EVENTS = ("stage_deviation", "bridge_lost", "left_descent")

T1P_OBSERVATION_NAMES = (
    "pos_h_x_15m", "pos_h_y_15m", "pos_h_z_15m", "vel_h_x_0p2", "vel_h_y_0p2", "vel_h_z_0p2",
    "port_axis_h_x", "port_axis_h_y", "port_axis_h_z", "body_z_h_x", "body_z_h_y", "body_z_h_z",
    "omega_h_x_2p7", "omega_h_y_2p7", "omega_h_z_2p7",
    "range_15m", "sin_latitude", "cos_latitude", "sin_phase", "cos_phase", "fov_margin_50deg", "time_300s",
    "in_cone", "mouth_region", "mode_stage", "mode_hold", "mode_bridge", "mode_descent",
    "goal_dq_h_x_2m", "goal_dq_h_y_2m", "goal_dq_h_z_2m", "goal_dv_h_x_0p2", "goal_dv_h_y_0p2", "goal_dv_h_z_0p2",
    "plan_terminal_eq_h_x_1m", "plan_terminal_eq_h_y_1m", "plan_terminal_eq_h_z_1m",
    "plan_terminal_ev_h_x_0p1", "plan_terminal_ev_h_y_0p1", "plan_terminal_ev_h_z_0p1",
    "stage_remaining_12s", "bridge_remaining_40s",
    "track_dq_h_x_1m", "track_dq_h_y_1m", "track_dq_h_z_1m", "track_dv_h_x_0p3", "track_dv_h_y_0p3", "track_dv_h_z_0p3",
    "last_request_accepted", "last_a_r", "last_a_t", "last_a_b",
    "mouth24_dq_h_x_15m", "mouth24_dq_h_y_15m", "mouth24_dq_h_z_15m",
    "review_due", "review_gap_6",
)
P_DUE_INDEX = T1P_OBSERVATION_NAMES.index("review_due")
P_GAP_INDEX = T1P_OBSERVATION_NAMES.index("review_gap_6")
P_MODE_INDEX = {m: T1P_OBSERVATION_NAMES.index(f"mode_{m}") for m in ("stage", "hold", "bridge", "descent")}


class T1PReviewEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None, stage_steps: int = STAGE_STEPS) -> None:
        self.config = config or T1GymConfig()
        self.env = T1Env(env_config or t1n_env_config())
        self.openings = openings
        self.stage_steps = int(stage_steps)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(T1P_OBSERVATION_NAMES),), dtype=np.float32)
        self.episode = 0
        self.current_opening: int | None = None
        self.generator = None
        self._init_review_state()

    def _init_review_state(self) -> None:
        self._due = True
        self._since_review = self.stage_steps
        self._review_index = 0
        self._last_a = np.zeros(3)
        self._last_outcome = "none"
        self._phi = 0.0

    def opening_seed(self) -> int:
        if self.openings is not None:
            return int(self.openings[self.episode % len(self.openings)])
        return int(self.config.seed_base + self.episode)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._init_review_state()
        opening = int(options["opening"]) if options and "opening" in options else self.opening_seed()
        self.episode += 1
        self.current_opening = opening
        self.env.reset(opening)
        self.generator = make_t1p_generator(self.env)
        self._phi = potential(self.env)
        return self._observation(), {"opening": opening}

    # ---------------------------------------------------------------- #
    def review_state(self) -> dict[str, Any]:
        return {"due": self._due, "since_review": self._since_review, "review_index": self._review_index,
                "last_a": self._last_a.copy(), "last_outcome": self._last_outcome, "phi": self._phi}

    def set_review_state(self, state: dict[str, Any]) -> None:
        self._due, self._since_review = bool(state["due"]), int(state["since_review"])
        self._review_index, self._last_a = int(state["review_index"]), np.array(state["last_a"])
        self._last_outcome, self._phi = str(state["last_outcome"]), float(state["phi"])

    def continue_action(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def _schedule(self, reviewed: bool) -> None:
        g = self.generator
        self._since_review = 1 if reviewed else self._since_review + 1
        if reviewed:
            self._review_index += 1
        far = g.mode in ("stage", "hold")
        self._due = far and (self._since_review >= self.stage_steps or g.mode == "hold" or g.event in EVENTS)

    def decide(self, action: np.ndarray, reviewed: bool) -> np.ndarray:
        """The decision of a step (also used by the shadow): review -> stage, else continue."""

        g = self.generator
        if reviewed:
            executed = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
            g.decide_stage(self.env, executed)
            self._last_a, self._last_outcome = executed.copy(), g.last_outcome
            return executed.astype(np.float32)
        g.continue_step(self.env)
        return self.continue_action()

    def step(self, action: np.ndarray):
        g = self.generator
        reviewed = self._due
        gap, review_index = self._since_review, self._review_index
        walls = len(g.plan_wall_s)
        executed = self.decide(action, reviewed)
        mode_before, status = g.mode, g.last_status
        request = None if not reviewed or g.request is None else {
            "q_goal": g.request["q_goal"].tolist(), "v_goal": g.request["v_goal"].tolist(),
            "status": g.request["result_status"],
            "terminal_position_error_m": g.request.get("terminal_position_error_m"),
            "terminal_velocity_error_m_s": g.request.get("terminal_velocity_error_m_s")}
        reward_raw, terminated, truncated, info = self.env.step(reference=g)
        event = info["event"]
        terminal = event in ("success", "unsafe", "timeout")
        if not (terminated or truncated):
            g.after_step(self.env)
        phi_next = 0.0 if terminal else potential(self.env)
        reward = float(reward_raw) + self.env.config.reward.gamma * phi_next - self._phi
        info = dict(info)
        info.update(reward_raw=float(reward_raw), potential=self._phi, potential_next=phi_next,
                    reward_train=reward, executed_action=executed)
        self._phi = phi_next
        self._schedule(reviewed)
        info["execution"] = {"mode_before": mode_before, "mode": g.mode, "status": status, "plan_status": g.last_status,
                             "generator_event": g.event, "outcome": g.last_outcome if reviewed else None,
                             "request": request, "bridge_tau_s": g.bridge_tau_s if g.mode == "bridge" else None,
                             "plan_wall_s": float(sum(g.plan_wall_s[walls:]))}
        info["review"] = {"reviewed": reviewed, "gap_steps": gap, "review_index": review_index,
                          "next_due": self._due, "plan_event": bool(self._due and self._since_review < self.stage_steps)}
        if terminated or truncated:
            m = self.env.metrics()
            info["episode_summary"] = {"event": event, **self.env.summary(), "opening": self.current_opening,
                                       "violations": dict(info.get("violations", {})), "end_phase": end_phase(m),
                                       "end_range_m": float(m.range_m)}
        return self._observation(), reward, bool(terminal), bool(event == "numerical_failure"), info

    # ---------------------------------------------------------------- #
    def frame(self) -> np.ndarray:
        a = self.env.anchor
        return np.vstack([a.e1, a.e2, a.h_axis])

    def _observation(self) -> np.ndarray:
        e, g = self.env, self.generator
        m = e.metrics()
        f = e.features(m)
        frame = self.frame()
        q, v = relative_inertial(e)
        rot = e.target_state.rotation
        lat, ph = np.deg2rad(f["latitude_deg"]), np.deg2rad(f["phase_deg"])
        goal_dq = goal_dv = term_q = term_v = np.zeros(3)
        if g.mode == "stage" and g.request is not None and g.plan is not None:
            goal_dq = frame @ (g.request["q_goal"] - q) / 2.0
            goal_dv = frame @ (g.request["v_goal"] - v) / 0.2
            term_q = frame @ (g.plan["q"][-1] - g.request["q_goal"]) / 1.0
            term_v = frame @ (g.plan["v"][-1] - g.request["v_goal"]) / 0.1
        track_q = track_v = np.zeros(3)
        if g.mode == "stage" and g.plan is not None:
            p, w = plan_state(g.plan, e.time_s - g.plan_start_s, g.mass)
            track_q, track_v = frame @ (p - q), frame @ (w - v) / 0.3
        elif g.mode == "bridge" and g.bridge.plan is not None and g.bridge.remaining_s(e) > 0.0:
            s = e.time_s - g.bridge.plan_start_s
            track_q = frame @ (_plan_position(g.bridge.plan, s, g.mass) - q)
            track_v = frame @ (_plan_velocity(g.bridge.plan, s, g.mass) - v) / 0.3
        bridge_remaining = g.bridge.remaining_s(e) if g.mode == "bridge" else 0.0
        values = [
            *(frame @ q / 15.0), *(frame @ v / 0.2), *(frame @ (rot @ e.config.task.approach_axis)),
            *(frame @ rot[:, 2]), *(frame @ (rot @ e.target_state.omega) / OMEGA_SCALE_RAD_S),
            f["range_m"] / 15.0, np.sin(lat), np.cos(lat), np.sin(ph), np.cos(ph), f["fov_margin_deg"] / 50.0,
            f["time_s"] / 300.0, float(m.in_cone), float(m.mouth_region),
            *(float(g.mode == k) for k in ("stage", "hold", "bridge", "descent")),
            *goal_dq, *goal_dv, *term_q, *term_v,
            g.stage_remaining_s(e) / STAGE_S, bridge_remaining / 40.0, *track_q, *track_v,
            float(self._last_outcome == "accepted"), *self._last_a,
            *(frame @ (mouth_point(e) - q) / 15.0),
            float(self._due), self._since_review / float(self.stage_steps),
        ]
        obs = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError(f"non-finite observation {obs}")
        return obs


def geometric_rule(env: T1PReviewEnv) -> np.ndarray:
    """The single pre-declared same-interface rule: the stage goal points at the 6.3 m entry point
    predicted 24 s ahead (the potential's anchor), scaled by d_max and clipped to the action box;
    the radial component is halved inside 10 m (conservative braking)."""

    from env.t1_p_stage import stage_goal

    g = stage_goal(env.env, np.zeros(3))
    delta = mouth_point(env.env) - g["q_coast"]
    a = np.array([-(delta @ g["basis"][0]), delta @ g["basis"][1], delta @ g["basis"][2]]) / g["d_max"]
    if env.env.metrics().range_m < 10.0:
        a[0] *= 0.5
    return np.clip(a, -1.0, 1.0)


def t1p_review_env(fuel_cost: float = 20.0, gym_config: T1GymConfig | None = None,
                   openings: list[int] | None = None) -> T1PReviewEnv:
    return T1PReviewEnv(gym_config, t1n_env_config(fuel_cost), openings)


__all__ = ["EVENTS", "INTERFACE_VERSION", "P_DUE_INDEX", "P_GAP_INDEX", "P_MODE_INDEX", "STAGE_STEPS",
           "T1PReviewEnv", "T1P_OBSERVATION_NAMES", "T1P_VERSION", "geometric_rule", "t1p_review_env"]
