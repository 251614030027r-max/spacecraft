"""The staged approach of the ECC method (``docs/paper/METHOD.md``): environment, execution chain and certification.

Decision layer. Every 16 s the policy gives a two-dimensional stage intent a = (a_p, a_z) in [-1, 1]^2, read in the
target's spin frame under the constant-rate model (``spin_frame``): e_r line of sight, e_pol towards the spin axis,
e_az towards the oncoming port:
    closing speed   v_c = 0.05 + 0.15 a_p    in [-0.10, 0.20] m/s,
    along-sweep     v_z = 0.15 a_z           in [-0.15, 0.15] m/s   (meet the port earlier / later),
    latitude        v_l = clip(0.02/s * r * (lam - beta), +-0.12)   (fixed regulation onto the port's sweep cone),
    v_goal = v + rate_limit(v_target - v, 2 d / H),  q_goal = q + H (v + v_goal) / 2,  d = 0.7 (1.6 N / m) H^2 / 2,
projected onto the stage safety set (range floor, radial braking envelope; ``_safe_stage_end``). ``geometry=False``
is the "w/o rotation guidance" ablation: closing along the line of sight only.

Execution chain (rotating-window guidance, RWG). The stage planner (T1-P local SCP) turns the goal into a 16 s plan;
beyond 7.5 m a feed-forward + PD law tracks it (the frozen MPC keeps no far-field translation) with a pointing torque;
inside 7.5 m the frozen MPC flies. At every 2 s the entry rule asks for the entry: range <= 12 m, radial envelope margin
>= 0.02 m/s, the forecast port within 35 deg for one of the arrival times 15-60 s, and a feasible conditional G* plan at
80 % of the force bound (``WindowGStar``: a failed re-plan of a live entry is re-timed 2 / 4 / 8 s later, then with a
95 % reserve, never beyond the deadline less 45 s; ``window=False`` is the rigid commitment). Entry, descent and
capture are flown by the frozen MPC inside one terminal macro-step. A stage that deviates by more than 3 m holds and
the policy is asked again at the next 2 s step.

Information boundary. Every planner, the stage frame, the entry rule and the observation forecast the target attitude
from the current attitude and body rate (an isotropic, i.e. estimated, inertia: ``InfoEnv``); the true inertia is used
by the truth propagation alone (RK45, central gravity and J2).

Certification (ECC). With ``certify`` the executor's opportunity evaluation (``_opportunity_eval``: the mode
prediction of nine probe intents, ``env/t2_probe.py``) certifies the intent before it is flown (``_certify``).

Mission: urgent (deadline U(140, 200) s, -0.10 per s, -10 per m/s) or economy (U(280, 340) s, -0.02 per s, -40 per m/s);
the episode times out at the deadline (-100); delta-v budget 3.0 m/s urgent, 2.0 m/s economy (resource exhausted -100);
capture +100, unsafe -150.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np

from dynamics.lie import so3_exp
from env.t1_env import T1Env, T1EnvConfig
from env.t1_n_task import ConditionalGStar, make_conditional_gstar
from env.t1_p_stage import LocalStagePlanner, T1PGenerator, plan_state, relative_inertial
from env.t1_task import T1InitialDistribution

T2_VERSION = "t2-stage-1.0.0"
STAGE_S = 16.0
STAGE_NODES = 16
PLAN_FORCE_N = 1.6
D_FRACTION = 0.70
ENTRY_RADIUS_M = 6.3
BRIDGE_RANGE_M = 12.0
BRIDGE_ANGLE_DEG = 35.0
BRIDGE_TAUS_S = (15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
BRIDGE_ENVELOPE_MARGIN_M_S = 0.02
BRIDGE_DEADLINE_MARGIN_S = 45.0
OBS_DIM = 24
#: the decision-layer interface: the policy's view drops the planner's answers (time to the port's pass, entry-window
#: reachability, deadline slack after the pass) and the executor's internal execution state (last stage's position /
#: velocity residual, hold flag); the executor feedback view z_E is the full ``observe`` vector (critic only)
OBS_PROFILE = "cut_priv"
PLANNER_FEATURES = (8, 17, 19)
EXEC_STATE_FEATURES = (14, 15, 16)
OBS_DIM_PRIV = 2 * OBS_DIM
#: arrival grid of the direct G* reference row (``start_direct_gstar``)
DIRECT_TAUS_S = (20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 140., 160., 180., 200., 220.)
#: the executor's entry-rule cadence (also the node grid of the mode prediction, ``env/t2_probe.py``)
ENTRY_NODE_S = 2.0
ENTRY_NODES_S = tuple(ENTRY_NODE_S * k for k in range(1, 9))
TERMINAL_EVENTS = ("success", "unsafe", "timeout", "resource_exhausted")
#: SMDP discount of one stage (0.999 per 2 s); the same constant is the learner's gamma and the
#: potential-shaping discount of every non-terminal real and model transition
GAMMA_STAGE = 0.999 ** (16.0 / 2.0)
#: a stage is re-planned from the actual state at the next decision (<= 16 s); only a gross deviation
#: holds (T1-P held at 1 m, and the MPC's 0.4-0.6 m tracking lag on a full-authority stage turned
#: most stages into holds that did not brake an outward drift)
STAGE_DEVIATION_HOLD_M = 3.0
#: far-field translational execution (outside this range, stage / hold modes): the frozen MPC's
#: rotating-frame formulation pushes outward when asked to hold an inertially fixed point from rest
#: (evidence/T2_PRECHECK hold test: +1.84 m, 0.19 m/s of delta-v in 20 s at 16 m; half at 8 m), the
#: root of T1-P's and D28's far-field drift. There the force is an inertial plan-tracking law
#: (feed-forward of the plan force + PD on the plan state); the MPC keeps the attitude (its torque)
#: and the whole bridge / descent.
FAR_FIELD_M = 7.5
TRACK_KP = 0.04      # 1/s^2  (omega_n 0.2 rad/s, critically damped)
TRACK_KD = 0.40      # 1/s
#: far-field attitude: the camera boresight is pointed at the port by a PD law (omega_n 0.6 rad/s)
#: and the MPC is not solved there at all (its translational output was discarded anyway; it was
#: 85 % of the wall time of a stage). The MPC flies the bridge and the descent (the near field).
POINT_KP = 0.36      # 1/s^2
POINT_KD = 1.2       # 1/s
FAR_FIELD_DIAGNOSTICS = SimpleNamespace(used_zero_fallback=False, maximum_slack=0.0, status="far_field_tracking")


@dataclass(frozen=True)
class T2Mission:
    urgent_deadline_s: tuple[float, float] = (170.0, 210.0)
    economy_deadline_s: tuple[float, float] = (280.0, 340.0)
    budget_m_s: float = 3.0
    #: economy missions' own budget (None: ``budget_m_s`` for both types)
    economy_budget_m_s: float | None = None
    #: mission preference (observed through the mission class): urgent values time, economy fuel
    urgent_time_cost_per_s: float = 0.10
    urgent_fuel_cost_per_m_s: float = 10.0
    economy_time_cost_per_s: float = 0.02
    economy_fuel_cost_per_m_s: float = 40.0

    def budget(self, urgent: bool) -> float:
        return self.budget_m_s if urgent or self.economy_budget_m_s is None else self.economy_budget_m_s

    def sample(self, opening: int) -> dict[str, Any]:
        rng = np.random.default_rng(np.random.SeedSequence([int(opening), 290021]))
        urgent = bool(rng.integers(0, 2))
        d = rng.uniform(*(self.urgent_deadline_s if urgent else self.economy_deadline_s))
        return {"deadline_s": float(round(d, 1)), "urgent": urgent, "budget_m_s": self.budget(urgent),
                "time_cost_per_s": self.urgent_time_cost_per_s if urgent else self.economy_time_cost_per_s,
                "fuel_cost_per_m_s": self.urgent_fuel_cost_per_m_s if urgent else self.economy_fuel_cost_per_m_s}



def window_half_width(lam: float, beta: float, theta_g: float) -> float | None:
    """Return the exact feasible phase half-width in radians, or None for no window."""
    offset = np.cos(lam) * np.cos(beta)
    amplitude = np.sin(lam) * np.sin(beta)
    target = np.cos(theta_g)
    if abs(amplitude) <= 1e-15:
        return float(np.pi) if offset >= target - 1e-15 else None
    c_star = (target - offset) / amplitude
    if c_star > 1.0 + 1e-15:
        return None
    if c_star <= -1.0:
        return float(np.pi)
    return float(np.arccos(np.clip(c_star, -1.0, 1.0)))


def opportunity_windows(q: np.ndarray, v: np.ndarray, rotation: np.ndarray, omega_body: np.ndarray, approach_axis: np.ndarray,
                        horizon_s: float = 900.0) -> dict[str, Any]:
    """Geometric window timetable of an initial state under the executor's constant-rate model (no policy, no learning):
    pass times t_k = t_pass0 + k P of the port through the line-of-sight meridian, window half-width h = 35 deg / |omega|,
    nominal approach time t_app (closing at the action box's 0.20 m/s to the 12 m entry range while the fixed latitude law
    brings the latitude error inside 35 deg) and the earliest arrival t_app + 15 s."""
    f = spin_frame(rotation, omega_body, q, approach_axis)
    w = float(f["rate_rad_s"]); period = 2.0 * np.pi / max(w, 1e-9)
    h = float(np.radians(BRIDGE_ANGLE_DEG) / max(w, 1e-9))
    v_max = CLOSING_M_S[0] + CLOSING_M_S[1]
    r, lat, t = float(f["r"]), float(f["lat_err_deg"]), 0.0
    while t < horizon_s and not (r <= BRIDGE_RANGE_M + 1e-9 and abs(lat) <= BRIDGE_ANGLE_DEG):
        dt = 0.5
        v_l = float(np.clip(LAT_GAIN_PER_S * r * np.radians(lat), -LAT_SPEED_M_S, LAT_SPEED_M_S))
        lat -= float(np.degrees(v_l * dt / max(r, 1e-6)))
        r = max(BRIDGE_RANGE_M, r - v_max * dt)
        t += dt
    t_arr = t + min(BRIDGE_TAUS_S)
    passes = [float(f["time_to_pass_s"]) + k * period for k in range(int(horizon_s / period) + 3)]
    return {"passes": passes, "period": period, "half_width": h, "t_app": t, "t_arr": t_arr}


@dataclass(frozen=True)
class MixMission(T2Mission):
    """The mission distribution with the urgent share ``urgent_share`` (training: 3/4), or a forced type ``force`` =
    'urgent' / 'economy' (evaluation conditions). Deadlines are drawn exactly as ``T2Mission.sample`` draws them for the
    same opening (same seed, the type draw consumed first), so a forced urgent opening has the deadline the natural
    mission gives it when urgent."""

    urgent_share: float = 0.5
    force: str | None = None

    def sample(self, opening: int) -> dict[str, Any]:
        rng = np.random.default_rng(np.random.SeedSequence([int(opening), 290021]))
        natural_urgent = bool(rng.integers(0, 2))
        if self.force is not None:
            urgent = self.force == "urgent"
        elif self.urgent_share == 0.5:
            urgent = natural_urgent
        else:
            urgent = bool(np.random.default_rng(np.random.SeedSequence([int(opening), 291109])).uniform() < self.urgent_share)
        d = rng.uniform(*(self.urgent_deadline_s if urgent else self.economy_deadline_s))
        return {"deadline_s": float(round(d, 1)), "urgent": urgent, "budget_m_s": self.budget(urgent),
                "time_cost_per_s": self.urgent_time_cost_per_s if urgent else self.economy_time_cost_per_s,
                "fuel_cost_per_m_s": self.urgent_fuel_cost_per_m_s if urgent else self.economy_fuel_cost_per_m_s}


def t2_env_config(mission: T2Mission = T2Mission()) -> T1EnvConfig:
    base = T1EnvConfig()
    initial = replace(T1InitialDistribution(), range_m=(12.0, 16.0), inertial_speed_max_m_s=0.05)
    return replace(base, initial=initial, max_time_s=max(mission.economy_deadline_s))



#: the paper's task distribution (``docs/paper/EXPERIMENTS.md`` section 2): urgent deadlines U(140, 200) s with the
#: 3.0 m/s budget; economy deadlines U(280, 340) s with the 2.0 m/s budget (user 2026-11-13, from the completed prescreen)
URGENT_DEADLINE_S = (140.0, 200.0)
ECONOMY_BUDGET_M_S = 2.0
_NAMED = dict(urgent_deadline_s=URGENT_DEADLINE_S, economy_budget_m_s=ECONOMY_BUDGET_M_S)


def mission_from_name(name: str) -> T2Mission:
    """'urgent_mix:<share>' (training: urgent with probability <share>), 'urgent' / 'economy' (forced type; evaluation
    conditions U and E), 'natural' (the natural 1:1 draw; condition N). Urgent missions carry 3.0 m/s, economy 2.0 m/s."""
    if isinstance(name, T2Mission):
        return name
    kind, _, arg = str(name).partition(':')
    if kind == 'urgent_mix':
        return MixMission(urgent_share=float(arg), **_NAMED)
    if kind in ('urgent', 'economy'):
        return MixMission(force=kind, **_NAMED)
    if kind == 'natural':
        return T2Mission(**_NAMED)
    raise ValueError(f'unknown mission {name!r}')


#: pass-window commitment (RWG): the entry is a commitment to the port's pass, not to an instant
WINDOW_SLIPS_S = (2.0, 4.0, 8.0)     # arrival re-timing inside the same pass (<= ~20 deg of port sweep)
WINDOW_RESERVE_FRACTION = 0.95       # authority reserve: the last re-plan may use 95 % of the force bound


@dataclass
class WindowGStar(ConditionalGStar):
    """Conditional G* with a pass-window commitment (RWG re-timing).

    The frozen MPC lags a bang-coast-bang entry plan systematically (T2.3 loss probe: the deviation grows
    ~0.05 m per 2 s until the 1 m re-plan threshold); the conditional G* then re-plans *exactly* the
    remaining arrival time at 80 % of the force bound, which is infeasible in most cases, and the entry is
    lost although the same pass is still reachable (2 s later arrival, or the 20 % authority reserve).
    Here a failed re-plan of a live entry is retried inside the same pass: later arrivals
    (``WINDOW_SLIPS_S``) at the plan bound, then the same arrival and slips with the authority reserve,
    never beyond the deadline. A new entry request (no live plan) is unchanged."""

    reserve_planner: Any = None
    window_slips_s: tuple = WINDOW_SLIPS_S
    elastic: bool = True
    recoveries: int = 0
    recovery_log: list = field(default_factory=list)   # (time, slip, reserve used, port sweep of the slip): same-pass evidence

    def decide(self, env, tau_s: float) -> None:
        live = self.plan is not None
        super().decide(env, tau_s)
        if not (self.elastic and live and self.plan is None and self.last_outcome == "replan_failed"):
            return
        from time import perf_counter
        from env.t1_gstar import free_body_rotations
        started = perf_counter()
        target, chaser = env.target_state, env.chaser_state
        q0 = chaser.position - target.position
        v0 = chaser.rotation @ chaser.velocity - target.rotation @ target.velocity
        horizon = float(env.config.max_time_s) - self.deadline_margin_s - float(env.time_s)
        trials = [(self.planner, s) for s in self.window_slips_s]
        if self.reserve_planner is not None:
            trials += [(self.reserve_planner, s) for s in (0.0, *self.window_slips_s)]
        for planner, slip in trials:
            tau = float(tau_s) + slip
            if tau <= 0.0 or tau > horizon + self.deadline_tolerance_s:
                continue
            history = free_body_rotations(target.rotation, target.omega, env.target_parameters.inertia, tau + 1.0)
            result = planner.solve(q0, v0, tau, history, None, entry_speed_m_s=self.entry_speed_m_s)
            if result["feasible"]:
                self.plan_wall_s.append(perf_counter() - started)
                self.plan_start_s = float(env.time_s)
                self.plan, self.last_status = result, f"window:{slip:g}:{'reserve' if planner is self.reserve_planner else 'plan'}"
                self.tau_s = self.tau_effective_s = float(result["tau_s"])
                self.hold_position, self.last_outcome = None, "accepted"
                self.recoveries += 1
                self.recovery_log.append({"t": float(env.time_s), "slip_s": float(slip), "reserve": planner is self.reserve_planner,
                                          "phase_deg": float(np.degrees(slip * np.linalg.norm(target.omega)))})
                return
        self.plan_wall_s.append(perf_counter() - started)


def make_window_gstar(env, elastic: bool = True, **kwargs: Any) -> WindowGStar:
    from env.t1_gstar import RendezvousPlanner
    base = make_conditional_gstar(env, **kwargs)
    reserve = RendezvousPlanner(env.config.task, env.chaser_parameters.mass, WINDOW_RESERVE_FRACTION * env.config.max_force_per_axis_n)
    return WindowGStar(kappa=base.kappa, planner=base.planner, recovery_planner=None, entry_speed_m_s=base.entry_speed_m_s,
                       deadline_margin_s=base.deadline_margin_s, reserve_planner=reserve, elastic=bool(elastic))


# ---------------------------------------------------------------------- information boundary
class InfoEnv:
    """What the onboard planners may read: the real environment, except that the target inertia is
    replaced by an isotropic one (constant body rate forecast). Truth never sees this object."""

    def __init__(self, env: T1Env) -> None:
        self._env = env
        inertia = np.asarray(env.target_parameters.inertia, dtype=np.float64)
        iso = np.eye(3) * float(np.trace(inertia)) / 3.0
        self.target_parameters = SimpleNamespace(**{**vars(env.target_parameters), "inertia": iso}) \
            if hasattr(env.target_parameters, "__dict__") else SimpleNamespace(inertia=iso)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def forecast_rotation(env, s: float) -> np.ndarray:
    t = env.target_state
    return t.rotation @ so3_exp(s * t.omega)


def port_forecast(env, s: float) -> np.ndarray:
    return forecast_rotation(env, s) @ env.config.task.approach_axis


LAT_GAIN_PER_S = 0.02          # latitude regulation: 1 / 50 s
LAT_SPEED_M_S = 0.12
CLOSING_M_S = (0.05, 0.15)     # v_c = 0.05 + 0.15 a_p in [-0.10, 0.20]
AZIMUTH_M_S = 0.15             # v_z = 0.15 a_z


def spin_frame(rotation: np.ndarray, omega_body: np.ndarray, q: np.ndarray, approach_axis: np.ndarray) -> dict[str, Any]:
    """Stage frame from a (possibly forecast) target attitude and body rate (constant-rate model):
    e_r (line of sight), e_pol (towards the spin axis), e_az (towards the oncoming port, against the
    sense in which the port sweeps past the line of sight); the port's cone half-angle beta about the
    spin axis, the line-of-sight polar angle lam, the phase the port still has to sweep to reach the
    line-of-sight meridian and the time to that pass."""

    s = rotation @ omega_body
    w = float(np.linalg.norm(s)); s_hat = s / max(w, 1e-12)
    r = float(np.linalg.norm(q)); e_r = q / r
    pol = s_hat - float(s_hat @ e_r) * e_r
    if np.linalg.norm(pol) < 1e-3:
        pol = np.cross(e_r, [0.0, 0.0, 1.0]) if abs(e_r[2]) < 0.9 else np.cross(e_r, [0.0, 1.0, 0.0])
    e_pol = pol / np.linalg.norm(pol)
    e_az = -np.cross(s_hat, e_r)
    e_az = e_az / np.linalg.norm(e_az) if np.linalg.norm(e_az) > 1e-3 else np.cross(e_r, e_pol)
    e_az = e_az - float(e_az @ e_pol) * e_pol; e_az /= np.linalg.norm(e_az)
    port = rotation @ approach_axis
    lam = angle_deg(e_r, s_hat); beta = angle_deg(port, s_hat)
    p_perp, r_perp = port - (port @ s_hat) * s_hat, e_r - (e_r @ s_hat) * s_hat
    phase = float(np.mod(np.arctan2(s_hat @ np.cross(p_perp, r_perp), p_perp @ r_perp), 2 * np.pi)) \
        if np.linalg.norm(p_perp) > 1e-6 and np.linalg.norm(r_perp) > 1e-6 else 0.0
    return {"basis": np.vstack([e_r, e_pol, e_az]), "lat_err_deg": lam - beta, "beta_deg": beta, "lam_deg": lam,
            "phase_rad": phase, "time_to_pass_s": phase / max(w, 1e-9), "rate_rad_s": w, "port": port, "r": r}


def frame_now(env, q: np.ndarray) -> dict[str, Any]:
    t = env.target_state
    return spin_frame(t.rotation, t.omega, q, env.config.task.approach_axis)


def latitude_velocity(frame: dict[str, Any]) -> float:
    """Fixed (non-decision) regulation of the line of sight onto the port's sweep cone."""
    return float(np.clip(LAT_GAIN_PER_S * frame["r"] * np.radians(frame["lat_err_deg"]), -LAT_SPEED_M_S, LAT_SPEED_M_S))


def d_authority(mass: float, h: float = STAGE_S) -> float:
    return float(D_FRACTION * (PLAN_FORCE_N / mass) * h * h / 2.0)


def target_velocity(frame: dict[str, Any], action: np.ndarray, geometry: bool = True) -> np.ndarray:
    """Stage-end velocity target. ``geometry=False`` is the rotation-unaware standard hierarchy (basic
    layering): closing along the line of sight only -- no sweep-cone regulation, no along-sweep intent."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
    b = frame["basis"]
    v_c = CLOSING_M_S[0] + CLOSING_M_S[1] * a[0]
    if not geometry:
        return -v_c * b[0]
    return -v_c * b[0] + latitude_velocity(frame) * b[1] + AZIMUTH_M_S * a[1] * b[2]


def action_for_velocity(frame: dict[str, Any], v_c: float, v_z: float) -> np.ndarray:
    """Action for a closing speed v_c and an azimuthal speed v_z (towards the oncoming port)."""
    return np.clip([(v_c - CLOSING_M_S[0]) / CLOSING_M_S[1], v_z / AZIMUTH_M_S], -1.0, 1.0)


#: executor safety projection of a stage goal: without it the stage layer flew intents that
#: closed faster than the truth's radial braking envelope near the far / near boundary (7.5-7.7 m, outside the cone)
#: and stages that ended inside the mouth band without the entry (6.3-6.5 m, in the cone, unsynchronized with the
#: rotating port). A stage now ends no closer than STAGE_RANGE_FLOOR_M (the operation sphere is entered by the entry
#: and descent, i.e. the MPC) and no faster inward than the envelope at its end range less the bridge's own margin.
STAGE_RANGE_FLOOR_M = 6.0 + 0.5 + 0.3          # operation sphere + mouth band + margin
STAGE_ENVELOPE_MARGIN_M_S = BRIDGE_ENVELOPE_MARGIN_M_S


def _safe_stage_end(q, v, v_goal, h, task):
    """Radial projection of the stage-end velocity: range floor first, then the braking envelope at the end range
    (two passes: the end range depends on the projected velocity). Only the inward radial component is reduced."""
    projected = False
    for _ in range(2):
        q_end = q + h * (v + v_goal) / 2.0
        r_end = float(np.linalg.norm(q_end))
        u = q_end / max(r_end, 1e-9)
        closing = -float(v_goal @ u)
        if r_end < STAGE_RANGE_FLOOR_M:            # keep the stage end outside the mouth band: shorten the inward travel
            need = (STAGE_RANGE_FLOOR_M - r_end) * 2.0 / h
            v_goal = v_goal + need * u; projected = True; continue
        cap = float(task.outer_radial_closing_speed_limit(r_end)) - STAGE_ENVELOPE_MARGIN_M_S
        if closing > max(cap, 0.0):
            v_goal = v_goal + (closing - max(cap, 0.0)) * u; projected = True
    return v_goal, projected


def stage_goal(q: np.ndarray, v: np.ndarray, frame: dict[str, Any], action: np.ndarray, mass: float,
               h: float = STAGE_S, geometry: bool = True, task=None) -> dict[str, Any]:
    """The stage layer's goal for an intent: the target velocity, rate-limited to the stage authority; with ``task``
    (the executor's own task, T2.8) projected onto the stage safety set (``_safe_stage_end``)."""
    d = d_authority(mass, h)
    v_target = target_velocity(frame, action, geometry)
    dv = v_target - v
    limit = 2.0 * d / h
    n = float(np.linalg.norm(dv))
    limited = n > limit
    if limited:
        dv = dv * (limit / n)
    v_goal = v + dv
    safe = False
    if task is not None:
        v_goal, safe = _safe_stage_end(q, v, v_goal, h, task)
    return {"a": np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1, 1), "v_target": v_target, "d": d,
            "rate_limited": bool(limited), "safety_projected": bool(safe), "q_goal": q + h * (v + v_goal) / 2.0, "v_goal": v_goal}


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def observe(env, q: np.ndarray, v: np.ndarray, t_s: float, dv_used: float, mission: dict, rotation: np.ndarray,
            omega_body: np.ndarray, last: dict | None) -> np.ndarray:
    """24 features from a (possibly forecast) state; the target attitude enters only through
    ``rotation`` and ``omega_body`` (constant-rate forecasts use the same function)."""

    task = env.config.task
    f = spin_frame(rotation, omega_body, q, task.approach_axis)
    r = f["r"]; b = f["basis"]; vb = b @ v
    def port_at(s):
        return rotation @ so3_exp(s * omega_body) @ task.approach_axis
    bridge = min(angle_deg(port_at(s), q) for s in BRIDGE_TAUS_S)
    last = last or {}
    closing = -float(q @ v) / max(r, 1e-9)
    margin = float(task.braking_speed_limit(max(r - float(task.operation_sphere_radius_m), 0.0))) - closing
    return np.array([
        r / 15.0, vb[0] / 0.2, vb[1] / 0.2, vb[2] / 0.2,
        angle_deg(f["port"], q) / 180.0, f["lat_err_deg"] / 90.0, np.sin(f["phase_rad"]), np.cos(f["phase_rad"]),
        f["time_to_pass_s"] / 150.0, np.degrees(f["rate_rad_s"]) / 2.7,
        (mission["deadline_s"] - t_s) / 300.0, float(mission["urgent"]), dv_used / 2.5, (mission["budget_m_s"] - dv_used) / 3.0,
        last.get("position_residual_m", 0.0), last.get("velocity_residual_m_s", 0.0) / 0.1, float(last.get("held", False)),
        bridge / 180.0, float(np.clip(margin / 0.1, -3.0, 3.0)),
        float(np.clip((mission["deadline_s"] - t_s - f["time_to_pass_s"] - 60.0) / 100.0, -3.0, 3.0)),   # deadline slack after the pass
        f["beta_deg"] / 90.0,
        *(b @ f["port"]),
    ], dtype=np.float32)


class T2StageEnv(gym.Env):
    """The staged approach as a decision process: one ``step`` is one stage decision (or a terminal macro-step).

    The observation is the ``cut_priv`` vector: the policy's view (``observe`` with the planner answers
    PLANNER_FEATURES and the executor's internal state EXEC_STATE_FEATURES zeroed, OBS_DIM values) followed by the
    executor feedback view z_E (the full ``observe`` vector, OBS_DIM values). The actor reads the first OBS_DIM values;
    a critic with ``critic_dim`` = OBS_DIM_PRIV reads both (``train/t2_mve_sac.py``)."""

    metadata: dict[str, Any] = {"render_modes": []}
    obs_profile = OBS_PROFILE

    def __init__(self, openings: list[int] | None = None, seed_base: int = 1_000_000, mission: T2Mission = T2Mission(),
                 shaping: bool = True, geometry: bool = True, window: bool = True, certify: bool = False) -> None:
        self.geometry = bool(geometry)
        self.window = bool(window)
        #: the certification map C(x, a) (``_certify``): the same map in training and execution
        self.certify = bool(certify)
        self.cert_prober = None
        self._opp = None                       # the decision state's opportunity evaluation (cached per decision)
        #: read-only decision / episode diagnostics in ``info`` (formal runs); never changes the trajectory
        self.audit_logging = False
        self.mission_distribution = mission
        self.base_config = t2_env_config(mission)
        self.env = T1Env(self.base_config)
        self.info_env = InfoEnv(self.env)
        self.openings, self.seed_base, self.shaping = openings, int(seed_base), bool(shaping)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.obs_dim = OBS_DIM_PRIV
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.episode = 0
        self.generator = None
        self._bridge = None
        self.gamma_stage = GAMMA_STAGE
        self.far_field_controls = 0
        original = self.env.controller.command

        def command(*args, **kwargs):
            force = self._far_field_force()
            if force is not None:
                self.far_field_controls += 1
                return np.r_[self._pointing_torque(), force], FAR_FIELD_DIAGNOSTICS
            return original(*args, **kwargs)
        self.env.controller.command = command

    # ---------------------------------------------------------------- helpers
    def opening(self) -> int:
        if self.openings is not None:
            return int(self.openings[self.episode % len(self.openings)])
        return self.seed_base + self.episode

    def observation(self) -> np.ndarray:
        q, v = relative_inertial(self.env)
        t = self.env.target_state
        o = observe(self.env, q, v, self.env.time_s, self.env._delta_v, self.mission, t.rotation, t.omega, self.last)
        r = o.copy(); r[list(PLANNER_FEATURES) + list(EXEC_STATE_FEATURES)] = 0.0
        return np.concatenate([r, o]).astype(np.float32)

    def potential(self, q: np.ndarray, t_rotation: np.ndarray, omega: np.ndarray) -> float:
        mouth = t_rotation @ so3_exp(24.0 * omega) @ (ENTRY_RADIUS_M * self.env.config.task.approach_axis)
        return -min(25.0, float(np.linalg.norm(q - mouth)))

    def _phi(self) -> float:
        q, _ = relative_inertial(self.env); t = self.env.target_state
        return self.potential(q, t.rotation, t.omega)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        opening = int(options["opening"]) if options and "opening" in options else self.opening()
        self.episode += 1
        self.current_opening = opening
        self.mission = self.mission_distribution.sample(opening)
        self.env.config = replace(self.base_config, max_time_s=self.mission["deadline_s"],
                                  reward=replace(self.base_config.reward, time_cost_per_s=self.mission["time_cost_per_s"],
                                                 delta_v_cost_per_m_s=self.mission["fuel_cost_per_m_s"]))
        self.env.reset(opening)
        self.info_env = InfoEnv(self.env)
        self.generator = T1PGenerator(planner=LocalStagePlanner(self.env.config.task, self.env.chaser_parameters.mass,
                                                                PLAN_FORCE_N, horizon_s=STAGE_S, nodes=STAGE_NODES),
                                      bridge=make_window_gstar(self.info_env, elastic=self.window, deadline_margin_s=BRIDGE_DEADLINE_MARGIN_S),
                                      mass=self.env.chaser_parameters.mass, replan_deviation_m=STAGE_DEVIATION_HOLD_M)
        self.last = {}
        self._bridge = None
        self.cert_prober = None                # the certification's planner instances are built per episode
        self._opp = None
        self.return_true = 0.0
        self.stats = {"stages": 0, "holds": 0, "rejected": 0, "bridge_attempts": 0, "macro": None}
        return self.observation(), {"opening": opening, "mission": dict(self.mission)}

    # ---------------------------------------------------------------- execution modes
    def start_direct_gstar(self) -> bool:
        """Reference row outside the stage action space: the conditional G* from the far field to the
        entry at the earliest feasible time of its grid (constant-rate model), re-planned on deviation."""

        g = self.generator
        horizon = self.env.config.max_time_s - BRIDGE_DEADLINE_MARGIN_S - self.env.time_s
        for tau in [t for t in DIRECT_TAUS_S if t <= horizon]:
            g.bridge.plan, g.bridge.hold_position = None, None
            g.bridge.decide(self.info_env, tau)
            if g.bridge.plan is not None:
                g.mode, g.plan, g.bridge_tau_s = "bridge", None, tau
                return True
        return False

    def _try_bridge(self, m) -> bool:
        """The entry rule at the 2 s cadence: range <= 12 m, radial envelope margin, a forecast port angle within
        35 deg for one of the arrival times, and a feasible window-committed conditional G* plan."""
        if m.range_m > BRIDGE_RANGE_M or m.radial_envelope_margin_m_s < BRIDGE_ENVELOPE_MARGIN_M_S:
            return False
        q, _ = relative_inertial(self.env)
        g = self.generator
        for tau in BRIDGE_TAUS_S:
            if angle_deg(port_forecast(self.env, tau), q) > BRIDGE_ANGLE_DEG:
                continue
            self.stats["bridge_attempts"] += 1
            g.bridge.plan, g.bridge.hold_position = None, None
            g.bridge.decide(self.info_env, tau)
            if g.bridge.plan is not None:
                g.mode, g.plan, g.bridge_tau_s = "bridge", None, tau
                self._bridge = {"t": self.env.time_s, "tau": float(g.bridge.plan["tau_s"])}
                return True
        return False

    def _close_bridge(self, event) -> None:
        """A lost entry (the plan dropped back to a hold) is counted; the episode goes on."""
        if self._bridge is not None:
            self.stats["entries_lost"] = self.stats.get("entries_lost", 0) + int(event == "lost")
            self._bridge = None

    def step_pure(self):
        """Paper baseline: one T1Env step of the shared MPC alone (fixed terminal set-point); the stage
        layer and its far-field execution law are switched off for the whole episode."""
        self.generator = None
        return self.env.step(pure=True)

    def _far_field_force(self):
        g = getattr(self, "generator", None)
        if g is None or g.mode not in ("stage", "hold"):
            return None
        e = self.env
        q, v = relative_inertial(e)
        if float(np.linalg.norm(q)) <= FAR_FIELD_M:
            return None
        mass = e.chaser_parameters.mass
        if g.mode == "stage" and g.plan is not None:
            s = e.time_s - g.plan_start_s
            q_ref, v_ref = plan_state(g.plan, s, mass)
            f = g.plan["f"]; i = int(np.clip(np.floor(max(s, 0.0) / g.plan["dt"]), 0, len(f) - 1))
            a_ff = f[i] / mass if s < g.plan["tau_s"] else np.zeros(3)
        else:
            if g.hold_position is None:
                g.hold_position = q.copy()
            q_ref, v_ref, a_ff = g.hold_position, np.zeros(3), np.zeros(3)
        a = a_ff + TRACK_KP * (q_ref - q) + TRACK_KD * (v_ref - v)
        limit = e.config.max_force_per_axis_n
        return np.clip(e.chaser_state.rotation.T @ (mass * a), -limit, limit)

    def _pointing_torque(self) -> np.ndarray:
        e = self.env; c, t, task = e.chaser_state, e.target_state, e.config.task
        d = t.position + t.rotation @ np.asarray(task.port_position) - c.position
        d = d / np.linalg.norm(d)
        b = c.rotation @ np.asarray(task.camera_boresight, dtype=np.float64)
        axis = np.cross(b, d); s = float(np.linalg.norm(axis))
        theta = axis / s * np.arctan2(s, float(b @ d)) if s > 1e-12 else np.zeros(3)
        J = np.asarray(e.chaser_parameters.inertia); w = np.asarray(c.omega)
        torque = J @ (POINT_KP * (c.rotation.T @ theta) - POINT_KD * w) + np.cross(w, J @ w)
        limit = e.config.max_torque_per_axis_nm
        return np.clip(torque, -limit, limit)

    def _reference(self, env, metrics):
        return self.generator(self.info_env, metrics)

    def _advance(self) -> tuple[float, str | None, dict]:
        """One 2 s truth step with mode bookkeeping; returns (true reward, event, info)."""

        g = self.generator
        if g.mode == "bridge":
            g.continue_step(self.info_env)
            if g.mode == "hold":
                self._close_bridge("lost")
        r, term, trunc, info = self.env.step(reference=self._reference)
        event = info.get("event")
        if event is None and self.env._delta_v >= self.mission["budget_m_s"]:
            event = "resource_exhausted"; r -= self.env.config.reward.timeout_penalty
        if event is None:
            m = self.env.metrics()
            # mode bookkeeping (T1-P after_step with this module's entry rule)
            g.event = None
            if g.descending(self.info_env, m):
                g.mode = "descent"; g.bridge.plan = None
            elif g.mode == "descent":
                g.mode, g.hold_position = "hold", None; g._enter_hold(self.info_env, "hold:left_descent")
                self._close_bridge("lost")
            else:
                if g.mode == "stage" and g.stage_deviation(self.info_env) > g.replan_deviation_m:
                    g.hold_position = None; g._enter_hold(self.info_env, "hold:stage_deviation")
                if g.mode in ("stage", "hold"):
                    self._try_bridge(m)
        return float(r), event, info

    def _macro_to_end(self):
        """A step that starts in the entry or descent (or the direct G* reference row): fly until the end."""
        e, g = self.env, self.generator
        t0, phi0, reward, event = e.time_s, self._phi(), 0.0, None
        while event is None:
            r, event, _ = self._advance(); reward += r
            if g.mode in ("stage", "hold"):
                break
        terminated = event in TERMINAL_EVENTS
        if terminated:
            self._close_bridge(event)
        self.return_true += reward
        shaped = reward + ((GAMMA_STAGE * (0.0 if terminated else self._phi()) - phi0) if self.shaping else 0.0)
        q1, _ = relative_inertial(e)
        info = {"event": event, "true_reward": reward, "stage_s": e.time_s - t0, "held": g.mode == "hold", "mode": g.mode,
                "plan_status": "macro", "rate_limited": False, "planned_dv": None, "executed_dv": None, "cert": None,
                "time_s": e.time_s, "delta_v": e._delta_v, "range_m": float(np.linalg.norm(q1)), "mission": dict(self.mission),
                "far_field_controls": self.far_field_controls}
        return self.observation(), float(shaped), terminated, event == "numerical_failure", info

    # ---------------------------------------------------------------- ECC: opportunity evaluation and certification
    def _opportunity_eval(self) -> dict:
        """The executor's opportunity evaluation of the current decision state (METHOD.md, ECC). For each probe intent b of
        PROBE_GRID the mode prediction (event, time, delta-v) from the executor's own model (``ExecutorProbe.probe_intent``),
        and whether b is certified: it switches to an entry mode within the horizon, the pass is binding (switch time + one
        spin period beyond the latest admissible arrival, deadline - BRIDGE_DEADLINE_MARGIN_S: missing it leaves no other
        pass) and the resource holds (delta-v used + stage plan + entry plan <= budget). A fixed function of the state:
        no value, no history, no learning."""
        from env.t2_probe import PROBE_GRID, ExecutorProbe
        e = self.env
        if self._opp is not None and self._opp["t"] == float(e.time_s) and self._opp["episode"] == self.episode:
            return self._opp
        if self.cert_prober is None:
            self.cert_prober = ExecutorProbe(self)
        q, v = relative_inertial(e); t = e.target_state
        args = (q, v, t.rotation.copy(), t.omega.copy(), float(e.time_s), float(e._delta_v), self.mission, self.geometry)
        period = 2.0 * np.pi / max(float(np.linalg.norm(t.omega)), 1e-9)
        latest = float(self.mission["deadline_s"]) - BRIDGE_DEADLINE_MARGIN_S
        budget, used = float(self.mission["budget_m_s"]), float(e._delta_v)
        rows = []
        for b in PROBE_GRID:
            d = self.cert_prober.probe_intent(np.asarray(b, np.float64), *args)
            d["certified"] = bool(d["event"] and float(e.time_s) + d["t"] + period > latest
                                  and used + d["dv"] + d.get("dv_entry", 0.0) <= budget)
            rows.append(d)
        self._opp = {"t": float(e.time_s), "episode": self.episode, "args": args, "rows": rows, "period": period, "latest": latest}
        return self._opp

    def _certify(self, a: np.ndarray) -> tuple[np.ndarray, dict]:
        """The certification map C(x, a): the intent is flown unless its own mode prediction has no entry-mode switch
        within the horizon while the certified set of the probe intents is non-empty; then the certified probe nearest
        the intent (ties: the earlier switch) is flown for this stage. The same map acts in training and execution."""
        from env.t2_probe import PROBE_GRID
        if self.audit_logging:
            from time import perf_counter

            started = perf_counter()
        opp = self._opportunity_eval()
        own = self.cert_prober.probe_intent(a, *opp["args"])
        rec = {"a": [float(x) for x in a], "own_event": int(own["event"]), "own_t": float(own["t"]), "intervened": False}
        if self.audit_logging:
            rec.update(
                own_kind=own["kind"], own_dv=float(own["dv"]),
                cands=[
                    {
                        "b": [float(x) for x in b],
                        "event": int(d["event"]), "kind": d["kind"], "t": float(d["t"]),
                        "dv": float(d["dv"]), "dv_entry": float(d.get("dv_entry", 0.0)),
                        "certified": bool(d["certified"]),
                        "binding": bool(opp["t"] + d["t"] + opp["period"] > opp["latest"]),
                    }
                    for b, d in zip(PROBE_GRID, opp["rows"])
                ],
            )
        if own["event"]:
            if self.audit_logging:
                rec["wall_s"] = perf_counter() - started
            return a, rec
        best = None
        for b, d in zip(PROBE_GRID, opp["rows"]):
            if not d["certified"]:
                continue
            key = (float(np.linalg.norm(np.asarray(b) - a)), float(d["t"]))
            if best is None or key < best[0]:
                best = (key, b, d)
        if self.audit_logging:
            rec["wall_s"] = perf_counter() - started
        if best is None:
            return a, rec
        self.stats["cert_interventions"] = self.stats.get("cert_interventions", 0) + 1
        rec.update(intervened=True, a_exec=[float(x) for x in best[1]], alt_t=float(best[2]["t"]), alt_kind=best[2]["kind"])
        return np.asarray(best[1], np.float64), rec


    # ---------------------------------------------------------------- read-only diagnostics (formal runs)
    def decision_audit(self, t, dv, frame):
        return {
            "t_start": float(t),
            "remaining_s": float(self.mission["deadline_s"] - t),
            "remaining_dv": float(self.mission["budget_m_s"] - dv),
            **{key: float(frame[key]) for key in ("phase_rad", "time_to_pass_s", "lat_err_deg")},
        }

    def episode_audit(self):
        from env.t1_env import audit_metrics

        return {
            "budget_m_s": float(self.mission["budget_m_s"]),
            "min_margins": dict(self.env._min_margins),
            "terminal": audit_metrics(self.env.metrics())["terminal"],
        }


    def step(self, action):
        e, g, mass = self.env, self.generator, self.env.chaser_parameters.mass
        if g.mode in ("bridge", "descent"):
            if not self.audit_logging:
                return self._macro_to_end()
            q, _ = relative_inertial(e)
            audit = self.decision_audit(e.time_s, e._delta_v, frame_now(e, q))
            mode = g.mode
            result = self._macro_to_end()
            result[-1]["decision_audit"] = {
                **audit, "macro_end": "entry" if mode == "bridge" else "descent",
                "true_mode_event_t": None,
            }
            if result[2] or result[3]:
                result[-1].update(self.episode_audit())
            return result
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
        cert = None
        if self.certify and g.mode in ("stage", "hold"):    # the executor certifies the intent before it is flown
            a, cert = self._certify(a)
        q, v = relative_inertial(e)
        frame = frame_now(e, q)
        goal = stage_goal(q, v, frame, a, mass, geometry=self.geometry, task=self.env.config.task)
        plan = g.planner.solve(q, v, goal["q_goal"], goal["v_goal"])
        t0, dv0, phi0 = e.time_s, e._delta_v, self._phi()
        self.stats["stages"] += 1
        self.stats["rate_limited"] = self.stats.get("rate_limited", 0) + int(goal["rate_limited"])
        if plan["feasible"]:
            g.mode, g.plan, g.plan_start_s, g.hold_position = "stage", plan, float(t0), None
            g.last_outcome = "accepted"
        else:
            self.stats["rejected"] += 1
            g._enter_hold(self.info_env, f"hold:rejected:{plan['status']}")
        reward, event, steps = 0.0, None, 0
        first_mode, first_mode_t = None, None
        while True:
            r, event, info = self._advance(); reward += r; steps += 1
            if self.audit_logging and first_mode is None and g.mode in ("bridge", "descent"):
                first_mode, first_mode_t = g.mode, float(e.time_s)
            if event is not None:
                break
            if g.mode in ("bridge", "descent"):
                continue                       # terminal macro-step: no decision until it ends
            if g.mode == "hold" and (g.last_status or "").startswith("hold:stage_deviation"):
                break                          # deviation: ask again now
            if e.time_s - t0 >= STAGE_S - 1e-6 and g.mode in ("stage", "hold"):
                break
        q1, v1 = relative_inertial(e)
        held = g.mode == "hold"
        last = {"held": held}
        if plan["feasible"]:
            q_plan, v_plan = plan_state(plan, e.time_s - t0, mass)
            last["position_residual_m"] = float(np.linalg.norm(q1 - q_plan))
            last["velocity_residual_m_s"] = float(np.linalg.norm(v1 - v_plan))
        self.last = last
        self.stats["holds"] += int(held)
        terminated = event in TERMINAL_EVENTS
        truncated = event == "numerical_failure"
        if terminated:
            self._close_bridge(event)
        self.return_true += reward
        shaped = reward
        if self.shaping:
            phi1 = 0.0 if terminated else self._phi()
            shaped = reward + GAMMA_STAGE * phi1 - phi0
        if g.mode in ("bridge", "descent"):
            self.stats["macro"] = g.mode
        info = {"event": event, "true_reward": reward, "stage_s": e.time_s - t0, "held": held, "mode": g.mode,
                "plan_status": plan["status"], "rate_limited": goal["rate_limited"], "planned_dv": plan.get("delta_v_m_s"),
                "executed_dv": e._delta_v - dv0, "cert": cert, "time_s": e.time_s, "delta_v": e._delta_v,
                "range_m": float(np.linalg.norm(q1)), "mission": dict(self.mission), "far_field_controls": self.far_field_controls}
        if self.audit_logging:
            macro_end = (
                "entry" if first_mode == "bridge" else "descent" if first_mode == "descent"
                else "terminal" if event is not None else
                "deviation_hold" if held and (g.last_status or "").startswith("hold:stage_deviation")
                else "stage_complete"
            )
            info["decision_audit"] = {
                **self.decision_audit(t0, dv0, frame),
                "macro_end": macro_end, "true_mode_event_t": first_mode_t,
            }
            if terminated or truncated:
                info.update(self.episode_audit())
        return self.observation(), float(shaped), terminated, truncated, info


__all__ = ["T2Mission", "MixMission", "mission_from_name", "ECONOMY_BUDGET_M_S", "T2StageEnv", "T2_VERSION", "STAGE_S", "OBS_DIM", "OBS_DIM_PRIV",
           "OBS_PROFILE", "PLANNER_FEATURES", "EXEC_STATE_FEATURES", "GAMMA_STAGE", "InfoEnv", "stage_goal", "spin_frame",
           "frame_now", "target_velocity", "action_for_velocity", "latitude_velocity", "d_authority", "observe",
           "t2_env_config", "window_half_width", "opportunity_windows"]
