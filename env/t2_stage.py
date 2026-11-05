"""T2 staged approach (research decision 2026-10-21): SAC sets a 16 s two-dimensional stage goal,
G* only turns it into an executable local reference (T1-P local SCP) and later bridges to the entry,
the shared frozen MPC flies everything; truth RK45 judges.

Information boundary. Every planner, the stage basis, the bridge and the observation forecast the
target attitude from the *current* attitude and body rate only, as a rigid body with unknown
(isotropic) inertia, i.e. with a constant body rate (``InfoEnv``). The true inertia is used by the
truth propagation alone. The frozen MPC is unchanged (it extrapolates the current rate over its
3.5 s horizon for the reference path; its internal target model is not used for any decision).

Stage action a = (a_p, a_z) in [-1, 1]^2: the stage-end relative velocity (H = 16 s) in the spin
frame of the target (constant-rate model): e_r line of sight, e_pol towards the spin axis, e_az
towards the oncoming port (``spin_frame``):
    closing speed   v_c = 0.05 + 0.15 a_p    in [-0.10, 0.20] m/s   (progress),
    azimuthal speed v_z = 0.15 a_z           in [-0.15, 0.15] m/s   (meet the port earlier / later),
    latitude        v_l = clip(0.02/s * r * (lam - beta), +-0.12)   (fixed regulation onto the port's
                    sweep cone; not a decision: without it the port never passes the line of sight);
    v_goal = v + rate_limit(v_target - v, 2 d / H),  q_goal = q + H (v + v_goal) / 2,
    d = 0.7 (1.6 N / m) H^2 / 2 (1.35 m; 0.17 m/s of velocity change per stage at 106 kg).
A velocity-level goal is a guidance law (the centripetal part of a swing is re-issued every stage);
a goal beyond the 16 s reachable velocity change is scaled onto it and reported (``rate_limited``).
a = 0 approaches slowly (0.05 m/s) and waits for the port's pass; the trade-off SAC owns is how fast to
close and whether to spend fuel swinging to meet the port earlier, against the mission deadline.

Chain: stage (SAC every 16 s) -> bridge (r <= 12 m and the port forecast within 35 deg of the
line of sight for one of the bridge arrival times 15-60 s: the T1-N conditional G*, constant-rate model)
-> T1 descent (kappa 1) -> capture. A stage with a > 1 m deviation, an infeasible goal or a lost
bridge returns to a hold and SAC is asked again at the next 2 s step. Bridge and descent run inside
one environment step (a terminal macro-step), so every SAC transition starts at a stage decision.

Mission (T1-B assets, soft fuel): urgent (deadline 170-210 s, -0.10 per s, -10 per m/s of delta-v) or
economy (deadline 280-340 s, -0.02 per s, -40 per m/s); the episode times out at the deadline (-100);
a 3.0 m/s safety budget (resource exhausted -100); capture +100, unsafe -150.

Executor model (EACC, T2.3; for the coupled learner, ``candidates`` > 0). After each executed stage, from
the same decision state: M1 -- the executed intent and K candidate intents are run through the
executor's own stage planner and its entry rule (2 s cadence: range, radial envelope, port window,
entry planner; inside the MPC's near field a braking excess removable within one node interval is
accepted and charged), over the intent's stage and one more holding it; an intent that opens the entry
gets a terminal label (stage part + entry outcome calibrated on this environment's executed entries,
from the first bridge to the terminal event: a lost bridge is not an outcome). M2 -- hindsight anchoring:
every intent sees the realized target attitude (the target's motion does not depend on the chaser),
and the executed intent's realized tracking residual path (every 2 s) and, for an entry that ended in
this decision, its realized outcome residual are transported to the candidates. The executed intent's
raw prediction (calibration only) is returned for the learner's precision weight (M3,
``train/t2_mve_sac.py``). Labels are issued only where the executed outcome covers the same interval
(its stage completed, or it opened the entry inside the stage).

Entry commitment (T2.3 sub-mechanism, ``WindowGStar``): a live entry whose fixed-time re-plan fails is
re-timed inside the same port pass (2 / 4 / 8 s later), then flown with the authority reserve, never
beyond the deadline (``window=False``: T2.2's rigid commitment).
"""
from __future__ import annotations

import hashlib
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

T2_VERSION = "t2-stage-0.13.0"
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
#: T2.9 decision-layer interface. 'engineered' (T2.0-T2.8): the 24 features of ``observe``, among them quantities the
#: executor's own models compute (time to the port's pass, entry-window reachability, braking margin, deadline slack
#: after the pass, last stage's execution residuals / hold) and frame-aligned phase angles. 'geometric': the same vector
#: with the seven executor-derived features set to zero (EXECUTOR_FEATURES) -- the target's geometric state (spin frame,
#: phase, rate, port direction, cone angles), the relative state, the mission clock and fuel stay, so the policy keeps
#: what it learns from; the executor's predictions reach it only through training (EACC). 'raw': ``observe_raw``, also
#: without the spin-aligned frame and phase angles (harder; not the T2.9 default).
#: 'reduced' (T2.10, research layer 2026-11-02: moderate change): only the three features that hand the policy the
#: planner's answer are zeroed -- time to the port's pass, entry-window reachability, deadline slack after the pass
#: (PLANNER_FEATURES); the relative state, target geometry, clock, fuel, braking margin and execution state stay.
#: 'reduced_priv' (T2.11 EACC-A): the 'reduced' vector followed by the executor's own state-wise judgements -- the full
#: engineered vector (pass time, window reachability, slack, braking margin, execution residuals / hold, ...) -- for
#: OBS_DIM_PRIV values in all. The actor reads only the first OBS_DIM (the reduced interface); a critic may read the
#: rest during training (the executor's information); deployment uses the actor alone.
#: 'reduced_priv_cert' (env 0.13.0, EACC with execution certificates): 'reduced_priv' followed by the executor-calibrated
#: counterfactual opportunity map of the decision -- for each intent of PREF_GRID (defined below) the executor's own
#: judgement ``opportunity_features``: entry feasible, success share of the entry outcome model (historical execution
#: replay) and value / CERT_VALUE_SCALE (clipped), then a 'calibration ready' flag (all zero until it is). It carries what
#: the executor has actually achieved, not only what it plans; for the critic during training, never for the actor.
#: 'reduced_dup' (final experiments, capacity control of U1): the 'reduced' vector twice -- the same 48-value critic
#: input width as 'reduced_priv', with the second half repeating the actor's view instead of the executor's judgements.
OBS_PROFILES = ("engineered", "reduced", "geometric", "raw", "reduced_priv", "reduced_priv_cert", "reduced_dup")
OBS_DIM_PRIV = 2 * 24
CERT_DIM = 3 * 9 + 1
OBS_DIM_CERT = OBS_DIM_PRIV + CERT_DIM
CERT_VALUE_SCALE = 200.0
PLANNER_FEATURES = (8, 17, 19)
#: indices in ``observe``: time to the port's pass, last stage's position / velocity residual and hold, entry-window
#: reachability (bridge angle), braking margin, deadline slack after the pass
EXECUTOR_FEATURES = (8, 14, 15, 16, 17, 18, 19)
CALIBRATION_MIN_STAGES = 10
CALIBRATION_WINDOW = 60
CALIBRATION_REGIME_MIN = 5       # stages of one execution regime before its own calibration is used
#: a stage starting within this range is (at least partly) flown by the near-field MPC
NEAR_REGIME_M = 7.5 + 2.0
CANDIDATE_RADIUS = 0.5          # local candidates (``candidate_actions``: 3 structured + 1 global + local)
#: EACC iteration (env 0.11.0, label arm e_rel only): candidates are compared in the executor's terms. An intent whose
#: stage goal (the executor's projected, rate-limited stage-end velocity; it fixes the executed stage plan) lies within
#: CANDIDATE_MIN_SEP_M_S of the executed intent's or of an already chosen candidate's is redrawn (its kind: local stays
#: local, a structured alternative falls back to a global draw), at most CANDIDATE_RESAMPLES times. 0.01 m/s = 0.08 m at
#: the stage end, above the anchored model's stage-end position error (P90 0.057 m, evidence/T2_12_PRECHECK): a smaller
#: difference is below the model's resolution and carries no comparison. 0 keeps the T2.11 / a3f4c65 generator exactly.
CANDIDATE_MIN_SEP_M_S = 0.01
CANDIDATE_RESAMPLES = 4
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
    #: mission preference (observed through the mission class): urgent values time, economy fuel
    urgent_time_cost_per_s: float = 0.10
    urgent_fuel_cost_per_m_s: float = 10.0
    economy_time_cost_per_s: float = 0.02
    economy_fuel_cost_per_m_s: float = 40.0

    def sample(self, opening: int) -> dict[str, Any]:
        rng = np.random.default_rng(np.random.SeedSequence([int(opening), 290021]))
        urgent = bool(rng.integers(0, 2))
        d = rng.uniform(*(self.urgent_deadline_s if urgent else self.economy_deadline_s))
        return {"deadline_s": float(round(d, 1)), "urgent": urgent, "budget_m_s": self.budget_m_s,
                "time_cost_per_s": self.urgent_time_cost_per_s if urgent else self.economy_time_cost_per_s,
                "fuel_cost_per_m_s": self.urgent_fuel_cost_per_m_s if urgent else self.economy_fuel_cost_per_m_s}


def t2_env_config(mission: T2Mission = T2Mission()) -> T1EnvConfig:
    base = T1EnvConfig()
    initial = replace(T1InitialDistribution(), range_m=(12.0, 16.0), inertial_speed_max_m_s=0.05)
    return replace(base, initial=initial, max_time_s=max(mission.economy_deadline_s))


#: T2.3 window commitment (sub-mechanism): the entry is a commitment to the port's pass, not to an instant
WINDOW_SLIPS_S = (2.0, 4.0, 8.0)     # arrival re-timing inside the same pass (<= ~20 deg of port sweep)
WINDOW_RESERVE_FRACTION = 0.95       # authority reserve: the last re-plan may use 95 % of the force bound


@dataclass
class WindowGStar(ConditionalGStar):
    """Conditional G* with a pass-window commitment (T2.3 sub-mechanism).

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


#: T2.8 executor safety projection of a stage goal (named defect, T2.7 evaluation): the stage layer flew intents that
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



RAW_FEATURES = 14


def observe_raw(env, q: np.ndarray, v: np.ndarray, t_s: float, dv_used: float, mission: dict, rotation: np.ndarray,
                omega_body: np.ndarray, last: dict | None = None) -> np.ndarray:
    """T2.9 'raw' interface (padded to OBS_DIM with zeros): range; relative velocity, the port direction and the
    target's spin vector in a neutral line-of-sight frame (e_r, the inertial z axis -- y near the pole -- made
    orthogonal to it, their cross product); deadline left, urgency, delta-v used and left. No spin-aligned frame,
    phase, pass time, window reachability, braking margin, slack or execution residual (``last`` is ignored)."""
    task = env.config.task
    r = float(np.linalg.norm(q)); e_r = q / max(r, 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(e_r[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = ref - float(ref @ e_r) * e_r; e2 = e2 / np.linalg.norm(e2)
    basis = np.vstack([e_r, e2, np.cross(e_r, e2)])
    spin = rotation @ omega_body
    port = rotation @ task.approach_axis
    x = np.zeros(OBS_DIM, dtype=np.float32)
    x[:RAW_FEATURES] = [r / 15.0, *(basis @ v / 0.2), *(basis @ port), *(basis @ spin / np.radians(2.7)),
                        (mission["deadline_s"] - t_s) / 300.0, float(mission["urgent"]), dv_used / 2.5,
                        (mission["budget_m_s"] - dv_used) / 3.0]
    return x

@dataclass
class Calibration:
    """Execution calibration of the stage planner on this environment's executed stages."""

    rows: list = field(default_factory=list)

    def add(self, planned_dv: float, executed_dv: float, dq_basis: np.ndarray, dv_basis: np.ndarray, near: bool = False) -> None:
        self.rows.append((planned_dv, executed_dv, np.asarray(dq_basis), np.asarray(dv_basis), bool(near)))
        self.rows = self.rows[-CALIBRATION_WINDOW:]

    def regime(self, near: bool) -> "Calibration":
        """Rows of one execution regime (near field: the MPC flies the stage; far field: the tracking law),
        when it has enough of them; otherwise all rows."""
        own = [r for r in self.rows if r[4] == bool(near)]
        return Calibration(rows=own) if len(own) >= CALIBRATION_REGIME_MIN else Calibration(rows=list(self.rows))

    @property
    def ready(self) -> bool:
        return len(self.rows) >= CALIBRATION_MIN_STAGES

    def frozen(self) -> "Calibration":
        """Copy used for one stage: the factual check and every candidate share it."""
        return Calibration(rows=list(self.rows))

    def coefficients(self) -> tuple[float, float]:
        p = np.array([r[0] for r in self.rows]); e = np.array([r[1] for r in self.rows])
        A = np.c_[np.ones_like(p), p]
        # ridge towards (0, 1) = "execution equals plan"
        lam = 1e-3
        coef = np.linalg.solve(A.T @ A + lam * np.eye(2), A.T @ e + lam * np.array([0.0, 1.0]))
        return float(coef[0]), float(coef[1])

    def residuals(self) -> tuple[np.ndarray, np.ndarray]:
        return np.mean([r[2] for r in self.rows], 0), np.mean([r[3] for r in self.rows], 0)

    def predict(self, plan: dict, basis: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray, float]:
        q_end, v_end = plan_state(plan, STAGE_S, mass)
        c0, c1 = self.coefficients(); dq, dv = self.residuals()
        return q_end + basis.T @ dq, v_end + basis.T @ dv, max(0.0, c0 + c1 * float(plan["delta_v_m_s"]))


#: entry-level executor model (T2.2): does the executor open the entry within the stage, and with what
#: calibrated outcome (``predict_intent``, ``EntryCalibration``)
ENTRY_MODEL = True
ENTRY_NODE_S = 2.0               # the executor's bridge cadence
ENTRY_NODES_S = tuple(ENTRY_NODE_S * k for k in range(1, 9))
ENTRY_SOLVES_PER_NODE = 7        # entry-planner solves per node: every arrival time, ascending, as the executor
ENTRY_MAX_SOLVES = 24            # per predicted action (over both stages)
ENTRY_CAL_MIN = 20               # executed entries before entry labels are issued (5 left the success rate to the prior: labels ~55 low, T2.3 audit)
#: horizon of a counterfactual label in stages (T2.5: 2). Stage 1 is the candidate intent's own 16 s stage (execution
#: anchored); if it does not open the entry, stage 2 is flown by an action a2 ~ mu(.|o1) of the policy's copy in this
#: environment (calibrated model): the learner forms the two-stage soft Retrace target with c = min(1, pi/mu). 1 = the
#: T2.4 one-stage label. T2.3's repeat-the-intent second stage (not a Bellman target) no longer exists.
ENTRY_STAGES = 2
#: T2.7 executor guidance (training only; ``track_intent``): the executor's conditional G* plans its earliest feasible
#: entry from the current state on the direct grid (start_direct_gstar's, arrivals before deadline - margin, the
#: estimated inertia), and the plan is projected onto the stage action space -- the intent whose stage goal lies nearest
#: the plan 16 s ahead. Sandbox (exploratory): flown at every decision it captured 19/19 urgent audit missions (rally
#: rule 9/19, direct G* 18/19) and matched the best converged SAC on the development openings.
#: T2.10 candidate intents of the opportunity preference (the 3 x 3 grid P4 validated) -- the executed intent is added
PREF_GRID = tuple((p, z) for p in (-0.5, 0.25, 1.0) for z in (-1.0, 0.0, 1.0))
assert CERT_DIM == 3 * len(PREF_GRID) + 1
OPPORTUNITY_TAUS_S = (20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 140., 160., 180., 200., 220.)
TRACK_GRID = tuple((float(p), float(z)) for p in np.linspace(-1.0, 1.0, 9) for z in np.linspace(-1.0, 1.0, 9))
TRACK_VELOCITY_WEIGHT = 10.0      # s: velocity mismatch of a stage goal weighed against position (m) in the projection


@dataclass
class EntryCalibration:
    """Entry outcome model (bridge + descent) from this environment's executed entries, each recorded from
    the first bridge to the terminal event: planned arrival, planned delta-v, realized duration and delta-v,
    outcome.

    T2.4: historical execution replay. A candidate entry's value is the mean over the recorded entries of
    the outcome that entry's realized execution deviation (time beyond its planned arrival, delta-v beyond
    its plan) would give on the candidate's own plan and remaining time: an unsafe record stays unsafe; any
    other record succeeds if its replayed duration fits the remaining time, else times out. T2.3 pooled one
    success rate with a Laplace prior: an entry with ample time was charged the rare late failures of others
    (labels without an executed outcome to anchor were 23 low; T2.3 audit, every candidate entry captured)."""

    rows: list = field(default_factory=list)

    def add(self, tau_plan: float, dv_plan: float, duration: float, dv: float, success: bool, penalty: float,
            remaining: float = np.inf) -> None:
        self.rows.append((float(tau_plan), float(dv_plan), float(duration), float(dv), bool(success), float(penalty), float(remaining)))
        self.rows = self.rows[-CALIBRATION_WINDOW:]

    @property
    def ready(self) -> bool:
        return len(self.rows) >= ENTRY_CAL_MIN

    def frozen(self) -> "EntryCalibration":
        return EntryCalibration(rows=list(self.rows))

    def predict(self, tau: float, dv_plan: float, reward, remaining: float = np.inf) -> dict:
        """A timed-out record is censored: its true time beyond the plan is at least what it had. It counts as
        a failure where the candidate has no more slack than that record had; otherwise it is left out (its
        outcome there is unknown). Records without a stored remaining time (before T2.4) count as failures."""
        rows = [x if len(x) > 6 else (*x, np.inf) for x in self.rows]
        r = np.array([x[:4] for x in rows], dtype=np.float64)
        rem_i = np.array([x[6] for x in rows], dtype=np.float64)
        unsafe = np.array([(not x[4]) and x[5] >= reward.unsafe_penalty for x in rows])
        timed_out = np.array([(not x[4]) and x[5] < reward.unsafe_penalty for x in rows])
        extra = r[:, 2] - r[:, 0]
        duration = tau + extra
        dv = np.maximum(0.0, dv_plan + (r[:, 3] - r[:, 1]))
        slack, slack_i = remaining - tau, rem_i - r[:, 0]
        informative = ~timed_out | (slack <= slack_i + 1e-9)
        fits = (duration <= remaining) & ~timed_out
        spent = np.minimum(duration, remaining)
        value = np.where(unsafe, -reward.unsafe_penalty, np.where(fits, reward.capture_reward, -reward.timeout_penalty)) \
            - reward.time_cost_per_s * spent - reward.delta_v_cost_per_m_s * dv
        m = informative if informative.any() else np.ones_like(informative)
        ok = (~unsafe) & fits
        return {"duration": float(spent[m].mean()), "dv": float(dv[m].mean()), "p_success": float(ok[m].mean()),
                "reward": float(value[m].mean()), "records": int(m.sum())}


def terminated_event(event) -> bool:
    return event in ("success", "unsafe", "timeout", "resource_exhausted")


class T2StageEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, openings: list[int] | None = None, seed_base: int = 1_000_000, candidates: int = 0,
                 mission: T2Mission = T2Mission(), shaping: bool = True, seed: int = 0, geometry: bool = True,
                 entry_stages: int = ENTRY_STAGES, window: bool = True, obs_profile: str = "engineered", pref: bool = False,
                 cand_min_sep: float = 0.0) -> None:
        if obs_profile not in OBS_PROFILES:
            raise ValueError(f"obs_profile must be one of {OBS_PROFILES}")
        self.obs_profile = obs_profile
        #: T2.10: at every stage decision, the executor-calibrated opportunity value of PREF_GRID and the executed intent
        self.pref = bool(pref)
        #: EACC iteration: minimum separation of the candidates' stage goals (m/s; 0 = the T2.11 generator)
        self.cand_min_sep = float(cand_min_sep)
        self.last_resampled = 0
        #: env 0.12.0 (EACC actor-side feedback): also predict the policy's own deterministic intent at the decision (set by
        #: the collector in ``policy_action``) through the candidates' treatment -- the reference of the execution advantage
        self.ref_labels = False
        #: T2.11: the policy's deterministic intent for the next decision (set by the collector; one decision only)
        self.policy_action = None
        self.geometry = bool(geometry)
        self.window = bool(window)
        self.diagnostics = False            # also predict the executed intent with its own realized residual (tests)
        self.entry_stages = int(entry_stages)
        #: obs -> (a2, log mu(a2 | obs)): the policy copy that chooses the second stage of a counterfactual label
        self.label_policy = None
        self.mission_distribution = mission
        self.base_config = t2_env_config(mission)
        self.env = T1Env(self.base_config)
        self.info_env = InfoEnv(self.env)
        self.openings, self.seed_base, self.candidates, self.shaping = openings, int(seed_base), int(candidates), bool(shaping)
        self.rng = np.random.default_rng(seed)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.obs_dim = {"reduced_priv": OBS_DIM_PRIV, "reduced_priv_cert": OBS_DIM_CERT, "reduced_dup": OBS_DIM_PRIV}.get(self.obs_profile, OBS_DIM)
        #: env 0.13.0: compute the certificate map (False: zeros -- evaluation, where only the actor acts)
        self.cert_live = True
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.episode = 0
        self.calibration = Calibration()
        self.entry_calibration = EntryCalibration()
        self.entry_residuals: list[float] = []   # realized minus predicted entry outcome (T2.4 entry reliability)
        self._bridge = None
        self._stage_entry = None
        self._entry_open = None
        self._trace = []
        self._model_t0 = 0.0
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

    def _metrics(self):
        return self.env.metrics()

    def observation(self) -> np.ndarray:
        q, v = relative_inertial(self.env)
        t = self.env.target_state
        o = self._observe(self.env, q, v, self.env.time_s, self.env._delta_v, self.mission, t.rotation, t.omega, self.last)
        if self.obs_profile == "reduced_priv_cert":       # env 0.13.0: the actual decision state carries its certificate map
            o[OBS_DIM_PRIV:] = self.certificate_map()
        return o

    def _observe(self, *args) -> np.ndarray:
        """The decision layer's observation in this environment's interface (actual and modelled states alike)."""
        if self.obs_profile == "raw":
            return observe_raw(*args)
        o = observe(*args)
        if self.obs_profile == "reduced_dup":
            r = o.copy(); r[list(PLANNER_FEATURES)] = 0.0
            return np.concatenate([r, r]).astype(np.float32)
        if self.obs_profile in ("reduced_priv", "reduced_priv_cert"):
            r = o.copy(); r[list(PLANNER_FEATURES)] = 0.0
            if self.obs_profile == "reduced_priv_cert":      # a modelled state carries no certificate (map and ready flag 0)
                return np.concatenate([r, o, np.zeros(CERT_DIM, np.float32)]).astype(np.float32)
            return np.concatenate([r, o]).astype(np.float32)
        if self.obs_profile == "geometric":
            o[list(EXECUTOR_FEATURES)] = 0.0
        elif self.obs_profile == "reduced":
            o[list(PLANNER_FEATURES)] = 0.0
        return o

    def certificate_map(self) -> np.ndarray:
        """env 0.13.0: the executor-calibrated counterfactual opportunity map of the current decision (CERT_DIM values;
        see 'reduced_priv_cert'). Zeros until the entry calibration is ready, or when ``cert_live`` is off."""
        out = np.zeros(CERT_DIM, np.float32)
        if not self.cert_live or not self.entry_calibration.ready or getattr(self, 'generator', None) is None:
            return out
        for k, x in enumerate(PREF_GRID):
            f = self.opportunity_features(x)
            if f is not None:
                out[3 * k:3 * k + 3] = (float(f["feasible"]), float(f["p_success"]),
                                        float(np.clip(f["value"] / CERT_VALUE_SCALE, -3.0, 3.0)))
        out[-1] = 1.0
        return out

    def opportunity_value(self, action) -> float | None:
        """T2.10 executor-calibrated opportunity value of a stage intent at the current decision (``opp2`` of
        evidence/T2_8_PRECHECK/脚本/opp2_probe.py, the signal P4 validates): from the intent's stage goal (the planner's
        own goal; target attitude 16 s ahead from the constant-rate model and the estimated inertia) the earliest
        feasible conditional G* entry arriving within the time left (no planning margin -- the executor's own entry has
        none), valued by the entry outcome model's historical execution replay (EntryCalibration.predict: capture or
        timeout, time and delta-v) less the stage's time cost; no feasible entry: -timeout penalty - time cost of the
        time left. None until the entry calibration is ready. Used only as a ranking between intents of one decision."""
        f = self.opportunity_features(action)
        return None if f is None else f["value"]

    def opportunity_features(self, action) -> dict | None:
        """T2.11 the executor's opportunity judgement of a stage intent with the raw quantities the decision-level
        execution calibration uses (``value`` = ``opportunity_value``): feasible, planned arrival tau, planned entry
        delta-v, the entry outcome model's success share p_success; plus the decision's context (time left at the
        stage end, range, urgency). None until the entry calibration is ready."""
        from env.t1_gstar import free_body_rotations
        if not self.entry_calibration.ready:
            return None
        e = self.env; w = e.config.reward
        q, v = relative_inertial(e)
        goal = stage_goal(q, v, frame_now(e, q), np.asarray(action, float).reshape(-1)[:2], e.chaser_parameters.mass,
                          geometry=self.geometry, task=e.config.task)
        remaining = float(self.mission["deadline_s"]) - (e.time_s + STAGE_S)
        ctx = {"remaining_s": remaining, "range_m": float(np.linalg.norm(q)), "urgent": bool(self.mission["urgent"])}
        none = {**ctx, "feasible": False, "tau": None, "dv_plan": None, "p_success": 0.0,
                "value": float(-w.timeout_penalty - w.time_cost_per_s * max(remaining + STAGE_S, 0.0))}
        taus = [t for t in OPPORTUNITY_TAUS_S if t <= remaining]
        if not taus:
            return none
        inertia = np.asarray(self.info_env.target_parameters.inertia)
        _, rots, rates = free_body_rotations(e.target_state.rotation, e.target_state.omega, inertia, STAGE_S)
        history = free_body_rotations(rots[-1], rates[-1], inertia, max(taus) + 1.0)
        g = self.generator.bridge
        for tau in taus:
            res = g.planner.solve(np.asarray(goal["q_goal"], float), np.asarray(goal["v_goal"], float), tau, history, None,
                                  entry_speed_m_s=g.entry_speed_m_s)
            if res["feasible"]:
                pr = self.entry_calibration.frozen().predict(float(res["tau_s"]), float(res["delta_v_m_s"]), w, remaining)
                return {**ctx, "feasible": True, "tau": float(res["tau_s"]), "dv_plan": float(res["delta_v_m_s"]),
                        "p_success": float(pr["p_success"]), "value": float(pr["reward"] - w.time_cost_per_s * STAGE_S)}
        return none

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
        self._entry_open = None
        self._trace = []
        self.return_true = 0.0
        self.stats = {"stages": 0, "holds": 0, "rejected": 0, "bridge_attempts": 0, "macro": None}
        return self.observation(), {"opening": opening, "mission": dict(self.mission)}

    def start_direct_gstar(self) -> bool:
        """Reference row outside the stage action space: the conditional G* from the far field to the
        entry at the earliest feasible time of its grid (constant-rate model), re-planned on deviation."""

        g = self.generator
        horizon = self.env.config.max_time_s - BRIDGE_DEADLINE_MARGIN_S - self.env.time_s
        for tau in [t for t in (20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 140., 160., 180., 200., 220.) if t <= horizon]:
            g.bridge.plan, g.bridge.hold_position = None, None
            g.bridge.decide(self.info_env, tau)
            if g.bridge.plan is not None:
                g.mode, g.plan, g.bridge_tau_s = "bridge", None, tau
                return True
        return False

    def _try_bridge(self, m) -> bool:
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
                q, v = relative_inertial(self.env)
                self._bridge = {"t": self.env.time_s, "dv": self.env._delta_v, "tau": float(g.bridge.plan["tau_s"]),
                                "dv_plan": float(g.bridge.plan["delta_v_m_s"]), "q": q, "v": v}
                if self._stage_entry is None:          # the decision's first entry
                    self._stage_entry = self._bridge
                if self._entry_open is None:           # an entry record runs from the first bridge to the terminal event
                    self._entry_open = {**self._bridge, "G": 0.0}
                    if self.entry_calibration.ready:   # predicted before the outcome is seen (predict-then-observe)
                        self._entry_open["pred"] = self.entry_calibration.predict(
                            self._bridge["tau"], self._bridge["dv_plan"], self.env.config.reward, self.mission["deadline_s"] - self.env.time_s)["reward"]
                return True
        return False

    def _close_bridge(self, event) -> None:
        """A lost bridge ends only the bridge: the entry record stays open and its eventual outcome counts
        (T2.2 counted a lost bridge as a failed entry; most were recovered, the labels were 69 too low).
        A terminal event closes the record: one entry-calibration row."""
        if self._bridge is not None:
            self.stats["entries_lost"] = self.stats.get("entries_lost", 0) + int(event == "lost")
            self._bridge = None
        o, e, w = self._entry_open, self.env, self.env.config.reward
        if o is None or event == "lost":
            return
        penalty = {"unsafe": w.unsafe_penalty, "timeout": w.timeout_penalty, "resource_exhausted": w.timeout_penalty}.get(event, w.timeout_penalty)
        self.entry_calibration.add(o["tau"], o["dv_plan"], e.time_s - o["t"], e._delta_v - o["dv"], event == "success", penalty,
                                   self.mission["deadline_s"] - o["t"])
        if "pred" in o:
            self.entry_residuals = (self.entry_residuals + [float(o["G"] - o["pred"])])[-CALIBRATION_WINDOW:]
        self._entry_open = None

    # ------------------------------------------------------------ EACC executor option model (M1 + M2)
    def _dv_model(self, plan, cal: "Calibration", s: float) -> float:
        """Calibrated delta-v of a stage plan flown for ``s`` seconds (executed against planned, fitted)."""
        mass = self.env.chaser_parameters.mass
        n = int(round(s / plan["dt"]))
        dv_plan_s = float(np.sum(np.linalg.norm(plan["f"][:n], axis=1)) * plan["dt"] / mass)
        c0, c1 = cal.coefficients()
        return max(0.0, c0 * s / STAGE_S + c1 * dv_plan_s)

    def _scan(self, plan, resid, attitude, offset: float, budget: list, brake_model: bool) -> dict | None:
        """The executor's bridge rule at its 2 s cadence along one stage plan corrected by ``resid`` (range,
        radial envelope, port window, entry planner -- the executor's own functions and inputs). The target
        attitude at each node is ``attitude`` (the realized one where known: it does not depend on the
        chaser's intent). ``brake_model``: a node violating the envelope inside the MPC's near field is
        accepted when the excess can be braked within one node interval (charged); the executor's MPC does so."""

        from env.t1_gstar import free_body_rotations
        e, g, task = self.env, self.generator, self.env.config.task
        mass = e.chaser_parameters.mass
        inertia = np.asarray(self.info_env.target_parameters.inertia)
        horizon = e.config.max_time_s - BRIDGE_DEADLINE_MARGIN_S
        t0 = self._model_t0
        for s in ENTRY_NODES_S:
            q, v = plan_state(plan, s, mass)
            dq, dv_, _ = resid(s)
            q = q + dq; v = v + dv_
            r = float(np.linalg.norm(q))
            if r > BRIDGE_RANGE_M:
                continue
            closing = -float(q @ v) / r
            cap = float(task.outer_radial_closing_speed_limit(r)) - BRIDGE_ENVELOPE_MARGIN_M_S
            brake = 0.0
            if closing > cap:
                brake = closing - cap
                if not brake_model or r > FAR_FIELD_M or brake > e.config.max_force_per_axis_n / mass * ENTRY_NODE_S:
                    continue
                v = v + brake * q / r
            rot, omega = attitude(offset + s)
            node_solves = 0
            for tau in BRIDGE_TAUS_S:
                if node_solves >= ENTRY_SOLVES_PER_NODE or budget[0] <= 0:
                    break
                if tau > horizon - (t0 + offset + s) + 1e-6:
                    continue
                if angle_deg(rot @ so3_exp(tau * omega) @ task.approach_axis, q) > BRIDGE_ANGLE_DEG:
                    continue
                history = free_body_rotations(rot, omega, inertia, tau + 1.0)
                res = g.bridge.planner.solve(q, v, tau, history, None, entry_speed_m_s=g.bridge.entry_speed_m_s)
                node_solves += 1; budget[0] -= 1
                if res["feasible"]:
                    return {"s": s, "tau": float(res["tau_s"]), "dv_plan": float(res["delta_v_m_s"]), "brake_dv": brake}
        return None

    def _opportunity(self, q, v, t_abs: float, rot, omega) -> dict | None:
        """T2.7: the executor's earliest feasible entry from a relative state at absolute time ``t_abs`` -- its
        conditional G* on the direct grid, arrivals up to deadline - margin, the constant-rate attitude model from
        ``rot`` / ``omega`` and the estimated (not the true) inertia: the executor's own inputs, as start_direct_gstar.
        Returns {tau, dv_plan, plan} or None (no capture opportunity left)."""
        from env.t1_gstar import free_body_rotations
        g = self.generator.bridge
        horizon = float(self.mission["deadline_s"]) - BRIDGE_DEADLINE_MARGIN_S - float(t_abs)
        taus = [t for t in OPPORTUNITY_TAUS_S if t <= horizon]
        if not taus:
            return None
        history = free_body_rotations(rot, omega, np.asarray(self.info_env.target_parameters.inertia), max(taus) + 1.0)
        for tau in taus:
            res = g.planner.solve(np.asarray(q, float), np.asarray(v, float), tau, history, None, entry_speed_m_s=g.entry_speed_m_s)
            if res["feasible"]:
                return {"tau": float(res["tau_s"]), "dv_plan": float(res["delta_v_m_s"]), "plan": res}
        return None

    def track_intent(self) -> np.ndarray | None:
        """T2.7 executor guidance: the executor's capture plan from the current state projected onto the stage action
        space (TRACK_GRID: the intent whose stage goal lies nearest the plan's state 16 s ahead); None without a plan or
        outside a stage decision. Used only by the training collector; it never replaces the policy at evaluation."""
        e = self.env
        if self.generator is None or self.generator.mode not in ("stage", "hold"):
            return None
        q, v = relative_inertial(e)
        opp = self._opportunity(q, v, e.time_s, e.target_state.rotation, e.target_state.omega)
        if opp is None:
            return None
        mass, frame = e.chaser_parameters.mass, frame_now(e, q)
        q16, v16 = plan_state(opp["plan"], STAGE_S, mass)
        best, cost = None, np.inf
        for a in TRACK_GRID:
            goal = stage_goal(q, v, frame, np.asarray(a), mass, geometry=self.geometry, task=self.env.config.task)
            c = float(np.linalg.norm(goal["q_goal"] - q16) + TRACK_VELOCITY_WEIGHT * np.linalg.norm(goal["v_goal"] - v16))
            if c < cost:
                best, cost = np.asarray(a, dtype=np.float64), c
        return best

    def _stage_cutoff(self, t_start: float, dv_start: float, dv_stage: float, span: float = STAGE_S):
        """T2.6 (label termination): a modelled stage without an entry ends where the truth would end it -- at the
        mission deadline (timeout) or when the delta-v budget is spent (resource exhausted), whichever comes first.
        ``span``: the modelled interval (a full stage, or T2.6.1 the part of a stage before its predicted entry, which
        delta-v spent before the entry ends just the same). Returns None, or (elapsed s, delta-v spent, event). The
        truth's terminal reward is the cost up to then minus the timeout penalty (both events), with no bootstrap;
        before T2.6 the label ran the full 16 s and bootstrapped."""
        ends = []
        left_t = float(self.mission["deadline_s"]) - float(t_start)
        if left_t < span - 1e-9:
            ends.append((max(left_t, 0.0), "timeout"))
        left_dv = float(self.mission["budget_m_s"]) - float(dv_start)
        if dv_stage > 1e-12 and dv_stage >= left_dv:
            ends.append((span * max(left_dv, 0.0) / dv_stage, "resource_exhausted"))
        if not ends:
            return None
        s_end, event = min(ends)
        return s_end, dv_stage * s_end / span, event

    def predict_intent(self, plan, action, start: dict, cal: "Calibration", ecal: "EntryCalibration", resid, attitude,
                       entry_offset: float = 0.0, brake_model: bool = True) -> dict:
        """Stage 1 of the executor option model of one intent from the decision state: its stage plan, corrected by
        ``resid`` (raw: the calibration's mean residual; anchored: the executed intent's realized residual path), run
        through the executor's entry rule. Returns the shaped label: an entry is terminal (stage part + calibrated
        entry outcome + ``entry_offset``), otherwise the stage end (reward, next observation with the realized target
        attitude); stage 2, if any, is ``predict_second_stage``."""

        e, mass, w = self.env, self.env.chaser_parameters.mass, self.env.config.reward
        cost = lambda s, dv: -w.time_cost_per_s * s - w.delta_v_cost_per_m_s * dv
        budget = [ENTRY_MAX_SOLVES]
        ecal_on = ecal is not None and ecal.ready
        hit = self._scan(plan, resid, attitude, 0.0, budget, brake_model) if ecal_on else None
        if hit is not None:
            dv_hit = self._dv_model(plan, cal, hit["s"]) + resid(hit["s"])[2] + hit["brake_dv"]
            cut = self._stage_cutoff(start["t"], start["dv"], dv_hit, hit["s"])   # the budget spent before the entry
            if cut is not None:
                true = cost(cut[0], cut[1]) - w.timeout_penalty
                return {"reward": float(true - (start["phi"] if self.shaping else 0.0)), "done": True, "entry": None, "event": cut[2],
                        "stages": 1, "true": float(true)}
            macro = ecal.predict(hit["tau"], hit["dv_plan"], w, self.mission["deadline_s"] - (start["t"] + hit["s"]))["reward"] + entry_offset
            true = cost(hit["s"], dv_hit) + macro
            return {"reward": float(true - (start["phi"] if self.shaping else 0.0)), "done": True, "entry": hit, "stages": 1, "true": float(true)}
        dq, dvv, ddv = resid(STAGE_S)
        q1, v1 = plan_state(plan, STAGE_S, mass)
        q1, v1 = q1 + dq, v1 + dvv
        dv1 = self._dv_model(plan, cal, STAGE_S) + ddv
        cut = self._stage_cutoff(start["t"], start["dv"], dv1)
        if cut is not None:
            true = cost(cut[0], cut[1]) - w.timeout_penalty
            return {"reward": float(true - (start["phi"] if self.shaping else 0.0)), "done": True, "entry": None, "event": cut[2],
                    "stages": 1, "true": float(true)}
        rot1, om1 = attitude(STAGE_S)
        r1 = cost(STAGE_S, dv1)
        obs = self._observe(e, q1, v1, start["t"] + STAGE_S, start["dv"] + dv1, self.mission, rot1, om1,
                      {"position_residual_m": float(np.linalg.norm(dq)), "velocity_residual_m_s": float(np.linalg.norm(dvv)), "held": False})
        phi1 = self.potential(q1, rot1, om1)
        out = {"reward": float(r1 + (GAMMA_STAGE * phi1 - start["phi"] if self.shaping else 0.0)), "obs": obs, "done": False,
               "q": q1, "v": v1, "dv": float(dv1), "r_end": float(np.linalg.norm(q1)), "true": float(r1), "phi": float(phi1),
               "rot": rot1, "omega": om1}
        return out

    def predict_second_stage(self, s1: dict, a2, start: dict, cal: "Calibration", ecal: "EntryCalibration", attitude,
                             brake_model: bool = True) -> dict | None:
        """Stage 2 of a counterfactual label (T2.5): from the stage-1 prediction ``s1`` the executor flies the policy's
        action ``a2`` for one stage (calibrated model; no realized residual exists beyond stage 1; the target attitude
        is realized where the factual trajectory recorded it, else the constant-rate forecast). Returns the shaped
        stage-2 reward (PBRS from Phi(s1)), its next observation and whether it ends in the entry (terminal; entry
        outcome from the replayed executed entries, never anchored to this decision's realized entry). None when the
        stage is outside the model's domain."""

        e, mass, w = self.env, self.env.chaser_parameters.mass, self.env.config.reward
        cost = lambda s, dv: -w.time_cost_per_s * s - w.delta_v_cost_per_m_s * dv
        q1, v1, rot1, om1 = s1["q"], s1["v"], s1["rot"], s1["omega"]
        if not (ecal is not None and ecal.ready) or float(np.linalg.norm(q1)) > BRIDGE_RANGE_M + 2 * d_authority(mass):
            return None
        frame1 = spin_frame(rot1, om1, q1, e.config.task.approach_axis)
        goal = stage_goal(q1, v1, frame1, a2, mass, geometry=self.geometry, task=self.env.config.task)
        nxt = self.generator.planner.solve(q1, v1, goal["q_goal"], goal["v_goal"])
        if not nxt["feasible"] or nxt.get("terminal_position_error_m", 1.0) > 0.3:
            return None
        dq_b, dv_b = cal.residuals(); b1 = frame1["basis"]
        raw = lambda s: (b1.T @ dq_b * (s / STAGE_S), b1.T @ dv_b * (s / STAGE_S), 0.0)
        t1 = start["t"] + STAGE_S
        hit = self._scan(nxt, raw, attitude, STAGE_S, [ENTRY_MAX_SOLVES], brake_model)
        if hit is not None:
            dv_hit = self._dv_model(nxt, cal, hit["s"]) + hit["brake_dv"]
            cut = self._stage_cutoff(t1, start["dv"] + s1["dv"], dv_hit, hit["s"])   # the budget spent before the entry
            if cut is not None:
                true = cost(cut[0], cut[1]) - w.timeout_penalty
                return {"reward": float(true - (s1["phi"] if self.shaping else 0.0)), "obs": np.zeros(self.obs_dim, np.float32), "done": True,
                        "true": float(true), "entry": None, "event": cut[2]}
            macro = ecal.predict(hit["tau"], hit["dv_plan"], w, self.mission["deadline_s"] - (t1 + hit["s"]))["reward"]
            true = cost(hit["s"], dv_hit) + macro
            return {"reward": float(true - (s1["phi"] if self.shaping else 0.0)), "obs": np.zeros(self.obs_dim, np.float32), "done": True,
                    "true": float(true), "entry": hit}
        q2, v2 = plan_state(nxt, STAGE_S, mass)
        q2, v2 = q2 + b1.T @ dq_b, v2 + b1.T @ dv_b
        dv2 = self._dv_model(nxt, cal, STAGE_S)
        cut = self._stage_cutoff(t1, start["dv"] + s1["dv"], dv2)
        if cut is not None:                    # the deadline or the budget ends the episode inside stage 2
            true = cost(cut[0], cut[1]) - w.timeout_penalty
            return {"reward": float(true - (s1["phi"] if self.shaping else 0.0)), "obs": np.zeros(self.obs_dim, np.float32), "done": True,
                    "true": float(true), "entry": None, "event": cut[2]}
        if not 6.5 <= float(np.linalg.norm(q2)) <= 25.0:
            return None
        rot2, om2 = attitude(2 * STAGE_S)
        obs = self._observe(e, q2, v2, t1 + STAGE_S, start["dv"] + s1["dv"] + dv2, self.mission, rot2, om2,
                      {"position_residual_m": float(np.linalg.norm(dq_b)), "velocity_residual_m_s": float(np.linalg.norm(dv_b)), "held": False})
        r2 = cost(STAGE_S, dv2)
        shaped = r2 + (GAMMA_STAGE * self.potential(q2, rot2, om2) - s1["phi"] if self.shaping else 0.0)
        return {"reward": float(shaped), "obs": obs, "done": False, "true": float(r2), "q": q2}

    def _realized_attitude(self, trace):
        """Target attitude at a time after the decision: realized where the factual trajectory recorded it
        (the target's motion does not depend on the chaser's intent), else the constant-rate forecast from
        the latest realized state (the executor's own model)."""
        rec = {round(x["s"], 1): (x["rot"], x["omega"]) for x in trace}
        keys = sorted(rec)
        def attitude(s):
            k = round(float(s), 1)
            if k in rec:
                return rec[k]
            last = max([x for x in keys if x <= k] or [keys[0]])
            rot, om = rec[last]
            return rot @ so3_exp((k - last) * om), om
        return attitude

    def _labels(self, plan, a, start, cal, ecal, trace, entered, entry_rec, event, pi_act=None) -> dict:
        """EACC labels of one decision, computed after the factual stage from the same decision state.

        M1 executor option model (``predict_intent``). M2 hindsight execution anchoring: (i) exogenous -- every
        intent sees the realized target attitude (common random numbers: the tumbling target's motion, the
        unknown-inertia nutation included, is the same whatever the chaser does); (ii) endogenous -- the
        executed intent's realized tracking residual path (truth minus its plan, every 2 s) is transported to
        each candidate's plan, and an executed entry that ended in this decision transports its realized
        outcome residual to the candidates' entry outcomes. The raw factual prediction (calibration only) is
        returned for the learner's precision weight (M3)."""

        e, mass = self.env, self.env.chaser_parameters.mass
        basis = start["basis"]
        attitude = self._realized_attitude(trace)
        dq_b, dv_b = cal.residuals()
        raw = lambda s: (basis.T @ dq_b * (s / STAGE_S), basis.T @ dv_b * (s / STAGE_S), 0.0)
        s_valid = (entered["t"] - start["t"]) if entered is not None else STAGE_S
        nodes = {}
        for x in trace:
            if x["s"] <= s_valid + 1e-6:
                qp, vp = plan_state(plan, x["s"], mass)
                nodes[round(x["s"], 1)] = (x["q"] - qp, x["v"] - vp, (x["dv"] - start["dv"]) - self._dv_model(plan, cal, x["s"]))
        keys = sorted(nodes)
        def anchored(s):
            k = round(float(s), 1)
            return nodes[k] if k in nodes else nodes[max([x for x in keys if x <= k] or [keys[0]])]
        self._model_t0 = start["t"]
        fact = self.predict_intent(plan, a, start, cal, ecal, raw, attitude)
        offset, anchored_outcome = 0.0, False
        if entry_rec is not None and entry_rec["t"] >= start["t"] - 1e-6 and terminated_event(event) and ecal.ready:
            anchored_outcome = True          # the executed entry's realized outcome residual is transported below
            offset = float(entry_rec["G"] - ecal.predict(entry_rec["tau"], entry_rec["dv_plan"], e.config.reward,
                                                         self.mission["deadline_s"] - entry_rec["t"])["reward"])
        # systematic error of the entry outcome model, measured predict-then-observe on this environment's own
        # executed entries (None until five are known). It is reported with every transition; the learner applies the
        # *current* value to every stored entry label that rests on the outcome model alone (kind 2), so old labels are
        # re-weighted when a bias appears. Entry labels whose outcome was anchored to this decision's realized entry
        # (kind 1) and stage labels (kind 0) carry no outcome-model bias.
        res = self.entry_residuals[-ENTRY_CAL_MIN:]
        entry_bias = float(np.mean(res)) if len(res) >= 5 else None
        cands = []
        for c in self._plan_candidates(start["q"], start["v"], start["frame"], a):
            p = self.predict_intent(c["plan"], c["a"], start, cal, ecal, anchored, attitude, entry_offset=offset)
            if p["done"]:                     # an entry (outcome model: kind 1 anchored / 2 not) or a deadline / budget end (kind 0)
                cand = {"a": c["a"], "reward": p["reward"], "next_obs": np.zeros(self.obs_dim, np.float32), "done": True,
                        "kind": (1 if anchored_outcome else 2) if p.get("entry") is not None else 0}
            elif c["plan"].get("terminal_position_error_m", 1.0) <= 0.3 and 6.5 <= p["r_end"] <= 25.0:   # model domain
                cand = {"a": c["a"], "reward": p["reward"], "next_obs": p["obs"], "done": False, "kind": 0, "s2": None}
                if self.entry_stages >= 2 and self.label_policy is not None:
                    a2, logp = self.label_policy(p["obs"])
                    p2 = self.predict_second_stage(p, np.asarray(a2, dtype=np.float64), start, cal, ecal, attitude)
                    if p2 is not None:
                        cand["s2"] = {"a": np.asarray(a2, np.float32), "logp": float(logp), "reward": p2["reward"], "next_obs": p2["obs"],
                                      "done": bool(p2["done"]), "kind": 2 if p2.get("entry") is not None else 0}
                        if self.diagnostics:
                            cand["stage2"] = p2
            else:
                continue
            if self.diagnostics:      # label audit against truth branches: the raw (calibration-only) prediction beside it
                cand["anchored"] = p; cand["raw"] = self.predict_intent(c["plan"], c["a"], start, cal, ecal, raw, attitude)
            cands.append(cand)
        diag = {"resampled": int(self.last_resampled), "fact_entry": fact.get("entry") is not None, "fact_cutoff": bool(fact["done"]) and fact.get("entry") is None, "entered": entered is not None, "entry_offset": offset, "candidates": len(cands),
                "entry_bias": entry_bias, "entry_records": len(ecal.rows)}
        # T2.12 (EACC relative labels): the executed intent through exactly the candidates' treatment (same function, the
        # same anchored residual path, entry rule with the brake model, entry outcome offset). A candidate's relative label
        # is its prediction minus this one, so for the executed intent itself the difference is zero by construction.
        same = self.predict_intent(plan, a, start, cal, ecal, anchored, attitude, entry_offset=offset)
        same_kind = (1 if anchored_outcome else 2) if same.get("entry") is not None else 0
        diag["fact_same_entry"] = same.get("entry") is not None
        ref = None
        if self.ref_labels and pi_act is not None:      # env 0.12.0: the policy's own intent, same treatment as the candidates
            pa = np.clip(np.asarray(pi_act, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
            g = stage_goal(start["q"], start["v"], start["frame"], pa, mass, geometry=self.geometry, task=self.env.config.task)
            rp = self.generator.planner.solve(start["q"], start["v"], g["q_goal"], g["v_goal"])
            if rp["feasible"] and rp.get("terminal_position_error_m", 1.0) <= 0.3:
                pr = self.predict_intent(rp, pa, start, cal, ecal, anchored, attitude, entry_offset=offset)
                if pr["done"] or 6.5 <= pr["r_end"] <= 25.0:
                    ref = {"a": pa.astype(np.float32), "reward": pr["reward"], "next_obs": pr.get("obs", np.zeros(self.obs_dim, np.float32)),
                           "done": bool(pr["done"]), "kind": ((1 if anchored_outcome else 2) if pr.get("entry") is not None else 0)}
        diag["ref"] = ref is not None
        if self.diagnostics:
            diag["anchored_fact"] = self.predict_intent(plan, a, start, cal, ecal, anchored, attitude, entry_offset=offset, brake_model=False)
            diag["raw_fact"] = fact
            diag["same_fact"] = same
        return {"fact": {"reward": fact["reward"], "next_obs": fact.get("obs", np.zeros(self.obs_dim, np.float32)), "done": bool(fact["done"]),
                         "same": {"reward": same["reward"], "next_obs": same.get("obs", np.zeros(self.obs_dim, np.float32)),
                                  "done": bool(same["done"]), "kind": same_kind}},
                "cands": cands, "ref": ref, "diag": diag}

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
        if self._entry_open is not None:
            self._entry_open["G"] += float(r)
        if self._trace:
            q, v = relative_inertial(self.env); t = self.env.target_state
            self._trace.append({"s": self.env.time_s - self._trace[0]["t_abs"], "q": q, "v": v,
                                "dv": self.env._delta_v, "rot": t.rotation.copy(), "omega": t.omega.copy()})
        if event is None:
            m = self._metrics()
            # mode bookkeeping (T1-P after_step with this module's bridge rule)
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

    def candidate_actions(self, action, goal_of=None) -> list[np.ndarray]:
        """K counterfactual intents: the decisive alternatives of the scenario (no along-sweep motion, the
        opposite along-sweep sense, a slower close), one anywhere in the box, the rest local (radius 0.5).
        A structured alternative within 0.15 of the executed intent is replaced by a global one.
        With ``cand_min_sep`` > 0 and ``goal_of`` (intent -> stage-goal velocity) the candidates are separated in the
        executor's terms instead (``_separated_candidates``)."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
        if self.cand_min_sep > 0 and goal_of is not None:
            return self._separated_candidates(a, goal_of)
        out = []
        for alt in ([a[0], 0.0], [a[0], -a[1] if abs(a[1]) > 0.25 else -0.5], [a[0] - 0.6, a[1]]):
            alt = np.clip(np.asarray(alt, dtype=np.float64), -1.0, 1.0)
            out.append(alt if np.linalg.norm(alt - a) > 0.15 else self.rng.uniform(-1.0, 1.0, size=2))
        out.append(self.rng.uniform(-1.0, 1.0, size=2))
        while len(out) < self.candidates:
            step = self.rng.normal(size=2); step = CANDIDATE_RADIUS * step / max(np.linalg.norm(step), 1e-9) * self.rng.uniform(0.3, 1.0)
            out.append(np.clip(a + step, -1.0, 1.0))
        return out[:self.candidates]

    def _separated_candidates(self, a, goal_of) -> list[np.ndarray]:
        """The same three kinds of candidates as ``candidate_actions``; a draw whose stage goal lies within
        ``cand_min_sep`` of the executed intent's or an accepted candidate's is redrawn (CANDIDATE_RESAMPLES at most;
        the last draw is kept so that every decision has K candidates). ``last_resampled`` counts the redraws."""
        def draw(kind):
            if kind == 'global':
                return self.rng.uniform(-1.0, 1.0, size=2)
            step = self.rng.normal(size=2); step = CANDIDATE_RADIUS * step / max(np.linalg.norm(step), 1e-9) * self.rng.uniform(0.3, 1.0)
            return np.clip(a + step, -1.0, 1.0)
        props = [('structured', np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0))
                 for x in ([a[0], 0.0], [a[0], -a[1] if abs(a[1]) > 0.25 else -0.5], [a[0] - 0.6, a[1]])]
        props.append(('global', None))
        props += [('local', None)] * max(self.candidates - 4, 0)
        goals = [np.asarray(goal_of(a), dtype=np.float64)]
        out, redraws = [], 0
        for kind, c in props[:self.candidates]:
            if c is None:
                c = draw(kind)
            fresh = 'global' if kind == 'structured' else kind
            g = np.asarray(goal_of(c), dtype=np.float64)
            tries = 0
            while min(float(np.linalg.norm(g - x)) for x in goals) < self.cand_min_sep and tries < CANDIDATE_RESAMPLES:
                c = draw(fresh); g = np.asarray(goal_of(c), dtype=np.float64); tries += 1
            redraws += tries
            out.append(c); goals.append(g)
        self.last_resampled = redraws
        return out

    def _plan_candidates(self, q, v, frame, action) -> list[dict]:
        if self.candidates <= 0:
            return []
        out, mass = [], self.env.chaser_parameters.mass
        goal_of = lambda x: stage_goal(q, v, frame, x, mass, geometry=self.geometry, task=self.env.config.task)["v_goal"]
        self.last_resampled = 0
        for ac in self.candidate_actions(action, goal_of if self.cand_min_sep > 0 else None):
            goal = stage_goal(q, v, frame, ac, mass, geometry=self.geometry, task=self.env.config.task)
            plan = self.generator.planner.solve(q, v, goal["q_goal"], goal["v_goal"])
            if not plan["feasible"]:
                continue
            out.append({"a": ac.astype(np.float32), "plan": plan})
        return out

    def _macro_to_end(self):
        """A step that starts in the bridge or descent (the G* reference row): fly until the end."""
        e, g = self.env, self.generator
        self._trace = []
        t0, phi0, reward, event = e.time_s, self._phi(), 0.0, None
        while event is None:
            r, event, _ = self._advance(); reward += r
            if g.mode in ("stage", "hold"):
                break
        terminated = event in ("success", "unsafe", "timeout", "resource_exhausted")
        if terminated:
            self._close_bridge(event)
        self.return_true += reward
        shaped = reward + ((GAMMA_STAGE * (0.0 if terminated else self._phi()) - phi0) if self.shaping else 0.0)
        q1, _ = relative_inertial(e)
        info = {"event": event, "true_reward": reward, "stage_s": e.time_s - t0, "held": g.mode == "hold", "mode": g.mode,
                "plan_status": "macro", "rate_limited": False, "planned_dv": None, "executed_dv": None,
                "model": {"fact": None, "cands": [], "diag": {"reason": "macro"}}, "time_s": e.time_s, "delta_v": e._delta_v,
                "range_m": float(np.linalg.norm(q1)), "mission": dict(self.mission), "far_field_controls": self.far_field_controls}
        return self.observation(), float(shaped), terminated, event == "numerical_failure", info

    # ---------------------------------------------------------------- step
    def step(self, action):
        e, g, mass = self.env, self.generator, self.env.chaser_parameters.mass
        if g.mode in ("bridge", "descent"):
            return self._macro_to_end()
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
        pref = None
        if self.pref and self.entry_calibration.ready:      # T2.10 / T2.11: before the stage is flown (the decision's own state)
            # candidates: the grid, the executed intent and (T2.11) the policy's own deterministic intent at this decision
            pi = a if self.policy_action is None else np.clip(np.asarray(self.policy_action, np.float64).reshape(-1)[:2], -1.0, 1.0)
            # (T2.10 / T2.11 opportunity judgements of the grid, the executed and the policy's intents)
            acts = [np.asarray(x, np.float64) for x in PREF_GRID] + [a, pi]
            feats = [self.opportunity_features(x) for x in acts]
            pref = {"a": [x.tolist() for x in acts], "v": [f["value"] for f in feats], "f": feats}
        pi_act = None if self.policy_action is None else np.clip(np.asarray(self.policy_action, np.float64).reshape(-1)[:2], -1.0, 1.0)
        self.policy_action = None
        q, v = relative_inertial(e)
        frame = frame_now(e, q); basis = frame["basis"]
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
        start = {"rotation": e.target_state.rotation.copy(), "omega": e.target_state.omega.copy(), "t": t0, "dv": dv0, "phi": phi0,
                 "q": q, "v": v, "basis": basis, "frame": frame}
        # the model of this decision uses the calibrations as they were at the decision (not this stage's own truth)
        model_on = bool(plan["feasible"] and self.candidates > 0 and self.calibration.ready)
        if model_on:
            cal = self.calibration.frozen().regime(float(np.linalg.norm(q)) <= NEAR_REGIME_M)
            ecal = self.entry_calibration.frozen()
        self._stage_entry = None
        self._trace = [{"s": 0.0, "t_abs": t0, "q": q.copy(), "v": v.copy(), "dv": dv0, "rot": start["rotation"], "omega": start["omega"]}]
        reward, event, steps = 0.0, None, 0
        while True:
            r, event, info = self._advance(); reward += r; steps += 1
            if event is not None:
                break
            if g.mode in ("bridge", "descent"):
                continue                       # terminal macro-step: no decision until it ends
            if g.mode == "hold" and steps * 2.0 >= 2.0 and (g.last_status or "").startswith("hold:stage_deviation"):
                break                          # deviation: ask again now
            if e.time_s - t0 >= STAGE_S - 1e-6 and g.mode in ("stage", "hold"):
                break
        q1, v1 = relative_inertial(e)
        held = g.mode == "hold"
        last = {"held": held}
        stage_complete = abs(e.time_s - t0 - STAGE_S) < 1e-6 and event is None and not held and g.mode == "stage"
        if plan["feasible"]:
            q_plan, v_plan = plan_state(plan, e.time_s - t0, mass)
            last["position_residual_m"] = float(np.linalg.norm(q1 - q_plan))
            last["velocity_residual_m_s"] = float(np.linalg.norm(v1 - v_plan))
            if stage_complete:
                self.calibration.add(float(plan["delta_v_m_s"]), float(e._delta_v - dv0), basis @ (q1 - q_plan), basis @ (v1 - v_plan),
                                     near=float(np.linalg.norm(q)) <= NEAR_REGIME_M)
        entered = self._stage_entry if (self._stage_entry is not None and self._stage_entry["t"] - t0 <= STAGE_S + 1e-6) else None
        entry_rec = self._entry_open
        self.last = last
        self.stats["holds"] += int(held)
        terminated = event in ("success", "unsafe", "timeout", "resource_exhausted")
        truncated = event == "numerical_failure"
        if terminated:
            self._close_bridge(event)
        # EACC labels: only where the executed intent's outcome covers the same interval as its siblings' --
        # its stage completed, or it opened the entry inside the stage
        model = {"fact": None, "cands": [], "diag": {"reason": "off"}}
        if model_on and (stage_complete or entered is not None) and ecal.ready:
            model = self._labels(plan, a, start, cal, ecal, self._trace, entered, entry_rec, event, pi_act=pi_act)
            model["diag"]["reason"] = "entry" if entered is not None else "stage"
        elif model_on:
            model["diag"]["reason"] = "not_comparable" if ecal.ready else "entry_calibration"
        self.return_true += reward
        shaped = reward
        if self.shaping:
            phi1 = 0.0 if terminated else self._phi()
            shaped = reward + GAMMA_STAGE * phi1 - phi0
        n_entry = sum(c["done"] for c in model["cands"])
        self.stats["entry_labels"] = self.stats.get("entry_labels", 0) + n_entry
        self.stats["model_candidates"] = self.stats.get("model_candidates", 0) + len(model["cands"])
        reason = model["diag"]["reason"]
        self.stats["model_reason"] = self.stats.get("model_reason", {})
        self.stats["model_reason"][reason] = self.stats["model_reason"].get(reason, 0) + 1
        if g.mode in ("bridge", "descent"):
            self.stats["macro"] = g.mode
        info = {"event": event, "true_reward": reward, "stage_s": e.time_s - t0, "held": held, "mode": g.mode,
                "plan_status": plan["status"], "rate_limited": goal["rate_limited"], "planned_dv": plan.get("delta_v_m_s"), "executed_dv": e._delta_v - dv0,
                "model": model, "pref": pref, "time_s": e.time_s, "delta_v": e._delta_v,
                "range_m": float(np.linalg.norm(q1)), "mission": dict(self.mission), "far_field_controls": self.far_field_controls}
        return self.observation(), float(shaped), terminated, truncated, info


__all__ = ["T2Mission", "T2StageEnv", "T2_VERSION", "STAGE_S", "OBS_DIM", "OBS_DIM_PRIV", "OBS_PROFILES", "EXECUTOR_FEATURES", "PLANNER_FEATURES", "PREF_GRID", "observe_raw", "GAMMA_STAGE", "InfoEnv", "stage_goal", "spin_frame", "frame_now", "target_velocity", "action_for_velocity", "latitude_velocity",
           "d_authority", "observe", "Calibration", "t2_env_config"]
