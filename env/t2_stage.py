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

Mission (T1-B assets, soft fuel): urgent deadline 170-210 s or economy 280-340 s (the episode times
out at the deadline, -100), fuel -20 per m/s of delta-v, a 3.0 m/s safety budget (resource
exhausted -100), -0.05 per s, capture +100, unsafe -150.

Stage model (for the coupled learner, ``candidates`` > 0): at each decision the local planner also
plans K candidate actions; each plan is corrected by this environment's *execution calibration*
(least squares of executed vs planned delta-v, mean end-position / end-velocity residual in the stage
basis, fitted on its own executed stages) and turned into a predicted reward and next observation.
At the end of the stage the executed action's calibrated prediction is compared with the truth;
the candidates of that stage are marked usable only if it matched (``gate``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np

from dynamics.lie import so3_exp
from env.t1_env import T1Env, T1EnvConfig, T1Reward
from env.t1_n_task import make_conditional_gstar
from env.t1_p_stage import LocalStagePlanner, T1PGenerator, plan_state, relative_inertial
from env.t1_task import T1InitialDistribution

T2_VERSION = "t2-stage-0.1"
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
#: stage-model gate (executed action: calibrated prediction vs truth at the stage end)
GATE_POSITION_M = 0.5
GATE_VELOCITY_M_S = 0.05
GATE_DELTA_V_M_S = 0.03
CALIBRATION_MIN_STAGES = 10
CALIBRATION_WINDOW = 60
CANDIDATE_RADIUS = 0.5
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


@dataclass(frozen=True)
class T2Mission:
    urgent_deadline_s: tuple[float, float] = (170.0, 210.0)
    economy_deadline_s: tuple[float, float] = (280.0, 340.0)
    fuel_cost_per_m_s: float = 20.0
    budget_m_s: float = 3.0

    def sample(self, opening: int) -> dict[str, Any]:
        rng = np.random.default_rng(np.random.SeedSequence([int(opening), 290021]))
        urgent = bool(rng.integers(0, 2))
        d = rng.uniform(*(self.urgent_deadline_s if urgent else self.economy_deadline_s))
        return {"deadline_s": float(round(d, 1)), "urgent": urgent, "budget_m_s": self.budget_m_s}


def t2_env_config(mission: T2Mission = T2Mission()) -> T1EnvConfig:
    base = T1EnvConfig()
    initial = replace(T1InitialDistribution(), range_m=(12.0, 16.0), inertial_speed_max_m_s=0.05)
    reward = replace(T1Reward(), delta_v_cost_per_m_s=mission.fuel_cost_per_m_s)
    return replace(base, initial=initial, reward=reward, max_time_s=max(mission.economy_deadline_s))


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


def target_velocity(frame: dict[str, Any], action: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
    b = frame["basis"]
    v_c = CLOSING_M_S[0] + CLOSING_M_S[1] * a[0]
    return -v_c * b[0] + latitude_velocity(frame) * b[1] + AZIMUTH_M_S * a[1] * b[2]


def action_for_velocity(frame: dict[str, Any], v_c: float, v_z: float) -> np.ndarray:
    """Action for a closing speed v_c and an azimuthal speed v_z (towards the oncoming port)."""
    return np.clip([(v_c - CLOSING_M_S[0]) / CLOSING_M_S[1], v_z / AZIMUTH_M_S], -1.0, 1.0)


def stage_goal(q: np.ndarray, v: np.ndarray, frame: dict[str, Any], action: np.ndarray, mass: float,
               h: float = STAGE_S) -> dict[str, Any]:
    d = d_authority(mass, h)
    v_target = target_velocity(frame, action)
    dv = v_target - v
    limit = 2.0 * d / h
    n = float(np.linalg.norm(dv))
    limited = n > limit
    if limited:
        dv = dv * (limit / n)
    v_goal = v + dv
    return {"a": np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1, 1), "v_target": v_target, "d": d,
            "rate_limited": bool(limited), "q_goal": q + h * (v + v_goal) / 2.0, "v_goal": v_goal}


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
        bridge / 180.0, float(np.clip(margin / 0.1, -3.0, 3.0)), np.cos(np.radians(f["lam_deg"])), f["beta_deg"] / 90.0,
        *(b @ f["port"]),
    ], dtype=np.float32)


@dataclass
class Calibration:
    """Execution calibration of the stage planner on this environment's executed stages."""

    rows: list = field(default_factory=list)

    def add(self, planned_dv: float, executed_dv: float, dq_basis: np.ndarray, dv_basis: np.ndarray) -> None:
        self.rows.append((planned_dv, executed_dv, np.asarray(dq_basis), np.asarray(dv_basis)))
        self.rows = self.rows[-CALIBRATION_WINDOW:]

    @property
    def ready(self) -> bool:
        return len(self.rows) >= CALIBRATION_MIN_STAGES

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


class T2StageEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, openings: list[int] | None = None, seed_base: int = 1_000_000, candidates: int = 0,
                 mission: T2Mission = T2Mission(), shaping: bool = True, seed: int = 0) -> None:
        self.mission_distribution = mission
        self.base_config = t2_env_config(mission)
        self.env = T1Env(self.base_config)
        self.info_env = InfoEnv(self.env)
        self.openings, self.seed_base, self.candidates, self.shaping = openings, int(seed_base), int(candidates), bool(shaping)
        self.rng = np.random.default_rng(seed)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.episode = 0
        self.calibration = Calibration()
        self.gamma_stage = 0.999 ** (STAGE_S / 2.0)
        self.far_field_controls = 0
        original = self.env.controller.command

        def command(*args, **kwargs):
            wrench, diagnostics = original(*args, **kwargs)
            force = self._far_field_force()
            if force is not None:
                wrench = np.array(wrench, dtype=np.float64); wrench[3:] = force
                self.far_field_controls += 1
            return wrench, diagnostics
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
        return observe(self.env, q, v, self.env.time_s, self.env._delta_v, self.mission, t.rotation, t.omega, self.last)

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
        self.env.config = replace(self.base_config, max_time_s=self.mission["deadline_s"])
        self.env.reset(opening)
        self.info_env = InfoEnv(self.env)
        self.generator = T1PGenerator(planner=LocalStagePlanner(self.env.config.task, self.env.chaser_parameters.mass,
                                                                PLAN_FORCE_N, horizon_s=STAGE_S, nodes=STAGE_NODES),
                                      bridge=make_conditional_gstar(self.info_env, deadline_margin_s=BRIDGE_DEADLINE_MARGIN_S),
                                      mass=self.env.chaser_parameters.mass, replan_deviation_m=STAGE_DEVIATION_HOLD_M)
        self.last = {}
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
                return True
        return False

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

    def _reference(self, env, metrics):
        return self.generator(self.info_env, metrics)

    def _advance(self) -> tuple[float, str | None, dict]:
        """One 2 s truth step with mode bookkeeping; returns (true reward, event, info)."""

        g = self.generator
        if g.mode == "bridge":
            g.continue_step(self.info_env)
        r, term, trunc, info = self.env.step(reference=self._reference)
        event = info.get("event")
        if event is None and self.env._delta_v >= self.mission["budget_m_s"]:
            event = "resource_exhausted"; r -= self.env.config.reward.timeout_penalty
        if event is None:
            m = self._metrics()
            # mode bookkeeping (T1-P after_step with this module's bridge rule)
            g.event = None
            if g.descending(self.info_env, m):
                g.mode = "descent"; g.bridge.plan = None
            elif g.mode == "descent":
                g.mode, g.hold_position = "hold", None; g._enter_hold(self.info_env, "hold:left_descent")
            else:
                if g.mode == "stage" and g.stage_deviation(self.info_env) > g.replan_deviation_m:
                    g.hold_position = None; g._enter_hold(self.info_env, "hold:stage_deviation")
                if g.mode in ("stage", "hold"):
                    self._try_bridge(m)
        return float(r), event, info

    def _plan_candidates(self, q, v, frame, action) -> list[dict]:
        if self.candidates <= 0 or not self.calibration.ready:
            return []
        out, mass = [], self.env.chaser_parameters.mass
        a = np.asarray(action, dtype=np.float64)
        for _ in range(self.candidates):
            step = self.rng.normal(size=2); step = CANDIDATE_RADIUS * step / max(np.linalg.norm(step), 1e-9) * self.rng.uniform(0.3, 1.0)
            ac = np.clip(a + step, -1.0, 1.0)
            goal = stage_goal(q, v, frame, ac, mass)
            plan = self.generator.planner.solve(q, v, goal["q_goal"], goal["v_goal"])
            if not plan["feasible"]:
                continue
            out.append({"a": ac.astype(np.float32), "plan": plan})
        return out

    def _predicted(self, plan, basis, t0, dv0, phi0) -> tuple[float, np.ndarray, float, float]:
        mass = self.env.chaser_parameters.mass
        q1, v1, dv = self.calibration.predict(plan, basis, mass)
        t = self.env.target_state
        rot1 = t.rotation @ so3_exp(STAGE_S * t.omega)
        reward = -self.env.config.reward.time_cost_per_s * STAGE_S - self.env.config.reward.delta_v_cost_per_m_s * dv
        dq, dvv = self.calibration.residuals()
        obs = observe(self.env, q1, v1, t0 + STAGE_S, dv0 + dv, self.mission, rot1, t.omega,
                      {"position_residual_m": float(np.linalg.norm(dq)), "velocity_residual_m_s": float(np.linalg.norm(dvv)), "held": False})
        shaped = reward + (self.gamma_stage * self.potential(q1, rot1, t.omega) - phi0 if self.shaping else 0.0)
        return float(shaped), obs, float(np.linalg.norm(q1)), dv

    # ---------------------------------------------------------------- step
    def step(self, action):
        e, g, mass = self.env, self.generator, self.env.chaser_parameters.mass
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
        q, v = relative_inertial(e)
        frame = frame_now(e, q); basis = frame["basis"]
        goal = stage_goal(q, v, frame, a, mass)
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
        factual = None
        if plan["feasible"] and self.calibration.ready:
            factual = self.calibration.predict(plan, basis, mass)
        cands = self._plan_candidates(q, v, frame, a) if plan["feasible"] else []
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
        gate = {"usable": False, "reason": "no_prediction"}
        stage_complete = abs(e.time_s - t0 - STAGE_S) < 1e-6 and event is None and not held and g.mode == "stage"
        if plan["feasible"]:
            q_plan, v_plan = plan_state(plan, e.time_s - t0, mass)
            last["position_residual_m"] = float(np.linalg.norm(q1 - q_plan))
            last["velocity_residual_m_s"] = float(np.linalg.norm(v1 - v_plan))
            if stage_complete:
                self.calibration.add(float(plan["delta_v_m_s"]), float(e._delta_v - dv0), basis @ (q1 - q_plan), basis @ (v1 - v_plan))
        if factual is not None:
            if not stage_complete:
                gate = {"usable": False, "reason": "stage_not_complete"}
            else:
                eq, ev, ed = float(np.linalg.norm(factual[0] - q1)), float(np.linalg.norm(factual[1] - v1)), abs(factual[2] - (e._delta_v - dv0))
                ok = eq <= GATE_POSITION_M and ev <= GATE_VELOCITY_M_S and ed <= GATE_DELTA_V_M_S
                gate = {"usable": bool(ok), "reason": "ok" if ok else "factual_mismatch", "position_error_m": eq,
                        "velocity_error_m_s": ev, "delta_v_error_m_s": ed,
                        "weight": float(max(0.0, 1.0 - max(eq / GATE_POSITION_M, ev / GATE_VELOCITY_M_S, ed / GATE_DELTA_V_M_S)))}
        self.last = last
        self.stats["holds"] += int(held)
        terminated = event in ("success", "unsafe", "timeout", "resource_exhausted")
        truncated = event == "numerical_failure"
        self.return_true += reward
        shaped = reward
        if self.shaping:
            phi1 = 0.0 if terminated else self._phi()
            shaped = reward + (0.999 ** ((e.time_s - t0) / 2.0)) * phi1 - phi0
        out_cands = []
        if gate["usable"]:
            for c in cands:
                if c["plan"].get("terminal_position_error_m", 1.0) > 0.3:
                    continue
                rh, obs, r_end, dv = self._predicted(c["plan"], basis, t0, dv0, phi0)
                if not (7.0 <= r_end <= 25.0):
                    continue
                out_cands.append({"a": c["a"], "reward": rh, "next_obs": obs, "weight": gate["weight"]})
        if g.mode in ("bridge", "descent"):
            self.stats["macro"] = g.mode
        info = {"event": event, "true_reward": reward, "stage_s": e.time_s - t0, "held": held, "mode": g.mode,
                "plan_status": plan["status"], "rate_limited": goal["rate_limited"], "planned_dv": plan.get("delta_v_m_s"), "executed_dv": e._delta_v - dv0,
                "gate": gate, "stage_model": out_cands, "time_s": e.time_s, "delta_v": e._delta_v,
                "range_m": float(np.linalg.norm(q1)), "mission": dict(self.mission)}
        return self.observation(), float(shaped), terminated, truncated, info


__all__ = ["T2Mission", "T2StageEnv", "T2_VERSION", "STAGE_S", "OBS_DIM", "InfoEnv", "stage_goal", "spin_frame", "frame_now", "target_velocity", "action_for_velocity", "latitude_velocity",
           "d_authority", "observe", "Calibration", "t2_env_config"]
