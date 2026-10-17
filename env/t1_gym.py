"""Gymnasium wrapper of T1 + G* (``GSTAR_VERSION``, 0.3.2) for SAC (M0 and M1 use it unchanged).

One ``step`` is one 2 s decision: the action becomes a G* request, the
generator decides (keep / re-plan / descent / hold), then 20 MPC control
steps of 0.1 s run under RK45 truth (``T1Env.step``).

Action ``a`` in [-1, 1]^3 (interface v4; a 2-D action is read with a[2] = 0):
  a[0] -> arrival-time request. With a live plan the request is relative:
          tau = remaining + shift(a[0]). Interface v2: |a[0]| <= ``keep_band``
          (0.5) asks to keep the plan (shift 0); beyond it the shift grows
          linearly to +-``tau_shift_s`` at the bound. v1 kept only for
          |a[0]| <= 0.1, and SAC at the action bound re-planned every 2 s
          (round-1 seed 0); "keep" is now half of the action range.
          Without a plan it is absolute: tau = ``tau_max_s`` (a[0] + 1) / 2,
          and tau <= 2 s asks for the earliest feasible arrival.
  a[1] -> descent pace kappa = kappa_min + (kappa_max - kappa_min)(a[1] + 1)/2.
  a[2] -> inward entry speed of the *next new* G* plan, v_in = 0.06 + 0.04 a[2] m/s
          (0.02..0.10; a terminal equality constraint of the planner). Applied only when a
          plan is actually (re)planned or recovered; a kept plan keeps its own v_in, and a
          different request never forces a re-plan. a[2] = 0 is G*'s former fixed 0.06.

Observation (``OBSERVATION_NAMES``): geometry and rates of the current truth
state as an onboard estimate would give them, plus the generator memory that
makes the decision process (approximately) Markov: whether a plan is live,
its remaining time, the last effective and requested arrival times, descent
and hold flags, and the last kappa. Interface v3 appends the current tracking error
against the live plan (position / 1 m, velocity / 0.3 m/s; 0 without a live plan), so the
policy and the critic can see whether a plan is being flown or about to be re-planned.
Interface v4 appends the live plan's entry speed ((v_in - 0.06) / 0.04, 0 without a plan).

Episode end: ``success`` / ``unsafe`` / ``timeout`` are terminal (the 300 s
horizon is part of the task, time is observed, the timeout penalty is a real
outcome); a numerical failure is a truncation.

``info["execution"]`` carries what the MPC actually did over the 2 s:
maximum deviation of the state from the live plan, maximum QP slack on solved
steps, fraction of force-saturated steps, zero-input fallbacks, whether the
request had to slip and by how much, and the plan solve time. M0 logs it; M1
may feed it into its training update. None of it is future truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from env.t1_env import T1Env, T1EnvConfig
from env.t1_gstar import GStarRendezvous, _plan_position, make_rendezvous

OBSERVATION_NAMES = (
    "range_10m", "sin_latitude", "cos_latitude", "sin_phase", "cos_phase", "rho_perp_10m",
    "port_los_180deg", "port_latitude_90deg", "omega_2p7deg_s", "fov_margin_50deg", "time_300s",
    "radial_closing_0p3m_s", "tangential_speed_0p5m_s", "in_cone", "mouth_region",
    "has_plan", "remaining_160s", "tau_effective_160s", "tau_request_160s", "descent", "hold", "kappa",
    # interface v3 (appended, earlier indices unchanged): the current state against the live
    # G* plan at the current time, ECI relative frame; 0 without a live plan
    "track_position_error_1m", "track_velocity_error_0p3m_s",
    # interface v4 (appended): inward entry speed of the live plan, (v_in - 0.06) / 0.04; 0
    # without a plan (has_plan tells the two apart)
    "plan_entry_speed",
)


def _plan_velocity(plan: dict[str, Any], s: float, mass: float) -> np.ndarray:
    """Plan velocity ``s`` seconds after its start (piecewise-constant force)."""

    v, f, dt = plan["v"], plan["f"], plan["dt"]
    n = len(f)
    if s >= n * dt:
        return np.asarray(v[-1], dtype=np.float64)
    i = int(np.clip(np.floor(max(s, 0.0) / dt), 0, n - 1))
    return v[i] + f[i] / mass * (max(s, 0.0) - i * dt)


def end_phase(m) -> str:
    """Where an episode ended, from the truth state (descriptive label, not a rule)."""

    if m.mouth_region or m.range_m < 6.0:
        return "near_field"
    if m.in_cone and m.range_m < 7.0:
        return "entry"
    return "far_field"


@dataclass(frozen=True)
class T1GymConfig:
    tau_max_s: float = 160.0
    tau_shift_s: float = 30.0
    keep_tolerance_s: float = 3.0
    #: interface v2: fraction of the a[0] range (around 0) that asks to keep the live plan
    keep_band: float = 0.5
    kappa_min: float = 0.3
    kappa_max: float = 1.0
    #: interface v4: a[2] -> inward entry speed of a new G* plan, v_in = nominal + span * a[2]
    #: (0.02..0.10 m/s; a[2] = 0 is G*'s fixed 0.06 m/s)
    entry_speed_nominal_m_s: float = 0.06
    entry_speed_span_m_s: float = 0.04
    #: scale of the ``omega_2p7deg_s`` observation (deg/s); T1 / T1-L use 2.7
    omega_scale_deg_s: float = 2.7
    #: training openings: seed_base + episode index (disjoint from R0 2900xx,
    #: R1 2910xx and the formal block >= 300000 when seed_base >= 1e6)
    seed_base: int = 1_000_000


class T1GStarEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: T1GymConfig | None = None, env_config: T1EnvConfig | None = None,
                 openings: list[int] | None = None) -> None:
        self.config = config or T1GymConfig()
        self.env = T1Env(env_config or T1EnvConfig())
        self.openings = openings
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(OBSERVATION_NAMES),), dtype=np.float32)
        self.episode = 0
        self.current_opening: int | None = None
        self.generator: GStarRendezvous | None = None
        self._last_request = 0.0
        self._last_kappa = 0.0
        self._step_log: list[tuple[float, bool, int, bool]] = []
        self.first_wrench: np.ndarray | None = None
        original = self.env.controller.command

        def command(*args: Any, **kwargs: Any):
            wrench, diagnostics = original(*args, **kwargs)
            if not self._step_log:
                # first control of the decision, as commanded (read-only record)
                self.first_wrench = np.array(wrench, dtype=np.float64)
            force = np.asarray(wrench, dtype=np.float64)[3:]
            self._step_log.append((
                float(diagnostics.maximum_slack) if not diagnostics.used_zero_fallback else np.nan,
                bool(np.max(np.abs(force)) >= 0.99 * self.env.config.max_force_per_axis_n),
                int(diagnostics.used_zero_fallback),
                diagnostics.status in ("optimal", "optimal_inaccurate", "solved", "solved_inaccurate"),
            ))
            return wrench, diagnostics

        self.env.controller.command = command  # type: ignore[method-assign]

    # ------------------------------------------------------------------ #
    def opening_seed(self) -> int:
        if self.openings is not None:
            return int(self.openings[self.episode % len(self.openings)])
        return int(self.config.seed_base + self.episode)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        opening = int(options["opening"]) if options and "opening" in options else self.opening_seed()
        self.episode += 1
        self.current_opening = opening
        self.env.reset(opening)
        kappa_mid = 0.5 * (self.config.kappa_min + self.config.kappa_max)
        self.generator = make_rendezvous(self.env, kappa_mid, replan_tau_tolerance_s=self.config.keep_tolerance_s)
        self._last_request, self._last_kappa = 0.0, kappa_mid
        return self._observation(), {"opening": opening}

    def shift_of(self, a0: float) -> float:
        """Arrival-time shift (s) asked by a[0] when a plan is live (interface v2)."""

        band, size = self.config.keep_band, abs(float(a0))
        if size <= band:
            return 0.0
        return float(np.sign(a0)) * self.config.tau_shift_s * (min(size, 1.0) - band) / (1.0 - band)

    def action_for_shift(self, shift_s: float) -> float:
        """Inverse of ``shift_of`` for a non-zero shift (|shift| <= tau_shift_s)."""

        band = self.config.keep_band
        return float(np.sign(shift_s)) * (band + (1.0 - band) * min(abs(shift_s) / self.config.tau_shift_s, 1.0))

    def intent(self, action: np.ndarray) -> tuple[float, float]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        assert self.generator is not None
        remaining = self.generator.remaining_s(self.env)
        if remaining > 0.0:
            tau = max(remaining + self.shift_of(float(a[0])), 0.0)
        else:
            tau = self.config.tau_max_s * 0.5 * (a[0] + 1.0)
        kappa = self.config.kappa_min + (self.config.kappa_max - self.config.kappa_min) * 0.5 * (a[1] + 1.0)
        return float(tau), float(kappa)

    def entry_speed_of(self, action: np.ndarray) -> float:
        """Inward entry speed (m/s) asked by a[2] for a new plan (a 2-D action asks the nominal)."""

        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        a2 = float(a[2]) if a.size > 2 else 0.0
        return float(self.config.entry_speed_nominal_m_s + self.config.entry_speed_span_m_s * a2)

    def apply_intent(self, action: np.ndarray) -> tuple[float, float, float]:
        """Set kappa and the entry-speed request on G* for this decision; (tau, kappa, v_in)."""

        assert self.generator is not None
        tau, kappa = self.intent(action)
        v_in = self.entry_speed_of(action)
        self.generator.kappa = kappa
        self.generator.entry_speed_m_s = v_in
        return tau, kappa, v_in

    def _plan_entry_speed(self) -> float | None:
        g = self.generator
        assert g is not None
        if g.plan is None:
            return None
        return float(g.plan.get("entry_speed_m_s", self.config.entry_speed_nominal_m_s))

    def _tracking_errors(self) -> tuple[float, float]:
        """Position and velocity error of the current state against the live plan (ECI relative)."""

        e, g = self.env, self.generator
        assert g is not None
        if g.plan is None or g.remaining_s(e) <= 0.0:
            return 0.0, 0.0
        s = e.time_s - g.plan_start_s
        q = e.chaser_state.position - e.target_state.position
        v = e.chaser_state.rotation @ e.chaser_state.velocity - e.target_state.rotation @ e.target_state.velocity
        return (float(np.linalg.norm(_plan_position(g.plan, s, g.planner.mass) - q)),
                float(np.linalg.norm(_plan_velocity(g.plan, s, g.planner.mass) - v)))

    def step(self, action: np.ndarray):
        assert self.generator is not None
        g = self.generator
        tau, kappa, v_in = self.apply_intent(action)
        # effective-action diagnostics (before the decision; read-only)
        remaining_before = g.remaining_s(self.env)
        deviation_before, _ = self._tracking_errors()
        usable_before = remaining_before > 0.0 and deviation_before <= g.replan_deviation_m
        if remaining_before <= 0.0:
            request_kind = "no_plan"
        else:
            shift = tau - remaining_before
            request_kind = ("keep" if abs(shift) <= g.replan_tau_tolerance_s
                            else "later" if shift > 0.0 else "earlier")
        walls_before = len(g.plan_wall_s)
        g.decide(self.env, tau)
        new_plan = g.last_status.startswith("optimal")
        trigger = None
        if g.last_status.startswith(("optimal", "recovery")):
            trigger = ("no_plan" if remaining_before <= 0.0 else "deviation" if not usable_before
                       else "deadline" if g.last_deadline == "plan_beyond" else "request")
        slip = max(g.tau_effective_s - tau, 0.0) if new_plan else 0.0
        self._last_request, self._last_kappa = tau, kappa
        self._step_log = []
        deviation = 0.0
        plan = g.plan
        reward, terminated, truncated, info = self.env.step(reference=g)
        if plan is not None and g.plan is plan:
            q = self.env.chaser_state.position - self.env.target_state.position
            deviation = float(np.linalg.norm(_plan_position(plan, self.env.time_s - g.plan_start_s, g.planner.mass) - q))
        slack = [s for s, _, _, solved in self._step_log if solved and np.isfinite(s)]
        info = dict(info)
        info["execution"] = {
            "tau_request_s": tau,
            "tau_effective_s": g.tau_effective_s,
            "plan_status": g.last_status,
            "new_plan": new_plan,
            "slip_s": slip,
            "plan_deviation_end_m": deviation,
            "max_slack_solved": float(max(slack)) if slack else 0.0,
            "force_saturated_fraction": float(np.mean([s for _, s, _, _ in self._step_log])) if self._step_log else 0.0,
            "zero_fallbacks": int(sum(f for _, _, f, _ in self._step_log)),
            "plan_wall_s": float(sum(g.plan_wall_s[walls_before:])),
            "request_kind": request_kind,
            "deviation_before_m": deviation_before,
            "new_plan_trigger": trigger,
            "deadline": g.last_deadline,
            # interface v4: the entry speed asked, whether a new plan took it, and the live plan's
            "entry_speed_request_m_s": v_in,
            "entry_speed_applied": bool(g.last_status.startswith(("optimal", "recovery"))),
            "entry_speed_plan_m_s": self._plan_entry_speed(),
        }
        info["kappa"] = kappa
        if terminated or truncated:
            # taken here, before any auto-reset by a vectorised wrapper can replace the state
            m = self.env.metrics()
            info["episode_summary"] = {
                "event": info["event"], **self.env.summary(), "opening": self.current_opening,
                "violations": dict(info.get("violations", {})), "end_phase": end_phase(m),
                "end_range_m": float(m.range_m),
            }
        terminal = info["event"] in ("success", "unsafe", "timeout")
        return self._observation(), float(reward), bool(terminal), bool(info["event"] == "numerical_failure"), info

    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        e, g = self.env, self.generator
        assert g is not None
        m = e.metrics()
        f = e.features(m)
        los = e.chaser_state.position - e.target_state.position
        rng = max(float(np.linalg.norm(los)), 1e-9)
        rel = e.chaser_state.rotation @ e.chaser_state.velocity - e.target_state.rotation @ e.target_state.velocity
        tangential = float(np.linalg.norm(rel - (rel @ los / rng) * los / rng))
        lat, ph = np.deg2rad(f["latitude_deg"]), np.deg2rad(f["phase_deg"])
        status = g.last_status
        values = [
            f["range_m"] / 10.0, np.sin(lat), np.cos(lat), np.sin(ph), np.cos(ph), f["rho_perp_m"] / 10.0,
            f["port_los_angle_deg"] / 180.0, f["port_latitude_deg"] / 90.0, f["omega_deg_s"] / self.config.omega_scale_deg_s,
            f["fov_margin_deg"] / 50.0, f["time_s"] / 300.0,
            float(m.radial_closing_speed_m_s) / 0.3, tangential / 0.5, float(m.in_cone), float(m.mouth_region),
            float(g.plan is not None), g.remaining_s(e) / 160.0, g.tau_effective_s / 160.0,
            self._last_request / 160.0, float(status == "descent"), float(status.startswith("hold")),
            self._last_kappa,
        ]
        position_error, velocity_error = self._tracking_errors()  # interface v3 (appended)
        values += [position_error / 1.0, velocity_error / 0.3]
        v_plan = self._plan_entry_speed()  # interface v4 (appended)
        c = self.config
        values += [0.0 if v_plan is None else (v_plan - c.entry_speed_nominal_m_s) / c.entry_speed_span_m_s]
        obs = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError(f"non-finite observation {obs}")
        return obs


GYM_INTERFACE_VERSION = "4"

__all__ = ["GYM_INTERFACE_VERSION", "OBSERVATION_NAMES", "T1GStarEnv", "T1GymConfig", "end_phase"]
