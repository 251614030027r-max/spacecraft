"""Gymnasium wrapper of T1 + G* (``GSTAR_VERSION``, 0.3.2) for SAC (M0 and M1 use it unchanged).

One ``step`` is one 2 s decision: the action becomes a G* request, the
generator decides (keep / re-plan / descent / hold), then 20 MPC control
steps of 0.1 s run under RK45 truth (``T1Env.step``).

Action ``a`` in [-1, 1]^2:
  a[0] -> arrival-time request. With a live plan the request is relative:
          tau = remaining + ``tau_shift_s`` * a[0], and the plan is kept while
          |tau_shift_s * a[0]| <= ``keep_tolerance_s`` (so "keep" is a set of
          actions of positive measure, reachable by a stochastic policy).
          Without a plan it is absolute: tau = ``tau_max_s`` (a[0] + 1) / 2,
          and tau <= 2 s asks for the earliest feasible arrival.
  a[1] -> descent pace kappa = kappa_min + (kappa_max - kappa_min)(a[1] + 1)/2.

Observation (``OBSERVATION_NAMES``): geometry and rates of the current truth
state as an onboard estimate would give them, plus the generator memory that
makes the decision process (approximately) Markov: whether a plan is live,
its remaining time, the last effective and requested arrival times, descent
and hold flags, and the last kappa.

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
)


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
    kappa_min: float = 0.3
    kappa_max: float = 1.0
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
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
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

    def intent(self, action: np.ndarray) -> tuple[float, float]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        assert self.generator is not None
        remaining = self.generator.remaining_s(self.env)
        if remaining > 0.0:
            tau = max(remaining + self.config.tau_shift_s * a[0], 0.0)
        else:
            tau = self.config.tau_max_s * 0.5 * (a[0] + 1.0)
        kappa = self.config.kappa_min + (self.config.kappa_max - self.config.kappa_min) * 0.5 * (a[1] + 1.0)
        return float(tau), float(kappa)

    def step(self, action: np.ndarray):
        assert self.generator is not None
        g = self.generator
        tau, kappa = self.intent(action)
        g.kappa = kappa
        walls_before = len(g.plan_wall_s)
        g.decide(self.env, tau)
        new_plan = g.last_status.startswith("optimal")
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
            f["port_los_angle_deg"] / 180.0, f["port_latitude_deg"] / 90.0, f["omega_deg_s"] / 2.7,
            f["fov_margin_deg"] / 50.0, f["time_s"] / 300.0,
            float(m.radial_closing_speed_m_s) / 0.3, tangential / 0.5, float(m.in_cone), float(m.mouth_region),
            float(g.plan is not None), g.remaining_s(e) / 160.0, g.tau_effective_s / 160.0,
            self._last_request / 160.0, float(status == "descent"), float(status.startswith("hold")),
            self._last_kappa,
        ]
        obs = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError(f"non-finite observation {obs}")
        return obs


__all__ = ["OBSERVATION_NAMES", "T1GStarEnv", "T1GymConfig", "end_phase"]
