"""T1 decision-level environment: RK45 truth, the shared MPC, and intent interface G.

One ``step`` is one 2 s decision (20 control steps of 0.1 s). The action is the
two-dimensional intent ``a = (a_lambda, a_sigma)``; ``step(..., pure=True)``
flies the Pure MPC row instead (fixed terminal setpoint, no reference path).
A constant ``a_sigma = +1`` is the Pure row by construction: the same waypoint,
no reference path, the same controller call.

This module does not touch the F01 environment. It reuses the truth
propagation, the MPC controller and its prediction models, and only adds the
T1 task semantics (``env/t1_task.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from controllers.mpc.config import MPCConfig, precapture_mpc_config
from controllers.mpc.controller import MPCController
from controllers.mpc.prediction import (
    LocalRelativePredictionModel,
    RelativePredictionModel,
    relative_to_vector,
)
from dynamics.disturbance import zero_disturbance
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings, propagate_rk45
from dynamics.lie import so3_exp, so3_log
from dynamics.relative import relative_state
from dynamics.types import GeneralizedForce, SpacecraftState
from env.scenarios import chaser_parameters
from env.t1_task import (
    EpisodeAnchor,
    T1InitialDistribution,
    T1IntentConfig,
    T1Metrics,
    T1TaskConfig,
    T1_TARGET_INERTIA_KG_M2,
    compute_t1_metrics,
    episode_anchor,
    intent_from_action,
    reference_position_target_frame,
    sample_t1_initial_states,
    t1_target_parameters,
)


@dataclass(frozen=True)
class T1Reward:
    time_cost_per_s: float = 0.05
    delta_v_cost_per_m_s: float = 5.0
    capture_reward: float = 100.0
    timeout_penalty: float = 100.0
    unsafe_penalty: float = 150.0
    gamma: float = 0.999


@dataclass(frozen=True)
class T1EnvConfig:
    task: T1TaskConfig = field(default_factory=T1TaskConfig)
    initial: T1InitialDistribution = field(default_factory=T1InitialDistribution)
    intent: T1IntentConfig = field(default_factory=T1IntentConfig)
    reward: T1Reward = field(default_factory=T1Reward)
    dt_s: float = 0.1
    decision_steps: int = 20
    max_time_s: float = 300.0
    max_force_per_axis_n: float = 2.0
    max_torque_per_axis_nm: float = 0.6
    include_j2: bool = True
    solver_rtol: float = 1.0e-7
    solver_atol: float = 1.0e-9
    horizon_steps: int = 35
    input_weight: float = 0.01
    terminal_weight: float = 1000.0


def t1_mpc_config(config: T1EnvConfig) -> MPCConfig:
    """The shared MPC for every T1 row. Only the force bound/scale differs from F01 (5 N -> 2 N)."""

    return replace(
        precapture_mpc_config(),
        precapture_task=config.task,
        reference_state=precapture_mpc_config().reference_state,
        horizon_steps=config.horizon_steps,
        outer_iterations=1,
        input_weight=config.input_weight,
        terminal_weight=config.terminal_weight,
        input_scales=np.array(
            [*([config.max_torque_per_axis_nm] * 3), *([config.max_force_per_axis_n] * 3)]
        ),
        reference_source="external_local",
        external_reference_frame="target",
        precapture_attitude_reference="frozen",
        external_reference_hold_steps=config.decision_steps,
        runtime_diagnostics=False,
        attitude_chart_unwrap=True,
    )


class T1Env:
    """Deterministic given the seed. Not a gym subclass yet (R1 needs none)."""

    def __init__(self, config: T1EnvConfig | None = None) -> None:
        self.config = config or T1EnvConfig()
        c = self.config
        self.target_parameters = t1_target_parameters()
        self.chaser_parameters = chaser_parameters()
        self._gravity = GravityOptions(include_j2=c.include_j2)
        self._solver = RK45Settings(rtol=c.solver_rtol, atol=c.solver_atol, max_step=c.dt_s)
        self.mpc_config = t1_mpc_config(c)
        reference_model = RelativePredictionModel(
            target_parameters=self.target_parameters,
            chaser_parameters=self.chaser_parameters,
            dt_s=c.dt_s,
            gravity_options=self._gravity,
            solver_settings=self._solver,
            unwrap_attitude=True,
        )
        self.controller = MPCController(
            self.mpc_config,
            LocalRelativePredictionModel(self.chaser_parameters, c.dt_s, unwrap_attitude=True),
            reference_model,
        )
        self.target_state: SpacecraftState | None = None
        self.chaser_state: SpacecraftState | None = None
        self.anchor: EpisodeAnchor | None = None
        self.time_s = 0.0
        self.decision = 0

    # ------------------------------------------------------------------ #
    def reset(self, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        self.target_state, self.chaser_state = sample_t1_initial_states(
            rng, self.config.initial, self.config.task
        )
        self.anchor = episode_anchor(self.target_state, self.chaser_state)
        self.controller.reset()
        self.time_s = 0.0
        self.decision = 0
        self._capture_streak = 0
        self._delta_v = 0.0
        self._fallbacks = 0
        self._control_steps = 0
        self._saturated_steps = 0
        self._max_range = 0.0
        self._min_margins: dict[str, float] = {}
        self._last_reference = None
        self._reference_jump_max = 0.0
        self._last_intent = (np.nan, np.nan)
        metrics = self.metrics()
        self._track(metrics)
        return {"metrics": metrics, "features": self.features(metrics)}

    def metrics(self) -> T1Metrics:
        assert self.target_state is not None and self.chaser_state is not None
        return compute_t1_metrics(self.target_state, self.chaser_state, self.config.task)

    # ------------------------------------------------------------------ #
    def features(self, metrics: T1Metrics | None = None) -> dict[str, float]:
        """Interpretable state features (latitude, phase, rho_perp, ...) in the H0 frame."""

        assert self.target_state is not None and self.chaser_state is not None
        assert self.anchor is not None
        m = metrics or self.metrics()
        h = self.anchor.h_axis
        los = self.chaser_state.position - self.target_state.position
        rng = float(np.linalg.norm(los))
        los_unit = los / rng
        port = self.target_state.rotation @ self.config.task.approach_axis
        latitude = float(np.arcsin(np.clip(los_unit @ h, -1, 1)))
        los_perp = los_unit - (los_unit @ h) * h
        port_perp = port - (port @ h) * h
        # angle the port still has to rotate (about +H) to reach the line of sight
        phase = float(
            np.mod(np.arctan2(h @ np.cross(port_perp, los_perp), port_perp @ los_perp), 2 * np.pi)
        )
        omega_inertial = self.target_state.rotation @ self.target_state.omega
        spin = omega_inertial / np.linalg.norm(omega_inertial)
        rho_perp = float(np.linalg.norm(los - (los @ spin) * spin))
        port_angle = float(np.degrees(np.arccos(np.clip(port @ los_unit, -1, 1))))
        return {
            "range_m": rng,
            "latitude_deg": float(np.degrees(latitude)),
            "phase_deg": float(np.degrees(phase)),
            "rho_perp_m": rho_perp,
            "port_los_angle_deg": port_angle,
            "port_latitude_deg": float(np.degrees(np.arcsin(np.clip(port @ h, -1, 1)))),
            "omega_deg_s": float(np.degrees(np.linalg.norm(self.target_state.omega))),
            "fov_margin_deg": float(np.degrees(m.fov_margin_rad)),
            "time_s": self.time_s,
        }

    # ------------------------------------------------------------------ #
    def _reference(self, latitude: float, sigma: float, pure: bool) -> tuple[np.ndarray, np.ndarray | None]:
        task, intent = self.config.task, self.config.intent
        if pure or sigma >= 1.0:
            return np.asarray(task.desired_position, dtype=np.float64), None
        assert self.target_state is not None and self.anchor is not None
        if sigma > 0.5:
            point = reference_position_target_frame(
                self.target_state.rotation, self.anchor, latitude, sigma, task, intent
            )
            return point, None
        n = self.mpc_config.horizon_steps
        path = np.zeros((3, n + 1))
        for j in range(n + 1):
            rotation = self.target_state.rotation @ so3_exp(j * self.config.dt_s * self.target_state.omega)
            path[:, j] = reference_position_target_frame(rotation, self.anchor, latitude, sigma, task, intent)
        return path[:, 0].copy(), path

    def _track(self, m: T1Metrics) -> None:
        self._max_range = max(self._max_range, m.range_m)
        for name, value in (
            ("collision_clearance_m", m.collision_clearance_m),
            ("union_margin_m", m.union_margin_m),
            ("fov_margin_deg", float(np.degrees(m.fov_margin_rad))),
            ("radial_envelope_margin_m_s", m.radial_envelope_margin_m_s),
            ("total_speed_margin_m_s", m.total_speed_margin_m_s),
            ("axial_envelope_margin_m_s", m.axial_envelope_margin_m_s),
        ):
            if np.isfinite(value):
                self._min_margins[name] = min(self._min_margins.get(name, np.inf), float(value))

    def step(self, action: np.ndarray | None = None, *, pure: bool = False) -> tuple[float, bool, bool, dict[str, Any]]:
        """One 2 s decision. Returns (reward, terminated, truncated, info)."""

        c = self.config
        assert self.target_state is not None and self.chaser_state is not None
        latitude, sigma = (np.nan, 1.0) if pure else intent_from_action(np.asarray(action))
        self._last_intent = (latitude, sigma)
        reward = 0.0
        event = None
        info: dict[str, Any] = {}
        for _ in range(c.decision_steps):
            m_before = self.metrics()
            waypoint, path = self._reference(latitude, sigma, pure)
            if self._last_reference is not None:
                self._reference_jump_max = max(
                    self._reference_jump_max, float(np.linalg.norm(waypoint - self._last_reference))
                )
            self._last_reference = waypoint.copy()
            relative = relative_state(self.target_state, self.chaser_state)
            try:
                wrench, diagnostics = self.controller.command(
                    relative_to_vector(relative),
                    target_state=self.target_state,
                    time_seconds=self.time_s,
                    external_reference=waypoint,
                    external_reference_path=path,
                    terminal_latched=bool(m_before.mouth_region),
                )
                limits = np.array([*([c.max_torque_per_axis_nm] * 3), *([c.max_force_per_axis_n] * 3)])
                wrench = np.clip(np.asarray(wrench, dtype=np.float64), -limits, limits)
                self._fallbacks += int(diagnostics.used_zero_fallback)
                control = GeneralizedForce.from_vector(wrench)
                target_next, _ = propagate_rk45(
                    self.target_state, self.target_parameters, self.time_s, c.dt_s,
                    control=GeneralizedForce.zeros(), disturbance=zero_disturbance,
                    gravity_options=self._gravity, settings=self._solver,
                )
                chaser_next, _ = propagate_rk45(
                    self.chaser_state, self.chaser_parameters, self.time_s, c.dt_s,
                    control=control, disturbance=zero_disturbance,
                    gravity_options=self._gravity, settings=self._solver,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                event = "numerical_failure"
                info["numerical_error"] = repr(error)
                break
            self.target_state, self.chaser_state = target_next, chaser_next
            self._control_steps += 1
            self.time_s = round(self.time_s + c.dt_s, 10)
            force = wrench[3:]
            dv = float(np.linalg.norm(force)) * c.dt_s / self.chaser_parameters.mass
            self._delta_v += dv
            self._saturated_steps += int(np.max(np.abs(force)) >= 0.99 * c.max_force_per_axis_n)
            reward -= c.reward.time_cost_per_s * c.dt_s + c.reward.delta_v_cost_per_m_s * dv
            m = self.metrics()
            self._track(m)
            self._capture_streak = (
                self._capture_streak + 1 if (m.instantaneous_capture and not m.hard_violation) else 0
            )
            if m.hard_violation:
                event = "unsafe"
                info["violations"] = dict(m.violations)
                break
            if m.range_m > c.task.max_distance_m:
                event = "unsafe"
                info["violations"] = {"distance": m.range_m - c.task.max_distance_m}
                break
            if self._capture_streak >= c.task.completion_required_steps(c.dt_s):
                event = "success"
                break
            if self.time_s >= c.max_time_s - 1e-9:
                event = "timeout"
                break
        self.decision += 1
        if event == "success":
            reward += c.reward.capture_reward
        elif event == "unsafe":
            reward -= c.reward.unsafe_penalty
        elif event == "timeout":
            reward -= c.reward.timeout_penalty
        terminated = event in {"success", "unsafe"}
        truncated = event in {"timeout", "numerical_failure"}
        info.update(event=event, latitude=latitude, sigma=sigma)
        return float(reward), terminated, truncated, info

    def summary(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "delta_v_m_s": self._delta_v,
            "decisions": self.decision,
            "qp_zero_fallback_steps": self._fallbacks,
            "control_steps": self._control_steps,
            "force_saturated_fraction": self._saturated_steps / max(self._control_steps, 1),
            "max_range_m": self._max_range,
            "min_margins": dict(self._min_margins),
            "reference_jump_max_m": self._reference_jump_max,
        }


__all__ = ["T1Env", "T1EnvConfig", "T1Reward", "t1_mpc_config"]
