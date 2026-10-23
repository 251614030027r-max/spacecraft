"""T1-N task (Research Plan R2, 2026-10-18): neutral arrival time chosen by SAC, conditional G*.

Physics, truth, MPC, safety rules and capture are T1's (``env/t1_env.py``, ``env/t1_task.py``).
What R2 changes, and nothing else:

* **Initial distribution** (``T1NInitialDistribution``, R2 sec. 2.2): range 12-18 m, tumble
  2.1-2.7 deg/s about an axis 10-30 deg off the body z axis (random azimuth), target attitude
  uniform on SO(3), chaser direction uniform on the LVLH sphere, and an *inertial relative*
  velocity built from the line of sight: a signed radial component in [-0.12, 0.12] m/s
  (positive = outward) and a tangential component of magnitude 0.03-0.20 m/s in a uniformly
  drawn direction of the plane normal to the line of sight (largest total 0.233 m/s < 0.25).
  Chaser attitude and rates are drawn as in T1 (aimed at the port, tilt <= 15 deg, random roll,
  +-0.005 rad/s per axis).
* **Conditional G*** (``ConditionalGStar``, R2 sec. 4): the same SCP planner (0.8 x 2 N, 30
  nodes, sphere half-spaces, braking-envelope chords, range cap) solves *exactly* the arrival
  time it is asked for. It no longer searches an earliest-feasible grid, slips a request later,
  clamps a request to the deadline or plans a full-force earliest recovery on its own: those
  chose a different long-term rendezvous in SAC's place. An unmeetable request keeps a usable
  live plan (no new task is chosen for SAC); without one the reference holds the current inertial
  position (brake and station-keep, the T1 hold) and the interface asks SAC again at the next
  2 s step. A live plan that the state no longer tracks (deviation > 1 m) is re-planned for the
  same remaining time; if that is infeasible the plan is dropped (event -> review).
* **Fixed descent settings**: kappa = 1.0 and entry speed 0.02 m/s for every plan and every arm
  (R2 sec. 3.1; the historical one-sided preferences), so SAC has one action: the arrival time.

``t1n_env_config()`` returns the physical configuration (``fuel_cost`` 20 by default, R2 sec. 2.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.random import Generator

from dynamics.constants import EARTH
from dynamics.lie import so3_exp
from dynamics.types import SpacecraftState
from env.scenarios import TARGET_ORBIT_ALTITUDE_M, TARGET_ORBIT_INCLINATION_RAD, _uniform_rotation
from env.t1_env import T1EnvConfig
from env.t1_gstar import GStarPlan, GStarRendezvous, RendezvousPlanner, _plan_position, free_body_rotations
from env.t1_task import T1InitialDistribution, T1TaskConfig, lvlh_basis

T1N_VERSION = "t1n-0.1"
#: descent pace and entry speed shared by every plan, arm and baseline (R2 sec. 3.1)
T1N_KAPPA = 1.0
T1N_ENTRY_SPEED_M_S = 0.02
T1N_FUEL_COST = 20.0


@dataclass(frozen=True)
class T1NInitialDistribution(T1InitialDistribution):
    """R2 sec. 2.2. ``inertial_speed_max_m_s`` of the parent is not used here."""

    tumble_offset_from_z_deg: tuple[float, float] = (10.0, 30.0)
    range_m: tuple[float, float] = (12.0, 18.0)
    radial_speed_m_s: tuple[float, float] = (-0.12, 0.12)
    tangential_speed_m_s: tuple[float, float] = (0.03, 0.20)
    total_speed_max_m_s: float = 0.25

    def sample(self, rng: Generator, task: T1TaskConfig) -> tuple[SpacecraftState, SpacecraftState]:
        return sample_t1n_initial_states(rng, self, task)


def sample_t1n_initial_states(
    rng: Generator, d: T1NInitialDistribution, task: T1TaskConfig
) -> tuple[SpacecraftState, SpacecraftState]:
    """Target (uniform SO(3), spin 10-30 deg off body z) and chaser (LVLH sphere, LOS-built speed).

    Draw order (fixed): target rotation, rate, offset, azimuth, LOS direction, range, radial
    speed, tangential direction, tangential magnitude, roll, tilt axis, tilt angle, chaser rates.
    """

    orbital_radius = EARTH.equatorial_radius + TARGET_ORBIT_ALTITUDE_M
    circular_speed = np.sqrt(EARTH.mu / orbital_radius)
    inc = TARGET_ORBIT_INCLINATION_RAD
    position_eci = np.array([orbital_radius, 0.0, 0.0])
    velocity_eci = circular_speed * np.array([0.0, np.cos(inc), np.sin(inc)])
    rotation = _uniform_rotation(rng)
    rate = np.deg2rad(rng.uniform(*d.tumble_rate_deg_s))
    offset = np.deg2rad(rng.uniform(*d.tumble_offset_from_z_deg))
    azimuth = rng.uniform(-np.pi, np.pi)
    omega = rate * np.array([np.sin(offset) * np.cos(azimuth), np.sin(offset) * np.sin(azimuth), np.cos(offset)])
    target = SpacecraftState(rotation=rotation, position=position_eci, omega=omega,
                             velocity=rotation.T @ velocity_eci)

    basis = lvlh_basis(target)
    direction_lvlh = rng.normal(size=3)
    direction_lvlh /= np.linalg.norm(direction_lvlh)
    range_m = rng.uniform(*d.range_m)
    los = basis @ direction_lvlh  # unit, ECI, target -> chaser
    chaser_position = position_eci + range_m * los
    radial = rng.uniform(*d.radial_speed_m_s)
    # tangential direction: uniform on the circle normal to the line of sight
    helper = np.array([0.0, 0.0, 1.0]) if abs(los[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(los, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(los, t1)
    angle = rng.uniform(-np.pi, np.pi)
    tangential = rng.uniform(*d.tangential_speed_m_s)
    relative_velocity = radial * los + tangential * (np.cos(angle) * t1 + np.sin(angle) * t2)
    speed = float(np.linalg.norm(relative_velocity))
    if speed > d.total_speed_max_m_s:  # never reached with the R2 bounds (0.233 m/s); kept as a rule
        relative_velocity *= d.total_speed_max_m_s / speed
    chaser_velocity = velocity_eci + relative_velocity  # inertial relative velocity, not co-rotating

    port_eci = position_eci + rotation @ task.port_position
    boresight = port_eci - chaser_position
    boresight /= np.linalg.norm(boresight)
    reference = np.array([0.0, 0.0, 1.0]) if abs(boresight[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference, boresight)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(boresight, y_axis)
    aimed = np.column_stack((boresight, y_axis, z_axis))
    roll = so3_exp(np.array([rng.uniform(-np.pi, np.pi), 0.0, 0.0]))
    tilt_axis = np.array([0.0, *rng.normal(size=2)])
    tilt_axis /= np.linalg.norm(tilt_axis)
    tilt = so3_exp(tilt_axis * np.deg2rad(rng.uniform(0.0, d.pointing_error_max_deg)))
    chaser_rotation = aimed @ tilt @ roll
    chaser = SpacecraftState(
        rotation=chaser_rotation, position=chaser_position,
        omega=rng.uniform(-d.chaser_omega_component_max_rad_s, d.chaser_omega_component_max_rad_s, size=3),
        velocity=chaser_rotation.T @ chaser_velocity,
    )
    return target, chaser


def t1n_env_config(fuel_cost: float = T1N_FUEL_COST) -> T1EnvConfig:
    base = T1EnvConfig()
    return replace(base, initial=T1NInitialDistribution(),
                   reward=replace(base.reward, delta_v_cost_per_m_s=float(fuel_cost)))


# --------------------------------------------------------------------------- #
# Conditional G*
# --------------------------------------------------------------------------- #


@dataclass
class ConditionalGStar(GStarRendezvous):
    """G* v0.6 planner and reference, without its own long-term task choice (module docstring).

    ``decide(env, tau)``: descent mode as before; a live plan that is tracked (deviation <= 1 m)
    and agrees with the request (within ``replan_tau_tolerance_s``) is kept; otherwise exactly
    ``tau`` is planned (warm start from the live plan, then cold). Statuses: ``optimal*`` (new
    plan), ``kept``, ``kept:infeasible_request`` (the request could not be met; the usable plan is
    kept), ``hold:rejected`` (no usable plan and the request could not be met: brake and hold).
    A request beyond the deadline (``max_time - deadline_margin``) is not planned (``deadline =
    "rejected_beyond"``): it is never cut to the deadline for SAC.
    """

    #: tolerance used when deciding whether a request lies beyond the deadline
    deadline_tolerance_s: float = 1e-6
    #: what the last decision did with the request: accepted (new plan or kept as asked),
    #: rejected (kept the old plan or hold), descent, or replan_failed (deviation re-plan)
    last_outcome: str = "none"

    def decide(self, env, tau_s: float) -> None:
        self.tau_s = float(tau_s)
        self.last_deadline = "none"
        self._descent = GStarPlan(tau_s=0.0, kappa=self.kappa)
        if self._descending(env, env.metrics()):
            self.plan, self.last_status, self.tau_effective_s = None, "descent", 0.0
            self.last_outcome = "descent"
            return
        started = perf_counter()
        target, chaser = env.target_state, env.chaser_state
        q0 = chaser.position - target.position
        v0 = chaser.rotation @ chaser.velocity - target.rotation @ target.velocity
        remaining = self.remaining_s(env)
        horizon = float(env.config.max_time_s) - self.deadline_margin_s - float(env.time_s)
        usable_plan = False
        if self.plan is not None:
            elapsed = env.time_s - self.plan_start_s
            deviation = float(np.linalg.norm(_plan_position(self.plan, elapsed, self.planner.mass) - q0))
            usable_plan = deviation <= self.replan_deviation_m and remaining > 0.0
            if usable_plan and abs(self.tau_s - remaining) <= self.replan_tau_tolerance_s:
                self.last_status, self.last_outcome = "kept", "accepted"
                return
        result: dict[str, Any] = {"feasible": False, "status": "not_solved"}
        if self.tau_s > horizon + self.deadline_tolerance_s or self.tau_s <= 0.0:
            self.last_deadline = "rejected_beyond" if self.tau_s > 0.0 else "none"
            result["status"] = "beyond_deadline" if self.tau_s > 0.0 else "non_positive"
        else:
            history = free_body_rotations(target.rotation, target.omega, env.target_parameters.inertia,
                                          self.tau_s + 1.0)
            guess = None
            if self.plan is not None:
                n, shift = self.planner.nodes, env.time_s - self.plan_start_s
                guess = np.array([_plan_position(self.plan, shift + k * self.tau_s / n, self.planner.mass)
                                  for k in range(n + 1)])
                guess[0] = q0
            result = self.planner.solve(q0, v0, self.tau_s, history, guess, entry_speed_m_s=self.entry_speed_m_s)
            if not result["feasible"] and guess is not None:
                result = self.planner.solve(q0, v0, self.tau_s, history, None, entry_speed_m_s=self.entry_speed_m_s)
        self.plan_wall_s.append(perf_counter() - started)
        if result["feasible"]:
            self.plan_start_s = float(env.time_s)
            self.plan, self.last_status = result, str(result["status"])
            self.tau_effective_s = float(result["tau_s"])
            self.hold_position = None
            self.last_outcome = "accepted"
            return
        if usable_plan:
            self.last_status, self.last_outcome = "kept:infeasible_request", "rejected"
            return
        # no usable plan and the request cannot be met: brake and hold the current inertial position
        self.last_outcome = "replan_failed" if self.plan is not None else "rejected"
        self.plan_start_s = float(env.time_s)
        self.plan, self.last_status, self.tau_effective_s = None, f"hold:rejected:{result['status']}", 0.0
        if self.hold_position is None:
            self.hold_position = q0.copy()


def make_conditional_gstar(env, kappa: float = T1N_KAPPA, entry_speed_m_s: float = T1N_ENTRY_SPEED_M_S,
                           plan_force_fraction: float = 0.8, **kwargs: Any) -> ConditionalGStar:
    planner = RendezvousPlanner(env.config.task, env.chaser_parameters.mass,
                                plan_force_fraction * env.config.max_force_per_axis_n)
    return ConditionalGStar(kappa=kappa, planner=planner, recovery_planner=None,
                            entry_speed_m_s=entry_speed_m_s, **kwargs)


__all__ = [
    "ConditionalGStar", "T1NInitialDistribution", "T1N_ENTRY_SPEED_M_S", "T1N_FUEL_COST", "T1N_KAPPA",
    "T1N_VERSION", "make_conditional_gstar", "sample_t1n_initial_states", "t1n_env_config",
]
