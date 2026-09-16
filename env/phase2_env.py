"""Canonical constructors for the current Phase-2 task.

This is the only public environment entry used by current training and
evaluation code.  The low-level environment remains in
``se3_rendezvous_env`` so the validated dynamics/step implementation is not
duplicated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from env.observation import (
    PHASE2_MISSION_OBSERVATION_SCHEMA,
    PHASE2_PERCEPTION_OBSERVATION_SCHEMA,
    PRECAPTURE_PLANNING_ESTIMATED_SCHEMA,
    PRECAPTURE_PLANNING_FULL_STATE_SCHEMA,
)
from env.perception import PerceptionConfig
from env.task import Phase2MissionConfig, PrecaptureTaskConfig


Phase2Mode = Literal[
    "phase1_pretrain", "full_mission", "single_phase", "single_phase_phase_sampled"
]

A3_PREDICTION_MODEL_MISMATCH = 0.20
A3_PREDICTION_MODEL_SEED = 260903


def phase2_s1v2_mission_config() -> Phase2MissionConfig:
    """Return the explicit S1-v2 acquisition task, isolated from legacy S1."""

    return replace(
        Phase2MissionConfig(),
        waypoint_semantics="acquisition_v2",
        waypoint_position_tolerance_m=1.5,
        waypoint_speed_tolerance_m_s=0.30,
        waypoint_fov_tolerance_rad=float(np.deg2rad(45.0)),
        initial_distance_min_m=10.0,
        initial_distance_max_m=14.0,
        initial_direction_half_angle_rad=float(np.deg2rad(15.0)),
        initial_attitude_limit_rad=float(np.deg2rad(20.0)),
        initial_speed_limit_m_s=0.05,
        initial_angular_velocity_component_limit_rad_s=0.01,
        phase1_cruise_speed_m_s=0.15,
    )


def phase2_environment_config(
    mode: Phase2Mode,
) -> SE3RendezvousConfig:
    """Return the canonical mission environment configuration.

    ``single_phase`` is the single-phase task: the same dynamics, geometry and
    constraint set, with the terminal constraints active from the first step
    and no approach waypoint, bonus or premature-entry guard.
    """

    if mode not in {
        "phase1_pretrain",
        "full_mission",
        "single_phase",
        "single_phase_phase_sampled",
    }:
        raise ValueError(f"unsupported Phase-2 mode: {mode}")
    # single_phase_phase_sampled is single_phase with one change: the target's
    # initial attitude and tumble direction are sampled per episode (magnitude
    # frozen). The training mode stays "single_phase" so every task semantic --
    # phase flag, live terminal constraints, no Waypoint -- is identical; the
    # only difference is the phase-sampling flag, which is the single factor.
    phase_sampled = mode == "single_phase_phase_sampled"
    training_mode = "single_phase" if phase_sampled else mode
    return replace(
        SE3RendezvousConfig(),
        phase2_enabled=True,
        phase2_mission_enabled=True,
        phase2_training_mode=training_mode,
        phase2_target_phase_sampling=phase_sampled,
        phase2_mission=phase2_s1v2_mission_config(),
        phase2_observation_schema=PHASE2_MISSION_OBSERVATION_SCHEMA,
        curriculum_enabled=False,
        phase2_target_tumble_scale=0.20,
        phase2_warmup_steps=0,
        observation_attitude_scale_rad=float(np.deg2rad(60.0)),
        observation_distance_scale_m=10.0,
        observation_angular_velocity_scale_rad_s=0.03,
        observation_velocity_scale_m_s=0.50,
    )


def make_phase2_env(
    mode: Phase2Mode = "full_mission",
) -> SE3RendezvousEnv:
    return SE3RendezvousEnv(phase2_environment_config(mode))


def phase2_perception_environment_config() -> SE3RendezvousConfig:
    """Derive A1 from frozen single_phase with perception as the sole change."""

    return replace(
        phase2_environment_config("single_phase"),
        perception=PerceptionConfig(),
        phase2_observation_schema=PHASE2_PERCEPTION_OBSERVATION_SCHEMA,
    )


def perception_guidance_free_environment_config() -> SE3RendezvousConfig:
    """Derive A2 from frozen A1 with guidance removal as the sole change."""

    return replace(
        phase2_perception_environment_config(),
        phase2_guidance_free=True,
    )


def deployable_planning_environment_config() -> SE3RendezvousConfig:
    """Derive A3 from A2 with one fixed, shared prediction-model mismatch."""

    return replace(
        perception_guidance_free_environment_config(),
        phase2_prediction_model_mismatch=A3_PREDICTION_MODEL_MISMATCH,
        phase2_prediction_model_seed=A3_PREDICTION_MODEL_SEED,
    )


def precapture_planning_environment_config() -> SE3RendezvousConfig:
    """Independent two-region planning task with full-state diagnostic input."""

    return replace(
        SE3RendezvousConfig(),
        max_time_s=300.0,
        max_distance_m=30.0,
        phase2_enabled=True,
        phase2_mission_enabled=False,
        precapture_planning_enabled=True,
        precapture_task=PrecaptureTaskConfig(),
        phase2_observation_schema=PRECAPTURE_PLANNING_FULL_STATE_SCHEMA,
        curriculum_enabled=False,
        phase2_target_phase_sampling=True,
        phase2_target_tumble_scale=0.20,
        phase2_warmup_steps=0,
        observation_attitude_scale_rad=float(np.deg2rad(60.0)),
        observation_distance_scale_m=20.0,
        observation_angular_velocity_scale_rad_s=0.05,
        observation_velocity_scale_m_s=1.10,
    )


def precapture_perception_environment_config() -> SE3RendezvousConfig:
    """Non-cooperative precapture: A1 camera + relative EKF as the sole change.

    Derived from ``precapture_planning_environment_config`` with perception
    enabled, so the upper policy observes the EKF-estimated relative state and
    its uncertainty (29D ``precapture_planning_estimated_v1_29d``) instead of the
    truth full-state 24D. Truth still drives dynamics, reward, termination,
    geometry and evaluation; ``perception=None`` reproduces the full-state task
    bitwise.
    """

    return replace(
        precapture_planning_environment_config(),
        perception=PerceptionConfig(),
        phase2_observation_schema=PRECAPTURE_PLANNING_ESTIMATED_SCHEMA,
    )


def precapture_staging_environment_config(
    entry_phase_gate_deg: float = 90.0,
) -> SE3RendezvousConfig:
    """Non-corotating staging + phase-gated capture (Path 2).

    Derived from ``precapture_planning_environment_config`` with the entry-phase
    gate enabled: a legal capture requires the tumbling port's outward normal to
    point within ``entry_phase_gate_deg`` of the fixed inertial staging direction
    (the reset-time approach line, held in inertial). The favourable window opens
    once per tumble, so "when to enter" becomes a real long-horizon decision.
    All other geometry, constraints, reward and the strong MPC are unchanged;
    ``entry_phase_gate_deg >= 180`` reproduces the frozen task.
    """

    base = precapture_planning_environment_config()
    gate_cos = float(np.cos(np.deg2rad(entry_phase_gate_deg)))
    return replace(
        base,
        precapture_task=replace(base.precapture_task, entry_phase_gate_cos=gate_cos),
    )


def precapture_timing_probe_environment_config(
    entry_phase_gate_deg: float = 180.0,
    initial_range_max_m: float = 28.0,
    initial_pointing_error_max_deg: float = 25.0,
) -> SE3RendezvousConfig:
    """Opened-distribution config for the zero-training timing-value A/B probe.

    Derived from ``precapture_planning_environment_config`` by enlarging only the
    *task-selection* space -- the spread of initial conditions the chaser has to
    pick an entry moment from -- while every knob that sets *control difficulty*
    stays frozen (the +-5 N / +-0.6 N*m authority, the 0.1 s update rate, the MPC
    model, the corridor/FOV/speed/terminal constraints, and the 0.041 rad/s
    tumble rate). This is the boundary the direction rides on: widening what has
    to be decided is legitimate, weakening what has to be executed would be a
    manufactured gap.

    What opens:

    * **Initial range** 15 -> ``initial_range_max_m`` m. Capped strictly below the
      30 m distance-failure boundary (default 28 m keeps a 2 m drift margin), so a
      literal "15-30 m" is intentionally not used -- starting on the failure
      boundary would fail on the first outward drift.
    * **Initial pointing error** 5 -> ``initial_pointing_error_max_deg`` deg,
      still inside the 50 deg FOV half-angle (the sampler rejects otherwise).
    * The **approach azimuth** is already fully open in
      ``sample_precapture_planning_chaser_state`` (uniform on [-pi, pi] with a
      polar cone from ~40 deg off-axis outward), and the **initial target phase**
      is already sampled per episode by ``phase2_target_phase_sampling`` -- that
      per-episode phase is what disperses the entry moment, so no change is
      needed there.

    The **entry-phase gate is OFF by default** (``entry_phase_gate_deg >= 180`` ->
    ``gate_cos <= -1``): the primary A/B must measure whether timing helps through
    physics alone (a fixture facing the approach makes the terminal geometry and
    the camera view easier), never because a hard gate rejected the immediate
    arm's crossings -- a gate-manufactured difference is not admissible evidence
    that timing is valuable. A gated variant (e.g. 90 deg) is one argument away
    and is a *diagnostic* guardrail only, not the paper mechanism.
    """

    base = precapture_planning_environment_config()
    gate_cos = float(np.cos(np.deg2rad(entry_phase_gate_deg)))
    pointing_error_max_rad = float(np.deg2rad(initial_pointing_error_max_deg))
    return replace(
        base,
        precapture_initial_range_min_m=15.0,
        precapture_initial_range_max_m=float(initial_range_max_m),
        precapture_initial_pointing_error_max_rad=pointing_error_max_rad,
        precapture_task=replace(base.precapture_task, entry_phase_gate_cos=gate_cos),
    )


def terminal_phase_environment_config() -> SE3RendezvousConfig:
    """Terminal-only config retained for P0 MPC validation and later P3 reuse."""

    return replace(
        SE3RendezvousConfig(),
        phase2_enabled=True,
        phase2_mission_enabled=False,
        curriculum_enabled=False,
        phase2_target_tumble_scale=0.20,
        phase2_warmup_steps=0,
        observation_attitude_scale_rad=float(np.deg2rad(60.0)),
        observation_distance_scale_m=5.0,
        observation_angular_velocity_scale_rad_s=0.03,
        observation_velocity_scale_m_s=0.35,
    )


__all__ = [
    "Phase2Mode",
    "A3_PREDICTION_MODEL_MISMATCH",
    "A3_PREDICTION_MODEL_SEED",
    "deployable_planning_environment_config",
    "make_phase2_env",
    "phase2_environment_config",
    "phase2_perception_environment_config",
    "perception_guidance_free_environment_config",
    "precapture_planning_environment_config",
    "precapture_perception_environment_config",
    "precapture_staging_environment_config",
    "precapture_timing_probe_environment_config",
    "phase2_s1v2_mission_config",
    "terminal_phase_environment_config",
]
