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
    "phase2_s1v2_mission_config",
    "terminal_phase_environment_config",
]
