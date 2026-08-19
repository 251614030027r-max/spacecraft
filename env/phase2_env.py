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
from env.observation import PHASE2_MISSION_OBSERVATION_SCHEMA
from env.task import Phase2MissionConfig


Phase2Mode = Literal["phase1_pretrain", "full_mission"]


def phase2_s1v2_mission_config() -> Phase2MissionConfig:
    """Return the explicit S1-v2 acquisition task, isolated from legacy S1."""

    return replace(
        Phase2MissionConfig(),
        gate_semantics="acquisition_v2",
        gate_position_tolerance_m=1.5,
        gate_speed_tolerance_m_s=0.30,
        gate_fov_tolerance_rad=float(np.deg2rad(45.0)),
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
    """Return the canonical two-phase mission environment configuration."""

    if mode not in {"phase1_pretrain", "full_mission"}:
        raise ValueError(f"unsupported Phase-2 mode: {mode}")
    return replace(
        SE3RendezvousConfig(),
        phase2_enabled=True,
        phase2_mission_enabled=True,
        phase2_training_mode=mode,
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
    "make_phase2_env",
    "phase2_environment_config",
    "phase2_s1v2_mission_config",
    "terminal_phase_environment_config",
]
