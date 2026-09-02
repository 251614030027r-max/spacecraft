"""Aggregate compatibility and 29D schema acceptance test for A1."""

from dataclasses import replace

import numpy as np

from dynamics.lie import adjoint
from env.observation import (
    PHASE2_MISSION_OBSERVATION_SCHEMA,
    PHASE2_PERCEPTION_OBSERVATION_SCHEMA,
    build_phase2_mission_observation,
)
from env.phase2_env import (
    phase2_environment_config,
    phase2_perception_environment_config,
)
from env.se3_rendezvous_env import SE3RendezvousEnv


def test_a1_perception_schema_and_disabled_path_compatibility() -> None:
    nominal_config = replace(
        phase2_environment_config("single_phase"),
        max_time_s=1.0,
        cache_target_trajectory=False,
    )
    perception_config = replace(
        phase2_perception_environment_config(),
        max_time_s=1.0,
        cache_target_trajectory=False,
    )
    derived_disabled_config = replace(
        perception_config,
        perception=None,
        phase2_observation_schema=PHASE2_MISSION_OBSERVATION_SCHEMA,
    )
    nominal = SE3RendezvousEnv(nominal_config)
    derived_disabled = SE3RendezvousEnv(derived_disabled_config)
    nominal_observation, _ = nominal.reset(seed=260902)
    disabled_observation, _ = derived_disabled.reset(seed=260902)
    np.testing.assert_array_equal(nominal_observation, disabled_observation)
    action = np.array([0.1, -0.2, 0.3, -0.4, 0.2, -0.1], dtype=np.float32)
    nominal_step = nominal.step(action)
    disabled_step = derived_disabled.step(action)
    np.testing.assert_array_equal(nominal_step[0], disabled_step[0])
    assert nominal_step[1:4] == disabled_step[1:4]
    np.testing.assert_array_equal(
        nominal.relative.transform, derived_disabled.relative.transform
    )
    np.testing.assert_array_equal(
        nominal.relative.twist, derived_disabled.relative.twist
    )

    env = SE3RendezvousEnv(perception_config)
    observation, info = env.reset(seed=260902)
    assert perception_config.phase2_observation_schema == (
        PHASE2_PERCEPTION_OBSERVATION_SCHEMA
    )
    assert env.observation_space.shape == (29,)
    assert observation.shape == (29,)
    assert env.observed_relative is not env.relative
    estimated_target_omega = (
        adjoint(env.observed_relative.transform)
        @ (env.chaser_state.twist - env.observed_relative.twist)
    )[:3]
    estimated_core = build_phase2_mission_observation(
        env.observed_relative,
        task=env.config.phase2_task,
        active_reference_position_m=env.config.phase2_task.desired_position,
        mission_phase=1,
        attitude_scale_rad=env.config.observation_attitude_scale_rad,
        distance_scale_m=env.config.observation_distance_scale_m,
        angular_velocity_scale_rad_s=(
            env.config.observation_angular_velocity_scale_rad_s
        ),
        velocity_scale_m_s=env.config.observation_velocity_scale_m_s,
        softsign_limit=env.config.observation_softsign_limit,
        target_angular_velocity_rad_s=estimated_target_omega,
        target_angular_velocity_scale_rad_s=(
            env.config.phase2_observation_target_angular_velocity_scale_rad_s
        ),
        translational_observation_frame="chaser_body",
        translational_velocity_observation="actual",
    )
    np.testing.assert_array_equal(observation[:24], estimated_core)
    truth_core = build_phase2_mission_observation(
        env.relative,
        task=env.config.phase2_task,
        active_reference_position_m=env.config.phase2_task.desired_position,
        mission_phase=1,
        attitude_scale_rad=env.config.observation_attitude_scale_rad,
        distance_scale_m=env.config.observation_distance_scale_m,
        angular_velocity_scale_rad_s=(
            env.config.observation_angular_velocity_scale_rad_s
        ),
        velocity_scale_m_s=env.config.observation_velocity_scale_m_s,
        softsign_limit=env.config.observation_softsign_limit,
        target_angular_velocity_rad_s=env.target_state.omega,
        target_angular_velocity_scale_rad_s=(
            env.config.phase2_observation_target_angular_velocity_scale_rad_s
        ),
        translational_observation_frame="chaser_body",
        translational_velocity_observation="actual",
    )
    assert not np.array_equal(observation[:24], truth_core)
    required = {
        "estimated_relative_vector",
        "estimator_covariance_diag",
        "estimation_attitude_error_rad",
        "estimation_position_error_m",
        "estimation_angular_velocity_error_rad_s",
        "estimation_velocity_error_m_s",
        "visible_feature_count",
        "visible_feature_fraction",
        "perception_measurement_used",
    }
    assert required <= info.keys()
    assert info["estimated_relative_vector"].shape == (12,)
    assert info["estimator_covariance_diag"].shape == (12,)
