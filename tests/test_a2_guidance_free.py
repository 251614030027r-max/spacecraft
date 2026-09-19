"""Aggregate A2 single-factor and endpoint-planning acceptance tests."""

from dataclasses import asdict

import numpy as np

from controllers.mpc import (
    MPCController,
    LocalRelativePredictionModel,
    planning_tracking_mpc_config,
    receding_plan_mpc_config,
)
from env.phase2_env import (
    perception_guidance_free_environment_config,
    phase2_perception_environment_config,
)
from env.scenarios import chaser_parameters
from env.se3_rendezvous_env import SE3RendezvousEnv


def test_a2_environment_is_a1_plus_guidance_flag_only() -> None:
    guided_config = phase2_perception_environment_config()
    free_config = perception_guidance_free_environment_config()
    guided_values = asdict(guided_config)
    free_values = asdict(free_config)
    assert not guided_values.pop("phase2_guidance_free")
    assert free_values.pop("phase2_guidance_free")
    assert guided_values == free_values

    guided = SE3RendezvousEnv(guided_config)
    free = SE3RendezvousEnv(free_config)
    try:
        guided_observation, _ = guided.reset(seed=262000)
        free_observation, _ = free.reset(seed=262000)
        np.testing.assert_array_equal(guided_observation, free_observation)
        assert np.linalg.norm(guided._reward.terminal_desired_velocity(guided.relative)) > 0
        np.testing.assert_array_equal(
            free._reward.terminal_desired_velocity(free.relative), np.zeros(3)
        )
    finally:
        guided.close()
        free.close()


def test_a2_endpoint_plans_start_from_observed_state_and_reset_cleanly() -> None:
    state = np.array(
        [0.1, -0.05, 0.02, -10.0, 0.4, -0.2, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
        dtype=np.float64,
    )
    local = LocalRelativePredictionModel(chaser_parameters(), 0.1)
    receding = MPCController(receding_plan_mpc_config(), local)
    tracked = MPCController(planning_tracking_mpc_config(), local)

    receding_reference = receding._reference_trajectory(state, 20.0)
    np.testing.assert_allclose(receding_reference[:, 0], state, atol=1.0e-12)
    tracked_initial = tracked._reference_trajectory(state, 20.0)
    np.testing.assert_allclose(tracked_initial[:, 0], state, atol=1.0e-12)
    tracked_later = tracked._reference_trajectory(state, 21.0)
    assert np.linalg.norm(tracked_later[3:6, 0] - state[3:6]) > 0.0
    tracked.reset()
    np.testing.assert_allclose(
        tracked._reference_trajectory(state, 40.0)[:, 0], state, atol=1.0e-12
    )
