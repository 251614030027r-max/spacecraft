"""Aggregate A3 mismatch and deployable-baseline acceptance tests."""

from dataclasses import asdict

import numpy as np

from controllers.mpc import LocalRelativePredictionModel, MPCController, online_endpoint_mpc_config
from env.phase2_env import (
    A3_PREDICTION_MODEL_MISMATCH,
    A3_PREDICTION_MODEL_SEED,
    deployable_planning_environment_config,
    perception_guidance_free_environment_config,
)
from env.scenarios import (
    chaser_parameters,
    fixed_prediction_target_parameters,
    target_parameters,
)
from env.se3_rendezvous_env import SE3RendezvousEnv
from experiments.evaluate_a3 import method_spec


def test_a3_uses_one_wrong_model_for_ekf_while_truth_stays_nominal() -> None:
    a2 = asdict(perception_guidance_free_environment_config())
    a3_config = deployable_planning_environment_config()
    a3 = asdict(a3_config)
    assert a2.pop("phase2_prediction_model_mismatch") == 0.0
    assert a2.pop("phase2_prediction_model_seed") == 0
    assert a3.pop("phase2_prediction_model_mismatch") == 0.20
    assert a3.pop("phase2_prediction_model_seed") == 260903
    assert a2 == a3

    expected = fixed_prediction_target_parameters(
        mismatch=A3_PREDICTION_MODEL_MISMATCH,
        seed=A3_PREDICTION_MODEL_SEED,
    )
    env = SE3RendezvousEnv(a3_config)
    try:
        env.reset(seed=262000)
        np.testing.assert_array_equal(
            env.target_parameters.inertia, target_parameters().inertia
        )
        assert env._relative_ekf is not None
        np.testing.assert_array_equal(
            env._relative_ekf._mean_prediction.target_parameters.inertia,
            expected.inertia,
        )
        assert not np.array_equal(expected.inertia, env.target_parameters.inertia)
    finally:
        env.close()


def test_a3_online_reference_is_endpoint_only_and_realistic_row_is_estimated() -> None:
    config, _, source = method_spec("deployable_online", deployable_horizon=10)
    assert source == "estimate"
    assert config.reference_source == "online_endpoint"
    assert config.linearization_source == "analytic_local"
    assert not config.runtime_diagnostics

    state = np.array(
        [0.1, -0.05, 0.02, -10.0, 0.4, -0.2, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
        dtype=np.float64,
    )
    controller = MPCController(
        online_endpoint_mpc_config(horizon_steps=10),
        LocalRelativePredictionModel(chaser_parameters(), 0.1),
    )
    reference = controller._reference_trajectory(state)
    np.testing.assert_allclose(reference[:6, 0], state[:6], atol=1.0e-12)
    assert np.linalg.norm(reference[9:12, 0]) > 0.0
    assert np.linalg.norm(reference[3:6, -1] - config.reference_state[3:6]) < (
        np.linalg.norm(reference[3:6, 0] - config.reference_state[3:6])
    )
