"""Aggregate deterministic estimator-chain acceptance test for A1."""

import numpy as np

from controllers.mpc.prediction import RelativePredictionModel, relative_to_vector
from dynamics.lie import make_transform, se3_exp
from dynamics.relative import RelativeState, reconstruct_chaser_state
from env.perception import PerceptionConfig, measure_visible_features
from env.scenarios import chaser_parameters, target_initial_state, target_parameters
from estimation.relative_ekf import RelativeStateEKF, local_error


def test_a1_relative_ekf_remains_finite_and_reduces_pose_error_over_20_s() -> None:
    rng = np.random.default_rng(20260902)
    perception = PerceptionConfig()
    chaser_params = chaser_parameters()
    target_params = target_parameters()
    truth_model = RelativePredictionModel(target_params, chaser_params, dt_s=0.1)
    ekf = RelativeStateEKF(
        chaser_params,
        perception,
        target_parameters=target_params,
        dt_s=0.1,
    )
    target = target_initial_state(tumble_scale=0.20)
    truth = RelativeState(
        make_transform(np.eye(3), np.array([-5.0, 0.0, 0.0])),
        np.zeros(6),
    )
    chaser = reconstruct_chaser_state(target, truth)
    ekf.initialize(truth, rng)
    initial_error = np.abs(local_error(truth, ekf.state))
    update_count = 0
    zero_wrench = np.zeros(6)

    for step in range(200):
        chaser_previous = chaser
        truth_vector, target = truth_model.predict(
            relative_to_vector(truth),
            zero_wrench,
            target,
            step * 0.1,
        )
        truth = RelativeState(se3_exp(truth_vector[:6]), truth_vector[6:])
        chaser = reconstruct_chaser_state(target, truth)
        ekf.predict(
            zero_wrench,
            chaser_state=chaser_previous,
            time_seconds=step * 0.1,
        )
        update_count += int(
            ekf.update(measure_visible_features(truth, perception, rng))
        )
        assert np.all(np.isfinite(relative_to_vector(ekf.state)))
        assert np.all(np.isfinite(ekf.covariance))

    final_error = np.abs(local_error(truth, ekf.state))
    assert update_count > 0
    assert np.linalg.norm(final_error[:3]) < np.linalg.norm(initial_error[:3])
    assert np.linalg.norm(final_error[3:6]) < np.linalg.norm(initial_error[3:6])
