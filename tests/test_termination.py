import numpy as np

from dynamics.lie import se3_exp
from dynamics.relative import RelativeState
from env.termination import SuccessThresholds, compute_error_metrics


def test_joint_success_requires_all_four_physical_errors() -> None:
    thresholds = SuccessThresholds()
    relative = RelativeState(
        se3_exp(np.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0])),
        np.array([0.01, 0.0, 0.0, 0.05, 0.0, 0.0]),
    )
    metrics = compute_error_metrics(relative, thresholds)
    assert metrics.attitude_success
    assert metrics.position_success
    assert metrics.joint_success


def test_excess_velocity_fails_position_success() -> None:
    relative = RelativeState(
        np.eye(4), np.array([0.0, 0.0, 0.0, 0.11, 0.0, 0.0])
    )
    metrics = compute_error_metrics(relative)
    assert not metrics.position_success
    assert not metrics.joint_success
