"""Single G0 guard: the estimated controller path cannot access truth."""

import numpy as np

from dynamics.relative import relative_state
from env.phase2_env import phase2_perception_environment_config
from env.se3_rendezvous_env import SE3RendezvousEnv
from experiments.evaluate_mpc import controller_inputs


def test_estimate_controller_inputs_do_not_read_environment_truth() -> None:
    source = SE3RendezvousEnv(phase2_perception_environment_config())
    try:
        source.reset(seed=262000)
        observed = source.observed_relative
        chaser = source.chaser_state
        assert chaser is not None

        class EstimateOnlyView:
            observed_relative = observed
            chaser_state = chaser

            @property
            def relative(self):
                raise AssertionError("estimated controller read env.relative")

            @property
            def target_state(self):
                raise AssertionError("estimated controller read env.target_state")

        vector, target_estimate = controller_inputs(EstimateOnlyView(), "estimate")
        reconstructed = relative_state(target_estimate, chaser)
        assert np.allclose(vector[:6], observed.exponential_coordinates)
        assert np.allclose(vector[6:], observed.twist)
        assert np.allclose(reconstructed.transform, observed.transform)
        assert np.allclose(reconstructed.twist, observed.twist)
    finally:
        source.close()
