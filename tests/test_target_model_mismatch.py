"""Tests for the target model-mismatch probe (truth params vs controller nominal).

Single factor = the amplitude of the truth-vs-nominal target inertia/mass
mismatch. These pin that the sampler is valid and reproducible, mismatch=0 is
byte-nominal, the trajectory cache keys on the mismatch seed, and the truth
trajectory (not the task geometry) is what changes.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from env.phase2_env import phase2_environment_config
from env.scenarios import sample_target_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousEnv, _cached_target_trajectory


def test_zero_mismatch_is_nominal() -> None:
    nominal = target_parameters()
    sampled = sample_target_parameters(mismatch=0.0, seed=1)
    assert sampled.mass == nominal.mass
    assert np.array_equal(sampled.inertia, nominal.inertia)


def test_sampled_parameters_are_spd_reproducible_and_bounded() -> None:
    nominal = target_parameters()
    a = sample_target_parameters(mismatch=0.30, seed=7)
    b = sample_target_parameters(mismatch=0.30, seed=7)
    c = sample_target_parameters(mismatch=0.30, seed=8)
    assert np.allclose(a.inertia, b.inertia) and a.mass == b.mass
    assert not np.allclose(a.inertia, c.inertia)
    for params in (a, c):
        assert np.allclose(params.inertia, params.inertia.T)
        assert float(np.linalg.eigvalsh(params.inertia).min()) > 0.0
        # Principal moments stay within the requested +-30% band.
        ratio = np.sort(np.linalg.eigvalsh(params.inertia)) / np.sort(
            np.linalg.eigvalsh(nominal.inertia)
        )
        assert ratio.min() >= 1.0 - 0.30 - 1e-9
        assert ratio.max() <= 1.0 + 0.30 + 1e-9


def test_mismatch_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        sample_target_parameters(mismatch=1.0, seed=0)


def test_trajectory_cache_keys_on_mismatch_seed() -> None:
    base = (5.0, 0.1, True, 1.0e-7, 1.0e-9, 0.20, None)
    nominal = _cached_target_trajectory(*base, 0.0, None)
    truth_a = _cached_target_trajectory(*base, 0.30, 11)
    truth_b = _cached_target_trajectory(*base, 0.30, 12)
    truth_a_again = _cached_target_trajectory(*base, 0.30, 11)
    assert truth_a is truth_a_again
    # A wrong inertia bends the free tumble, so the trajectories diverge in time.
    assert not np.allclose(truth_a[-1].rotation, nominal[-1].rotation)
    assert not np.allclose(truth_a[-1].rotation, truth_b[-1].rotation)


def test_env_uses_truth_parameters_and_nominal_is_unchanged() -> None:
    env = SE3RendezvousEnv(
        replace(
            phase2_environment_config("single_phase"),
            phase2_target_model_mismatch=0.30,
        )
    )
    try:
        _, info = env.reset(seed=970000)
        assert info["episode_target_model_mismatch"] == pytest.approx(0.30)
        assert info["episode_target_model_mismatch_seed"] is not None
        # env.target_parameters is the truth: perturbed away from nominal.
        assert not np.allclose(
            env.target_parameters.inertia, target_parameters().inertia
        )
    finally:
        env.close()

    nominal_env = SE3RendezvousEnv(phase2_environment_config("single_phase"))
    try:
        _, info = nominal_env.reset(seed=970000)
        assert info["episode_target_model_mismatch"] == 0.0
        assert info["episode_target_model_mismatch_seed"] is None
        assert np.array_equal(
            nominal_env.target_parameters.inertia, target_parameters().inertia
        )
    finally:
        nominal_env.close()


def test_evaluate_accepts_truth_and_nominal_controller_models() -> None:
    from controllers.mpc import corridor_tracking_mpc_config
    from experiments.evaluate_mpc import evaluate

    env_config = replace(
        phase2_environment_config("single_phase"),
        phase2_target_model_mismatch=0.30,
        max_time_s=1.0,
    )
    for source in ("nominal", "truth"):
        result = evaluate(
            corridor_tracking_mpc_config(horizon_steps=5),
            episodes=1,
            seed=970000,
            environment_config=env_config,
            controller_model_source=source,
        )
        assert result["main_table"]["episodes"] == 1
    with pytest.raises(ValueError):
        evaluate(
            corridor_tracking_mpc_config(horizon_steps=5),
            episodes=1,
            seed=970000,
            environment_config=env_config,
            controller_model_source="bogus",
        )
