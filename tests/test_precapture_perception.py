"""Non-cooperative precapture observation path (estimated state via A1 EKF).

The single factor is the observation source: ``perception=None`` keeps the 24D
truth full-state observation, perception enabled swaps it for the 29D estimated
schema. Truth still drives dynamics, reward, termination and geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics.relative import reconstruct_target_state, relative_state
from env.observation import (
    PRECAPTURE_PLANNING_ESTIMATED_SCHEMA,
    PRECAPTURE_PLANNING_FULL_STATE_SCHEMA,
)
from env.phase2_env import (
    precapture_perception_environment_config,
    precapture_planning_environment_config,
)
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv


def _rollout_obs(config, seed, steps):
    env = SE3RendezvousEnv(config)
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(np.zeros(6, dtype=np.float32))
        trace.append(obs)
        if term or trunc:
            break
    return env, trace


def test_truth_path_is_24d_and_finite():
    env, trace = _rollout_obs(precapture_planning_environment_config(), 262000, 10)
    assert env.config.perception is None
    assert env.config.phase2_observation_schema == PRECAPTURE_PLANNING_FULL_STATE_SCHEMA
    for obs in trace:
        assert obs.shape == (24,)
        assert np.all(np.isfinite(obs))


def test_estimated_path_is_29d_and_finite():
    config = precapture_perception_environment_config()
    assert config.perception is not None
    assert config.phase2_observation_schema == PRECAPTURE_PLANNING_ESTIMATED_SCHEMA
    env, trace = _rollout_obs(config, 262000, 30)
    for obs in trace:
        assert obs.shape == (29,)
        assert np.all(np.isfinite(obs))
    # the EKF is live and produces a measurement every frame
    assert env._relative_ekf is not None
    assert env._perception_measurement is not None


def test_estimate_differs_from_truth():
    """The whole point of the non-cooperative path: the policy sees an estimate,
    not the truth, so the two must actually diverge."""

    env, _ = _rollout_obs(precapture_perception_environment_config(), 262000, 20)
    truth = env.relative
    estimate = env.observed_relative
    position_error = float(np.linalg.norm(estimate.position - truth.position))
    assert position_error > 1.0e-3


def test_reconstruct_target_roundtrip():
    """The estimated observation reconstructs the target pose from the known
    chaser and the estimated relative state; that inverse must be exact on the
    truth pair."""

    env = SE3RendezvousEnv(precapture_planning_environment_config())
    env.reset(seed=262001)
    reconstructed = reconstruct_target_state(env.chaser_state, env.relative)
    round_trip = relative_state(reconstructed, env.chaser_state)
    assert np.allclose(round_trip.transform, env.relative.transform, atol=1.0e-9)
    assert np.allclose(round_trip.twist, env.relative.twist, atol=1.0e-9)


def test_estimated_observation_is_deterministic():
    """Same seed, same 29D observation, bitwise -- perception draws must be
    reproducible so a trained policy loads against the observation it saw."""

    _, trace_a = _rollout_obs(precapture_perception_environment_config(), 262000, 15)
    _, trace_b = _rollout_obs(precapture_perception_environment_config(), 262000, 15)
    assert len(trace_a) == len(trace_b)
    for a, b in zip(trace_a, trace_b):
        assert np.array_equal(a, b)


def test_truth_path_unchanged_by_perception_branch():
    """The estimated branch must not perturb the truth observation: with
    perception off the observation is identical across constructions."""

    _, trace_a = _rollout_obs(precapture_planning_environment_config(), 262002, 12)
    _, trace_b = _rollout_obs(precapture_planning_environment_config(), 262002, 12)
    for a, b in zip(trace_a, trace_b):
        assert np.array_equal(a, b)


def test_config_guards_estimated_schema():
    from dataclasses import replace

    from env.perception import PerceptionConfig

    # perception without the estimated schema is rejected
    bad = replace(precapture_perception_environment_config(),
                  phase2_observation_schema=PRECAPTURE_PLANNING_FULL_STATE_SCHEMA)
    with pytest.raises(ValueError):
        SE3RendezvousEnv(bad)

    # the estimated schema without a perception config is rejected
    bad2 = replace(precapture_planning_environment_config(),
                   phase2_observation_schema=PRECAPTURE_PLANNING_ESTIMATED_SCHEMA)
    with pytest.raises(ValueError):
        SE3RendezvousEnv(bad2)
