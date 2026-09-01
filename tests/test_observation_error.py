"""Tests for the target state-observation-error probe (partial observability).

Single factor = the amplitude of the controller's target-pose estimation error
(attitude bias, delay, low update rate). Truth is untouched: the estimator only
changes what the controller is told; the env still judges violations on real
geometry. These pin the estimator's mechanics and that the eval path threads a
corrupted estimate to the controller while leaving truth metrics intact.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dynamics.lie import so3_log
from env.observation_error import TargetStateEstimator
from env.phase2_env import phase2_environment_config
from env.scenarios import target_initial_state


def _state(step_rotation=None):
    s = target_initial_state(tumble_scale=0.20)
    if step_rotation is not None:
        s = replace(s, rotation=step_rotation)
    return s


def test_no_error_is_identity_passthrough() -> None:
    est = TargetStateEstimator()
    s = _state()
    out = est.estimate(s, 0)
    assert np.array_equal(out.rotation, s.rotation)
    assert np.array_equal(out.omega, s.omega)


def test_bias_is_a_fixed_rotation_of_the_requested_size() -> None:
    bias = np.deg2rad(2.0)
    est = TargetStateEstimator(bias_rad=bias, bias_seed=3)
    s = _state()
    out = est.estimate(s, 0)
    # The applied rotation offset has exactly the requested angle.
    offset = out.rotation @ s.rotation.T
    assert np.linalg.norm(so3_log(offset)) == pytest.approx(bias, abs=1e-9)
    # Constant across steps (persistent bias, not per-step noise).
    out2 = est.estimate(s, 1)
    assert np.allclose(out.rotation, out2.rotation)


def test_bias_seed_is_reproducible_and_varies() -> None:
    s = _state()
    a = TargetStateEstimator(bias_rad=0.02, bias_seed=7).estimate(s, 0).rotation
    b = TargetStateEstimator(bias_rad=0.02, bias_seed=7).estimate(s, 0).rotation
    c = TargetStateEstimator(bias_rad=0.02, bias_seed=8).estimate(s, 0).rotation
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_delay_lags_the_truth() -> None:
    est = TargetStateEstimator(delay_steps=2)
    states = []
    rng = np.random.default_rng(0)
    from env.scenarios import _uniform_rotation

    for step in range(5):
        s = replace(_state(), rotation=_uniform_rotation(rng))
        states.append(s)
        out = est.estimate(s, step)
        # With delay 2, step k sees truth from max(0, k-2).
        expected = states[max(0, step - 2)]
        assert np.allclose(out.rotation, expected.rotation)


def test_low_update_rate_holds_between_refreshes() -> None:
    est = TargetStateEstimator(update_every=3)
    rng = np.random.default_rng(1)
    from env.scenarios import _uniform_rotation

    outs = []
    for step in range(6):
        s = replace(_state(), rotation=_uniform_rotation(rng))
        outs.append(est.estimate(s, step).rotation.copy())
    # Held on steps 1,2 (== step 0's fix) and 4,5 (== step 3's fix).
    assert np.allclose(outs[0], outs[1]) and np.allclose(outs[1], outs[2])
    assert np.allclose(outs[3], outs[4]) and np.allclose(outs[4], outs[5])
    assert not np.allclose(outs[2], outs[3])  # refreshed at step 3


def test_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        TargetStateEstimator(bias_rad=-1.0)
    with pytest.raises(ValueError):
        TargetStateEstimator(delay_steps=-1)
    with pytest.raises(ValueError):
        TargetStateEstimator(update_every=0)


def test_ekf_propagate_filter_reduces_lag_but_keeps_bias() -> None:
    """The predict-step filter de-lags a delayed fix; a constant bias survives."""

    from dynamics.lie import so3_log
    from controllers.mpc.prediction import RelativePredictionModel
    from env.scenarios import chaser_parameters, target_parameters
    from env.se3_rendezvous_env import _cached_target_trajectory

    truth = _cached_target_trajectory(3.0, 0.1, True, 1.0e-7, 1.0e-9, 0.20)
    model = RelativePredictionModel(
        target_parameters=target_parameters(),
        chaser_parameters=chaser_parameters(),
        dt_s=0.1,
    )
    delay = 4
    hold = TargetStateEstimator(delay_steps=delay, filter_mode="hold")
    ekf = TargetStateEstimator(
        delay_steps=delay, filter_mode="propagate",
        propagate_step=model.propagate_target, dt_s=0.1,
    )
    step = 20
    for k in range(step + 1):
        h = hold.estimate(truth[k], k, k * 0.1)
        e = ekf.estimate(truth[k], k, k * 0.1)
    now = truth[step]

    def ang(a, b):
        return float(np.linalg.norm(so3_log(a.rotation.T @ b.rotation)))

    # The held fix lags by `delay` steps of tumble; the propagated one is close.
    assert ang(h, now) > ang(e, now)
    assert ang(e, now) < 0.5 * ang(h, now)


def test_ekf_propagate_requires_a_callable() -> None:
    with pytest.raises(ValueError):
        TargetStateEstimator(delay_steps=2, filter_mode="propagate")


def test_corridor_speed_fraction_lowers_the_reference_speed() -> None:
    from controllers.mpc import corridor_tracking_mpc_config

    fast = corridor_tracking_mpc_config(horizon_steps=6, corridor_speed_fraction=0.6)
    slow = corridor_tracking_mpc_config(horizon_steps=6, corridor_speed_fraction=0.4)
    assert slow.corridor_speed_fraction == 0.4
    with pytest.raises(ValueError):
        corridor_tracking_mpc_config(corridor_speed_fraction=0.0)


def test_env_observation_error_is_off_by_default_and_corrupts_when_on() -> None:
    """Env-level obs error (for training) must be bitwise-off by default."""

    from env.phase2_env import phase2_environment_config
    from env.se3_rendezvous_env import SE3RendezvousEnv

    base = phase2_environment_config("single_phase")
    actions = [np.zeros(6), 0.4 * np.ones(6), -0.2 * np.ones(6)]

    def rollout(config):
        env = SE3RendezvousEnv(config)
        try:
            obs, info = env.reset(seed=970000)
            seq = [obs]
            rewards = []
            for a in actions:
                obs, r, _, _, _ = env.step(a)
                seq.append(obs)
                rewards.append(r)
        finally:
            env.close()
        return seq, rewards, info

    off_seq, off_rewards, off_info = rollout(base)
    off_seq2, _, _ = rollout(base)
    assert all(np.array_equal(a, b) for a, b in zip(off_seq, off_seq2))
    assert off_info["observation_error_active"] is False

    on_config = replace(
        base,
        phase2_observation_bias_rad=np.deg2rad(2.0),
        phase2_observation_delay_steps=3,
    )
    on_seq, on_rewards, on_info = rollout(on_config)
    on_seq_again, _, _ = rollout(on_config)
    assert on_info["observation_error_active"] is True
    assert not np.array_equal(off_seq[0], on_seq[0])  # observation is corrupted
    assert on_seq[0].shape == (24,)  # schema preserved
    # Reproducible from the episode seed.
    assert all(np.array_equal(a, b) for a, b in zip(on_seq, on_seq_again))
    # Reward comes from truth: identical zero-action rewards regardless of what
    # the observation was corrupted to.
    assert np.allclose(off_rewards[0], on_rewards[0])


def test_evaluate_runs_with_observation_error_and_truth_metrics() -> None:
    from controllers.mpc import corridor_tracking_mpc_config
    from experiments.evaluate_mpc import evaluate

    env_config = replace(phase2_environment_config("single_phase"), max_time_s=1.0)
    result = evaluate(
        corridor_tracking_mpc_config(horizon_steps=5),
        episodes=1,
        seed=970000,
        environment_config=env_config,
        observation_bias_rad=np.deg2rad(2.0),
        observation_delay_steps=3,
        observation_update_every=3,
    )
    assert result["observation_error"]["delay_steps"] == 3
    assert result["main_table"]["episodes"] == 1
