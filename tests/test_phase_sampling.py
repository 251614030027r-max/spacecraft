"""Tests for the single_phase_phase_sampled sub-task (route A, stage 0).

The single factor is the target's initial phase: attitude and tumble direction
sampled per episode with the rate magnitude frozen. These pin that the nominal
task is byte-unchanged, the sampling is reproducible and magnitude-preserving,
the trajectory cache keys on the phase seed, and the eval paths accept the task.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from env.phase2_env import make_phase2_env, phase2_environment_config
from env.scenarios import TARGET_BASE_TUMBLE_RAD_S, target_initial_state
from env.se3_rendezvous_env import _cached_target_trajectory


def test_nominal_state_is_unchanged() -> None:
    """phase_seed=None must reproduce the single deterministic realisation."""

    state = target_initial_state(tumble_scale=0.20)
    assert np.array_equal(state.rotation, np.eye(3))
    assert np.allclose(state.omega, 0.20 * TARGET_BASE_TUMBLE_RAD_S)


def test_sampling_freezes_magnitude_and_is_reproducible() -> None:
    magnitude = float(np.linalg.norm(0.20 * TARGET_BASE_TUMBLE_RAD_S))
    a = target_initial_state(tumble_scale=0.20, phase_seed=12345)
    b = target_initial_state(tumble_scale=0.20, phase_seed=12345)
    c = target_initial_state(tumble_scale=0.20, phase_seed=999)

    assert np.allclose(a.rotation, b.rotation) and np.allclose(a.omega, b.omega)
    assert not np.allclose(a.rotation, c.rotation)
    for state in (a, c):
        assert abs(float(np.linalg.norm(state.omega)) - magnitude) < 1e-12
        assert not np.allclose(state.rotation, np.eye(3))
        assert abs(float(np.linalg.det(state.rotation)) - 1.0) < 1e-9
        assert np.allclose(state.rotation @ state.rotation.T, np.eye(3), atol=1e-9)


def test_trajectory_cache_keys_on_phase_seed() -> None:
    """Different phases must give different trajectories; the same phase, one."""

    args = (5.0, 0.1, True, 1.0e-7, 1.0e-9, 0.20)
    nominal = _cached_target_trajectory(*args, None)
    phase_a = _cached_target_trajectory(*args, 7)
    phase_b = _cached_target_trajectory(*args, 8)
    phase_a_again = _cached_target_trajectory(*args, 7)

    assert phase_a is phase_a_again  # cache hit on the same key
    assert not np.allclose(phase_a[0].rotation, nominal[0].rotation)
    assert not np.allclose(phase_a[-1].rotation, phase_b[-1].rotation)


def test_sampled_config_keeps_single_phase_semantics() -> None:
    config = phase2_environment_config("single_phase_phase_sampled")
    # Only the sampling flag differs from single_phase; the training mode -- and
    # with it every task semantic -- is unchanged, so it stays the single factor.
    assert config.phase2_training_mode == "single_phase"
    assert config.phase2_target_phase_sampling is True
    assert phase2_environment_config("single_phase").phase2_target_phase_sampling is False


def test_sampled_env_varies_per_episode_and_nominal_does_not() -> None:
    env = make_phase2_env("single_phase_phase_sampled")
    try:
        _, info_a = env.reset(seed=990000)
        rot_a = env.target_state.rotation.copy()
        _, info_b = env.reset(seed=990001)
        rot_b = env.target_state.rotation.copy()
        _, info_a2 = env.reset(seed=990000)
    finally:
        env.close()
    assert info_a["episode_target_phase_seed"] is not None
    assert info_a["episode_target_phase_seed"] != info_b["episode_target_phase_seed"]
    # Reproducible from the episode seed.
    assert info_a["episode_target_phase_seed"] == info_a2["episode_target_phase_seed"]
    assert not np.allclose(rot_a, rot_b)
    assert not np.allclose(rot_a, np.eye(3))

    nominal = make_phase2_env("single_phase")
    try:
        _, info = nominal.reset(seed=990000)
        assert np.allclose(nominal.target_state.rotation, np.eye(3))
        assert info["episode_target_phase_seed"] is None
    finally:
        nominal.close()


def test_mpc_evaluate_runs_on_the_phase_sampled_task() -> None:
    from controllers.mpc import corridor_tracking_mpc_config
    from experiments.evaluate_mpc import evaluate

    env_config = replace(
        phase2_environment_config("single_phase_phase_sampled"), max_time_s=1.0
    )
    result = evaluate(
        corridor_tracking_mpc_config(horizon_steps=5),
        episodes=1,
        seed=990000,
        environment_config=env_config,
    )
    assert result["main_table"]["episodes"] == 1
    assert result["environment"]["phase2_target_phase_sampling"] is True
