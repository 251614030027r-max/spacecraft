"""The coupled task's time limit is a terminal, not a bootstrapped cut-off.

The normalised remaining time is in every coupled observation, so a state at
the 300 s limit has no continuation. The SAC replay target there must be the
reward alone, which is what the V3 value heads regress on as well.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

from train.hybrid_configs import SAC_MPC_HYBRID, hybrid_model_kwargs


def _buffer_after_timeout(**kwargs) -> ReplayBuffer:
    buffer = ReplayBuffer(
        8,
        spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        device="cpu",
        **kwargs,
    )
    # A VecEnv reports done = terminated or truncated and flags a pure
    # time-limit end as TimeLimit.truncated.
    buffer.add(
        np.zeros((1, 2)),  # observation
        np.zeros((1, 2)),  # next observation
        np.zeros((1, 2)),  # action
        np.array([1.0]),  # reward
        np.array([True]),
        [{"TimeLimit.truncated": True}],
    )
    return buffer


def test_the_hybrid_config_makes_the_time_limit_terminal() -> None:
    assert SAC_MPC_HYBRID.timeout_is_terminal
    kwargs = hybrid_model_kwargs(SAC_MPC_HYBRID)
    assert kwargs["replay_buffer_kwargs"] == {"handle_timeout_termination": False}
    assert "timeout_is_terminal" not in kwargs  # not an SB3 argument


def test_a_timeout_transition_is_not_bootstrapped() -> None:
    kwargs = hybrid_model_kwargs(SAC_MPC_HYBRID)["replay_buffer_kwargs"]
    sample = _buffer_after_timeout(**kwargs)._get_samples(np.array([0]))
    assert float(sample.dones[0, 0]) == 1.0
    # SB3's default would have bootstrapped through the same transition.
    default = _buffer_after_timeout()._get_samples(np.array([0]))
    assert float(default.dones[0, 0]) == 0.0
