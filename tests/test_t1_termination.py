"""P0-2: the four ways a T1 decision ends, and what SB3 stores for each."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC

from dynamics.lie import make_transform
from dynamics.relative import RelativeState, reconstruct_chaser_state
from env.t1_env import T1Reward
from env.t1_gym import T1GStarEnv
from train.train_t1 import T1_SAC, model_kwargs

KEEP = np.array([0.0, 0.0])


def _fresh(opening=291000):
    env = T1GStarEnv(openings=[opening])
    env.reset()
    env.step(np.array([-0.2, 0.0]))
    return env


def _costs(env, before):
    s = env.env.summary()
    steps = s["control_steps"] - before["control_steps"]
    dv = s["delta_v_m_s"] - before["delta_v_m_s"]
    r = T1Reward()
    return steps, -r.time_cost_per_s * 0.1 * steps - r.delta_v_cost_per_m_s * dv


def test_timeout_is_terminal_and_charged_for_the_steps_actually_flown():
    env = _fresh()
    env.env.time_s = 299.0
    before = env.env.summary()
    _, reward, terminated, truncated, info = env.step(KEEP)
    steps, costs = _costs(env, before)
    assert info["event"] == "timeout" and terminated and not truncated
    assert steps == 10
    assert reward == pytest.approx(costs - T1Reward().timeout_penalty, abs=1e-9)


def test_unsafe_distance_is_terminal_mid_decision_with_the_penalty():
    env = _fresh()
    t, c = env.env.target_state, env.env.chaser_state
    u = (c.position - t.position) / np.linalg.norm(c.position - t.position)
    c.position = t.position + 29.98 * u
    c.velocity = c.rotation.T @ (t.rotation @ t.velocity + 0.5 * u)
    before = env.env.summary()
    _, reward, terminated, truncated, info = env.step(KEEP)
    steps, costs = _costs(env, before)
    assert info["event"] == "unsafe" and "distance" in info["violations"] and terminated and not truncated
    assert 1 <= steps < 20
    assert reward == pytest.approx(costs - T1Reward().unsafe_penalty, abs=1e-9)


def test_success_is_terminal_after_the_hold_with_the_capture_reward():
    env = _fresh()
    task = env.env.config.task
    t = env.env.target_state
    env.env.chaser_state = reconstruct_chaser_state(
        t, RelativeState(make_transform(np.eye(3), task.desired_position), np.zeros(6)))
    before = env.env.summary()
    _, reward, terminated, truncated, info = env.step(KEEP)
    steps, costs = _costs(env, before)
    assert info["event"] == "success" and terminated and not truncated
    assert steps == task.completion_required_steps(0.1)
    assert reward == pytest.approx(costs + T1Reward().capture_reward, abs=1e-9)


def test_numerical_failure_is_a_truncation_without_penalty_and_without_the_failed_step():
    env = _fresh()
    command = env.env.controller.command
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise FloatingPointError("injected")
        return command(*args, **kwargs)

    env.env.controller.command = failing
    before = env.env.summary()
    _, reward, terminated, truncated, info = env.step(KEEP)
    steps, costs = _costs(env, before)
    assert info["event"] == "numerical_failure" and truncated and not terminated
    assert steps == 2 and reward == pytest.approx(costs, abs=1e-9)


class _Scripted(gym.Env):
    """Episode of 3 steps ending as scripted; obs = [episode, step, 0...]."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, endings):
        self.endings, self.episode, self.k = list(endings), -1, 0

    def _obs(self):
        o = np.zeros(22, dtype=np.float32)
        o[0], o[1] = self.episode, self.k
        return o

    def reset(self, *, seed=None, options=None):
        self.episode += 1
        self.k = 0
        return self._obs(), {}

    def step(self, action):
        self.k += 1
        end = self.endings[self.episode] if self.k == 3 else None
        return self._obs(), float(-self.k), end == "terminated", end == "truncated", {}


def test_sb3_stores_terminals_without_bootstrap_and_truncations_with_it():
    assert T1_SAC.gamma == T1Reward().gamma == 0.999
    kwargs = model_kwargs(T1_SAC, [8])
    kwargs.update(learning_starts=10_000, buffer_size=100)
    model = SAC("MlpPolicy", _Scripted(["terminated", "truncated", "terminated"]), seed=0, device="cpu", **kwargs)
    model.learn(total_timesteps=9)
    b = model.replay_buffer
    obs, nxt = b.observations[:9, 0], b.next_observations[:9, 0]
    dones, timeouts = b.dones[:9, 0], b.timeouts[:9, 0]
    for i in range(9):
        assert nxt[i][0] == obs[i][0] and nxt[i][1] == obs[i][1] + 1  # next obs is the terminal one, not the reset one
    assert dones.tolist() == [0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert timeouts.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0]  # only the truncation
    sampled_done = dones * (1 - timeouts)
    assert sampled_done.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 1]  # truncation bootstraps, terminals do not
    assert b.rewards[:9, 0].tolist() == [-1, -2, -3] * 3


def test_terminal_record_is_taken_before_any_auto_reset():
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: T1GStarEnv(openings=[291000, 291013])])
    vec.reset()
    inner = vec.envs[0]
    inner.env.time_s = 299.0
    _, _, dones, infos = vec.step(np.array([[0.0, 0.0]]))
    summary = infos[0]["episode_summary"]
    assert dones[0] and summary["event"] == "timeout"
    assert summary["opening"] == 291000 and inner.current_opening == 291013  # vec env already reset
    assert summary["end_phase"] in ("far_field", "entry", "near_field") and summary["end_range_m"] > 0
