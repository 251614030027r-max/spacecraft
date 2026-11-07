"""Acceptance C (learning signal): M1 replay, critic loss normalisation, exact M0 fallback."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from train.t1_m1 import M1ReplayBuffer, M1SAC, auxiliary_critic_loss

OBS, ACT = 22, 2


class _Stub(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS, dtype=np.float32), 0.0, False, False, {}


def _model(cls, **kw):
    th.manual_seed(0)
    return cls("MlpPolicy", _Stub(), seed=0, device="cpu", buffer_size=500, learning_starts=0, batch_size=32,
               ent_coef=0.005, gamma=0.999, policy_kwargs={"net_arch": [16, 16]}, **kw)


def _fill(buffer, weights, rng):
    for w in weights:
        obs, nxt = rng.normal(size=OBS).astype(np.float32), rng.normal(size=OBS).astype(np.float32)
        act = rng.uniform(-1, 1, ACT).astype(np.float32)
        info = {"counterfactual": {"w": int(w), "a_tilde": rng.uniform(-1, 1, ACT).astype(np.float32),
                                   "r_hat": float(rng.normal()), "z_hat": rng.normal(size=OBS).astype(np.float32)}}
        buffer.add(obs[None], nxt[None], act[None], np.array([rng.normal()]), np.array([rng.uniform() < 0.1]), [info])


def _params(model):
    return th.cat([p.detach().flatten() for p in model.policy.parameters()])


def test_buffer_writes_and_reads_the_cached_counterfactual():
    rng = np.random.default_rng(1)
    m = _model(M1SAC)
    infos = []
    for k in range(5):
        info = {"counterfactual": {"w": k % 2, "a_tilde": np.full(ACT, k / 10, np.float32), "r_hat": -k,
                                   "z_hat": np.full(OBS, k, np.float32)}}
        infos.append(info)
        m.replay_buffer.add(np.zeros((1, OBS)), np.zeros((1, OBS)), np.zeros((1, ACT)), np.zeros(1), np.zeros(1), [info])
    m.replay_buffer._get_samples(np.array([0, 1, 3, 4]))
    cf = m.replay_buffer.counterfactual()
    assert cf.weights.flatten().tolist() == [0, 1, 1, 0]
    assert cf.rewards.flatten().tolist() == [0, -1, -3, 0]  # w = 0 rows are zeroed
    assert np.allclose(cf.next_observations[2].numpy(), 3.0) and np.allclose(cf.actions[1].numpy(), 0.1)


def test_auxiliary_loss_is_averaged_over_the_whole_batch_not_over_sum_w():
    q = [th.tensor([[1.0], [2.0], [3.0], [4.0]]), th.tensor([[0.0], [0.0], [0.0], [0.0]])]
    y = th.zeros(4, 1)
    w = th.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = auxiliary_critic_loss(q, y, w, lam=0.5)
    expected = 0.5 * (0.5 * (1 + 9) / 4 + 0.0)
    assert loss.item() == pytest.approx(expected)
    assert auxiliary_critic_loss(q, y, th.zeros(4, 1), 0.5).item() == 0.0


def test_all_zero_weights_reproduce_the_m0_update_exactly():
    rng = np.random.default_rng(2)
    m0, m1 = _model(SAC), _model(M1SAC, cf_warmup_fraction=0.0)
    assert th.equal(_params(m0), _params(m1))
    _fill(m0.replay_buffer, [0] * 200, np.random.default_rng(2))
    _fill(m1.replay_buffer, [0] * 200, np.random.default_rng(2))
    for model in (m0, m1):
        model._total_timesteps, model.num_timesteps = 1000, 500
        model._setup_learn(1000)
    np.random.seed(5); th.manual_seed(5)
    m0.train(gradient_steps=3, batch_size=32)
    np.random.seed(5); th.manual_seed(5)
    m1.train(gradient_steps=3, batch_size=32)
    assert th.equal(_params(m0), _params(m1))


def test_valid_counterfactuals_move_both_critics_and_through_them_the_actor():
    runs = {}
    for lam in (0.0, 0.5):
        m = _model(M1SAC, cf_lambda=lam, cf_warmup_fraction=0.0)
        _fill(m.replay_buffer, [1] * 200, np.random.default_rng(3))
        m._setup_learn(1000)
        m._total_timesteps, m.num_timesteps = 1000, 500
        np.random.seed(7); th.manual_seed(7)
        m.train(gradient_steps=2, batch_size=32)
        runs[lam] = m
    q0 = [th.cat([p.flatten() for p in net.parameters()]) for net in runs[0.0].critic.q_networks]
    q1 = [th.cat([p.flatten() for p in net.parameters()]) for net in runs[0.5].critic.q_networks]
    assert all(not th.equal(a, b) for a, b in zip(q0, q1))
    a0 = th.cat([p.flatten() for p in runs[0.0].actor.parameters()])
    a1 = th.cat([p.flatten() for p in runs[0.5].actor.parameters()])
    assert not th.equal(a0, a1)
    assert runs[0.5].cf_stats["aux_loss"][-1] > 0.0 and runs[0.0].cf_stats["aux_loss"][-1] == 0.0


def test_warmup_keeps_lambda_zero():
    m = _model(M1SAC, cf_lambda=0.5, cf_warmup_fraction=0.15)
    m._total_timesteps = 10_000
    m.num_timesteps = 1_499
    assert m.current_lambda() == 0.0
    m.num_timesteps = 1_500
    assert m.current_lambda() == 0.5


def test_m1_buffer_is_a_plain_replay_buffer_otherwise():
    assert issubclass(M1ReplayBuffer, ReplayBuffer)


def test_buffer_keeps_each_environments_counterfactual_apart():
    import gymnasium as gym_
    space_o = gym_.spaces.Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float32)
    space_a = gym_.spaces.Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32)
    b = M1ReplayBuffer(50, space_o, space_a, device="cpu", n_envs=2)
    for k in range(10):
        infos = [{"counterfactual": {"w": 1, "a_tilde": np.full(ACT, e + k / 100, np.float32), "r_hat": 10 * e + k,
                                     "z_hat": np.full(OBS, e, np.float32)}} if (k + e) % 2 == 0 else {}
                 for e in range(2)]
        obs = np.stack([np.full(OBS, 10 * e + k, np.float32) for e in range(2)])
        b.add(obs, obs, np.zeros((2, ACT)), np.zeros(2), np.zeros(2), infos)
    np.random.seed(0)
    samples = b.sample(40)
    cf = b.counterfactual()
    for j in range(40):
        o = samples.observations[j, 0].item()
        e, k = int(o // 10), int(o % 10)
        valid = (k + e) % 2 == 0
        assert cf.weights[j, 0].item() == (1.0 if valid else 0.0)
        if valid:
            assert cf.rewards[j, 0].item() == 10 * e + k and cf.next_observations[j, 0].item() == e
