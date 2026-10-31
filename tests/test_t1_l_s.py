"""T1-L-S contracts (plan 2026-10-16, sec. 7 B-D): review-interval replay, target routing, trainer
clocks and learning rate, and the old update path."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from env.t1_task_review import DUE_INDEX, GAP_INDEX, REVIEW_OBSERVATION_NAMES
from train.t1_option_sac import OptionSAC
from train.t1_tcpf import CosineLearningRate, IntervalReplayBuffer, TCPFSAC

OBS = len(REVIEW_OBSERVATION_NAMES)
G = 0.999


def _buffer(size=50, n_envs=1):
    b = IntervalReplayBuffer(size, gym.spaces.Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float32),
                             gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32), device="cpu", n_envs=n_envs)
    b.gamma = G
    return b


def _o(tag, due):
    o = np.zeros(OBS, dtype=np.float32)
    o[0] = tag
    o[DUE_INDEX] = float(due)
    return o


def _add(b, obs, nxt, rewards, dones, truncated=None):
    n = b.n_envs
    truncated = truncated or [False] * n
    infos = [{"TimeLimit.truncated": bool(t)} if d else {} for d, t in zip(dones, truncated)]
    b.add(np.stack(obs), np.stack(nxt), np.zeros((n, 3)), np.asarray(rewards, float), np.asarray(dones, float), infos)


def _run(b, dues, rewards, end_done=None, truncated=False):
    """One env: states tagged k, transition k: s_k -> s_{k+1}; optional terminal at the last step."""

    for k, (due, r) in enumerate(zip(dues, rewards)):
        last = k == len(dues) - 1
        done = bool(end_done) and last
        _add(b, [_o(k, due)], [_o(k + 1, 0 if done else dues[k + 1] if not last else 1)], [r], [done],
             [truncated and done])


def test_six_second_interval_folds_three_rewards_and_ends_at_the_next_review():
    b = _buffer()
    _run(b, [1, 0, 0, 1, 0, 0, 1], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert b.interval_ready[0, 0] and b.interval_steps[0, 0] == 3
    assert b.interval_return[0, 0, 0] == pytest.approx(1 + G * 2 + G * G * 3)
    assert b.interval_discount[0, 0, 0] == pytest.approx(G ** 3)
    assert b.interval_end_obs[0, 0, 0] == 3.0 and b.interval_done[0, 0, 0] == 0.0
    assert b.interval_ready[3, 0] and b.interval_return[3, 0, 0] == pytest.approx(4 + G * 5 + G * G * 6)
    assert not b.interval_ready[1:3, 0].any() and not b.interval_ready[6, 0]  # continues; the open last review
    assert b.interval_lengths == {3: 2}


def test_early_review_closes_a_shorter_interval():
    b = _buffer()
    _run(b, [1, 0, 1, 0, 0, 1], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert b.interval_steps[0, 0] == 2 and b.interval_return[0, 0, 0] == pytest.approx(1 + G * 2)
    assert b.interval_end_obs[0, 0, 0] == 2.0 and b.interval_discount[0, 0, 0] == pytest.approx(G ** 2)


@pytest.mark.parametrize("truncated", [False, True])
def test_episode_end_closes_with_the_terminal_observation(truncated):
    b = _buffer()
    _run(b, [1, 0, 0], [1.0, 2.0, -100.0], end_done=True, truncated=truncated)
    assert b.interval_ready[0, 0] and b.interval_steps[0, 0] == 3
    assert b.interval_return[0, 0, 0] == pytest.approx(1 + G * 2 - G * G * 100)
    assert b.interval_end_obs[0, 0, 0] == 3.0  # the terminal observation, not a reset
    # success / unsafe / timeout: no bootstrap; numerical truncation: bootstrap (SB3's one-step rule)
    assert b.interval_done[0, 0, 0] == (0.0 if truncated else 1.0)
    assert b.interval_counts["closed_by_truncation" if truncated else "closed_by_terminal"] == 1


def test_review_that_ends_the_episode_at_once_is_a_one_step_interval():
    b = _buffer()
    _run(b, [1], [100.0], end_done=True)
    assert b.interval_ready[0, 0] and b.interval_steps[0, 0] == 1 and b.interval_done[0, 0, 0] == 1.0
    assert b.interval_return[0, 0, 0] == pytest.approx(100.0)


def test_every_step_review_degenerates_to_one_step():
    b = _buffer()
    _run(b, [1, 1, 1, 1], [1.0, 2.0, 3.0, 4.0])
    for k in range(3):
        assert b.interval_ready[k, 0] and b.interval_steps[k, 0] == 1
        assert b.interval_return[k, 0, 0] == pytest.approx(k + 1.0) and b.interval_end_obs[k, 0, 0] == k + 1
        assert b.interval_discount[k, 0, 0] == pytest.approx(G)


def test_overwritten_start_slot_is_dropped_never_backfilled():
    b = _buffer(size=4)
    _run(b, [1, 0, 0, 0, 0, 1], [1.0] * 6)  # an interval of 5 steps in a 4-slot ring
    assert b.interval_counts["dropped_overwritten"] == 1 and not b.interval_ready.any()


def test_two_environments_never_mix_intervals():
    b = _buffer(n_envs=2)
    dues = [[1, 1], [0, 1], [0, 0], [1, 0], [0, 1]]
    for k, (d0, d1) in enumerate(dues):
        nxt = dues[k + 1] if k + 1 < len(dues) else [1, 1]
        _add(b, [_o(10 + k, d0), _o(20 + k, d1)], [_o(11 + k, nxt[0]), _o(21 + k, nxt[1])],
             [1.0 + k, 100.0 + k], [False, False])
    # env 0: review at 0 closes at 3 (d = 3); env 1: review at 0 (d = 1), review at 1 closes at 4 (d = 3)
    assert b.interval_steps[0, 0] == 3 and b.interval_return[0, 0, 0] == pytest.approx(1 + G * 2 + G * G * 3)
    assert b.interval_end_obs[0, 0, 0] == 13.0
    assert b.interval_steps[0, 1] == 1 and b.interval_return[0, 1, 0] == pytest.approx(100.0)
    assert b.interval_steps[1, 1] == 3 and b.interval_return[1, 1, 0] == pytest.approx(101 + G * 102 + G * G * 103)
    assert b.interval_end_obs[1, 1, 0] == 24.0


# --------------------------------------------------------------------------- learner
class _Stub(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS, dtype=np.float32), 0.0, False, False, {}


def _model(cls=TCPFSAC, n_envs=1, **kw):
    th.manual_seed(0)
    return cls("MlpPolicy", _Stub(), seed=0, device="cpu", buffer_size=600, learning_starts=0, batch_size=32,
               ent_coef=0.005, gamma=G, policy_kwargs={"net_arch": [16, 16]}, **kw)


def _fill(buffer, rng, n=240):
    for k in range(n):
        due = k % 3 == 0
        o = rng.normal(size=OBS).astype(np.float32)
        o[DUE_INDEX], o[GAP_INDEX] = float(due), 1.0
        nxt = rng.normal(size=OBS).astype(np.float32)
        nxt[DUE_INDEX], nxt[GAP_INDEX] = float(k % 3 == 2), 1.0 / 3
        buffer.add(o[None], nxt[None], rng.uniform(-1, 1, (1, 3)).astype(np.float32), np.array([rng.normal()]),
                   np.array([rng.uniform() < 0.03]), [{"TimeLimit.truncated": False}])


def _params(m):
    return th.cat([p.detach().flatten() for p in m.policy.parameters()])


def _train(m, steps=3, timesteps=5000):
    m._setup_learn(10_000)
    m._total_timesteps, m.num_timesteps = 10_000, timesteps
    np.random.seed(3); th.manual_seed(3)
    m.train(gradient_steps=steps, batch_size=32)


def test_interval_off_is_the_option_learner_bitwise():
    a = _model(OptionSAC)
    b = _model(TCPFSAC, interval_bellman=False, actor_every_transitions=None)
    _fill(a.replay_buffer, np.random.default_rng(1))
    _fill(b.replay_buffer, np.random.default_rng(1))
    _train(a)
    _train(b)
    assert th.equal(_params(a), _params(b))


def test_target_routing_review_folded_continue_one_step_open_review_masked():
    m = _model()
    _fill(m.replay_buffer, np.random.default_rng(2), n=60)
    m.target_value = lambda obs: obs[:, 1:2] * 0 + 7.0  # a fixed successor value
    np.random.seed(0)
    data = m.replay_buffer.sample(64)
    target, mask = m._td_targets(data, m.target_value(data.next_observations))
    ready, ret, disc, _, done = m.replay_buffer.interval()
    review = data.observations[:, DUE_INDEX] > 0.5
    one = data.rewards + (1 - data.dones) * G * 7.0
    folded = ret + disc * (1 - done) * 7.0
    assert th.allclose(target[~review], one[~review]) and th.all(mask[~review] == 1.0)
    r_ready = review & (ready[:, 0] > 0)
    assert r_ready.any() and th.allclose(target[r_ready], folded[r_ready]) and th.all(mask[r_ready] == 1.0)
    r_open = review & (ready[:, 0] == 0)
    assert th.all(mask[r_open] == 0.0)


def test_cosine_learning_rate_runs_on_real_transition_progress():
    lr = CosineLearningRate(1e-4, 1e-5)
    assert lr(1.0) == pytest.approx(1e-4) and lr(0.5) == pytest.approx(5.5e-5) and lr(0.0) == pytest.approx(1e-5)
    m = _model(learning_rate=lr, actor_every_transitions=3)
    _fill(m.replay_buffer, np.random.default_rng(3))
    m._setup_learn(40_000)
    m._total_timesteps = 40_000
    m._update_current_progress_remaining(20_000, 40_000)
    m.num_timesteps = 20_000
    m.train(gradient_steps=1, batch_size=32)
    assert m.actor.optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-5)
    assert m.critic.optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-5)


@pytest.mark.parametrize("n_envs", [1, 6])
def test_actor_clock_counts_global_real_transitions(n_envs):
    m = _model(actor_every_transitions=3)
    _fill(m.replay_buffer, np.random.default_rng(4))
    m._setup_learn(10_000)
    m.learning_starts = 1000
    per_call = []
    t = 1000
    for _ in range(30):
        t += n_envs  # SB3 adds n_envs transitions per vector step, then calls train once
        m.num_timesteps = t
        before = m.actor_updates
        m.train(gradient_steps=n_envs, batch_size=32)  # trainer B: one critic update per transition
        per_call.append(m.actor_updates - before)
    assert m.actor_updates == (t - 1000) // 3
    if n_envs == 6:
        assert set(per_call) == {2}
    else:
        assert per_call.count(1) == 10 and set(per_call) <= {0, 1}


def test_trainer_a_updates_the_actor_with_every_critic_update_on_closed_reviews():
    m = _model(actor_every_transitions=None)
    _fill(m.replay_buffer, np.random.default_rng(5))
    _train(m, steps=4)
    assert m.actor_updates == 4
    np.random.seed(1)
    batch = m.replay_buffer.sample_closed_review(64)
    assert th.all(batch[:, DUE_INDEX] == 1.0)


def test_relative_term_off_ignores_counterfactual_data_with_interval_targets():
    a, b = _model(), _model()
    rng_a, rng_b = np.random.default_rng(6), np.random.default_rng(6)
    for buf, rng, cf in ((a.replay_buffer, rng_a, True), (b.replay_buffer, rng_b, False)):
        for k in range(150):
            o = rng.normal(size=OBS).astype(np.float32)
            o[DUE_INDEX], o[GAP_INDEX] = float(k % 3 == 0), 1.0
            nxt = rng.normal(size=OBS).astype(np.float32)
            alt = {"w": 1, "a_tilde": rng.uniform(-1, 1, 3).astype(np.float32), "r_hat": float(rng.normal()),
                   "z_hat": rng.normal(size=OBS).astype(np.float32)}
            info = {"TimeLimit.truncated": False, **({"counterfactual": alt} if cf else {})}
            buf.add(o[None], nxt[None], rng.uniform(-1, 1, (1, 3)).astype(np.float32), np.array([rng.normal()]),
                    np.array([False]), [info])
    _train(a)
    _train(b)
    assert th.equal(_params(a), _params(b))


# --------------------------------------------------------------------------- trainer choice rule
def _run_dir(tmp, name, zero, late, q=0.0, later_share=0.1):
    import json as _json

    run = tmp / name
    for mark, (cap, ret) in {0: zero, **late}.items():
        d = run / "dev_eval" / str(mark)
        d.mkdir(parents=True)
        (d / "summary.json").write_text(_json.dumps({"clean_capture": cap, "mean_discounted_return": ret}))
    (run / "fix_stats.json").write_text(_json.dumps({"audit_decisions": [31000, 35000], "q_minus_g_plan": [q, q]}))
    n_later = int(round(100 * later_share))
    lines = [{"t": 32000, "rv": {"due": 1}, "req": "later"}] * n_later + [{"t": 32000, "rv": {"due": 1}, "req": "keep"}] * (100 - n_later)
    (run / "decisions.jsonl").write_text("\n".join(_json.dumps(x) for x in lines) + "\n")
    return run


def test_trainer_choice_rule(tmp_path):
    from experiments.t1_l_s_trainer_choice import choose

    stable = {25000: (16, 81.0), 30000: (16, 80.5), 35000: (15, 79.0), 40000: (16, 81.5)}
    assert choose(_run_dir(tmp_path, "s", (16, 79.7), stable), None)["trainer"] == "A"
    v4_like = {25000: (12, 22.6), 30000: (8, -29.2), 35000: (13, 31.1), 40000: (10, -3.7)}
    assert choose(_run_dir(tmp_path, "v", (16, 80.2), v4_like), None)["trainer"] == "B"
    drop = {25000: (16, 66.0), 30000: (16, 68.0)}  # (b) only: captures kept, return 12.7 below 0k
    assert choose(_run_dir(tmp_path, "d", (16, 79.7), drop), None)["trainer"] == "B"
    critic = _run_dir(tmp_path, "c", (16, 79.7), stable, q=80.0, later_share=0.45)  # (c) only
    assert choose(critic, None)["trainer"] == "B"
    assert choose(_run_dir(tmp_path, "c2", (16, 79.7), stable, q=80.0, later_share=0.2), None)["trainer"] == "A"
    out = choose(_run_dir(tmp_path, "n", (16, 79.7), stable), _run_dir(tmp_path, "nd", (16, 79.7), v4_like))
    assert out["trainer"] == "A" and out["ndelta_only_degraded"]
    assert choose(_run_dir(tmp_path, "u", (16, 79.7), {25000: (16, 80.0)}), None)["trainer"] == "undetermined"
