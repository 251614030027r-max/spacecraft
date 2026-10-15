"""T1-L blocking contracts (plan 2026-10-15, sec. 4.2): task profile, anchor + delay, reviews and
continue, phase-aware learner, relative MPC term, shadow of the review wrapper."""

from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from env.t1_env import T1EnvConfig
from env.t1_gym import T1GStarEnv
from env.t1_l_task import t1l_env_config
from env.t1_task_review import (DUE_INDEX, GAP_INDEX, REVIEW_OBSERVATION_NAMES, T1LReviewEnv,
                                continue_action_from_observation)
from train.t1_option_sac import OptionReplayBuffer, OptionSAC

OBS = len(REVIEW_OBSERVATION_NAMES)


def test_t1l_changes_only_the_initial_range():
    a, b = asdict(T1EnvConfig()), asdict(t1l_env_config())
    assert b["initial"]["range_m"] == (12.0, 18.0) and a["initial"]["range_m"] == (9.0, 15.0)
    a["initial"].pop("range_m"), b["initial"].pop("range_m")
    assert a == b


@pytest.fixture(scope="module")
def env3():
    return T1LReviewEnv(openings=[297000], review_period_steps=3)


def test_first_review_is_anchor_plus_delay_and_minus_one_is_earliest(env3):
    env3.reset()
    g, e = env3.generator, env3.env
    tau_min, _ = g.earliest_feasible(e, 0.06)
    _, _, _, _, info = env3.step(np.array([-1.0, 0.0, 0.0]))
    rv, x = info["review"], info["execution"]
    assert rv["reviewed"] and rv["tau_min_feasible_s"] == tau_min and rv["delay_s"] == 0.0
    assert x["tau_request_s"] == tau_min and x["tau_effective_s"] == tau_min and x["new_plan"]
    env3.reset()
    _, _, _, _, info = env3.step(np.array([0.0, 0.0, 0.0]))
    rv, x = info["review"], info["execution"]
    assert rv["delay_s"] == pytest.approx(30.0) and x["tau_request_s"] == pytest.approx(tau_min + 30.0)
    assert x["tau_effective_s"] >= x["tau_request_s"] - 1e-9  # a slip is G*'s bounded correction, logged


def test_anchor_depends_on_the_entry_speed_through_the_same_planner(env3):
    env3.reset()
    g, e = env3.generator, env3.env
    for v in (0.02, 0.10):
        tau_min, result = g.earliest_feasible(e, v)
        assert result["feasible"] and result["entry_speed_m_s"] == v and result["tau_s"] == tau_min


def test_review_clock_and_continue_ignore_the_policy():
    a = T1LReviewEnv(openings=[297000], review_period_steps=3)
    b = T1LReviewEnv(openings=[297000], review_period_steps=3)
    oa, _ = a.reset()
    ob, _ = b.reset()
    first = np.array([0.0, 0.3, -0.4])
    pattern = []
    rng = np.random.default_rng(0)
    for k in range(6):
        due = bool(oa[DUE_INDEX] > 0.5)
        pattern.append(int(due))
        if due:
            pa = pb = first if k == 0 else np.array([0.1, 0.3, -0.4])
        else:
            pa, pb = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)  # different proposals: ignored
            expected = continue_action_from_observation(oa)
        oa, ra, *_, ia = a.step(pa)
        ob, rb, *_, ib = b.step(pb)
        assert np.array_equal(oa, ob) and ra == rb
        if not due:
            assert np.array_equal(ia["executed_action"], expected) and not ia["review"]["reviewed"]
            assert ia["execution"]["plan_status"] in ("kept", "descent") or ia["execution"]["new_plan_trigger"] == "deviation"
    assert pattern == [1, 0, 0, 1, 0, 0]


def test_period_one_reviews_every_step_and_plan_events_force_a_review():
    env = T1LReviewEnv(openings=[297000], review_period_steps=1)
    obs, _ = env.reset()
    for _ in range(3):
        assert obs[DUE_INDEX] == 1.0 and round(float(obs[GAP_INDEX]) * 3) == 1 or _ == 0
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0]))
    env3 = T1LReviewEnv(openings=[297000], review_period_steps=3)
    env3.reset()
    env3.step(np.array([0.0, 0.0, 0.0]))
    assert not env3._due
    env3.generator.plan = None  # the plan is lost outside the descent: a plan event
    env3.advance_review(False, 0.65, 0.06)
    assert env3._due and env3._since_review == 2


def test_zero_delay_review_reproduces_the_earliest_rule_on_t1():
    """B2 interface with delay 0 on the original T1 profile: same physics and G* plan as the v4 rule."""

    old = T1GStarEnv(openings=[295003])
    new = T1LReviewEnv(env_config=T1EnvConfig(), openings=[295003], review_period_steps=1)
    old.reset(); new.reset()
    request = REVIEW_OBSERVATION_NAMES.index("tau_request_160s")
    same = [i for i in range(25) if i != request]
    for k in range(4):
        live = old.generator.remaining_s(old.env) > 0.0
        o1, r1, *_, i1 = old.step(np.array([0.0 if live else -1.0, -0.142857, 0.0]))
        o2, r2, *_, i2 = new.step(np.array([0.0 if live else -1.0, -0.142857, 0.0]))
        assert np.array_equal(o1[same], o2[same]) and r1 == r2
        assert i1["execution"]["tau_effective_s"] == i2["execution"]["tau_effective_s"]
        assert np.array_equal(old.env.chaser_state.position, new.env.chaser_state.position)
        if k == 0:  # the old earliest request was logged as 0 s; the new one is the anchor itself
            assert o1[request] == 0.0 and o2[request] * 160.0 == pytest.approx(i2["review"]["tau_min_feasible_s"])
        else:
            assert o1[request] == o2[request]


# --------------------------------------------------------------------------- learner
class _Stub(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS, dtype=np.float32), 0.0, False, False, {}


def _model(**kw):
    th.manual_seed(0)
    return OptionSAC("MlpPolicy", _Stub(), seed=0, device="cpu", buffer_size=500, learning_starts=0, batch_size=32,
                     ent_coef=0.005, gamma=0.999, policy_kwargs={"net_arch": [16, 16]}, **kw)


def _obs(rng, due):
    o = rng.normal(size=OBS).astype(np.float32)
    o[DUE_INDEX] = float(due)
    o[GAP_INDEX] = rng.choice([1, 2, 3]) / 3.0
    return o


def _fill(buffer, rng, cf=True, n=200):
    for k in range(n):
        obs, nxt = _obs(rng, k % 3 == 0), _obs(rng, k % 3 == 2)
        act = rng.uniform(-1, 1, 3).astype(np.float32)
        info = {"TimeLimit.truncated": False}
        if k % 3 == 0:  # drawn either way, so both fills hold the same transitions
            alt = {"w": 1, "a_tilde": rng.uniform(-1, 1, 3).astype(np.float32),
                   "r_hat": float(rng.normal()), "z_hat": _obs(rng, 0)}
            if cf:
                info["counterfactual"] = alt
        buffer.add(obs[None], nxt[None], act[None], np.array([rng.normal()]), np.array([rng.uniform() < 0.05]), [info])


def _params(m):
    return th.cat([p.detach().flatten() for p in m.policy.parameters()])


def _train(m, steps=3, timesteps=5000):
    m._total_timesteps, m.num_timesteps = 10_000, timesteps
    m._setup_learn(10_000)
    np.random.seed(3); th.manual_seed(3)
    m.train(gradient_steps=steps, batch_size=32)


def test_lambda_delta_zero_ignores_counterfactual_data_bitwise():
    a, b = _model(delta_lambda=0.0), _model(delta_lambda=0.0)
    _fill(a.replay_buffer, np.random.default_rng(1), cf=True)
    _fill(b.replay_buffer, np.random.default_rng(1), cf=False)
    _train(a)
    _train(b)
    assert th.equal(_params(a), _params(b))


def test_relative_term_moves_both_critics_with_finite_gradients():
    a, b = _model(delta_lambda=0.0), _model(delta_lambda=0.1, delta_warmup_decisions=0, delta_diagnostics_every=1)
    for m in (a, b):
        _fill(m.replay_buffer, np.random.default_rng(2))
    _train(a)
    _train(b)
    for qa, qb in zip(a.critic.q_networks, b.critic.q_networks):
        assert not th.equal(th.cat([p.flatten() for p in qa.parameters()]), th.cat([p.flatten() for p in qb.parameters()]))
    s = b.delta_stats
    assert len(s["g_delta"]) == 3 and all(np.isfinite(s["g_delta"])) and min(s["g_delta"]) > 0.0


def test_warmup_keeps_the_relative_term_off():
    a, b = _model(delta_lambda=0.0), _model(delta_lambda=0.1, delta_warmup_decisions=6000)
    for m in (a, b):
        _fill(m.replay_buffer, np.random.default_rng(4))
    _train(a, timesteps=5000)
    _train(b, timesteps=5000)
    assert th.equal(_params(a), _params(b))


def test_review_only_actor_batch_continue_target_and_alpha():
    m = _model()
    rng = np.random.default_rng(5)
    _fill(m.replay_buffer, rng)
    np.random.seed(0)
    review = m.replay_buffer.sample_due(64)
    assert th.all(review[:, DUE_INDEX] == 1.0)
    obs = th.as_tensor(np.stack([_obs(rng, 0) for _ in range(8)]))
    with th.no_grad():
        expected = th.min(th.cat(m.critic_target(obs, m.continue_actions(obs)), dim=1), dim=1, keepdim=True)[0]
    assert th.equal(m.target_value(obs), expected)  # continue successors: the continue action, no entropy
    o = th.zeros((3, OBS))
    o[:, GAP_INDEX] = th.tensor([1 / 3, 2 / 3, 1.0])
    assert th.allclose(m.alpha_of(o)[:, 0], th.tensor([0.005, 0.010, 0.015]))
    single = np.zeros(OBS, dtype=np.float32)
    single[REVIEW_OBSERVATION_NAMES.index("kappa")] = 0.79
    single[REVIEW_OBSERVATION_NAMES.index("has_plan")] = 1.0
    single[REVIEW_OBSERVATION_NAMES.index("plan_entry_speed")] = -0.5
    assert np.allclose(m.continue_actions(th.as_tensor(single[None])).numpy()[0], continue_action_from_observation(single))


def test_replay_stores_the_executed_action():
    m = _model()
    obs = np.zeros((1, OBS), dtype=np.float32)
    m._last_obs = obs
    m._last_original_obs = obs
    executed = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    m._store_transition(m.replay_buffer, np.array([[0.9, -0.9, 0.9]], dtype=np.float32), obs, np.array([0.0]),
                        np.array([False]), [{"executed_action": executed}])
    assert np.array_equal(m.replay_buffer.actions[0, 0], executed)


def test_option_buffer_is_an_m1_buffer():
    from train.t1_m1 import M1ReplayBuffer

    assert issubclass(OptionReplayBuffer, M1ReplayBuffer)


# --------------------------------------------------------------------------- shadow
def test_review_shadow_reproduces_review_and_continue_steps_bitwise():
    from env.t1_option_cf import ReviewCounterfactualPredictor

    real = T1LReviewEnv(openings=[297002], review_period_steps=3)
    predictor = ReviewCounterfactualPredictor(shadow_factory=lambda: T1LReviewEnv(review_period_steps=3))
    real.reset()
    for action in (np.array([0.2, 0.1, 0.4]), np.array([0.9, -0.3, -0.8]), np.array([-0.6, 0.5, 0.0]),
                   np.array([0.0, 0.1, 0.4])):
        predictor.sync(real)
        if real._due:
            p = predictor.predict(action)
            assert p.observation[DUE_INDEX] == 0.0 and round(float(p.observation[GAP_INDEX]) * 3) == 1
            predictor.sync(real)
        shadow = predictor.closed_loop(action)
        obs, reward, *_ = real.step(action)
        assert np.array_equal(shadow["observation"], obs) and shadow["reward"] == reward


def test_cf_env_transitions_equal_the_plain_review_env():
    from env.t1_option_cf import T1LReviewCFEnv

    plain = T1LReviewEnv(openings=[297003], review_period_steps=3)
    cf = T1LReviewCFEnv(openings=[297003], review_period_steps=3)
    plain.reset(); cf.reset()
    kinds = []
    for action in (np.array([0.0, 0.1, 0.6]), np.array([0.3, 0.1, 0.6]), np.array([0.3, 0.1, 0.6]),
                   np.array([0.8, 0.1, -0.6])):
        o1, r1, *_ = plain.step(action)
        o2, r2, _, _, info = cf.step(action)
        assert np.array_equal(o1, o2) and r1 == r2
        kinds.append(info["counterfactual"]["kind"])
    assert kinds[1] == kinds[2] == "continue" and kinds[0] != "continue" and kinds[3] != "continue"
