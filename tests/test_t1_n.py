"""T1-N contracts (Research Plan R2): initial sampling, neutral time action, conditional G*,
review interface, paired shadow isolation and the learner's paired label."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from env.t1_env import T1Env
from env.t1_n_review import (
    N_DUE_INDEX, N_HAS_PLAN_INDEX, T1N_OBSERVATION_NAMES, T1NReviewEnv, action_for_tau, neutral_tau, tau_high,
)
from env.t1_n_task import T1N_ENTRY_SPEED_M_S, T1N_KAPPA, t1n_env_config

OPENING = 292000  # development block of the T1-N technical checks


def _relative(env):
    t, c = env.target_state, env.chaser_state
    q = c.position - t.position
    v = c.rotation @ c.velocity - t.rotation @ t.velocity
    return q, v


def test_sampler_is_deterministic_and_within_the_r2_bounds():
    env = T1Env(t1n_env_config())
    env.reset(OPENING)
    q1, v1 = _relative(env)
    env.reset(OPENING)
    q2, v2 = _relative(env)
    assert np.array_equal(q1, q2) and np.array_equal(v1, v2)
    for k in range(OPENING, OPENING + 60):
        env.reset(k)
        q, v = _relative(env)
        r = np.linalg.norm(q)
        radial = float(v @ q / r)
        tangential = float(np.linalg.norm(v - radial * q / r))
        offset = np.degrees(np.arccos(env.target_state.omega[2] / np.linalg.norm(env.target_state.omega)))
        assert 12.0 <= r <= 18.0
        assert -0.12 <= radial <= 0.12 and 0.03 - 1e-12 <= tangential <= 0.20 + 1e-12
        assert np.linalg.norm(v) <= 0.25
        assert 10.0 - 1e-9 <= offset <= 30.0 + 1e-9
        assert 2.1 <= np.degrees(np.linalg.norm(env.target_state.omega)) <= 2.7
        assert not env.metrics().violations  # legal at t = 0


def test_t1_sampler_unchanged_for_the_old_profiles():
    from env.t1_l_task import t1l_env_config
    from env.t1_task import sample_t1_initial_states

    env = T1Env(t1l_env_config())
    env.reset(297000)
    rng = np.random.default_rng(297000)
    target, chaser = sample_t1_initial_states(rng, t1l_env_config().initial, t1l_env_config().task)
    assert np.array_equal(env.chaser_state.position, chaser.position)
    assert np.array_equal(env.target_state.rotation, target.rotation)


def test_neutral_time_mapping_and_inverse():
    assert neutral_tau(-1.0, 0.0) == pytest.approx(20.0)
    assert neutral_tau(1.0, 0.0) == pytest.approx(200.0)
    assert tau_high(100.0) == pytest.approx(140.0)
    assert neutral_tau(1.0, 100.0) == pytest.approx(140.0)
    for t in (0.0, 50.0, 150.0):
        for a in (-0.7, 0.0, 0.4):
            assert action_for_tau(neutral_tau(a, t), t) == pytest.approx(a)
    assert neutral_tau(0.0, 230.0) == pytest.approx(10.0)  # tau_high below tau_low: tau_high


def test_conditional_gstar_plans_exactly_the_request_or_rejects():
    env = T1NReviewEnv(openings=[OPENING], earliest_summary=False)
    env.reset()
    g, e = env.generator, env.env
    earliest, _ = g.earliest_feasible(e, T1N_ENTRY_SPEED_M_S)
    assert earliest is not None and earliest > 20.0
    g.decide(e, 20.0)  # too early from 12-18 m: rejected, no plan, hold
    assert g.plan is None and g.last_status.startswith("hold:rejected") and g.last_outcome == "rejected"
    g.decide(e, earliest + 7.0)  # exactly this arrival (no grid, no slip)
    assert g.last_status.startswith("optimal") and g.tau_effective_s == pytest.approx(earliest + 7.0)
    plan = g.plan
    g.decide(e, 20.0)  # unmeetable earlier request with a usable plan: kept
    assert g.plan is plan and g.last_status == "kept:infeasible_request" and g.last_outcome == "rejected"
    g.decide(e, 239.0 - e.time_s + 5.0)  # beyond the deadline: never cut to it
    assert g.plan is plan and g.last_deadline == "rejected_beyond"
    assert g.kappa == T1N_KAPPA and g.entry_speed_m_s == T1N_ENTRY_SPEED_M_S


def test_review_interface_rejection_is_an_event_and_continue_keeps():
    env = T1NReviewEnv(openings=[OPENING])
    obs, _ = env.reset()
    assert obs.shape == (len(T1N_OBSERVATION_NAMES),) and np.all(np.isfinite(obs))
    assert obs[N_DUE_INDEX] == 1.0 and obs[N_HAS_PLAN_INDEX] == 0.0
    obs, _, _, _, info = env.step(np.array([-1.0]))  # tau_req = 20 s: rejected
    assert info["review"]["outcome"] == "rejected" and info["execution"]["plan_status"].startswith("hold")
    assert obs[N_DUE_INDEX] == 1.0  # asked again at the next step
    earliest = env.earliest_feasible_summary()
    obs, _, _, _, info = env.step(np.array([action_for_tau(earliest + 10.0, env.env.time_s)]))
    assert info["review"]["outcome"] == "accepted" and obs[N_HAS_PLAN_INDEX] == 1.0
    assert obs[N_DUE_INDEX] == 0.0
    obs, _, _, _, info = env.step(np.array([0.9]))  # a continue step ignores the proposal
    assert not info["review"]["reviewed"] and np.array_equal(info["executed_action"], np.zeros(1, np.float32))
    assert info["execution"]["plan_status"] == "kept"


def test_paired_shadow_leaves_the_real_transitions_bitwise_unchanged():
    from env.t1_n_cf import T1NReviewCFEnv, T1NShadowConfig

    plain = T1NReviewEnv(openings=[OPENING])
    cf = T1NReviewCFEnv(openings=[OPENING], shadow=T1NShadowConfig(horizon_steps=2))
    o1, _ = plain.reset()
    o2, _ = cf.reset()
    assert np.array_equal(o1, o2)
    earliest = plain.earliest_feasible_summary()
    actions = [np.array([action_for_tau(earliest + 15.0, 0.0)]), np.zeros(1), np.zeros(1), np.array([0.8])]
    saw_pair = False
    for a in actions:
        o1, r1, *_ = plain.step(a)
        o2, r2, _, _, info = cf.step(a)
        assert np.array_equal(o1, o2) and r1 == r2
        c = info["counterfactual"]
        if c["a_tilde"] is not None:
            saw_pair = True
            assert c["h_eff"] >= 1 and c["z_hat"].shape == o2.shape and c["z_hat_real"].shape == o2.shape
            assert c["first_reward_error"] is not None
    assert saw_pair
    assert np.array_equal(plain.env.chaser_state.position, cf.env.chaser_state.position)


def test_paired_label_and_one_dimensional_continue_action():
    from train.t1_n_sac import PairedBatch, T1NReplayBuffer

    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(T1N_OBSERVATION_NAMES),), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    b = T1NReplayBuffer(20, obs_space, act_space, device="cpu", n_envs=1)
    o = np.zeros((1, len(T1N_OBSERVATION_NAMES)), dtype=np.float32)
    o[0, N_DUE_INDEX] = 1.0
    z_alt, z_real = o.copy()[0] + 1.0, o.copy()[0] + 2.0
    info = {"counterfactual": {"w": 1, "a_tilde": np.array([0.3], np.float32), "r_hat": -1.0, "r_hat_real": -1.5,
                               "z_hat": z_alt, "z_hat_real": z_real, "h_eff": 2}}
    b.add(o, o, np.zeros((1, 1)), np.array([-0.7]), np.array([0.0]), [info])
    assert b.cf_real_rewards[0, 0, 0] == pytest.approx(-1.5)
    assert b.cf_discount[0, 0, 0] == pytest.approx(b.gamma ** 2)
    assert np.array_equal(b.cf_real_next_observations[0, 0], z_real)
    b._last = (np.array([0]), np.array([0]))
    batch = b.counterfactual()
    assert isinstance(batch, PairedBatch) and float(batch.weights[0, 0]) == 1.0

    class _Stub:
        def target_value(self, x):
            return x[:, :1] * 10.0

    from train.t1_n_sac import T1NSAC

    delta = T1NSAC._delta_target(_Stub(), None, batch, th.zeros(1, 1))
    # (r^alt - r^real) + gamma^2 (V(z_alt) - V(z_real)) = 0.5 + gamma^2 (10 - 20)
    assert float(delta[0, 0]) == pytest.approx(0.5 + b.gamma ** 2 * (10.0 - 20.0), rel=1e-6)
    assert T1NSAC.continue_actions(None, th.zeros(4, 3)).shape == (4, 1)
