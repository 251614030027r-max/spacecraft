"""T1-P contracts: stage action mapping, local planner safety between nodes, potential telescoping,
pending stage-pair weights, the learner's configuration and the counterfactual's isolation."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from env.t1_p_review import P_DUE_INDEX, T1P_OBSERVATION_NAMES, T1PReviewEnv, geometric_rule
from env.t1_p_stage import LocalStagePlanner, d_max, plan_state, stage_basis, stage_goal
from env.t1_task import T1TaskConfig

OPENING = 292000


def test_stage_goal_is_the_coast_plus_a_bounded_reachable_offset():
    assert d_max(106.0) == pytest.approx(0.70 * 1.6 / 106.0 * 72.0)
    env = T1PReviewEnv(openings=[OPENING])
    env.reset()
    b = stage_basis(env.env)
    assert np.allclose(b @ b.T, np.eye(3), atol=1e-12) and np.linalg.det(b) == pytest.approx(1.0)
    g0 = stage_goal(env.env, np.zeros(3))
    assert np.allclose(g0["q_goal"], g0["q_coast"]) and np.allclose(g0["v_goal"], g0["v_coast"])
    g = stage_goal(env.env, np.array([1.0, 1.0, 1.0]))
    assert np.linalg.norm(g["q_goal"] - g["q_coast"]) == pytest.approx(g["d_max"])  # |a| > 1 is normalised
    assert np.allclose(g["v_goal"] - g["v_coast"], 2.0 * (g["q_goal"] - g["q_coast"]) / 12.0)
    r = stage_goal(env.env, np.array([1.0, 0.0, 0.0]))
    assert float((r["q_goal"] - r["q_coast"]) @ b[0]) == pytest.approx(-g["d_max"])  # a_r > 0 is inward
    assert np.all(np.abs(geometric_rule(env)) <= 1.0)


def test_local_planner_keeps_the_sphere_and_envelope_between_nodes():
    task = T1TaskConfig()
    planner = LocalStagePlanner(task, 106.0)
    q0, v0 = np.array([7.6, 0.0, 0.0]), np.array([-0.08, 0.05, 0.0])
    goal_q, goal_v = np.array([5.0, 1.0, 0.0]), np.array([-0.2, 0.0, 0.0])  # asks to cut into the sphere
    res = planner.solve(q0, v0, goal_q, goal_v)
    assert res["feasible"], res["status"]
    assert planner.verify(res) == []
    for k in range(121):
        p, w = plan_state(res, 0.1 * k, 106.0)
        r = np.linalg.norm(p)
        assert r >= 6.05
        assert -(p @ w) / r <= task.braking_speed_limit(r - 6.0) + 1e-6
    assert res["terminal_position_error_m"] > 0.5  # soft terminal: the unreachable goal is not met
    assert np.all(np.linalg.norm(res["f"], axis=1) <= 1.6 + 1e-6)
    bad = {**res, "q": res["q"].copy()}
    bad["q"][3] = np.array([5.0, 0.0, 0.0])
    assert "sphere" in planner.verify(bad)


def test_potential_shaping_telescopes():
    env = T1PReviewEnv(openings=[OPENING])
    env.reset()
    g = env.env.config.reward.gamma
    phi0 = env._phi
    raw = train = 0.0
    for k in range(3):
        _, r, term, trunc, info = env.step(geometric_rule(env) if env._due else np.zeros(3))
        assert info["potential_next"] <= 0.0 and info["reward_train"] == r
        raw += g ** k * info["reward_raw"]
        train += g ** k * r
    assert train == pytest.approx(raw + g ** 3 * env._phi - phi0, abs=1e-9)


def test_pending_stage_pair_weight_is_set_when_the_stage_closes():
    from train.t1_p_sac import T1PReplayBuffer

    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(T1P_OBSERVATION_NAMES),), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    b = T1PReplayBuffer(30, obs_space, act_space, device="cpu", n_envs=1)
    o = np.zeros((1, len(T1P_OBSERVATION_NAMES)), dtype=np.float32)
    due = o.copy()
    due[0, P_DUE_INDEX] = 1.0
    z = np.zeros(len(T1P_OBSERVATION_NAMES), np.float32)
    pair = {"counterfactual": {"w": 1, "a_tilde": np.array([0.6, 0, 0], np.float32), "r_hat": -3.0, "z_hat": z}}
    b.add(due, o, np.zeros((1, 3)), np.array([-1.0]), np.array([0.0]), [pair])
    assert b.cf_weights[0, 0, 0] == 0.0  # pending
    for k in range(4):
        b.add(o, o, np.zeros((1, 3)), np.array([-1.0]), np.array([0.0]), [{}])
    b.add(o, due, np.zeros((1, 3)), np.array([-1.0]), np.array([0.0]), [{"cf_final": {"weight": 1.0}}])
    assert b.cf_weights[0, 0, 0] == 1.0 and b.cf_counts == {"pairs": 1, "weighted": 1, "cancelled_overwritten": 0}
    b.add(due, o, np.zeros((1, 3)), np.array([-1.0]), np.array([0.0]), [pair])
    b.add(o, due, np.zeros((1, 3)), np.array([-1.0]), np.array([0.0]), [{"cf_final": {"weight": 0.0}}])
    assert b.cf_weights[6, 0, 0] == 0.0


def test_learner_configuration():
    from train.t1_p_sac import T1PSAC

    assert T1PSAC.continue_actions(None, th.zeros(5, 2)).shape == (5, 3)
    assert float(T1PSAC._eq_mask(None, th.ones(4, len(T1P_OBSERVATION_NAMES))).sum()) == 0.0

    class _E(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(T1P_OBSERVATION_NAMES),), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    with pytest.raises(ValueError):
        T1PSAC("MlpPolicy", _E(), eq_lambda=0.1, ent_coef=0.005)
    m = T1PSAC("MlpPolicy", _E(), eq_lambda=0.0, ent_coef=0.005, policy_kwargs={"net_arch": [8]})
    obs = th.zeros(2, len(T1P_OBSERVATION_NAMES))
    obs[:, -1] = th.tensor([1 / 6, 1.0])
    assert th.allclose(m.alpha_of(obs)[:, 0], th.tensor([0.005, 0.03]))


def test_stage_counterfactual_leaves_the_real_transitions_bitwise_unchanged():
    from env.t1_p_cf import T1PReviewCFEnv, review_hash

    opening = 292002
    assert review_hash(opening, 0) % 4 == 0  # the first review is sampled
    plain, cf = T1PReviewEnv(openings=[opening]), T1PReviewCFEnv(openings=[opening])
    o1, _ = plain.reset()
    o2, _ = cf.reset()
    finals = []
    for k in range(6):
        a = geometric_rule(plain) if plain._due else np.zeros(3)
        o1, r1, *_ = plain.step(a)
        o2, r2, _, _, info = cf.step(a)
        assert np.array_equal(o1, o2) and r1 == r2
        if k == 0:
            c = info["counterfactual"]
            assert c["kind"] == "stage_pair" and c["a_tilde"] is not None
            assert np.all(np.isfinite(c["z_hat"])) and c["z_hat"][P_DUE_INDEX] == 1.0
        if "cf_final" in info:
            finals.append(info["cf_final"])
    assert len(finals) == 1 and set(finals[0]) >= {"weight", "e_q_m", "e_v_m_s", "e_R"}
