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


def test_ranking_signal_leaves_the_real_transitions_bitwise_unchanged():
    from env.t1_n_cf import T1NReviewCFEnv

    plain = T1NReviewEnv(openings=[OPENING])
    cf = T1NReviewCFEnv(openings=[OPENING])
    o1, _ = plain.reset()
    o2, _ = cf.reset()
    assert np.array_equal(o1, o2)
    earliest = plain.earliest_feasible_summary()
    actions = [np.array([action_for_tau(earliest + 15.0, 0.0)]), np.zeros(1), np.zeros(1), np.array([0.8])]
    kinds = []
    for a in actions:
        o1, r1, *_ = plain.step(a)
        o2, r2, _, _, info = cf.step(a)
        assert np.array_equal(o1, o2) and r1 == r2
        c = info["counterfactual"]
        kinds.append(c["kind"])
        if c["a_tilde"] is not None:
            assert c["signal"] is not None and c["signal_plan_only"] is not None
            assert 0.0 <= c["rank_weight"] <= 1.0 and c["first_reward_error"] < 1e-6 + 0.25
            # D differs from the plan-only D exactly by the two MPC corrections
            assert c["signal"] - c["signal_plan_only"] == pytest.approx(c["mpc_correction_alt"] - c["mpc_correction_real"])
            if c["w"]:
                assert c["r_hat"] == c["signal"] and not np.any(c["z_hat"])  # nothing predicted is bootstrapped
    assert kinds[0] == "request_shift" and kinds[3] == "replan_vs_keep"
    assert np.array_equal(plain.env.chaser_state.position, cf.env.chaser_state.position)


def test_plan_step_rewards_partition_the_plan_and_cut_at_arrival():
    from env.t1_n_cf import branch_values, plan_step_rewards

    mass = 106.0
    f = np.zeros((10, 3))
    f[0] = [1.6, 0.0, 0.0]
    f[9] = [0.0, 1.6, 0.0]
    plan = {"f": f, "dt": 1.5, "tau_s": 15.0}
    steps = plan_step_rewards(plan, 0.0, mass, 0.05, 20.0)
    assert len(steps) == 8  # 7 full steps of 2 s and a partial 1 s step
    total = -0.05 * 15.0 - 20.0 * (1.6 * 1.5 + 1.6 * 1.5) / mass
    assert sum(steps) == pytest.approx(total)
    assert steps[0] == pytest.approx(-0.1 - 20.0 * 1.6 * 1.5 / mass)
    later = plan_step_rewards(plan, 5.0, mass, 0.05, 20.0)  # a kept plan 5 s into it
    assert sum(later) == pytest.approx(-0.05 * 10.0 - 20.0 * 1.6 * 1.5 / mass)
    from env.t1_n_task import t1n_env_config

    v = branch_values({"plan": plan, "plan_elapsed_s": 0.0, "rewards": [-0.5]}, t1n_env_config(), mass)
    g = 0.999
    assert v["plan_only"] == pytest.approx(sum(g ** j * r for j, r in enumerate(steps)))
    assert v["combined"] == pytest.approx(-0.5 + sum(g ** j * r for j, r in enumerate(steps) if j >= 1))
    assert v["mpc_correction"] == pytest.approx(-0.5 - steps[0])


def test_ranking_loss_uses_only_the_direction_and_reaches_the_critic():
    from train.t1_n_sac import T1NReplayBuffer, T1NSAC

    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(T1N_OBSERVATION_NAMES),), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    b = T1NReplayBuffer(20, obs_space, act_space, device="cpu", n_envs=1)
    o = np.zeros((1, len(T1N_OBSERVATION_NAMES)), dtype=np.float32)
    o[0, N_DUE_INDEX] = 1.0
    info = {"counterfactual": {"w": 1, "rank_weight": 0.4, "a_tilde": np.array([0.3], np.float32), "r_hat": -2.5,
                               "z_hat": np.zeros(len(T1N_OBSERVATION_NAMES), np.float32)}}
    b.add(o, o, np.zeros((1, 1)), np.array([-0.7]), np.array([0.0]), [info])
    b.add(o, o, np.zeros((1, 1)), np.array([-0.7]), np.array([0.0]), [{"counterfactual": {"w": 0, "rank_weight": 0.0}}])
    assert b.cf_weights[0, 0, 0] == pytest.approx(0.4) and b.cf_weights[1, 0, 0] == 0.0
    assert b.cf_rewards[0, 0, 0] == pytest.approx(-2.5)

    class _Stub:
        rank_margin = 0.5

    w = th.tensor([[1.0], [1.0], [0.0]])
    d = th.tensor([[-2.5], [3.0], [5.0]])
    qr = [th.zeros(3, 1, requires_grad=True)]
    qa = [th.tensor([[-1.0], [0.1], [9.0]], requires_grad=True)]
    loss = T1NSAC._relative_loss(_Stub(), qa, qr, d, w)
    # row 0: D < 0 and Q(a~) - Q(a) = -1 <= -0.5: satisfied; row 1: D > 0, gap 0.1 < 0.5: 0.4; row 2: w = 0
    assert loss.item() == pytest.approx(0.5 * 0.4 / 3)
    loss.backward()
    assert qa[0].grad[1, 0] < 0.0 and qa[0].grad[0, 0] == 0.0 and qa[0].grad[2, 0] == 0.0
    # the magnitude of D does not matter, only its sign
    loss2 = T1NSAC._relative_loss(_Stub(), [qa[0].detach()], [qr[0].detach()], d * 100.0, w)
    assert loss2.item() == pytest.approx(loss.item())
    assert T1NSAC.continue_actions(None, th.zeros(4, 3)).shape == (4, 1)
    assert T1NSAC._delta_target(None, None, type("C", (), {"rewards": d})(), None) is d
