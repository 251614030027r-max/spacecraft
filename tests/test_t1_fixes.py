"""Learnability fixes (2026-10-15): MC return labels, keep-band equivalence, lam = 0 regression,
G* v0.5 deadline, interface v3 tracking errors and effective-action diagnostics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import SAC

from env.t1_gym import OBSERVATION_NAMES, T1GStarEnv
from train.t1_m1 import DESCENT_INDEX, PLAN_REMAINING_INDEX, M1ReplayBuffer, M1SAC, eq_mask

OBS, ACT = len(OBSERVATION_NAMES), 2
GAMMA = 0.999


def _buffer(size=50, n_envs=1, window=10_000):
    b = M1ReplayBuffer(size, gym.spaces.Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float32),
                       gym.spaces.Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32), device="cpu", n_envs=n_envs)
    b.gamma, b.mc_window = GAMMA, window
    return b


def _add(b, rewards, dones, truncated=None):
    n = b.n_envs
    truncated = truncated or [False] * n
    infos = [{"TimeLimit.truncated": bool(t)} if d else {} for d, t in zip(dones, truncated)]
    obs = np.zeros((n, OBS), np.float32)
    b.add(obs, obs, np.zeros((n, ACT)), np.asarray(rewards, float), np.asarray(dones, float), infos)


def _returns(rewards):
    g, out = 0.0, []
    for r in reversed(rewards):
        g = r + GAMMA * g
        out.append(g)
    return out[::-1]


def test_mc_returns_interleaved_envs_terminal_truncated_and_unfinished():
    b = _buffer(n_envs=2)
    # env 0: episode of 3 (normal end), then a numerical truncation of 2, then 1 unfinished step
    # env 1: episode of 4 (normal end: timeout/unsafe look the same to the buffer), then 2 unfinished
    e0 = [(-1, 0, 0), (-2, 0, 0), (100, 1, 0), (-3, 0, 0), (-4, 1, 1), (-5, 0, 0)]
    e1 = [(-0.5, 0, 0), (-0.5, 0, 0), (-0.5, 0, 0), (-150, 1, 0), (-1, 0, 0), (-1, 0, 0)]
    for (r0, d0, t0), (r1, d1, t1) in zip(e0, e1):
        _add(b, [r0, r1], [d0, d1], [t0, t1])
    g0, g1 = _returns([-1, -2, 100]), _returns([-0.5, -0.5, -0.5, -150])
    assert b.mc_mask[:6, 0, 0].tolist() == [1, 1, 1, 0, 0, 0]
    assert b.mc_mask[:6, 1, 0].tolist() == [1, 1, 1, 1, 0, 0]
    assert np.allclose(b.mc_returns[:3, 0, 0], g0, rtol=1e-6)
    assert np.allclose(b.mc_returns[:4, 1, 0], g1, rtol=1e-6)


def test_mc_returns_skip_overwritten_slots():
    b = _buffer(size=4)
    rewards = [-1.0, -2.0, -3.0, -4.0, -5.0, 50.0]
    for k, r in enumerate(rewards):
        _add(b, [r], [k == len(rewards) - 1])
    g = _returns(rewards)
    # slots 0, 1 now hold steps 4, 5; slots 2, 3 hold steps 2, 3; steps 0, 1 were overwritten
    assert b.mc_mask[:, 0, 0].tolist() == [1, 1, 1, 1]
    assert np.allclose(b.mc_returns[:, 0, 0], [g[4], g[5], g[2], g[3]], rtol=1e-6)


def test_mc_label_of_an_overwritten_unfinished_slot_is_cleared():
    b = _buffer(size=3)
    _add(b, [1.0], [False]); _add(b, [1.0], [True])  # finished episode in slots 0, 1
    assert b.mc_mask[:2, 0, 0].tolist() == [1, 1]
    _add(b, [1.0], [False]); _add(b, [1.0], [False])  # slot 2, then slot 0 again (unfinished)
    assert b.mc_mask[:, 0, 0].tolist() == [0, 1, 0]


def test_mc_window_masks_stale_transitions():
    b = _buffer(window=3)
    for k in range(5):
        _add(b, [-1.0], [k == 4])
    b._get_samples(np.arange(5))
    b._last = (np.arange(5), np.zeros(5, dtype=int))
    _, mask = b.monte_carlo()
    assert mask.flatten().tolist() == [0, 0, 1, 1, 1]


class _Scripted(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32)

    def __init__(self, endings):
        self.endings, self.episode, self.k = list(endings), -1, 0

    def reset(self, *, seed=None, options=None):
        self.episode += 1
        self.k = 0
        return np.zeros(OBS, np.float32), {}

    def step(self, action):
        self.k += 1
        end = self.endings[self.episode] if self.k == 3 else None
        return np.zeros(OBS, np.float32), float(-self.k), end == "terminated", end == "truncated", {}


def test_sb3_path_labels_terminals_and_not_truncations():
    model = M1SAC("MlpPolicy", _Scripted(["terminated", "truncated", "terminated"]), seed=0, device="cpu",
                  learning_starts=10_000, buffer_size=100, gamma=GAMMA, cf_lambda=0.0,
                  policy_kwargs={"net_arch": [8]})
    model.learn(total_timesteps=9)
    b = model.replay_buffer
    assert b.mc_mask[:9, 0, 0].tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1]
    g = _returns([-1, -2, -3])
    assert np.allclose(b.mc_returns[:3, 0, 0], g) and np.allclose(b.mc_returns[6:9, 0, 0], g)


def test_eq_mask_live_plan_only_not_descent():
    obs = th.zeros((4, OBS))
    obs[0, PLAN_REMAINING_INDEX] = 0.3                                   # live plan
    obs[1, PLAN_REMAINING_INDEX] = 0.0                                   # no plan
    obs[2, PLAN_REMAINING_INDEX], obs[2, DESCENT_INDEX] = 0.3, 1.0       # descent
    obs[3, PLAN_REMAINING_INDEX] = -0.1
    assert eq_mask(obs).flatten().tolist() == [1, 0, 0, 0]
    assert OBSERVATION_NAMES[PLAN_REMAINING_INDEX] == "remaining_160s"
    assert OBSERVATION_NAMES[DESCENT_INDEX] == "descent"


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
               ent_coef=0.005, gamma=GAMMA, policy_kwargs={"net_arch": [16, 16]}, **kw)


def _fill(buffer, rng, n=200):
    for _ in range(n):
        obs, nxt = rng.normal(size=OBS).astype(np.float32), rng.normal(size=OBS).astype(np.float32)
        act = rng.uniform(-1, 1, ACT).astype(np.float32)
        done = rng.uniform() < 0.1
        buffer.add(obs[None], nxt[None], act[None], np.array([rng.normal()]), np.array([done]),
                   [{"TimeLimit.truncated": False}])


def _params(model):
    return th.cat([p.detach().flatten() for p in model.policy.parameters()])


def _train(model, steps=4):
    model._total_timesteps, model.num_timesteps = 1000, 500
    model._setup_learn(1000)
    np.random.seed(5); th.manual_seed(5)
    model.train(gradient_steps=steps, batch_size=32)


@pytest.mark.parametrize("probe", [False, True])
def test_lambda_zero_is_plain_sac_bitwise_even_with_probe_and_audit(probe):
    m0 = _model(SAC)
    kw = {"grad_probe_every": 1, "audit_every": 1} if probe else {}
    m = _model(M1SAC, cf_lambda=0.0, mc_lambda=0.0, eq_lambda=0.0, **kw)
    _fill(m0.replay_buffer, np.random.default_rng(2))
    _fill(m.replay_buffer, np.random.default_rng(2))
    _train(m0)
    _train(m)
    assert th.equal(_params(m0), _params(m))
    if probe:
        s = m.fix_stats
        assert len(s["g_td"]) == 4 and all(np.isfinite(s["g_td"])) and min(s["g_mc"]) >= 0.0
        assert 0.0 < np.mean(s["mc_fraction"]) <= 1.0 and 0.0 < np.mean(s["eq_fraction"]) < 1.0
        assert len(s["band_q_spread"]) == 4 and len(s["q_minus_g_plan"]) == 4


class _Stub3(_Stub):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)


def test_probe_and_audit_with_a_three_dimensional_action():
    th.manual_seed(0)
    m = M1SAC("MlpPolicy", _Stub3(), seed=0, device="cpu", buffer_size=500, learning_starts=0, batch_size=32,
              ent_coef=0.005, gamma=GAMMA, policy_kwargs={"net_arch": [16, 16]}, cf_lambda=0.0,
              mc_lambda=0.1, eq_lambda=0.1, grad_probe_every=1, audit_every=1)
    rng = np.random.default_rng(6)
    for _ in range(200):
        obs, nxt = rng.normal(size=OBS).astype(np.float32), rng.normal(size=OBS).astype(np.float32)
        m.replay_buffer.add(obs[None], nxt[None], rng.uniform(-1, 1, (1, 3)).astype(np.float32),
                            np.array([rng.normal()]), np.array([rng.uniform() < 0.1]), [{"TimeLimit.truncated": False}])
    _train(m, steps=2)
    assert len(m.fix_stats["band_q_spread"]) == 2 and np.isfinite(m.fix_stats["band_q_spread"]).all()
    assert np.isfinite(m.fix_stats["g_eq"]).all()


@pytest.mark.parametrize("which", ["mc_lambda", "eq_lambda"])
def test_each_fix_moves_the_critic(which):
    m0 = _model(M1SAC, cf_lambda=0.0)
    m = _model(M1SAC, cf_lambda=0.0, **{which: 0.3})
    _fill(m0.replay_buffer, np.random.default_rng(3))
    _fill(m.replay_buffer, np.random.default_rng(3))
    _train(m0)
    _train(m)
    q = lambda model: th.cat([p.detach().flatten() for p in model.critic.parameters()])  # noqa: E731
    assert not th.equal(q(m0), q(m))


def test_saved_model_reloads_with_fix_settings(tmp_path):
    m = _model(M1SAC, cf_lambda=0.0, mc_lambda=0.1, eq_lambda=0.03, grad_probe_every=1)
    _fill(m.replay_buffer, np.random.default_rng(4))
    _train(m, steps=1)
    m.save(tmp_path / "m")
    r = M1SAC.load(tmp_path / "m", device="cpu")
    assert (r.mc_lambda, r.eq_lambda) == (0.1, 0.03)
    obs = np.random.default_rng(0).normal(size=OBS).astype(np.float32)
    assert np.array_equal(m.predict(obs, deterministic=True)[0], r.predict(obs, deterministic=True)[0])


# --------------------------------------------------------------------------- environment
@pytest.fixture(scope="module")
def env():
    return T1GStarEnv(openings=[1_000_000])


def test_tracking_errors_zero_without_plan_and_appended(env):
    assert OBSERVATION_NAMES[22:24] == ("track_position_error_1m", "track_velocity_error_0p3m_s")
    obs, _ = env.reset()
    assert obs.shape == (25,) and obs[22] == 0.0 and obs[23] == 0.0
    obs, _, _, _, info = env.step(np.array([0.0, 0.0]))
    x = info["execution"]
    assert x["request_kind"] == "no_plan" and x["new_plan_trigger"] == "no_plan"
    assert x["deviation_before_m"] == 0.0 and x["deadline"] == "none"
    assert np.isfinite(obs[22:24]).all() and 0.0 <= obs[22] < 1.0  # flying a fresh plan
    _, _, _, _, info = env.step(np.array([0.45, 0.0]))
    assert info["execution"]["request_kind"] == "keep" and info["execution"]["new_plan_trigger"] is None
    _, _, _, _, info = env.step(np.array([1.0, 0.0]))
    assert info["execution"]["request_kind"] == "later"


def test_keep_band_one_step_equivalence():
    a, b = T1GStarEnv(openings=[1_000_000]), T1GStarEnv(openings=[1_000_000])
    for e in (a, b):
        e.reset()
        e.step(np.array([0.0, 0.2]))
    oa, ra, ta, ua, _ = a.step(np.array([0.0, 0.2]))
    ob, rb, tb, ub, _ = b.step(np.array([0.3, 0.2]))
    assert np.array_equal(oa, ob) and ra == rb and (ta, ua) == (tb, ub)


def test_deadline_clamps_a_late_request(env):
    env.reset()
    g, e = env.generator, env.env
    e.time_s = 200.0  # 100 s request at t = 200 s: the deadline (300 - 60) leaves 40 s
    g.decide(e, 100.0)
    assert g.tau_s == pytest.approx(40.0)
    assert g.last_deadline in ("clamped", "infeasible_within")
    if g.last_deadline == "clamped" and g.plan is not None:
        assert g.tau_effective_s <= 40.0 + 1e-9


def test_deadline_replans_a_live_plan_arriving_too_late(env):
    env.reset()
    g, e = env.generator, env.env
    g.decide(e, 150.0)
    assert g.plan is not None and g.last_deadline == "none"
    remaining = g.remaining_s(e)
    g.deadline_margin_s = 200.0  # horizon 100 s < remaining: the plan is now beyond the deadline
    try:
        g.decide(e, remaining)
        assert g.last_deadline in ("plan_beyond", "infeasible_within")
        assert g.last_status != "kept"
    finally:
        g.deadline_margin_s = 60.0


def test_earliest_request_search_is_unaffected_by_the_deadline(env):
    env.reset()
    g, e = env.generator, env.env
    g.decide(e, 0.0)
    first = (g.last_status, g.tau_effective_s)
    env.reset()
    g = env.generator
    g.deadline_margin_s = 0.0
    g.decide(env.env, 0.0)
    assert (g.last_status, g.tau_effective_s) == first


def test_calibration_rule_revision_2():
    from experiments.t1_fix_calibration import calibrate

    n = 200

    def stats(mc, eq=1.0, fraction=0.3):
        return {"probe_decisions": list(np.linspace(0, 2500, n)), "g_td": [1.0] * n, "g_mc": [mc] * n,
                "g_eq": [eq] * n, "g_mc_dot_eq": [0.0] * n, "mc_fraction": [fraction] * n, "eq_fraction": [0.5] * n}

    assert calibrate(stats(1.0))["frozen"] == {"lambda_mc": 0.3, "lambda_eq": 0.1}
    assert calibrate(stats(2.0))["frozen"]["lambda_mc"] == 0.1
    assert calibrate(stats(100.0))["frozen"]["lambda_mc"] == 0.03   # strong: floor, never 0
    assert calibrate(stats(0.01))["frozen"]["lambda_mc"] == 0.3     # weak: capped share, at most 0.3
    assert calibrate(stats(1.0, fraction=0.0))["frozen"]["lambda_mc"] == 0.03  # no labels yet: floor
    out = calibrate(stats(1.0, eq=4.0))  # combined sqrt(0.09 + 0.16) = 0.5: not above the limit
    assert out["frozen"]["lambda_mc"] == 0.3 and out["combined_share_median"] == pytest.approx(0.5)
    out = calibrate(stats(1.0, eq=5.0))  # combined > 0.5: one step down
    assert out["frozen"]["lambda_mc"] == 0.1
    bad = stats(1.0)
    bad["g_td"] = [float("nan")] * n
    assert "defect" in calibrate(bad) and "frozen" not in calibrate(bad)


# --------------------------------------------------------------------------- interface v4 (entry speed)
def test_entry_speed_is_applied_only_to_new_plans_and_observed(env):
    obs, _ = env.reset()
    assert OBSERVATION_NAMES[24] == "plan_entry_speed" and obs[24] == 0.0
    assert env.entry_speed_of(np.array([0.0, 0.0])) == 0.06 and env.entry_speed_of(np.array([0, 0, 0.0])) == 0.06
    obs, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0]))
    x = info["execution"]
    assert x["entry_speed_applied"] and x["entry_speed_request_m_s"] == pytest.approx(0.10)
    assert x["entry_speed_plan_m_s"] == pytest.approx(0.10) and obs[24] == pytest.approx(1.0)
    plan = env.generator.plan
    obs, _, _, _, info = env.step(np.array([0.2, 0.0, -1.0]))  # keep band: the plan is kept
    x = info["execution"]
    assert x["plan_status"] == "kept" and not x["entry_speed_applied"]
    assert env.generator.plan is plan and x["entry_speed_plan_m_s"] == pytest.approx(0.10)
    assert obs[24] == pytest.approx(1.0)


def test_planner_terminal_velocity_carries_the_entry_speed(env):
    from env.t1_gstar import _rotation_at, free_body_rotations

    env.reset()
    e, g = env.env, env.generator
    target, chaser = e.target_state, e.chaser_state
    q0 = chaser.position - target.position
    v0 = chaser.rotation @ chaser.velocity - target.rotation @ target.velocity
    tau = 100.0
    history = free_body_rotations(target.rotation, target.omega, e.target_parameters.inertia, tau + 1.0)
    rotation, omega = _rotation_at(history, tau)
    for speed in (0.02, 0.06, 0.10):
        r = g.planner.solve(q0, v0, tau, history, None, entry_speed_m_s=speed)
        assert r["feasible"] and r["entry_speed_m_s"] == speed
        entry = r["q"][-1]
        inward = -(r["v"][-1] - np.cross(omega, entry)) @ (entry / np.linalg.norm(entry))
        assert inward == pytest.approx(speed, abs=1e-5)
    default = g.planner.solve(q0, v0, tau, history, None)
    assert default["entry_speed_m_s"] == 0.06


def test_zero_entry_action_reproduces_the_two_dimensional_action_bitwise():
    a, b = T1GStarEnv(openings=[1_000_000]), T1GStarEnv(openings=[1_000_000])
    a.reset(); b.reset()
    for two in (np.array([-0.4, 0.2]), np.array([0.1, 0.2]), np.array([0.8, -0.3])):
        oa, ra, *_ = a.step(two)
        ob, rb, *_ = b.step(np.array([*two, 0.0]))
        assert np.array_equal(oa, ob) and ra == rb


def test_keep_band_equivalence_holds_with_an_entry_speed_component():
    a, b = T1GStarEnv(openings=[1_000_000]), T1GStarEnv(openings=[1_000_000])
    for e in (a, b):
        e.reset()
        e.step(np.array([0.0, 0.2, 0.5]))
    oa, ra, *_ = a.step(np.array([0.0, 0.2, -0.7]))
    ob, rb, *_ = b.step(np.array([0.35, 0.2, -0.7]))
    assert np.array_equal(oa, ob) and ra == rb


def test_entry_speed_counterfactual_alternative_and_gate():
    from types import SimpleNamespace

    from env.t1_counterfactual import CounterfactualPredictor

    predictor = CounterfactualPredictor()
    real = T1GStarEnv(openings=[1_000_000])
    real.reset()  # episode 1, decision 0: odd parity -> tau alternative
    p_new = SimpleNamespace(extra={"new_plan": True, "entry_speed_plan_m_s": 0.08}, tau_effective_s=50.0)
    alt, kind = predictor.alternative(real, np.array([0.0, 0.1, 0.5]), p_new)
    assert kind != "entry_speed"
    real.episode = 2  # even parity
    alt, kind = predictor.alternative(real, np.array([0.0, 0.1, 0.5]), p_new)
    assert kind == "entry_speed" and np.allclose(alt, [0.0, 0.1, -0.5])
    alt, _ = predictor.alternative(real, np.array([0.0, 0.1, -0.2]), p_new)
    assert np.allclose(alt, [0.0, 0.1, 0.8])
    p_kept = SimpleNamespace(extra={"new_plan": False}, tau_effective_s=50.0)
    assert predictor.alternative(real, np.array([0.0, 0.1, 0.5]), p_kept)[1] != "entry_speed"

    def pred(v, new=True):
        return SimpleNamespace(extra={"new_plan": new, "entry_speed_plan_m_s": v}, tau_effective_s=50.0,
                               plan_status="optimal", solved=True, solve_slack=0.0, window_mode_switch=False,
                               window_violation=False, window_violation_kinds=[], saturation=0.0)

    errors = {"e_r": 0.0, "e_p": 0.0, "e_o": 0.0}
    w, reasons = predictor.gate(real, pred(0.08), pred(0.04), errors, "entry_speed")
    assert w == 1 and reasons == []  # same tau, different entry speed: physically distinct
    w, reasons = predictor.gate(real, pred(0.08), pred(0.07), errors, "entry_speed")
    assert w == 0 and "same_entry_speed" in reasons
    w, reasons = predictor.gate(real, pred(0.08), pred(0.04), errors, "keep_vs_shift")
    assert "same_effective_tau" in reasons


def test_shadow_predicts_the_entry_speed_and_leaves_the_real_branch_alone():
    from env.t1_counterfactual import CounterfactualPredictor, T1CounterfactualEnv

    plain = T1GStarEnv(openings=[1_000_000])
    cf = T1CounterfactualEnv(openings=[1_000_000])
    plain.reset(); cf.reset()
    for action in (np.array([0.0, 0.1, 1.0]), np.array([0.0, 0.1, -1.0]), np.array([0.7, 0.1, -1.0])):
        o1, r1, *_ = plain.step(action)
        o2, r2, _, _, info = cf.step(action)
        assert np.array_equal(o1, o2) and r1 == r2
        assert info["counterfactual"]["entry_speed_real_plan_m_s"] == plain._plan_entry_speed()
    predictor = CounterfactualPredictor()
    real = T1GStarEnv(openings=[1_000_001])
    real.reset()
    action = np.array([0.0, 0.0, -1.0])
    predictor.sync(real)
    p = predictor.predict(action)
    predictor.sync(real)
    shadow = predictor.closed_loop(action)
    obs, reward, *_ = real.step(action)
    assert np.array_equal(shadow["observation"], obs) and shadow["reward"] == reward
    assert p.extra["new_plan"] and p.extra["entry_speed_plan_m_s"] == pytest.approx(0.02)
    assert p.observation[24] == pytest.approx(-1.0) and obs[24] == pytest.approx(-1.0)
