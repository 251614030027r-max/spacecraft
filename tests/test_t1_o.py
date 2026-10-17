"""T1-O profiles, interface hooks and the physical screen (research decision 2026-10-17)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from env.t1_env import T1EnvConfig
from env.t1_gym import T1GymConfig
from env.t1_l_task import t1l_env_config
from env.t1_o_task import T1O_DELAY_MAX_S, T1O_PROFILES, t1o_env_config, t1o_review_env
from env.t1_task_review import DELAY_MAX_S, T1LReviewEnv
from experiments.t1o_screen import CANDIDATES, FixedCandidate, subset16

ROOT = Path(__file__).resolve().parents[1]
SCREEN_SHA = {
    "p1": "b255d145c85bf947e69b5ede86958c62ceadfc12e7feb110b1a8ebab3c1b35ff",
    "p2": "d95babeb32754f9a731214e04bec647d1cc6890f962f35c4590ccc153bb59960",
}


@pytest.mark.parametrize("profile", ["p1", "p2", "o"])
def test_profiles_change_only_initial_range_tumble_and_fuel_weight(profile):
    base, cfg = asdict(T1EnvConfig()), asdict(t1o_env_config(profile, fuel_cost=15.0))
    spec = T1O_PROFILES[profile]
    assert cfg["initial"]["range_m"] == spec["range_m"]
    assert cfg["initial"]["tumble_rate_deg_s"] == spec["tumble_rate_deg_s"]
    assert cfg["reward"]["delta_v_cost_per_m_s"] == 15.0
    for key in ("initial", "reward"):
        cfg[key] = {k: v for k, v in cfg[key].items() if k not in ("range_m", "tumble_rate_deg_s", "delta_v_cost_per_m_s")}
        base[key] = {k: v for k, v in base[key].items() if k not in ("range_m", "tumble_rate_deg_s", "delta_v_cost_per_m_s")}
    assert cfg == base


def test_old_interface_defaults_are_unchanged():
    assert T1GymConfig().omega_scale_deg_s == 2.7
    assert DELAY_MAX_S == 60.0
    env = T1LReviewEnv(env_config=t1l_env_config())
    assert env.delay_max_s == 60.0
    assert t1o_review_env("p1").delay_max_s == T1O_DELAY_MAX_S == 90.0


def test_delay_range_is_0_to_90_s_after_the_feasible_anchor():
    env = t1o_review_env("p1")
    env.reset(options={"opening": 299990})
    for a0, delay in ((-1.0, 0.0), (0.0, 45.0), (1.0 / 3.0, 60.0), (1.0, 90.0)):
        tau, _, _ = env.apply_intent(np.array([a0, 1.0, -1.0]))
        assert env._anchor["delay_s"] == pytest.approx(delay)
        assert tau == pytest.approx(env._anchor["tau_min_feasible_s"] + delay)


def test_step_delta_v_is_the_decision_increment_of_the_episode_total():
    env = t1o_review_env("p1")
    env.reset(options={"opening": 299990})
    before = env.env._delta_v
    _, reward, _, _, info = env.step(np.array([-1.0, 1.0, -1.0]))
    dv = info["delta_v_step_m_s"]
    assert dv == pytest.approx(env.env._delta_v - before, abs=1e-15)
    c = env.env.config
    # the step reward is time cost + fuel cost exactly (no terminal event in the first 2 s)
    assert reward == pytest.approx(-c.reward.time_cost_per_s * 2.0 - c.reward.delta_v_cost_per_m_s * dv, abs=1e-9)


def test_omega_observation_scale_is_a_parameter():
    a = t1o_review_env("p1")
    b = t1o_review_env("p1", gym_config=T1GymConfig(omega_scale_deg_s=3.0))
    oa, _ = a.reset(options={"opening": 299990})
    ob, _ = b.reset(options={"opening": 299990})
    from env.t1_gym import OBSERVATION_NAMES

    i = OBSERVATION_NAMES.index("omega_2p7deg_s")
    assert ob[i] == pytest.approx(oa[i] * 2.7 / 3.0, rel=1e-6)
    assert np.array_equal(np.delete(oa, i), np.delete(ob, i))


@pytest.mark.parametrize("profile", ["p1", "p2"])
def test_screen_openings_are_frozen_stratified_and_disjoint(profile):
    path = ROOT / f"experiments/t1o_screen_{profile}_openings.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SCREEN_SHA[profile]
    doc = json.loads(path.read_text())
    cells = {}
    for o in doc["openings"]:
        cells.setdefault((o["latitude_bin"], o["phase_bin"]), []).append(o["seed"])
    assert len(cells) == 16 and all(len(v) == 2 for v in cells.values())
    assert doc["subset16"] == subset16(doc) and len(set(doc["subset16"])) == 16
    for s in doc["seeds"]:
        assert 299_500 <= s < 299_900  # outside 297xxx-2986xx, 2990xx-2991xx, 2999xx and >= 300000


def test_fixed_candidate_delay_applies_to_the_first_plan_only():
    env = t1o_review_env("p1")
    env.reset(options={"opening": 299990})
    f3 = FixedCandidate("F3")
    a = f3.action(env)
    assert a[0] == pytest.approx(2.0 * 60.0 / 90.0 - 1.0)
    assert a[1] == pytest.approx(1.0) and a[2] == pytest.approx(-1.0)
    _, _, _, _, info = env.step(a)
    f3.executed(env, info)
    assert f3.first_done
    env.generator.plan = None  # no live plan any more: the uniform rule asks for the earliest
    assert f3.action(env)[0] == -1.0
    assert {f"F{i}" for i in range(1, 9)} <= set(CANDIDATES)


# --------------------------------------------------------------------------- 2 x 2 sample eligibility
from types import SimpleNamespace  # noqa: E402

import torch as th  # noqa: E402

from env.t1_counterfactual import CounterfactualConfig  # noqa: E402
from env.t1_option_cf import near_kappa_alternative, near_kappa_gate  # noqa: E402
from env.t1_task_review import DUE_INDEX  # noqa: E402
from tests.test_t1_l_s import G, _fill, _model  # noqa: E402


def _batch(m, seed=0):
    np.random.seed(seed)
    data = m.replay_buffer.sample(64)
    return data, m._td_targets(data, data.rewards * 0 + 7.0)


def test_r1_and_r3_share_one_eligibility_and_differ_only_in_the_review_target():
    r1 = _model(interval_bellman=False, uniform_eligibility=True)
    r3 = _model(interval_bellman=True, uniform_eligibility=True)
    for m in (r1, r3):
        _fill(m.replay_buffer, np.random.default_rng(2), n=60)
        m.target_value = lambda obs: obs[:, 1:2] * 0 + 7.0
    d1, (t1, k1) = _batch(r1)
    d3, (t3, k3) = _batch(r3)
    assert th.equal(d1.observations, d3.observations) and th.equal(k1, k3)
    assert th.equal(r1._eligible, k1) and th.equal(r3._eligible, k3)
    ready, ret, disc, _, done = r3.replay_buffer.interval()
    review = d3.observations[:, DUE_INDEX] > 0.5
    one = d3.rewards + (1 - d3.dones) * G * 7.0
    assert th.allclose(t1, one)  # R1: one-step everywhere
    eligible_review = review & (ready[:, 0] > 0)
    assert eligible_review.any() and th.allclose(t3[eligible_review], (ret + disc * (1 - done) * 7.0)[eligible_review])
    assert th.allclose(t3[~review], one[~review])
    assert th.all(k1[review & (ready[:, 0] == 0)] == 0.0) and th.all(k1[~review] == 1.0)


def test_eligibility_masks_the_mc_equivalence_and_relative_terms():
    m = _model(interval_bellman=False, uniform_eligibility=True)
    _fill(m.replay_buffer, np.random.default_rng(3), n=90)
    data, _ = _batch(m, seed=1)
    q = m.critic(data.observations, data.actions)
    m._eligible = th.zeros_like(m._eligible)
    assert m._mc_loss(data, q)[0].item() == 0.0
    assert m._eq_loss(data)[0].item() == 0.0
    cf = SimpleNamespace(weights=th.ones_like(m._eligible))
    assert float(m._cf_weights(cf).sum()) == 0.0
    off = _model()  # T1-L-S default: no uniform eligibility, the 672ba79 behaviour
    _fill(off.replay_buffer, np.random.default_rng(3), n=90)
    _batch(off, seed=1)
    assert off._eligible is None and off._cf_weights(cf) is cf.weights


def test_r1_actor_samples_closed_reviews_only():
    m = _model(interval_bellman=False, uniform_eligibility=True)
    _fill(m.replay_buffer, np.random.default_rng(4))
    np.random.seed(0)
    batch = m._review_batch(64)
    assert th.all(batch[:, DUE_INDEX] == 1.0) and m.actor_updates == 1
    closed = m.replay_buffer.due & m.replay_buffer.interval_ready
    assert closed.sum() < m.replay_buffer.due[: m.replay_buffer.pos].sum()  # the open latest review is excluded


# --------------------------------------------------------------------------- near-field kappa counterfactual
@pytest.mark.parametrize("kappa, alt", [(1.0, 0.8), (0.7, 0.5), (0.65, 0.85), (0.3, 0.5)])
def test_near_kappa_alternative_moves_kappa_by_0p2_towards_the_middle(kappa, alt):
    env = t1o_review_env("p1")
    c = env.config
    a1 = 2.0 * (kappa - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
    a = near_kappa_alternative(env, np.array([0.2, a1, -0.5]))
    assert c.kappa_min + (c.kappa_max - c.kappa_min) * 0.5 * (a[1] + 1.0) == pytest.approx(alt)
    assert a[0] == pytest.approx(0.2) and a[2] == pytest.approx(-0.5)


def _pred(controls, status="descent", switch=False):
    return SimpleNamespace(extra={"controls": np.asarray(controls, float), "near_mode_switch": switch},
                           plan_status=status, solved=True, solve_slack=0.0, window_violation=False,
                           window_violation_kinds=[], saturation=0.0)


def test_near_kappa_gate_requires_distinct_execution_descent_mode_and_no_mode_change():
    env = t1o_review_env("p1")
    env.reset(options={"opening": 299990})
    cfg, errors = CounterfactualConfig(), {"e_r": 0.0, "e_p": 0.0, "e_o": 0.0}
    base = np.zeros((20, 6))
    moved = base.copy()
    moved[:, 3] = 0.01
    assert near_kappa_gate(env, cfg, _pred(base), _pred(moved), errors) == (1, [])
    assert "same_execution" in near_kappa_gate(env, cfg, _pred(base), _pred(base), errors)[1]
    assert "alt_mode_switch" in near_kappa_gate(env, cfg, _pred(base), _pred(moved, switch=True), errors)[1]
    assert "real_not_descent:kept" in near_kappa_gate(env, cfg, _pred(base, "kept"), _pred(moved), errors)[1]


def test_delay_counterfactual_step_is_15_s_of_the_90_s_range():
    from env.t1_option_cf import T1LReviewCFEnv

    env = T1LReviewCFEnv(env_config=t1o_env_config("p1"), delay_max_s=T1O_DELAY_MAX_S, near_field_kappa=True)
    env.reset(options={"opening": 299990})
    assert env.predictor.shadow.delay_max_s == T1O_DELAY_MAX_S
    alt, kind = env.predictor.alternative(env, np.array([0.0, 1.0, -1.0]), None)
    if kind == "delay_shift":
        assert alt[0] == pytest.approx(1.0 / 3.0)
    else:  # the parity picked the entry-speed alternative; the other parity gives the delay
        env._review_index += 1
        alt, kind = env.predictor.alternative(env, np.array([0.0, 1.0, -1.0]), None)
        assert kind == "delay_shift" and alt[0] == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- snapshot / restore
def _state(env):
    e = env.env
    return (e.target_state.position.copy(), e.target_state.rotation.copy(), e.chaser_state.position.copy(),
            e.chaser_state.rotation.copy(), e.chaser_state.velocity.copy(), e.chaser_state.omega.copy(), e.time_s,
            e._delta_v, env.generator.last_status, env.generator.tau_effective_s)


def test_branches_from_one_snapshot_are_bitwise_equal_and_close_to_the_real_continuation():
    """Gate 1b / Gate 3 branching: two shadows synced from the same real state with fresh
    solvers give the same closed loop bit for bit under the same actions (state, clocks, G*
    plan memory, MPC warm start, review clock are all cloned); the real continuation differs
    only by the solver-cache numerics (measured, ~1e-8), which is the repeatability error."""

    from env.t1_option_cf import ReviewCounterfactualPredictor

    cfg = t1o_env_config("p1")
    real = t1o_review_env("p1")
    real.reset(options={"opening": 299990})
    policy = FixedCandidate("F3")
    for _ in range(2):  # advance into a live plan, with a solver cache in the real MPC
        a = policy.action(real)
        _, _, _, _, info = real.step(a)
        policy.executed(real, info)
    branches = [ReviewCounterfactualPredictor(None, cfg, None, shadow_factory=lambda: t1o_review_env("p1"))
                for _ in range(2)]
    for b in branches:
        b.sync(real, fresh_solvers=True)
    for k in range(3):
        a = policy.action(real)
        o_r, r_r, *_ = real.step(a)
        (o1, r1, *_), (o2, r2, *_) = (b.shadow.step(a) for b in branches)
        assert r1 == r2 and np.array_equal(o1, o2), k
        for x, y in zip(_state(branches[0].shadow), _state(branches[1].shadow)):
            assert np.array_equal(np.asarray(x), np.asarray(y)), k
        assert abs(r1 - r_r) < 1e-6 and float(np.max(np.abs(o1 - o_r))) < 1e-5


# --------------------------------------------------------------------------- V2 Gate A and the state rule
from experiments.t1o_v2 import GATE_A, STATE_FEATURES, apply_state_rule, fit_state_rule, gate_a  # noqa: E402


def _row(event="success", t=100.0, dv=1.0, j=80.0, obs=None):
    from env.t1_task_review import REVIEW_OBSERVATION_NAMES

    return {"event": event, "time_s": t, "delta_v_m_s": dv, "end_phase": "near_field", "hold_steps": 0,
            "discounted_return_by_fuel_weight": {str(l): j - (l - 5.0) * dv for l in (5.0, 10.0, 15.0, 20.0)},
            "decision_log": [{"deadline": "none"}],
            "initial_observation": obs if obs is not None else [0.0] * len(REVIEW_OBSERVATION_NAMES)}


def _synthetic(n=16):
    from env.t1_task_review import REVIEW_OBSERVATION_NAMES

    i = REVIEW_OBSERVATION_NAMES.index("sin_phase")
    rows = {c: {} for c in GATE_A}
    for s in range(n):
        obs = [0.0] * len(REVIEW_OBSERVATION_NAMES)
        obs[i] = -1.0 + 2.0 * s / (n - 1)
        wait_good = obs[i] < 0.0  # waiting pays where sin(phase) < 0: slower but cheaper
        for c in GATE_A:
            rows[c][s] = _row(t=100.0, dv=1.5, j=80.0, obs=obs)
        rows["A30"][s] = _row(t=130.0, dv=0.8 if wait_good else 1.6, j=90.0 if wait_good else 75.0, obs=obs)
        rows["K0"][s] = _row(t=110.0, dv=1.5, j=78.0 if s % 2 else 82.0, obs=obs)
    return rows


def test_gate_a_recognises_a_state_dependent_tradeoff_and_the_state_rule_recovers_it():
    rows, seeds = _synthetic(), list(range(16))
    result = gate_a(rows, seeds)
    assert result["A1"]["pass"] and len(result["pareto_openings"]) >= 4
    w = result["per_weight"]["5.0"]
    assert w["checks"]["A3_i"] and w["checks"]["A3_iii_gap"] and w["checks"]["A4_dimension"]
    rule = fit_state_rule(rows, seeds, 5.0)
    assert rule["form"] != "constant" and "sin_phase" in (rule.get("feature"), rule.get("feature2"))
    obs_wait = rows["A30"][0]["initial_observation"]
    assert apply_state_rule(rule, obs_wait) == "A30"
    assert all(f in STATE_FEATURES for f in (rule.get("feature"), rule.get("feature2")) if f)


def test_gate_a_without_opportunity_is_no_go_and_leaves_respect_the_minimum_size():
    rows = {c: {s: _row() for s in range(16)} for c in GATE_A}
    result = gate_a(rows, list(range(16)))
    assert result["verdict"] == "LEARNING_SPACE_NO_GO"
    assert fit_state_rule(rows, list(range(16)), 5.0)["form"] == "constant"
