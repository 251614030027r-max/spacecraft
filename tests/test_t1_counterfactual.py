"""M1 counterfactual contract: isolation, same-source first QP, observation closure."""

from __future__ import annotations

import numpy as np
import pytest

from env.t1_counterfactual import CounterfactualPredictor, T1CounterfactualEnv
from env.t1_gym import OBSERVATION_NAMES, T1GStarEnv

TAU = OBSERVATION_NAMES.index("tau_request_160s")
KAPPA = OBSERVATION_NAMES.index("kappa")
REMAINING = OBSERVATION_NAMES.index("remaining_160s")
HAS_PLAN = OBSERVATION_NAMES.index("has_plan")
TIME = OBSERVATION_NAMES.index("time_300s")


@pytest.fixture(scope="module")
def started():
    env = T1GStarEnv(openings=[291000])
    env.reset()
    env.step(np.array([-0.2, 0.0]))  # a live plan
    return env


def test_observation_closure_uses_the_alternative_request_and_kappa(started):
    predictor = CounterfactualPredictor()
    predictor.sync(started)
    real_action = np.array([0.0, -0.5])
    alt_action = np.array([0.5, 0.8])  # different tau shift and different kappa
    tau_real, kappa_real = started.intent(real_action)
    tau_alt, kappa_alt = started.intent(alt_action)
    p = predictor.predict(alt_action)
    z = p.observation
    assert z[TAU] == pytest.approx(tau_alt / 160.0) and z[TAU] != pytest.approx(tau_real / 160.0)
    assert z[KAPPA] == pytest.approx(kappa_alt) and z[KAPPA] != pytest.approx(kappa_real)
    assert z[HAS_PLAN] == 1.0
    assert z[REMAINING] == pytest.approx((p.tau_effective_s - 2.0) / 160.0, abs=1e-6)
    assert z[TIME] == pytest.approx((started.env.time_s + 2.0) / 300.0)
    # the real wrapper is untouched by the prediction
    assert started._last_request != tau_alt and started._last_kappa != kappa_alt


def test_shadow_first_qp_equals_the_real_first_command_and_real_branch_is_unchanged():
    plain = T1GStarEnv(openings=[291013])
    cf = T1CounterfactualEnv(openings=[291013])
    plain.reset()
    cf.reset()
    for action in (np.array([-0.3, 0.1]), np.array([0.0, 0.1]), np.array([0.6, -0.4])):
        o1, r1, *_ = plain.step(action)
        o2, r2, _, _, info = cf.step(action)
        assert np.array_equal(o1, o2) and r1 == r2
        assert np.array_equal(plain.env.chaser_state.position, cf.env.chaser_state.position)
        assert np.array_equal(info["counterfactual"]["real_first_wrench"], cf.first_wrench)


def test_alternative_rule_matches_the_design_table(started):
    predictor = CounterfactualPredictor()
    predictor.sync(started)
    p = predictor.predict(np.array([0.0, 0.1]))
    alt, kind = predictor.alternative(started, np.array([0.5, 0.1]), p)
    assert kind == "replan_vs_keep" and alt[0] == 0.0 and alt[1] == 0.1
    alt, kind = predictor.alternative(started, np.array([0.05, 0.1]), p)
    assert kind == "keep_vs_shift" and abs(alt[0]) == pytest.approx(0.5) and alt[1] == 0.1


def test_shadow_closed_loop_reproduces_the_real_step_bitwise_including_the_first_decision():
    predictor = CounterfactualPredictor()
    real = T1GStarEnv(openings=[291043])
    real.reset()
    predictor.sync(real)
    predictor.predict(np.array([0.3, 0.0]))  # shadow solvers now hold caches from another request
    real = T1GStarEnv(openings=[291027])  # a fresh episode: real controller and planner without cache
    real.reset()
    for action in (np.array([-0.125, 0.14]), np.array([0.0, 0.14]), np.array([0.4, -0.2])):
        predictor.sync(real)
        predictor.predict(action)  # leaves a solver cache in the shadow
        predictor.sync(real)
        shadow = predictor.closed_loop(action)
        obs, reward, *_ = real.step(action)
        assert np.array_equal(shadow["observation"], obs) and shadow["reward"] == reward
