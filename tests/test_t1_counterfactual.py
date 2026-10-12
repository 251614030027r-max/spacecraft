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


def _window_inputs(env):
    from controllers.mpc.prediction import relative_to_vector
    from dynamics.relative import relative_state

    model = env.env.controller.reference_model
    targets = [env.env.target_state]
    for i in range(20):
        targets.append(model.propagate_target(targets[-1], env.env.time_s + 0.1 * i))
    x0 = relative_to_vector(relative_state(env.env.target_state, env.env.chaser_state))
    # hold the current target-frame relative state over the window (a valid far-field state)
    return targets, [x0.copy() for _ in range(21)]


def _with_transform(x, rotation=None, position_scale=1.0):
    from dynamics.lie import se3_exp, se3_log

    t = se3_exp(x[:6])
    if rotation is not None:
        t[:3, :3] = rotation @ t[:3, :3]
    t[:3, 3] = t[:3, 3] * position_scale
    y = x.copy()
    y[:6] = se3_log(t)
    return y


def test_window_check_rejects_a_mid_window_violation_that_recovers_by_the_end(started):
    from dynamics.lie import so3_exp
    from env.t1_counterfactual import check_predicted_window

    targets, states = _window_inputs(started)
    clean = check_predicted_window(started.env, started.generator, targets, states)
    assert not clean["window_violation"] and not clean["window_mode_switch"]
    r = np.linalg.norm(started.env.chaser_state.position - started.env.target_state.position)
    far = list(states)
    far[10] = _with_transform(states[10], position_scale=31.0 / r)  # beyond 30 m at 1.0 s only
    out = check_predicted_window(started.env, started.generator, targets, far)
    assert out["window_violation"] and out["window_first_violation_step"] == 10 and "distance" in out["window_violation_kinds"]
    turned = list(states)
    turned[7] = _with_transform(states[7], rotation=so3_exp(np.array([0.0, np.pi / 1.5, 0.0])))  # camera off target
    out = check_predicted_window(started.env, started.generator, targets, turned)
    assert out["window_violation"] and out["window_first_violation_step"] == 7 and "fov" in out["window_violation_kinds"]
    inside = list(states)
    inside[12] = _with_transform(states[12], position_scale=3.0 / r)  # into the sphere, then back out
    out = check_predicted_window(started.env, started.generator, targets, inside)
    assert out["window_violation"] or out["window_mode_switch"]
    assert out["window_first_violation_step"] == 12 or out["window_first_mode_switch_step"] == 12


def test_prediction_carries_the_window_verdict(started):
    predictor = CounterfactualPredictor()
    predictor.sync(started)
    p = predictor.predict(np.array([0.0, 0.0]))
    assert p.window_violation in (False, True) and isinstance(p.window_violation_kinds, list)
    assert p.window_violation or not p.end_violation  # the window includes its end
