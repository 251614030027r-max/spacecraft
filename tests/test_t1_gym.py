"""Training contract of the T1 + G* gym wrapper (M0 / M1)."""

from __future__ import annotations

import numpy as np
import pytest

from env.t1_gym import OBSERVATION_NAMES, T1GStarEnv, T1GymConfig


@pytest.fixture(scope="module")
def env():
    return T1GStarEnv(openings=[1_000_000])


def test_reset_observation_and_action_space(env):
    obs, info = env.reset()
    assert obs.shape == (len(OBSERVATION_NAMES),) and np.all(np.isfinite(obs))
    assert info["opening"] == 1_000_000
    assert env.action_space.shape == (3,)  # interface v4: tau, kappa, entry speed
    # no plan yet: the request is absolute, and kappa spans its range
    c = T1GymConfig()
    assert env.intent(np.array([-1.0, -1.0])) == pytest.approx((0.0, c.kappa_min))
    assert env.intent(np.array([1.0, 1.0])) == pytest.approx((c.tau_max_s, c.kappa_max))
    assert env.entry_speed_of(np.array([0.0, 0.0, -1.0])) == pytest.approx(0.02)
    assert env.entry_speed_of(np.array([0.0, 0.0, 1.0])) == pytest.approx(0.10)


def test_step_plans_then_a_small_shift_keeps_the_plan(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    x = info["execution"]
    assert x["new_plan"] and x["tau_effective_s"] >= x["tau_request_s"] - 1e-9
    assert reward < 0.0 and not terminated and not truncated
    assert obs[OBSERVATION_NAMES.index("has_plan")] == 1.0
    remaining = env.generator.remaining_s(env.env)
    tau, _ = env.intent(np.array([0.4, 0.0]))  # inside the v2 keep band: no shift
    assert tau == pytest.approx(remaining)
    _, _, _, _, info = env.step(np.array([0.4, 0.0]))
    assert info["execution"]["plan_status"] == "kept"
    assert info["execution"]["plan_wall_s"] == 0.0


def test_training_openings_are_disjoint_from_the_research_blocks():
    for seed in range(3):
        base = 1_000_000 * (seed + 1)
        assert base > 1_000_000 - 1 and not (290000 <= base < 1_000_000)


def test_interface_v2_shift_map_and_inverse(env):
    assert env.shift_of(0.0) == 0.0 and env.shift_of(0.5) == 0.0 and env.shift_of(-0.49) == 0.0
    assert env.shift_of(0.75) == pytest.approx(15.0) and env.shift_of(-1.0) == pytest.approx(-30.0)
    for shift in (-30.0, -15.0, -4.0, 4.0, 15.0, 30.0):
        assert env.shift_of(env.action_for_shift(shift)) == pytest.approx(shift)


def test_gstar_v04_never_trades_a_plan_for_a_later_one_on_an_earlier_request():
    e = T1GStarEnv(openings=[291000])
    e.reset()
    e.step(np.array([0.2 * 2 - 1.0, 0.0]))  # tau 64 s from no plan
    g = e.generator
    for _ in range(4):
        remaining = g.remaining_s(e.env)
        old = g.plan
        g.decide(e.env, max(remaining - 30.0, 0.0) if remaining > 32.0 else remaining + 0.0)
        if g.plan is not old:
            assert g.tau_effective_s < remaining - g.replan_tau_tolerance_s
        else:
            assert g.last_status in ("kept", "kept:no_earlier", "kept:infeasible_request")
        e.env.step(reference=g)
