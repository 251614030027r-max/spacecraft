"""Diagnostics (research only): intervention modes and the planned-arrival bookkeeping."""

from __future__ import annotations

import numpy as np

from diagnostics.t1_rootcause.common import mode_action, rule_a1
from diagnostics.t1_rootcause.critic_probe import _log_requests
from env.t1_gym import T1GStarEnv


def test_modes_without_and_with_a_live_plan():
    env = T1GStarEnv(openings=[295009])
    env.reset()
    actor = np.array([0.9, -0.3])
    assert np.allclose(mode_action("A", env, actor), actor)
    assert np.allclose(mode_action("B", env, actor), actor)          # no plan: the actor's request
    assert np.allclose(mode_action("C", env, actor), [-1.0, -0.3])   # no plan: earliest
    assert np.allclose(mode_action("D", env, actor), [-1.0, -0.3])
    assert np.allclose(mode_action("E", env, None), [-1.0, rule_a1(env)])
    env.step(np.array([-1.0, 0.0]))
    assert env.generator.remaining_s(env.env) > 0.0
    assert np.allclose(mode_action("B", env, actor), [0.0, -0.3])    # live plan: keep
    assert np.allclose(mode_action("C", env, actor), actor)
    assert np.allclose(mode_action("D", env, actor), [0.0, -0.3])
    assert env.shift_of(0.0) == 0.0


def test_planned_arrival_is_plan_start_plus_total_tau():
    log = [{"status": "optimal", "tau_effective_s": 100.0, "tau_request_s": 100.0},
           {"status": "kept", "tau_effective_s": 100.0, "tau_request_s": 98.0},
           {"status": "optimal", "tau_effective_s": 120.0, "tau_request_s": 120.0},
           {"status": "kept:no_earlier", "tau_effective_s": 120.0, "tau_request_s": 90.0}]
    q = _log_requests(log)
    assert [x["arrival_after_s"] for x in q] == [100.0, 100.0, 124.0, 124.0]
    assert [x["kind"] for x in q] == ["no_plan", "keep", "later", "earlier"]
    assert q[2]["remaining_s"] == 96.0
