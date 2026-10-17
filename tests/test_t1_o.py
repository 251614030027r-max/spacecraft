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
    assert set(CANDIDATES) == {f"F{i}" for i in range(1, 9)}
