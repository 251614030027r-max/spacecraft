import numpy as np
import torch as th

from env.t1_b_feasible_v1 import FeasibleArrivalEnv
from env.t1_b_feasible_v2 import FeasibleArrivalEnvV2
from train.t1_b_v2_sac import FeasibleSupportSACV2


class _Env:
    _review_index = 0


def test_v1_alternative_rule_is_unchanged_farthest():
    e = _Env()
    assert FeasibleArrivalEnv.alternative_index(e, [80., 100., 120.], 0) == 2
    assert FeasibleArrivalEnv.alternative_index(e, [80., 100., 120.], 2) == 0
    assert FeasibleArrivalEnv.alternative_index(e, [80., 100.], 1) == 0


def test_v2_alternative_is_fixed_hash_and_covers_both_sides():
    e = _Env(); seen = set()
    for op in range(292140, 292200):
        e.current_opening = op
        j = FeasibleArrivalEnvV2.alternative_index(e, [80., 100., 120.], 1)
        assert j in (0, 2) and j == FeasibleArrivalEnvV2.alternative_index(e, [80., 100., 120.], 1)
        seen.add(j)
    assert seen == {0, 2}


def test_v2_temperature_is_one_constant_per_decision():
    m = FeasibleSupportSACV2.__new__(FeasibleSupportSACV2); m.candidate_alpha = 0.5
    z = th.zeros((4, 52)); z[:, 40] = th.tensor([1 / 3, 2 / 3, 1., 0.])
    assert th.equal(m.alpha_of(z), th.full((4, 1), 0.5))


def test_soft_optimal_policy_is_interior_at_v2_and_degenerate_at_v1_temperature():
    q = th.tensor([0., -0.5, -1.0])
    assert float(th.softmax(q / 0.015, 0)[1]) < 1e-12
    assert float(th.softmax(q / 0.5, 0)[1]) > 0.2


def test_v3_decides_only_without_a_live_plan_and_keeps_it():
    import json
    from env.t1_b_task import MissionDistribution
    from env.t1_b_feasible_v1 import bin_center
    from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3
    from env.t1_n_review import N_DUE_INDEX
    prof = json.load(open('evidence/T1B_V1/profile_mission_v3.json', encoding='utf-8-sig'))
    env = FeasibleArrivalEnvV3(openings=[292140], mission=MissionDistribution(**prof)); obs, _ = env.reset()
    n = int((obs[49:52] > .5).sum())
    assert obs[N_DUE_INDEX] > .5 and n == 3 and obs[27] < .5
    obs, *_, info = env.step(np.array([bin_center(2, 3)], np.float32))
    tau = info['support_transition']['tau_candidates'][2]
    assert info['support_transition']['learning_decision'] and info['support_transition']['candidate_index'] == 2
    for _ in range(9):
        assert obs[N_DUE_INDEX] < .5 and obs[27] > .5
        obs, r, term, trunc, info = env.step(np.array([0.9], np.float32))   # ignored: no decision while the plan is live
        assert not info['support_transition']['learning_decision']
    assert abs(env.generator.remaining_s(env.env) + env.env.time_s - tau) < 1.0


def test_v4_learner_uses_two_counterfactual_slots_and_exploration_half():
    import gymnasium as gym
    from train.t1_b_v4_sac import AllAlternativesReplay, FeasibleSupportSACV4, V4_BEHAVIOUR_EPSILON
    b = AllAlternativesReplay(8, gym.spaces.Box(-np.inf, np.inf, (52,), np.float32), gym.spaces.Box(-1, 1, (1,), np.float32), device='cpu')
    b._sets[1]['cf_weights'][3, 0] = 0.7; b._sets[0]['cf_weights'][3, 0] = 0.2
    b.use(1); assert float(b.cf_weights[3, 0]) == np.float32(0.7)
    b.use(0); assert float(b.cf_weights[3, 0]) == np.float32(0.2)
    assert V4_BEHAVIOUR_EPSILON == 0.5 and AllAlternativesReplay.SLOTS == 2


def test_v5_full_horizon_gate_admits_only_reproduced_real_outcomes():
    from train.t1_b_v5_sac import FullHorizonReplay
    g = FullHorizonReplay.calibrate
    pay = {'factual': {'done': True, 'event': 'success', 'return': 60.0, 'steps': 50, 'outcome': 'accepted'},
           'alternative': {'done': True, 'event': 'success', 'return': 63.0, 'steps': 58, 'outcome': 'accepted'}}
    ok = g(None, pay, 61.0, None, 52, 'success', False)
    assert ok['weight'] == 0.8 and not ok['reasons']
    assert g(None, pay, 61.0, None, 52, 'resource_exhausted', False)['weight'] == 0.
    assert g(None, pay, 50.0, None, 52, 'success', False)['weight'] == 0.
    assert g(None, pay, 61.0, None, 70, 'success', False)['weight'] == 0.
    assert FullHorizonReplay.close_after(None, pay) == 60


def test_v5_gate_admits_a_reproduced_plan_loss():
    from train.t1_b_v5_sac import FullHorizonReplay
    pay = {'factual': {'done': False, 'event': None, 'return': -14.6, 'steps': 25, 'outcome': 'accepted'},
           'alternative': {'done': False, 'event': None, 'return': -23.3, 'steps': 46, 'outcome': 'accepted'}}
    due = np.zeros(52, np.float32); due[39] = 1.
    assert FullHorizonReplay.calibrate(None, pay, -15.0, due, 25, None, False)['weight'] > 0.9
    assert FullHorizonReplay.calibrate(None, pay, -15.0, np.zeros(52, np.float32), 25, None, False)['weight'] == 0.
    assert FullHorizonReplay.calibrate(None, pay, -15.0, due, 24, None, False)['weight'] == 0.


def test_v6_intent_replay_samples_only_closed_decisions_and_sets_last():
    import gymnasium as gym
    from train.t1_b_v6_sac import IntentReplay
    b = IntentReplay(16, gym.spaces.Box(-np.inf, np.inf, (52,), np.float32), gym.spaces.Box(-1, 1, (1,), np.float32), device='cpu')
    assert b.sample_decisions(8) is None
    b.pos = 6; b.due[[1, 4], 0] = True; b.interval_ready[[1, 4], 0] = True; b.due[3, 0] = True   # 3: still open
    b.observations[1, 0, 0] = 1.; b.observations[4, 0, 0] = 4.
    d = b.sample_decisions(64)
    assert set(d.observations[:, 0].tolist()) <= {1., 4.} and len(set(b._last[0].tolist())) == 2
