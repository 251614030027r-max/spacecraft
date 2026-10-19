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
