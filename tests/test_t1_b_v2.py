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
