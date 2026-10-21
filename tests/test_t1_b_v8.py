import json

import numpy as np
import torch as th

from env.t1_b_feasible_v1 import bin_center
from env.t1_n_review import N_DUE_INDEX
from experiments.t1b_v8 import choose, trigger
from train.t1_b_v8_sac import event_aware_targets


def test_event_aware_targets_suppress_failures_rank_successes_and_skip_all_fail():
    G = th.tensor([[50., 54., -130.], [-130., -129., -128.], [40., 0., 0.]])
    S = th.tensor([[True, True, False], [False, False, False], [True, False, False]])
    M = th.tensor([[True, True, True], [True, True, True], [True, True, False]])
    p, usable = event_aware_targets(G, S, M, temperature=2.)
    assert usable.tolist() == [True, False, True]
    assert float(p[0, 2]) == 0. and float(p[0, 1]) > float(p[0, 0]) > 0.
    assert abs(float(p[0, 1] / p[0, 0]) - float(np.exp(2.))) < 1e-4
    assert p[2].tolist() == [1., 0., 0.]


def test_deploy_trigger_and_choice_rules():
    assert trigger('off', 2, -500.) == (False, None)
    assert trigger('all', 0, 50.) == (True, 'all')
    assert trigger('deviation', 0, -500.) == (False, None) and trigger('deviation', 1, 50.)[0]
    assert trigger('risk', 0, -1., 0.) == (True, 'risk') and trigger('risk', 0, 1., 0.) == (False, None)
    assert trigger('risk', 2, 80., 0.) == (True, 'deviation')
    assert trigger('random', 2, -500., u=.7, rho=.5) == (False, None) and trigger('random', 0, 0., u=.2, rho=.5)[0]
    assert choose(1, [(50., 'success'), (60., 'timeout'), (55., 'success')]) == 2
    assert choose(1, [(-130., 'timeout'), (-129., 'resource_exhausted')]) == 1


def _v8(arm='e_ecpg'):
    from experiments.t1b_v2 import INTERFACES, mission, model_for
    env = INTERFACES['v8'][0](openings=[292140], mission=mission()); obs, _ = env.reset()
    return model_for(env, arm, 0, smoke=True, interface='v8'), env, obs


def test_v8_head_starts_uniform_and_deploys_earliest_from_v1_init():
    m, env, obs = _v8()
    assert obs[N_DUE_INDEX] > .5 and int((obs[49:52] > .5).sum()) == 3
    p = m.support_probabilities(th.as_tensor(obs[None]))
    assert th.allclose(p, th.full((1, 3), 1 / 3))
    assert float(m.predict(obs, deterministic=True)[0][0]) == np.float32(bin_center(0, 3))
    z = obs.copy(); z[51] = 0.   # two candidates: the third logit is masked out
    assert th.allclose(m.support_probabilities(th.as_tensor(z[None])), th.tensor([[.5, .5, 0.]]))
    assert m.ce_weight == 1. and _v8('e_sac')[0].ce_weight == 0.


def test_v8_cross_entropy_moves_the_head_toward_the_model_preference():
    m, env, obs = _v8()
    z = th.as_tensor(np.repeat(obs[None], 8, 0))
    G = th.tensor([[-129.8, -129.6, 56.0]] * 8); S = th.tensor([[False, False, True]] * 8); M = th.ones((8, 3), dtype=bool)
    before = float(m.support_probabilities(z[:1])[0, 2].detach())
    for _ in range(200):   # learning rate 1e-4 (the actor's)
        loss, sac, ce, used, *_ = m.actor_objective(z, (z, G, S, M, th.ones(8)))
        m.intent_optimizer.zero_grad(); loss.backward(); m.intent_optimizer.step()
    assert used == 8 and ce is not None
    assert float(m.support_probabilities(z[:1])[0, 2]) > before + .2


def test_v8_preference_replay_maps_branches_to_candidate_indices():
    m, env, obs = _v8()
    b = m.replay_buffer
    b.observations[0, 0] = obs; b.due[0, 0] = True; b.add_index[0, 0] = 7; b.pos = 1
    fact = {'return': 50., 'event': 'success', 'done': True, 'steps': 40, 'outcome': 'accepted', 'qp_zero_fallbacks': 0, 'discount': .96, 'observation': obs}
    alts = [dict(fact, **{'return': -129., 'event': 'resource_exhausted'}), dict(fact, **{'return': 56., 'event': 'success', 'steps': 55})]
    pays = [{'a_tilde': np.array([0.], np.float32), 'factual': fact, 'alternative': a,
             'sampling': {'selected_index': 1, 'alternative_index': j}} for j, a in zip((0, 2), alts)]
    b._release(0, [0, 7, 50.5, 40, pays], obs, 'success', False)
    assert b.preference_counts == {'released': 1, 'complete': 1}
    obs_t, G, S, M, W = b.sample_preferences(4)
    assert G[0].tolist() == [-129., 50., 56.] and S[0].tolist() == [False, True, True] and bool(M.all())
    assert abs(float(W[0]) - .9) < 1e-6   # gate weight 1 - |50 - 50.5| / 5
    p, usable = event_aware_targets(G, S, M)
    assert bool(usable.all()) and float(p[0, 0]) == 0. and float(p[0, 2]) > float(p[0, 1])


def test_v8_save_load_keeps_the_head(tmp_path):
    from train.t1_b_v8_sac import FeasibleSupportSACV8
    m, env, obs = _v8()
    with th.no_grad():
        m.policy.intent_head[-1].bias.copy_(th.tensor([0., 0., 3.]))
    m.save(tmp_path / 'v8.zip')
    r = FeasibleSupportSACV8.load(tmp_path / 'v8.zip', device='cpu', custom_objects={'learning_rate': 1e-4, 'lr_schedule': lambda _: 1e-4})
    assert float(r.predict(obs, deterministic=True)[0][0]) == np.float32(bin_center(2, 3))
    json.dumps(r.support_stats)
