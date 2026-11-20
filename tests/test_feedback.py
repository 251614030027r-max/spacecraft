"""Execution-grounded value arbitration and execution-equivalence experience (``docs/final_method/FEEDBACK_DEV.md``)."""
import json

import numpy as np
import pytest
import torch as th

from env.t2_stage import OBS_DIM, T2StageEnv


class _Own:
    def __init__(self, ev):
        self.ev = ev

    def probe_intent(self, x, *args):
        return {'event': self.ev, 't': 20.0 if self.ev else 48.0, 'kind': 'entry' if self.ev else None, 'dv': 0.1}


def _env(monkeypatch, **kw):
    from env.t2_probe import PROBE_GRID
    env = T2StageEnv(openings=[1000004], coordination=True, **kw); env.reset(options={'opening': 1000004})
    env.prober = _Own(0)
    monkeypatch.setattr(env, '_pass_clock', lambda: (150.0, 1e9))

    def set_rows(critical, events=()):
        rs = [{'event': 0, 't': 48.0, 'dv': 0.1, 'kind': None, 'critical': False} for _ in PROBE_GRID]
        for i in events:
            rs[i] = {'event': 1, 't': 30.0, 'dv': 0.1, 'kind': 'entry', 'critical': False}
        for i, t in critical.items():
            rs[i] = {'event': 1, 't': t, 'dv': 0.2, 'kind': 'entry', 'critical': True}
        env._opp = {'t': float(env.env.time_s), 'episode': env.episode, 'args': (), 'rows': rs, 'period': 150.0, 'latest': 100.0}
    return env, set_rows, PROBE_GRID


def test_defaults_keep_the_round2_rule():
    env = T2StageEnv(openings=[1000004])
    assert env.arbitration == 'nearest' and env.equivalence is False and env.ranker is None
    with pytest.raises(ValueError):
        T2StageEnv(openings=[1000004], arbitration='best')


def test_value_arbitration_picks_the_highest_critic_value_among_critical_intents(monkeypatch):
    env, set_rows, g = _env(monkeypatch, arbitration='value')
    i_near, i_far = g.index((0.0, 0.0)), g.index((1.0, 1.0))
    set_rows({i_near: 30.0, i_far: 10.0})
    seen = []

    def ranker(obs, cands):
        seen.append((obs.shape, cands.tolist()))
        return [5.0 if tuple(c) == (1.0, 1.0) else 1.0 for c in cands]
    env.ranker = ranker
    x, rec = env._coordinate(np.array([0.2, 0.1]))
    assert tuple(x) == (1.0, 1.0) and rec['kind'] == 'correct' and seen[0][0] == (OBS_DIM,)
    assert sorted(map(tuple, seen[0][1])) == [(0.0, 0.0), (1.0, 1.0)]
    assert env.stats['value_arbitrations'] == 1 and env.stats['arbitration_changed'] == 1
    env.ranker = lambda obs, cands: [1.0 for _ in cands]                         # ties: the earlier event, then grid order
    x, _ = env._coordinate(np.array([0.2, 0.1]))
    assert tuple(x) == (1.0, 1.0) and env.stats['arbitration_changed'] == 2
    env.ranker = None
    with pytest.raises(RuntimeError):
        env._coordinate(np.array([0.2, 0.1]))
    env.prober = _Own(1)                                                          # the intent reaches an event: kept
    x, rec = env._coordinate(np.array([0.2, 0.1]))
    assert rec['kind'] == 'keep' and tuple(x) == (0.2, 0.1)


def test_equivalent_set_under_value_arbitration_is_every_event_less_intent_and_the_executed_one(monkeypatch):
    env, set_rows, g = _env(monkeypatch, arbitration='value', equivalence=True)
    i_a, i_b, i_ev = g.index((0.0, 0.0)), g.index((1.0, 1.0)), g.index((-1.0, -1.0))
    set_rows({i_a: 30.0, i_b: 10.0}, events=[i_ev])
    env.ranker = lambda obs, cands: [3.0 if tuple(c) == (0.0, 0.0) else 1.0 for c in cands]
    x, rec = env._coordinate(np.array([0.2, 0.1]))
    eq = sorted(map(tuple, rec['equivalent']))
    expect = sorted([(0.0, 0.0)] + [b for i, b in enumerate(g) if i not in (i_a, i_b, i_ev)])
    assert tuple(x) == (0.0, 0.0) and eq == expect and env.stats['equivalent_samples'] == len(expect)


def test_equivalent_set_under_the_nearest_rule_is_the_voronoi_cell_of_the_executed_intent(monkeypatch):
    env, set_rows, g = _env(monkeypatch, equivalence=True)
    i_a, i_b = g.index((-1.0, -1.0)), g.index((1.0, 1.0))
    set_rows({i_a: 30.0, i_b: 10.0})
    x, rec = env._coordinate(np.array([-0.8, -0.6]))
    assert tuple(x) == (-1.0, -1.0)
    cell = []
    for i, b in enumerate(g):
        if i == i_b:
            continue
        da, db = np.linalg.norm(np.subtract(b, g[i_a])), np.linalg.norm(np.subtract(b, g[i_b]))
        if i == i_a or da < db or (da == db and 30.0 < 10.0):
            cell.append(tuple(b))
    assert sorted(map(tuple, rec['equivalent'])) == sorted(cell)


def test_launch_shares_the_transition_with_every_event_less_intent(monkeypatch):
    import env.t2_stage as T
    env, set_rows, g = _env(monkeypatch, equivalence=True)
    set_rows({}, events=[0])
    t = float(env.env.time_s)
    monkeypatch.setattr(env, '_pass_clock', lambda: (150.0, t + 10.0 + 150.0 - 1.0))
    monkeypatch.setattr(T, 'frame_now', lambda e, q: {'time_to_pass_s': 10.0})

    def launch():
        env.generator.bridge.plan = {'tau_s': 40.0}; return True
    monkeypatch.setattr(env, 'start_direct_gstar', launch)
    _, rec = env._coordinate(np.array([0.2, 0.1]))
    assert rec['kind'] == 'launch' and sorted(map(tuple, rec['equivalent'])) == sorted(map(tuple, g[1:]))


def test_critic_ranker_is_min_of_twin_critics():
    import train.t2_mve_sac as t
    m = t.make_model(t.SpacesEnv(OBS_DIM), 'sac', 0, smoke=True)
    rank = t.critic_ranker(m)
    obs = np.linspace(-1, 1, OBS_DIM).astype(np.float32); cands = np.array([[0, 0], [1, 1], [-1, 0.5]], np.float32)
    q = rank(obs, cands)
    with th.no_grad():
        q1, q2 = m.critic(th.as_tensor(np.repeat(obs[None], 3, 0)), th.as_tensor(cands))
    assert np.allclose(q, th.min(q1, q2).squeeze(1).numpy(), atol=1e-6) and len(q) == 3


def test_equivalent_actions_reach_the_learner_and_do_not_count_as_decisions():
    import train.t2_mve_sac as t
    info = {'event': None, 'coord': {'kind': 'correct', 'equivalent': [[0.0, 0.0], [1.0, -1.0]]}, 'true_reward': -1.0}
    assert t._slim(info)['equivalent'] == [[0.0, 0.0], [1.0, -1.0]]
    assert 'equivalent' not in t._slim({'event': None, 'coord': {'kind': 'keep'}})
    import inspect
    src = inspect.getsource(t.train)
    assert "for b in info.get('equivalent') or []" in src and "msg[0] == 'rank'" in src and 'model.eq_buffer.add' in src
    assert src.index("for b in info.get('equivalent')") < src.index('model.num_timesteps += 1')


def _filled(n_real, n_eq):
    import train.t2_mve_sac as t
    from stable_baselines3.common.logger import configure
    m = t.make_model(t.SpacesEnv(OBS_DIM), 'sac', 0, smoke=True)
    m.set_logger(configure(None, [])); m._current_progress_remaining = 1.0
    rng = np.random.default_rng(0)
    for i in range(n_real):
        o = rng.normal(size=(1, OBS_DIM)).astype(np.float32)
        m.replay_buffer.add(o, o, rng.uniform(-1, 1, size=(1, 2)).astype(np.float32), np.array([-1.0]), np.array([False]), [{}])
    if n_eq is not None:
        m.enable_equivalence()
        for i in range(n_eq):
            o = rng.normal(size=(1, OBS_DIM)).astype(np.float32)
            m.eq_buffer.add(o, o, rng.uniform(-1, 1, size=(1, 2)).astype(np.float32), np.array([5.0]), np.array([False]), [{}])
    return m


def test_equivalence_samples_enter_the_critic_batch_only_at_the_natural_ratio(monkeypatch):
    m = _filled(32, 16)
    assert m.equivalence_batch(8) == 4 and _filled(32, 200).equivalence_batch(8) == 8 and _filled(32, None).equivalence_batch(8) == 0
    sizes = {}
    orig_eq, orig_actor = m.eq_buffer.sample, m.actor.action_log_prob
    monkeypatch.setattr(m.eq_buffer, 'sample', lambda k, env=None: sizes.setdefault('eq', k) and orig_eq(k, env=env))
    calls = []
    monkeypatch.setattr(m.actor, 'action_log_prob', lambda obs: calls.append(obs.shape[0]) or orig_actor(obs))
    critic_shapes = []
    orig_critic = m.critic.forward
    monkeypatch.setattr(m.critic, 'forward', lambda obs, act: critic_shapes.append(obs.shape[0]) or orig_critic(obs, act))
    m.train(gradient_steps=1, batch_size=8)
    assert sizes['eq'] == 4
    assert calls[0] == 8                                   # the actor's batch: real samples only
    assert critic_shapes[0] == 12                          # the critic regression: 8 real + 4 equivalent
    assert m.model_stats[-1]['equivalence_batch'] == 4


def test_without_equivalence_the_update_is_the_baseline_update():
    a, b = _filled(32, None), _filled(32, None)
    np.random.seed(3); th.manual_seed(3); a.train(gradient_steps=2, batch_size=8)
    np.random.seed(3); th.manual_seed(3); b.train(gradient_steps=2, batch_size=8)
    assert all(th.equal(x, y) for x, y in zip(a.critic.parameters(), b.critic.parameters()))
    assert a.eq_buffer is None


def test_registry_feedback_configurations():
    from experiments import final
    t = final.registry()['trained']
    assert (t['ECC_FB']['arbitration'], t['ECC_FB']['equivalence']) == ('value', True)
    assert all((c['arbitration'], c['equivalence']) == ('nearest', False) for k, c in t.items() if k != 'ECC_FB')
    assert all(final.env_kwargs(t['ECC_FB'])[k] == final.env_kwargs(t['ECC'])[k] for k in ('coordination', 'correction', 'long_entry', 'potential'))
    assert final.stage_of('dev')['trained'] == ['ECC', 'ECC_FB'] and final.stage_of('dev')['eval_switches'] == ['nearest_arbitration']
