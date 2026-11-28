"""Round-3 diagnostic D1 (docs/final_method/ROUND3.md section 1): evaluation-only switches, neutral at their defaults."""
import json

import numpy as np
import pytest

from experiments import final
from experiments import r3_diag_readout as DR
from tests.test_pack import _result


def _flight(steps=1, **kw):
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[296003], shaping=False, coordination=True, **kw)
    env.reset(options={'opening': 296003})
    recs = []
    for _ in range(steps):
        _, _, term, trunc, info = env.step(np.array([0.6, 0.2]))
        recs.append(info.get('coord'))
        if term or trunc:
            break
    c = env.env.chaser_state
    return np.concatenate([c.position, c.velocity, c.omega, c.rotation.ravel()]), env.env._delta_v, recs, env


def test_defaults_are_the_round2_method():
    from env.t2_stage import STAGE_S, WINDOW_SLIPS_S
    x0, dv0, r0, e0 = _flight()
    x1, dv1, r1, e1 = _flight(coord_rule='nearest', launch_lead_s=STAGE_S)
    assert np.array_equal(x0, x1) and dv0 == dv1 and [r.get('kind') for r in r0 if r] == [r.get('kind') for r in r1 if r]
    b = e0.generator.bridge
    assert e0.entry_kwargs == {} and b.window_slips_s == WINDOW_SLIPS_S and b.replan_deviation_m == 1.0
    assert b.planner.max_force_n == pytest.approx(0.8 * e0.env.config.max_force_per_axis_n)


def test_entry_parameters_reach_execution_and_prediction():
    from env.t2_probe import ExecutorProbe
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[296003], shaping=False, coordination=True, entry_force_fraction=0.7,
                     entry_replan_deviation_m=0.5, retime_slips=[2, 4, 8, 12, 16])
    env.reset(options={'opening': 296003})
    b = env.generator.bridge
    assert b.window_slips_s == (2.0, 4.0, 8.0, 12.0, 16.0) and b.replan_deviation_m == 0.5
    assert b.planner.max_force_n == pytest.approx(0.7 * env.env.config.max_force_per_axis_n)
    assert ExecutorProbe(env).entry_planner.max_force_n == pytest.approx(b.planner.max_force_n)
    with pytest.raises(ValueError):
        T2StageEnv(coord_rule='best')


def test_launch_first_starts_the_direct_entry_before_a_correction():
    from env.t2_probe import PROBE_GRID
    from env.t2_stage import T2StageEnv

    class Probe:
        def probe_intent(self, a, *args):
            return {'event': 0, 't': 48.0, 'kind': None, 'dv': 0.1}

    kinds = {}
    for rule in ('nearest', 'launch_first'):
        env = T2StageEnv(openings=[296003], shaping=False, coordination=True, coord_rule=rule)
        env.reset(options={'opening': 296003})
        env.prober = Probe()
        rows = [{'event': 1, 'kind': 'entry', 't': 20.0, 'dv': 0.1, 'critical': i == 4} for i in range(len(PROBE_GRID))]
        env._opportunity_eval = lambda rows=rows: {'args': (), 'rows': rows}
        env._launch_direct = lambda: True
        _, rec = env._coordinate(np.array([0.9, 0.9]))
        kinds[rule] = rec['kind']
    assert kinds == {'nearest': 'correct', 'launch_first': 'launch'}


def test_diag_stage_uses_its_own_unused_block():
    st = final.stage_of('diag3')
    assert st['models_from'] == 'formal' and final.block(st['eval_openings']) == list(range(296000, 296096))
    assert not set(final.block(st['eval_openings'])) & final._forbidden_eval('diag3')


def _fake_diag(root):
    reg = final.registry(); st = final.stage_of('diag3'); ops = final.block(st['eval_openings'])
    for cfg in st['trained']:
        for seed in st['seeds']:
            labels = [f'{cfg}_s{seed}'] + [f'{cfg}_s{seed}_{sw}' for sw in st['eval_switches'] if cfg in reg['eval_switches'][sw]['applies_to']]
            for label in labels:
                d = root / 'C' / label; d.mkdir(parents=True)
                n = 4 if label.endswith('launch_first') else (3 if cfg != 'SAC_MPC' else 2)
                for k, o in enumerate(ops):
                    r = _result('timeout' if k % n == 0 else 'success', k)
                    r.update(entries_lost=k % 2, window_recoveries=[{}] * (k % 3), terminal={'in_cone': k % 4 == 0})
                    (d / f'{label}_{o}.json').write_text(json.dumps(r))


def test_diag_readout(tmp_path):
    _fake_diag(tmp_path / 'R')
    res = DR.write_all([tmp_path / 'R'], tmp_path / 'out')
    assert res['verdict'] == 'complete'
    r = json.loads((tmp_path / 'out' / 'READOUT.json').read_text())
    pairs = {(p['a'], p['b'], p['metric']): p for p in r['paired']}
    assert ('ECC+r3_launch_first', 'ECC', 'capture') in pairs and ('ECC+r3_hs_slips', 'ECC+r3_hs', 'capture') in pairs
    t = {(x['arm'], x['seed']): x for x in r['table']}
    assert t[('ECC', '30')]['timeout_in_cone'] == sum(1 for k in range(96) if k % 3 == 0 and k % 4 == 0)
    assert t[('ECC', '30')]['episodes'] == 96 and t[('ECC', '30')]['entries_lost'] == 48
    next((tmp_path / 'R' / 'C' / 'ECC_s31_r3_hs').glob('*.json')).unlink()
    assert DR.write_all([tmp_path / 'R'], tmp_path / 'out2')['verdict'] == 'incomplete'


def test_intervention_cost_lowers_only_the_shaped_reward_of_a_corrected_decision():
    from env.t2_stage import T2StageEnv
    out = []
    for cost in (0.0, 2.0):
        env = T2StageEnv(openings=[296003], shaping=True, coordination=True, intervention_cost=cost)
        env.reset(options={'opening': 296003})
        env._coordinate = lambda a: (a, {'kind': 'correct'})
        _, shaped, _, _, info = env.step(np.array([0.6, 0.2]))
        out.append((shaped, info['true_reward']))
    (s0, r0), (s1, r1) = out
    assert r0 == r1 and s0 - s1 == pytest.approx(2.0)


def test_entry_order_option():
    from env.t2_stage import T2StageEnv
    assert T2StageEnv(entry_order='aligned').entry_order == 'aligned'
    with pytest.raises(ValueError):
        T2StageEnv(entry_order='latest')


def test_dev3_configurations_and_old_manifests():
    reg = final.registry(); st = final.stage_of('dev3')
    assert st['seeds'] == [40, 41, 42] and not set(st['seeds']) & {20, 21, 22, 30, 31, 32}
    kw = {c: final.env_kwargs(reg['trained'][c]) for c in st['trained']}
    assert kw['ECC_V3A']['coord_rule'] == 'launch_first' and kw['ECC_V3B']['intervention_cost'] == 2.0
    assert kw['ECC_NC3']['correction'] is False and kw['SAC_MPC']['coord_rule'] == 'nearest'
    m = {'algo': 'sac', 'geometry': True, 'window': True, 'coordination': True, 'correction': True, 'long_entry': True,
         'potential': 'opportunity'}                      # a round-2 ECC manifest
    final.verify_manifest(m, 'ECC')
    assert not set(final.block(final.stage_of('diag3b')['eval_openings'])) & final._forbidden_eval('diag3b')


def test_d2_readout(tmp_path):
    reg = final.registry(); ops = final.block(final.stage_of('diag3b')['eval_openings'])
    root = tmp_path / 'R'
    for cfg in ('ECC', 'SAC_MPC'):
        for seed in (30, 31, 32):
            for label in [f'{cfg}_s{seed}'] + [f'{cfg}_s{seed}_{sw}' for sw in final.stage_of('diag3b')['eval_switches']]:
                d = root / 'C' / label; d.mkdir(parents=True)
                for k, o in enumerate(ops):
                    (d / f'{label}_{o}.json').write_text(json.dumps(_result('timeout' if k % 3 == 0 else 'success', k)))
    res = DR.write_all([root], tmp_path / 'out', 'diag3b')
    assert res['verdict'] == 'complete'
    pairs = {(p['a'], p['b']) for p in json.loads((tmp_path / 'out' / 'READOUT.json').read_text())['paired']}
    assert ('ECC+r3b_hs_aligned', 'ECC+r3b_hs') in pairs and ('ECC+r3b_aligned', 'SAC_MPC+r3b_aligned') in pairs


def test_window_rule_and_feedback():
    from env.t2_probe import PROBE_GRID
    from env.t2_stage import T2StageEnv

    class Probe:
        def probe_intent(self, a, *args):
            return {'event': 0, 't': 48.0, 'kind': None, 'dv': 0.1}

    def run(last_window, launch_ok):
        env = T2StageEnv(openings=[296203], shaping=False, coordination=True, coord_rule='window')
        env.reset(options={'opening': 296203})
        env.prober = Probe()
        rows = [{'event': int(i == 2), 'kind': 'entry' if i == 2 else None, 't': 20.0, 'dv': 0.1, 'critical': False}
                for i in range(len(PROBE_GRID))]
        env._opportunity_eval = lambda rows=rows: {'args': (), 'rows': rows}
        env._last_window = lambda: last_window
        env._launch_direct = lambda check_window=True: launch_ok
        a, rec = env._coordinate(np.array([0.9, 0.9]))
        return rec['kind'], a
    assert run(True, True)[0] == 'launch'
    kind, a = run(True, False)
    assert kind == 'correct' and np.array_equal(a, np.asarray(PROBE_GRID[2], float))   # not critical-flagged: state-level window
    assert run(False, True)[0] == 'none'
    from env.t2_stage import T2StageEnv as E
    out = []
    for cost in (0.0, 2.0):
        env = E(openings=[296203], shaping=True, coordination=True, coord_rule='window', intervention_cost=cost)
        env.reset(options={'opening': 296203})
        env._coordinate = lambda a: (a, {'kind': 'launch'})
        out.append(env.step(np.array([0.6, 0.2]))[1])
    assert out[0] - out[1] == pytest.approx(2.0)                                       # a launch is a replacement too


def test_window_bounded_slips():
    from env.t2_stage import BRIDGE_ANGLE_DEG, T2StageEnv, WINDOW_SLIP_CAP_S, angle_deg, port_forecast
    env = T2StageEnv(openings=[296203], shaping=False, retime='window')
    env.reset(options={'opening': 296203})
    assert env.generator.bridge.window_bounded is True
    assert T2StageEnv(openings=[296203], shaping=False).entry_kwargs == {}
    from env.t1_p_stage import relative_inertial
    q, _ = relative_inertial(env.env)
    for tau in (0.0, 10.0, 30.0, 60.0):
        sl = env.generator.bridge._window_slips(env.info_env, tau, q)
        assert all(angle_deg(port_forecast(env.info_env, tau + s), q) <= BRIDGE_ANGLE_DEG for s in sl)
        assert all(b - a == pytest.approx(2.0) for a, b in zip(sl, sl[1:])) and (not sl or sl[0] == 2.0)
        assert not sl or sl[-1] <= WINDOW_SLIP_CAP_S
        if sl and sl[-1] < WINDOW_SLIP_CAP_S:
            assert angle_deg(port_forecast(env.info_env, tau + sl[-1] + 2.0), q) > BRIDGE_ANGLE_DEG
    with pytest.raises(ValueError):
        T2StageEnv(retime='adaptive')


def test_v4_registry_and_diag3c():
    reg = final.registry(); kw = final.env_kwargs(reg['trained']['ECC_V4'])
    assert kw['coord_rule'] == 'window' and kw['retime'] == 'window' and kw['intervention_cost'] == 2.0
    assert final.env_kwargs(reg['trained']['SAC_MPC'])['retime'] == 'fixed'
    st = final.stage_of('diag3c')
    assert st['models_from'] == 'dev3' and st['eval_openings'] == final.stage_of('dev3')['eval_openings']
    assert not set(final.block(st['eval_openings'])) & final._forbidden_eval('diag3c')


def test_dev3_runs_may_come_from_the_two_registered_commits(tmp_path):
    reg = final.registry(); st = final.stage_of('dev3')
    for cfg, head in (('ECC_V3B', '89b1e784aaaa'), ('ECC_V4', '5432ecceaaaa')):
        run = tmp_path / 'B' / f'TRAIN_{cfg}_s40'; run.mkdir(parents=True)
        m = {'algo': 'sac', **final.env_kwargs(reg['trained'][cfg]), 'seed': 40, 'steps': st['steps'], 'git_head': head,
             'final_config': cfg, 'final_stage': 'dev3'}
        (run / 'manifest.json').write_text(json.dumps(m)); (run / 'final.zip').write_bytes(cfg.encode())
    res = final.resolve('dev3', tmp_path)
    assert res['science_sha'] == '5432ecceaaaa, 89b1e784aaaa' and len(res['runs']) == 2
    bad = tmp_path / 'B' / 'TRAIN_SAC_MPC_s40'; bad.mkdir()
    m = {'algo': 'sac', **final.env_kwargs(reg['trained']['SAC_MPC']), 'seed': 40, 'steps': st['steps'], 'git_head': 'deadbeef',
         'final_config': 'SAC_MPC', 'final_stage': 'dev3'}
    (bad / 'manifest.json').write_text(json.dumps(m)); (bad / 'final.zip').write_bytes(b'x')
    assert len(final.resolve('dev3', tmp_path)['runs']) == 3          # one further session is accepted
    worse = tmp_path / 'B' / 'TRAIN_ECC_NC3_s40'; worse.mkdir()
    m.update(final_config='ECC_NC3', **final.env_kwargs(reg['trained']['ECC_NC3']), git_head='cafebabe')
    (worse / 'manifest.json').write_text(json.dumps(m)); (worse / 'final.zip').write_bytes(b'y')
    with pytest.raises(SystemExit):                                   # a second unregistered one is not
        final.resolve('dev3', tmp_path)


def test_v4f_is_v4_with_the_round2_retiming():
    reg = final.registry()
    a, b = final.env_kwargs(reg['trained']['ECC_V4F']), final.env_kwargs(reg['trained']['ECC_V4'])
    assert {k for k in a if a[k] != b[k]} == {'retime'} and a['retime'] == 'fixed'
    st = final.stage_of('diag3d')
    assert st['models_from'] == 'dev3' and 'ECC_V4F' in reg['eval_switches']['dev_hs']['applies_to']
