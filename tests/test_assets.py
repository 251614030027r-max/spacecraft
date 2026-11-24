"""Supplementary assets (docs/final_method/FORMAL.md section 10): evaluation-only distribution switches, the read-only
per-control-step trace, and the assets read-out."""
import json
from argparse import Namespace
from dataclasses import asdict

import numpy as np
import pytest

from experiments import final
from experiments import r2_assets_readout as AR
from tests.test_pack import _fake_formal, _result


def test_assets_stage_reuses_the_formal_models_and_openings():
    st = final.stage_of('assets'); fm = final.stage_of('formal')
    assert st['models_from'] == 'formal' and st['seeds'] == fm['seeds'] and st['steps'] == fm['steps']
    assert st['eval_openings'] == fm['eval_openings']
    assert not set(final.block(st['eval_openings'])) & final._forbidden_eval('assets')
    assert set(final.block(fm['eval_openings'])) & final._forbidden_eval('dev')      # other stages still keep it out
    reg = final.registry()
    for sw in st['eval_switches']:
        assert set(reg['eval_switches'][sw]['applies_to']) == {'ECC', 'SAC_MPC'}
    a = Namespace(openings=None, stage='assets')
    assert final.checked_openings(a, st, smoke=False) == final.block(fm['eval_openings'])


def _initial(env, opening):
    env.reset(options={'opening': opening})
    t, c = env.env.target_state, env.env.chaser_state
    return env.mission, [np.asarray(x) for x in (t.rotation, t.position, t.omega, c.rotation, c.position, c.velocity, c.omega)]


def test_switches_at_their_formal_values_change_nothing():
    from env.t2_stage import T2StageEnv
    m0, x0 = _initial(T2StageEnv(openings=[301007], shaping=False), 301007)
    m1, x1 = _initial(T2StageEnv(openings=[301007], shaping=False, deadline_s=(140.0, 200.0), tumble_rate_deg_s=(2.1, 2.7),
                                 tumble_offset_deg=(10.0, 20.0)), 301007)
    assert m0 == m1 and all(np.array_equal(a, b) for a, b in zip(x0, x1))


def test_switches_move_only_their_quantity():
    from env.t2_stage import T2StageEnv
    m0, x0 = _initial(T2StageEnv(openings=[301007], shaping=False), 301007)
    for kw, check in (({'deadline_s': (120.0, 140.0)}, lambda m, x: 120.0 <= m['deadline_s'] <= 140.0),
                      ({'tumble_rate_deg_s': (2.7, 3.3)}, lambda m, x: 2.7 <= np.degrees(np.linalg.norm(x[2])) <= 3.3 + 1e-9),
                      ({'tumble_rate_deg_s': (1.5, 2.1)}, lambda m, x: 1.5 - 1e-9 <= np.degrees(np.linalg.norm(x[2])) <= 2.1)):
        m, x = _initial(T2StageEnv(openings=[301007], shaping=False, **kw), 301007)
        assert check(m, x)
        assert np.array_equal(x[4] - x[1], x0[4] - x0[1])            # same chaser offset from the target
    m, x = _initial(T2StageEnv(openings=[301007], shaping=False, tumble_offset_deg=(20.0, 40.0)), 301007)
    assert m == m0 and not np.array_equal(x[2], x0[2])


def test_the_fine_trace_does_not_change_the_flight():
    from env.t2_stage import T2StageEnv
    runs = []
    for fine in (False, True):
        env = T2StageEnv(openings=[301001], shaping=False)
        env.reset(options={'opening': 301001})
        env.env.audit_trace = [] if fine else None
        for _ in range(2):
            env.step(np.array([1.0, 0.0]))
        c = env.env.chaser_state
        runs.append((np.concatenate([c.position, c.velocity, c.omega, c.rotation.ravel()]), env.env._delta_v, env.env.time_s,
                     env.env.audit_trace))
    (x0, dv0, t0, _), (x1, dv1, t1, trace) = runs
    assert np.array_equal(x0, x1) and dv0 == dv1 and t0 == t1
    assert len(trace) == round(t1 / 0.1) and {'force_n', 'torque_nm', 'margins', 'terminal', 'delta_v'} <= set(trace[0])


def _fake_assets(root):
    _fake_formal(root)
    for f in root.glob('[CS]/*/*.json'):
        r = json.loads(f.read_text())
        r['terminal'] = {'position_error_m': 0.1, 'attitude_error_rad': 0.05, 'speed_m_s': 0.02, 'angular_velocity_error_rad_s': 0.005,
                         'in_cone': True}
        r['min_margins'] = {'fov_margin_deg': 10.0}
        f.write_text(json.dumps(r))
    ops = final.block(final.stage_of('assets')['eval_openings'])
    for sw in final.stage_of('assets')['eval_switches']:
        for cfg in ('ECC', 'SAC_MPC'):
            for seed in final.stage_of('assets')['seeds']:
                d = root / 'C' / f'{cfg}_s{seed}_{sw}'; d.mkdir(parents=True)
                for k, o in enumerate(ops):
                    ev = 'timeout' if k % (5 if cfg == 'ECC' else 3) == 0 else 'success'
                    (d / f'{d.name}_{o}.json').write_text(json.dumps(_result(ev, k, deadline=130.0, spin=3.0)))
    (root / 'traj_fine').mkdir()
    (root / 'traj_fine' / 'traj_ECC_fine_301001.json').write_text('{}')


def test_assets_readout_writes_every_table(tmp_path):
    root = tmp_path / 'R'; _fake_assets(root)
    res = AR.write_all([root], tmp_path / 'out')
    assert res['verdict'] == 'complete' and res['fine_replays'] == 1
    out = tmp_path / 'out'
    for name in ('GEN_TABLE.csv', 'GEN_PAIRED.csv', 'GEN_EPISODES.csv', 'TERMINAL.csv', 'TERMINAL_SUMMARY.csv', 'READOUT.json',
                 'SHA256SUMS', 'traj_fine/traj_ECC_fine_301001.json'):
        assert (out / name).is_file(), name
    r = json.loads((out / 'READOUT.json').read_text())
    conds = {p['condition'] for p in r['gen_paired']}
    assert conds == {'nominal', *final.stage_of('assets')['eval_switches']}
    assert r['gen_capture']['gen_spin_high/ECC'] == pytest.approx(1 - 39 / 192)
    s = {(x['subset'], x['method']): x for x in r['terminal_summary']}
    assert s[('own captures', 'pure')]['attitude_error_deg_median'] == pytest.approx(np.degrees(0.05))


def test_assets_readout_refuses_missing_results(tmp_path):
    root = tmp_path / 'R'; _fake_assets(root)
    next(root.glob('C/ECC_s31_gen_nutation/*.json')).unlink()
    res = AR.write_all([root], tmp_path / 'out')
    assert res['verdict'] == 'incomplete' and not (tmp_path / 'out' / 'GEN_TABLE.csv').exists()
