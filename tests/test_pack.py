"""One-step delivery (``experiments.final preflight`` / ``pack``) and the formal readout (``docs/final_method/FORMAL.md``)."""
import json
import shutil
from hashlib import sha256

import pytest

from experiments import final
from experiments import formal_r2_readout as FR


def _result(event, k, corrections=0, kinds=('keep',), deadline=170.0, spin=2.4):
    import numpy as np
    w = np.radians(spin)
    return {'event': event, 'return': 50.0 if event == 'success' else -120.0, 'time_s': 120.0 + k % 7, 'delta_v': 1.0 + 0.01 * (k % 5),
            'corrections': corrections, 'long_entries': 0, 'mission': {'deadline_s': deadline},
            'initial': {'omega': [0.0, 0.0, float(w)]},
            'decisions': [{'coord': {'kind': kd, 'wall_s': 0.5, **({'alt_kind': 'entry'} if kd == 'correct' else {})}} for kd in kinds]}


#: capture probability per arm in the fake round (deterministic pattern over openings)
CAPTURE_EVERY = {'ECC': 10, 'SAC_MPC': 4, 'ECC_NO_CORE': 6, 'TD3_MPC': 3, 'ECC+rigid_entry': 8, 'pure': 2, 'c_10_0': 5, 'b_10': 3}


def _event(arm, k):
    n = CAPTURE_EVERY[arm]
    return 'timeout' if k % n == 0 else 'success'


def _fake_formal(root, skip=None):
    reg = final.registry(); st = final.stage_of('formal'); ops = final.block(st['eval_openings'])
    for cfg in st['trained']:
        for seed in st['seeds']:
            run = root / 'B' / f'TRAIN_{cfg}_s{seed}'; (run / 'tb').mkdir(parents=True)
            m = {'algo': reg['trained'][cfg]['algo'], **final.env_kwargs(reg['trained'][cfg]), 'seed': seed, 'steps': st['steps'],
                 'git_head': 'abc123', 'final_config': cfg, 'final_stage': 'formal'}
            (run / 'manifest.json').write_text(json.dumps(m))
            (run / 'finished.json').write_text(json.dumps({'steps': st['steps'], 'wall_s': 3600.0 * 16}))
            eps = [{'steps': 10 * (i + 1), 'event': 'success' if i % 3 else 'timeout', 'return_true': 40.0,
                    'exec_stats': {'corrections': 1 if i < 150 else 0}} for i in range(300)]
            (run / 'episodes.json').write_text(json.dumps(eps))
            (run / 'tb' / 'events.out.tfevents.1').write_bytes(b'tb')
            (run / 'final.zip').write_bytes(f'{cfg}{seed}'.encode())
            arms = [(cfg, f'{cfg}_s{seed}')] + ([('ECC+rigid_entry', f'{cfg}_s{seed}_rigid_entry')] if cfg == 'ECC' else [])
            for arm, label in arms:
                d = root / 'C' / label; d.mkdir(parents=True)
                for k, o in enumerate(ops):
                    if skip == (label, o):
                        continue
                    ev = _event(arm, k)
                    corr = 1 if (arm == 'ECC' and k % 3 == 1) else 0
                    r = _result(ev, k, corrections=corr, kinds=('correct', 'keep') if corr else ('keep',),
                                deadline=140.0 + (k % 60), spin=2.1 + 0.6 * (k % 10) / 10)
                    (d / f'{label}_{o}.json').write_text(json.dumps(r))
    for pol in st['fixed']:
        d = root / 'S' / pol; d.mkdir(parents=True)
        for k, o in enumerate(ops):
            r = _result(_event(pol, k), k, kinds=())
            (d / f'{pol}_{o}.json').write_text(json.dumps(r))


def test_formal_readout_gives_every_figure_source(tmp_path):
    root, out = tmp_path / 'R', tmp_path / 'OUT'
    _fake_formal(root)
    r = FR.write_all(root, out)
    assert not r['missing_or_error']
    for name in ('READOUT.json', 'MAIN_TABLE.csv', 'PAIRED.csv', 'EPISODES_WIDE.csv', 'TRAINING_CURVES.csv', 'COORDINATION_USAGE.csv',
                 'STRATIFIED.csv', 'COMPUTE.csv', 'SAFETY.json', 'CASES.json'):
        assert (out / name).is_file(), name
    o = r['outcomes']
    assert o['ECC']['pooled']['n'] == 576 and o['pure']['pooled']['n'] == 192 and set(o['ECC']['per_seed']) == {'30', '31', '32'}
    cap = r['paired']['ECC vs SAC_MPC']['capture']
    assert cap['n'] == 576 and cap['estimate'] > 0 and cap['ci95'][0] <= cap['estimate'] <= cap['ci95'][1]
    assert r['paired']['ECC vs pure']['capture']['n'] == 576                     # the fixed arm pairs with every seed
    assert r['paired']['c_10_0 vs b_10']['capture']['n'] == 192
    assert set(r['paired']) >= {'ECC vs SAC_MPC', 'ECC vs TD3_MPC', 'ECC vs ECC_NO_CORE', 'ECC_NO_CORE vs SAC_MPC', 'ECC vs ECC+rigid_entry'}
    case = r['cases']
    assert case['seed'] == 30 and case['case_A_typical'] is not None and case['case_B_correction'] is not None
    curves = (out / 'TRAINING_CURVES.csv').read_text()
    assert 'corrections_per_episode' in curves and 'decisions' in curves
    usage = (out / 'COORDINATION_USAGE.csv').read_text()
    assert 'correct_share' in usage and 'ECC_NO_CORE' in usage


def test_case_rule_is_the_first_qualifying_opening(tmp_path):
    rows = {('ECC', '30', 5): {'event': 'success', 'corrections': 0}, ('SAC_MPC', '30', 5): {'event': 'timeout', 'corrections': 0},
            ('ECC', '30', 6): {'event': 'success', 'corrections': 0}, ('SAC_MPC', '30', 6): {'event': 'success', 'corrections': 0},
            ('ECC', '30', 7): {'event': 'success', 'corrections': 2}, ('SAC_MPC', '30', 7): {'event': 'timeout', 'corrections': 0}}
    assert FR.cases(rows, '30') == {'seed': 30, 'case_A_typical': 6, 'case_B_correction': 7}


def test_pack_formal_builds_the_whole_delivery(tmp_path):
    root, out = tmp_path / 'R', tmp_path / 'OUT'
    _fake_formal(root)
    (root / 'PREFLIGHT.json').write_text(json.dumps({'pytest': {'exit_code': 0, 'summary': '180 passed'}, 'clean_tree': True}))
    res = final.pack('formal', root, out)
    assert res['raw_files_hashed'] == (12 + 3 + 3) * 192 + 12
    for name in ('CHECK.json', 'REGISTRY_RESOLVED.json', 'READOUT.json', 'PAIRED.csv', 'CASES.json', 'CASE_COMMANDS.txt',
                 'RAW_SHA256SUMS', 'RESULT.md', 'SHA256SUMS', 'PREFLIGHT.json', 'runs/TRAIN_TD3_MPC_s32/tb/events.out.tfevents.1'):
        assert (out / name).is_file(), name
    md = (out / 'RESULT.md').read_text()
    assert '主比较 ECC 对 SAC_MPC' in md and '| ECC_NO_CORE | 31 |' in md and '异常与补跑' in md
    cmd = (out / 'CASE_COMMANDS.txt').read_text()
    assert 'traj --entry ECC=' in cmd and 'TRAIN_SAC_MPC_s30' in cmd
    sums = dict(reversed(line.split('  ', 1)) for line in (out / 'SHA256SUMS').read_text().splitlines())
    assert sums['RESULT.md'] == sha256((out / 'RESULT.md').read_bytes()).hexdigest() and 'SHA256SUMS' not in sums
    (out / 'traj').mkdir(); (out / 'traj' / 'traj_ECC_301000.json').write_text('{}')        # replayed cases, then pack again
    final.pack('formal', root, out)
    assert 'traj/traj_ECC_301000.json' in (out / 'SHA256SUMS').read_text()


def test_pack_refuses_an_incomplete_round(tmp_path):
    root, out = tmp_path / 'R', tmp_path / 'OUT'
    _fake_formal(root, skip=('ECC_s30_rigid_entry', 301005))
    with pytest.raises(SystemExit):
        final.pack('formal', root, out)
    assert not json.loads((out / 'CHECK.json').read_text())['complete'] and not (out / 'RESULT.md').exists()


def _move_run(src_root, dst_root, run, labels):
    shutil.move(str(src_root / 'B' / run), str(dst_root / 'B' / run))
    model = dst_root / 'B' / run / 'final.zip'
    (dst_root / 'B' / run / 'final.zip.sha256').write_text(sha256(model.read_bytes()).hexdigest() + '  final.zip\n')
    model.unlink()
    for label in labels:
        shutil.move(str(src_root / 'C' / label), str(dst_root / 'C' / label))


def test_pack_merges_local_and_cloud_roots(tmp_path):
    local, cloud, out = tmp_path / 'local', tmp_path / 'cloud', tmp_path / 'OUT'
    _fake_formal(local)
    (cloud / 'B').mkdir(parents=True); (cloud / 'C').mkdir()
    for cfg, seed in (('SAC_MPC', 31), ('SAC_MPC', 32), ('TD3_MPC', 31), ('TD3_MPC', 32)):   # the cloud's share (FORMAL.md 6)
        _move_run(local, cloud, f'TRAIN_{cfg}_s{seed}', [f'{cfg}_s{seed}'])
    with pytest.raises(SystemExit):
        final.pack('formal', [local], out)                    # the local root alone is incomplete
    final.pack('formal', [local, cloud], out)
    raw = (out / 'RAW_SHA256SUMS').read_text()
    assert 'cloud/B/TRAIN_SAC_MPC_s31/final.zip  (recorded)' in raw and 'local/C/ECC_s30/' in raw
    runs = json.loads((out / 'REGISTRY_RESOLVED.json').read_text())['runs']
    assert runs['TRAIN_TD3_MPC_s32']['model_hash_source'] == 'final.zip.sha256' and runs['TRAIN_ECC_s30']['location'].startswith('local/')
    assert '| cloud |' in (out / 'RESULT.md').read_text()


def test_the_same_result_in_two_roots_is_refused(tmp_path):
    local, cloud = tmp_path / 'local', tmp_path / 'cloud'
    _fake_formal(local)
    (cloud / 'C').mkdir(parents=True)
    shutil.copytree(local / 'C' / 'ECC_s30', cloud / 'C' / 'ECC_s30')
    rep = final.check_stage([local, cloud], 'formal')
    assert not rep['complete'] and any('duplicate' in e for e in rep['errors'])


def test_pack_dev_stage_still_works(tmp_path):
    reg = final.registry(); st = final.stage_of('dev'); root = tmp_path / 'R'
    for seed in st['seeds']:
        run = root / 'B' / f'TRAIN_ECC_s{seed}'; (run / 'tb').mkdir(parents=True)
        (run / 'manifest.json').write_text(json.dumps({'algo': 'sac', **final.env_kwargs(reg['trained']['ECC']), 'seed': seed,
                                                       'steps': st['steps'], 'git_head': 'abc', 'final_config': 'ECC', 'final_stage': 'dev'}))
        (run / 'finished.json').write_text(json.dumps({'steps': st['steps'], 'wall_s': 10.0}))
        (run / 'episodes.json').write_text(json.dumps([{'event': 'success', 'return_true': 1.0}] * 8))
        (run / 'tb' / 'events.out.tfevents.1').write_bytes(b'tb'); (run / 'final.zip').write_bytes(b'm')
        d = root / 'C' / f'ECC_s{seed}'; d.mkdir(parents=True)
        for k, o in enumerate(final.block(st['eval_openings'])):
            (d / f'ECC_s{seed}_{o}.json').write_text(json.dumps(_result('success', k)))
    final.pack('dev', root, tmp_path / 'OUT')
    assert (tmp_path / 'OUT' / 'SUMMARY.json').is_file() and '| ECC | 0 |' in (tmp_path / 'OUT' / 'RESULT.md').read_text()


def test_preflight_records_the_state(tmp_path):
    rec = final.preflight(tmp_path, run_tests=False)
    assert {'science_sha', 'clean_tree', 'logical_cores', 'versions', 'started'} <= rec.keys()
    assert json.loads((tmp_path / 'PREFLIGHT.json').read_text())['science_sha'] == rec['science_sha']
