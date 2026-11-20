"""One-step delivery (``experiments.final preflight`` / ``pack``)."""
import json
from hashlib import sha256

import pytest

from experiments import final


def _fake_round(root, complete=True):
    reg = final.registry(); st = final.stage_of('dev')
    for cfg in st['trained']:
        for seed in st['seeds']:
            run = root / 'B' / f'TRAIN_{cfg}_s{seed}'; (run / 'tb').mkdir(parents=True)
            m = {'algo': reg['trained'][cfg]['algo'], **final.env_kwargs(reg['trained'][cfg]), 'seed': seed, 'steps': st['steps'],
                 'git_head': 'abc123', 'final_config': cfg, 'final_stage': 'dev'}
            (run / 'manifest.json').write_text(json.dumps(m))
            (run / 'finished.json').write_text(json.dumps({'steps': st['steps'], 'wall_s': 7200.0}))
            (run / 'episodes.json').write_text(json.dumps([{'event': 'success', 'return_true': 50.0}] * 8))
            (run / 'tb' / 'events.out.tfevents.1').write_bytes(b'tb')
            (run / 'final.zip').write_bytes(f'{cfg}{seed}'.encode())
            labels = [f'{cfg}_s{seed}'] + ([f'{cfg}_s{seed}_nearest_arbitration'] if cfg == 'ECC_FB' else [])
            for label in labels:
                d = root / 'C' / label; d.mkdir(parents=True)
                for k, o in enumerate(final.block(st['eval_openings'])):
                    if not complete and k == 0 and label == 'ECC_s0':
                        continue
                    (d / f'{label}_{o}.json').write_text(json.dumps(
                        {'event': 'success' if k % 5 else 'timeout', 'return': 1.0, 'time_s': 100.0, 'delta_v': 1.0,
                         'corrections': 1, 'long_entries': 0, 'arbitration_changed': 0, 'mission': {'deadline_s': 170.0}}))


def test_pack_builds_the_whole_delivery(tmp_path):
    root, out = tmp_path / 'R', tmp_path / 'OUT'
    _fake_round(root)
    (root / 'PREFLIGHT.json').write_text(json.dumps({'pytest': {'exit_code': 0, 'summary': '171 passed'}, 'clean_tree': True}))
    res = final.pack('dev', root, out)
    assert res['raw_files_hashed'] == 9 * 192 + 6 and res['verdict'] is not None
    for name in ('CHECK.json', 'REGISTRY_RESOLVED.json', 'SUMMARY.json', 'EPISODES.csv', 'PREFLIGHT.json', 'RAW_SHA256SUMS',
                 'RESULT.md', 'SHA256SUMS', 'runs/TRAIN_ECC_FB_s2/tb/events.out.tfevents.1', 'runs/TRAIN_ECC_s0/episodes.json'):
        assert (out / name).is_file(), name
    sums = dict(reversed(line.split('  ', 1)) for line in (out / 'SHA256SUMS').read_text().splitlines())
    assert sums['RESULT.md'] == sha256((out / 'RESULT.md').read_bytes()).hexdigest() and 'SHA256SUMS' not in sums
    md = (out / 'RESULT.md').read_text()
    assert '| ECC_FB | 1 |' in md and '预登记判定' in md and '异常与补跑' in md
    assert json.loads((out / 'REGISTRY_RESOLVED.json').read_text())['science_sha'] == 'abc123'


def test_pack_refuses_an_incomplete_round(tmp_path):
    root, out = tmp_path / 'R', tmp_path / 'OUT'
    _fake_round(root, complete=False)
    with pytest.raises(SystemExit):
        final.pack('dev', root, out)
    assert not json.loads((out / 'CHECK.json').read_text())['complete'] and not (out / 'RESULT.md').exists()


def test_preflight_records_the_state(tmp_path):
    rec = final.preflight(tmp_path, run_tests=False)
    assert {'science_sha', 'clean_tree', 'logical_cores', 'versions', 'started'} <= rec.keys()
    assert json.loads((tmp_path / 'PREFLIGHT.json').read_text())['science_sha'] == rec['science_sha']


def _move_run(src_root, dst_root, run, labels, keep_model=False):
    import shutil
    shutil.move(str(src_root / 'B' / run), str(dst_root / 'B' / run))
    model = dst_root / 'B' / run / 'final.zip'
    if not keep_model:
        (dst_root / 'B' / run / 'final.zip.sha256').write_text(sha256(model.read_bytes()).hexdigest() + '  final.zip\n')
        model.unlink()
    for label in labels:
        shutil.move(str(src_root / 'C' / label), str(dst_root / 'C' / label))


def test_pack_merges_local_and_cloud_roots(tmp_path):
    local, cloud, out = tmp_path / 'local', tmp_path / 'cloud', tmp_path / 'OUT'
    _fake_round(local)
    (cloud / 'B').mkdir(parents=True); (cloud / 'C').mkdir()
    for seed in (1, 2):                                       # the cloud ran ECC_FB seeds 1 and 2
        _move_run(local, cloud, f'TRAIN_ECC_FB_s{seed}', [f'ECC_FB_s{seed}', f'ECC_FB_s{seed}_nearest_arbitration'])
    with pytest.raises(SystemExit):
        final.pack('dev', [local], out)                       # the local root alone is incomplete
    res = final.pack('dev', [local, cloud], out)
    assert res['verdict'] is not None and res['raw_files_hashed'] == 9 * 192 + 6
    raw = (out / 'RAW_SHA256SUMS').read_text()
    assert 'cloud/B/TRAIN_ECC_FB_s1/final.zip  (recorded)' in raw and 'local/C/ECC_s0/' in raw
    runs = json.loads((out / 'REGISTRY_RESOLVED.json').read_text())['runs']
    assert runs['TRAIN_ECC_FB_s2']['model_hash_source'] == 'final.zip.sha256' and runs['TRAIN_ECC_s0']['location'].startswith('local/')
    assert (out / 'runs' / 'TRAIN_ECC_FB_s1' / 'episodes.json').is_file() and '| 机器 |' in (out / 'RESULT.md').read_text()


def test_the_same_result_in_two_roots_is_refused(tmp_path):
    import shutil
    local, cloud = tmp_path / 'local', tmp_path / 'cloud'
    _fake_round(local)
    (cloud / 'C').mkdir(parents=True)
    shutil.copytree(local / 'C' / 'ECC_s0', cloud / 'C' / 'ECC_s0')
    rep = final.check_stage([local, cloud], 'dev')
    assert not rep['complete'] and any('duplicate' in e for e in rep['errors'])
