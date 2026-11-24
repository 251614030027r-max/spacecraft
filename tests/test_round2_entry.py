"""The round-2 entry and registry (``docs/final_method/ROUND2.md``, ``experiments/final.py``)."""
from argparse import Namespace
from hashlib import sha256
import json

import pytest

from experiments import final


def test_registered_configurations_match_the_spec():
    reg = final.registry()
    t = reg['trained']
    assert t['ECC'] == {**t['ECC'], 'algo': 'sac', 'coordination': True, 'correction': True, 'long_entry': True, 'potential': 'opportunity'}
    assert t['ECC_NO_CORE']['correction'] is False and t['ECC_NO_CORE']['long_entry'] is True
    assert not t['SAC_MPC']['coordination'] and t['TD3_MPC']['algo'] == 'td3' and not t['TD3_MPC']['coordination']
    assert set(t) == {'ECC', 'ECC_NO_CORE', 'SAC_MPC', 'TD3_MPC', 'ECC_FINE'} and {c['potential'] for c in t.values()} == {'opportunity'}
    assert t['ECC_FINE']['extended_lattice'] is True and all(final.env_kwargs(t['ECC_FINE'])[k] == final.env_kwargs(t['ECC'])[k]
                                                               for k in ('coordination', 'correction', 'long_entry', 'potential', 'window'))
    assert not any(c['extended_lattice'] for k, c in t.items() if k != 'ECC_FINE')
    assert 'no_long_entry' not in reg['eval_switches']                       # the long-range entry belongs to the coordination
    assert reg['eval_switches']['rigid_entry']['env'] == {'window': False} and set(reg['fixed']) == {'pure', 'c_10_0', 'b_10'}
    dev = final.stage_of('dev')
    assert dev['steps'] == 3000 and dev['seeds'] == [0, 1, 2] and dev['trained'] == ['ECC']
    assert final.block(dev['eval_openings']) == list(range(295300, 295492))
    fo = final.stage_of('formal')
    assert fo['steps'] == 6000 and fo['seeds'] == [30, 31, 32] and fo['trained'] == ['ECC', 'SAC_MPC', 'ECC_NO_CORE', 'TD3_MPC']
    assert final.block(fo['eval_openings']) == list(range(301000, 301192)) and fo['fixed'] == ['pure', 'c_10_0', 'b_10']
    assert fo['eval_switches'] == ['rigid_entry'] and reg['eval_switches']['rigid_entry']['applies_to'] == ['ECC', 'ECC_FINE']
    assert fo['variants'] == {'ECC': ['ECC', 'ECC_FINE']} and dev['eval_switches'] == ['extended_lattice']


def test_stage_blocks_and_seeds_avoid_the_first_draft():
    reg = final.registry(); p = reg['protocol']
    old = set(final.block(p['reserved']['never_reuse']['openings']))
    assert old == set(range(300000, 300296)) and set(p['reserved']['never_reuse']['seeds']) == {20, 21, 22}
    # stages that train their own models use disjoint blocks; an evaluation-only stage that takes its models from
    # another stage (``models_from``, FORMAL.md section 10) evaluates on exactly that stage's block
    own = {k: st for k, st in p['stages'].items() if 'models_from' not in st}
    blocks = [set(final.block(st['eval_openings'])) for st in own.values()]
    for i, b in enumerate(blocks):
        assert not b & old and all(not b & c for c in blocks[i + 1:])
    for st in p['stages'].values():
        if 'models_from' in st:
            assert st['eval_openings'] == own[st['models_from']]['eval_openings']
    for st in p['stages'].values():
        assert not set(st['seeds']) & {20, 21, 22}


def _args(**kw):
    return Namespace(**{'openings': None, 'stage': 'dev', **kw})


def test_evaluation_openings_and_smoke_boundaries():
    st = final.stage_of('dev')
    assert final.checked_openings(_args(), st, smoke=False) == list(range(295300, 295492))
    assert final.checked_openings(_args(openings='295300-295301'), st, smoke=False) == [295300, 295301]
    for bad in ('300000-300001', '295299-295300', '295491-295492', '294300-294301'):
        with pytest.raises(SystemExit):
            final.checked_openings(_args(openings=bad), st, smoke=False)
    assert final.checked_openings(_args(openings='294300-294301'), st, smoke=True) == [294300, 294301]
    with pytest.raises(SystemExit):
        final.checked_openings(_args(), st, smoke=True)


def test_switches_and_smoke_seeds(tmp_path):
    st = final.stage_of('dev')
    with pytest.raises(SystemExit):
        final.switch_env(st, 'ECC', 'rigid_entry')              # not registered for the development stage
    assert final.switch_env(st, 'ECC', 'extended_lattice') == {'extended_lattice': True}
    assert final.switch_env(final.stage_of('formal'), 'ECC', 'rigid_entry') == {'window': False}
    assert final.switch_env({**st, 'eval_switches': ['rigid_entry']}, 'ECC', 'rigid_entry') == {'window': False}
    with pytest.raises(SystemExit):
        final.switch_env({**st, 'eval_switches': ['rigid_entry']}, 'SAC_MPC', 'rigid_entry')
    for seed in (0, 20):
        with pytest.raises(SystemExit):
            final.cmd_train(Namespace(stage='dev', config='ECC', seed=seed, output=str(tmp_path / f's{seed}'), workers=1, smoke_steps=4))


def test_manifest_must_match_the_registry():
    m = {'algo': 'sac', **final.env_kwargs(final.registry()['trained']['ECC'])}
    final.verify_manifest(m, 'ECC')
    with pytest.raises(SystemExit):
        final.verify_manifest({**m, 'potential': 'lead24'}, 'ECC')
    with pytest.raises(SystemExit):
        final.verify_manifest({**m, 'correction': False}, 'ECC')


def test_resume_verifies_existing_bytes_before_skip(tmp_path):
    out = tmp_path / 'C' / 'ECC_s0'; out.mkdir(parents=True)
    f = out / 'ECC_s0_295300.json'; f.write_text('{"event": "success"}', encoding='utf-8')
    sums = tmp_path / 'SHA256SUMS'
    sums.write_text(f'{sha256(f.read_bytes()).hexdigest()}  C/ECC_s0/ECC_s0_295300.json\n', encoding='utf-8')
    final.verify_existing(out, sums, [295300, 295301], 'ECC_s0')
    f.write_text('{"event": "timeout"}', encoding='utf-8')
    with pytest.raises(SystemExit):
        final.verify_existing(out, sums, [295300], 'ECC_s0')


def test_completeness_report_requires_runs_results_and_tensorboard(tmp_path):
    rep = final.check_stage(tmp_path, 'dev')
    assert not rep['complete'] and len(rep['missing_runs']) == 3 and len(rep['evaluations']) == 6
    for cfg in ('ECC',):
        for seed in (0, 1, 2):
            run = tmp_path / 'B' / f'TRAIN_{cfg}_s{seed}'; (run / 'tb').mkdir(parents=True)
            (run / 'manifest.json').write_text('{}'); (run / 'finished.json').write_text(json.dumps({'steps': 3000}))
            (run / 'tb' / 'events.out.tfevents.1').write_text('x')
            for label in [f'{cfg}_s{seed}', f'{cfg}_s{seed}_extended_lattice']:
                c = tmp_path / 'C' / label; c.mkdir(parents=True)
                for o in range(295300, 295492):
                    (c / f'{label}_{o}.json').write_text('{}')
    assert final.check_stage(tmp_path, 'dev')['complete']


def test_rwg_fixed_intent_pair_differs_only_in_rotation_guidance():
    from experiments.t2 import fixed_intent
    import numpy as np
    a, geo = fixed_intent('c_10_0'); b, nogeo = fixed_intent('b_10')
    assert np.array_equal(a, b) and np.array_equal(a, [1.0, 0.0]) and geo is True and nogeo is False
    with pytest.raises(ValueError):
        fixed_intent('rally')
