import json

from experiments import r2_dev_readout as R


def _fill(root, config, seed, captures, unsafe=0):
    d = root / 'C' / f'{config}_s{seed}'; d.mkdir(parents=True)
    for k, o in enumerate(range(295300, 295492)):
        event = 'success' if k < captures else ('unsafe' if k < captures + unsafe else 'timeout')
        (d / f'{config}_s{seed}_{o}.json').write_text(json.dumps(
            {'event': event, 'return': 50.0 if event == 'success' else -120.0, 'time_s': 120.0, 'delta_v': 1.2,
             'corrections': 1, 'long_entries': 0, 'mission': {'deadline_s': 170.0}}))
    run = root / 'B' / f'TRAIN_{config}_s{seed}'; run.mkdir(parents=True)
    eps = [{'event': 'timeout', 'return_true': -120.0}] * 40 + [{'event': 'success', 'return_true': 60.0}] * 40
    (run / 'episodes.json').write_text(json.dumps(eps))


def test_summary_counts_outcomes_per_seed_and_training_quarters(tmp_path):
    for seed in (0, 1, 2):
        _fill(tmp_path, 'ECC', seed, 80 - (4 if seed == 0 else 0), unsafe=1)
    s, table = R.readout(tmp_path)
    c = s['configs']['ECC']
    assert len(table) == 576 and c['complete'] and c['total']['captures'] == 236 and c['total']['unsafe'] == 3
    assert c['per_seed']['0']['captures'] == 76 and c['per_seed']['0']['corrections_per_episode'] == 1.0
    t = c['per_seed']['0']['training']
    assert t['first_quarter']['capture_share'] == 0.0 and t['last_quarter']['capture_share'] == 1.0 and t['hover_signal'] is False
    assert 'verdict' not in s and R.feedback_verdict(s, None) is None


def test_missing_results_are_listed(tmp_path):
    for seed in (0, 1, 2):
        _fill(tmp_path, 'ECC', seed, 80)
    (tmp_path / 'C' / 'ECC_s1' / 'ECC_s1_295310.json').unlink()
    s = R.readout(tmp_path)[0]
    assert not s['configs']['ECC']['complete'] and s['configs']['ECC']['per_seed']['1']['missing_or_error'] == [295310]


OPS = range(295300, 295492)


def _fill2(root, label, seed, captures, unsafe=0, changed=0):
    d = root / 'C' / label; d.mkdir(parents=True)
    for k, o in enumerate(OPS):
        event = 'success' if k < captures else ('unsafe' if k < captures + unsafe else 'timeout')
        (d / f'{label}_{o}.json').write_text(json.dumps(
            {'event': event, 'return': 50.0 if event == 'success' else -120.0, 'time_s': 120.0, 'delta_v': 1.2,
             'corrections': 1, 'long_entries': 0, 'arbitration_changed': changed, 'mission': {'deadline_s': 170.0}}))


def _round(tmp_path, base, fb, nearest):
    for seed in range(3):
        _fill2(tmp_path, f'ECC_s{seed}', seed, base[seed])
        _fill2(tmp_path, f'ECC_FB_s{seed}', seed, fb[seed], changed=1)
        _fill2(tmp_path, f'ECC_FB_s{seed}_nearest_arbitration', seed, nearest[seed])
    s = R.readout(tmp_path)[0]
    pair = R.paired(tmp_path)
    return R.feedback_verdict(s, pair), pair


def test_primary_pass_adopts_fb(tmp_path):
    v, pair = _round(tmp_path, (160, 160, 160), (166, 163, 159), (164, 162, 159))    # +8? no: 488 vs 480
    assert not v['P']['pass']
    v, pair = _round(tmp_path / 'b', (160, 160, 160), (168, 164, 159), (160, 160, 159))
    assert v['P']['pass'] and v['formal_ecc'] == 'ECC_FB' and v['P']['seeds_not_worse'] == 2


def test_mechanism_check_is_paired_on_the_same_models(tmp_path):
    v, pair = _round(tmp_path, (170, 170, 170), (168, 168, 168), (165, 166, 167))
    # same openings captured in order: value captures 504, nearest 498, only_on 6, only_off 0
    assert pair['pairs'] == 576 and (pair['captures_on'], pair['captures_off'], pair['only_on'], pair['only_off']) == (504, 498, 6, 0)
    assert not v['P']['pass'] and v['M']['pass'] and v['formal_ecc'] is None and v['conclusion'].startswith('report')
    v, _ = _round(tmp_path / 'b', (170, 170, 170), (168, 168, 168), (167, 167, 168))
    assert not v['M']['pass'] and v['formal_ecc'] == 'ECC'


def test_incomplete_round_has_no_verdict(tmp_path):
    for seed in range(3):
        _fill2(tmp_path, f'ECC_s{seed}', seed, 160); _fill2(tmp_path, f'ECC_FB_s{seed}', seed, 160)
    assert R.paired(tmp_path) is None and R.feedback_verdict(R.readout(tmp_path)[0], None) is None
