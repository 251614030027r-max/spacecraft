import json

from experiments import r2_dev_readout as R


def _fill(root, config, seed, captures, unsafe=0):
    d = root / 'C' / f'{config}_s{seed}'; d.mkdir(parents=True)
    for k, o in enumerate(range(295300, 295396)):
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
    assert len(table) == 288 and c['complete'] and c['total']['captures'] == 236 and c['total']['unsafe'] == 3
    assert c['per_seed']['0']['captures'] == 76 and c['per_seed']['0']['corrections_per_episode'] == 1.0
    t = c['per_seed']['0']['training']
    assert t['first_quarter']['capture_share'] == 0.0 and t['last_quarter']['capture_share'] == 1.0 and t['hover_signal'] is False
    assert 'verdict' not in s and R.feedback_verdict(s) == {**R.feedback_verdict(s), 'ECC_FB': None, 'formal_ecc': None}


def test_missing_results_are_listed(tmp_path):
    for seed in (0, 1, 2):
        _fill(tmp_path, 'ECC', seed, 80)
    (tmp_path / 'C' / 'ECC_s1' / 'ECC_s1_295310.json').unlink()
    s = R.readout(tmp_path)[0]
    assert not s['configs']['ECC']['complete'] and s['configs']['ECC']['per_seed']['1']['missing_or_error'] == [295310]


def _round(tmp_path, base, fb, eq=None):
    for seed in range(3):
        _fill(tmp_path, 'ECC', seed, base[seed]); _fill(tmp_path, 'ECC_FB', seed, fb[seed])
        if eq is not None:
            _fill(tmp_path, 'ECC_EQ', seed, eq[seed])
    return R.feedback_verdict(R.readout(tmp_path)[0])


def test_feedback_verdict_adopts_fb_when_it_passes(tmp_path):
    v = _round(tmp_path, (80, 80, 80), (84, 82, 79))                       # 245 >= 240 + 5, two seeds not worse
    assert v['ECC_FB']['pass'] and v['ECC_FB']['seeds_not_worse'] == 2 and v['formal_ecc'] == 'ECC_FB'


def test_feedback_verdict_needs_the_margin_and_two_seeds(tmp_path):
    v = _round(tmp_path / 'a', (80, 80, 80), (83, 82, 79))                 # 244 < 245
    assert not v['ECC_FB']['pass'] and v['formal_ecc'] is None and v['next'] == 'run ECC_EQ'
    v = _round(tmp_path / 'b', (80, 80, 80), (90, 79, 79))                 # 248, but one seed not worse
    assert not v['ECC_FB']['pass']


def test_feedback_verdict_falls_back_to_eq_then_to_ecc(tmp_path):
    assert _round(tmp_path / 'a', (80, 80, 80), (80, 80, 80), (82, 82, 82))['formal_ecc'] == 'ECC_EQ'
    assert _round(tmp_path / 'b', (80, 80, 80), (80, 80, 80), (81, 81, 81))['formal_ecc'] == 'ECC'
