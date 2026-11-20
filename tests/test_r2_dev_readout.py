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
    assert 'verdict' not in s


def test_missing_results_are_listed(tmp_path):
    for seed in (0, 1, 2):
        _fill(tmp_path, 'ECC', seed, 80)
    (tmp_path / 'C' / 'ECC_s1' / 'ECC_s1_295310.json').unlink()
    s = R.readout(tmp_path)[0]
    assert not s['configs']['ECC']['complete'] and s['configs']['ECC']['per_seed']['1']['missing_or_error'] == [295310]
