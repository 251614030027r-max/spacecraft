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


def test_verdict_follows_the_registered_rule(tmp_path):
    for seed in (0, 1, 2):
        _fill(tmp_path, 'ECC', seed, 80 - (4 if seed == 0 else 0))
        _fill(tmp_path, 'ECC_LEAD24', seed, 82, unsafe=0)
    s, table = R.readout(tmp_path)
    assert len(table) == 576 and s['configs']['ECC']['total']['captures'] == 236
    assert s['verdict']['potential_for_formal'] == 'opportunity'                       # 236 >= 246 - 10
    assert s['configs']['ECC']['per_seed']['0']['training']['hover_signal'] is False


def test_worse_by_more_than_ten_keeps_the_first_draft_potential(tmp_path):
    for seed in (0, 1, 2):
        _fill(tmp_path, 'ECC', seed, 78)
        _fill(tmp_path, 'ECC_LEAD24', seed, 82)
    assert R.readout(tmp_path)[0]['verdict']['potential_for_formal'] == 'lead24'      # 234 < 246 - 10


def test_missing_results_give_no_verdict(tmp_path):
    for seed in (0, 1, 2):
        _fill(tmp_path, 'ECC', seed, 80)
        _fill(tmp_path, 'ECC_LEAD24', seed, 80)
    (tmp_path / 'C' / 'ECC_s1' / 'ECC_s1_295310.json').unlink()
    s = R.readout(tmp_path)[0]
    assert s['verdict'] is None and s['configs']['ECC']['per_seed']['1']['missing_or_error'] == [295310]
