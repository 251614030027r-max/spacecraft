"""Gate 0 readout logic (``experiments/ecc2_gate0.py``) on synthetic records; no simulation."""
import json

from experiments import ecc2_gate0 as g


def _cands(events):
    grid = [(p, z) for p in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)]
    return [{'b': list(b), 'event': int(e is not None), 'kind': (e or ('none', 48.0, 0.0))[0], 't': (e or ('none', 48.0, 0.0))[1],
             'dv': (e or ('none', 48.0, 0.0))[2], 'dv_entry': 0.0, 'certified': False, 'binding': False}
            for b, e in zip(grid, events)]


def _write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(x), encoding='utf-8')


def test_report_counts(tmp_path):
    ev = [('bridge', 20.0, 0.3), ('bridge', 21.0, 0.31), None, None, None, None, None, None, ('descent', 40.0, 0.5)]
    dec = {'cert': {'a': [0.3, 0.2], 'own_event': 0, 'own_t': 48.0, 'intervened': False, 'cands': _cands(ev)}}
    for o, base_event in ((294000, 'timeout'), (294001, 'success'), (294002, 'timeout')):
        _write(tmp_path / 'base' / f'L_{o}.json', {'label': 'L', 'opening': o, 'event': base_event,
                                                   'return': 50.0 if base_event == 'success' else -100.0, 'decisions': [dec]})
        caps = {0: 1, 1: 1, 8: 0} if o != 294002 else {0: 1, 1: 0, 8: 0}
        for j in range(9):
            _write(tmp_path / 'branch' / f'L_{o}_k0_b{j}.json',
                   {'label': 'L', 'opening': o, 'k': 0, 'b': j, 'event': 'success' if caps.get(j) else 'timeout',
                    'return': 40.0 if caps.get(j) else -100.0})
        _write(tmp_path / 'branch' / f'L_{o}_k0_ref.json', {'label': 'L', 'opening': o, 'k': 0, 'b': None, 'event': base_event,
                                                            'return': 50.0 if base_event == 'success' else -100.0})
    r = g.report(tmp_path)
    assert r['determinism']['pass'] and r['determinism']['reference_branches'] == 3
    assert r['B1_headroom']['rescued'] == 2 and r['B1_headroom']['rescue_rate_of_failed'] == 1.0
    b2 = r['B2_consequence_sufficiency']
    assert b2['similar_pairs'] == 3 and abs(b2['similar_discordant'] - 1 / 3) < 1e-9     # (0,1) similar in each decision
    assert b2['other_pairs'] == 6
    assert r['B4_grid_coverage']['off_grid_captured_decisions'] == 1 and r['B4_grid_coverage']['capture_covered'] == 1.0


def test_logs_anatomy(tmp_path):
    no_opp = {'cert': {'a': [0, 0], 'own_event': 0, 'own_t': 48.0, 'intervened': False, 'cands': _cands([None] * 9)}, 'macro_end': 'stage_complete'}
    acted = {'cert': {'a': [0, 0], 'own_event': 0, 'own_t': 48.0, 'intervened': True, 'alt_t': 10.0, 'alt_kind': 'bridge',
                      'cands': _cands([('bridge', 10.0, 0.2)] + [None] * 8)}, 'macro_end': 'entry'}
    own = {'cert': {'a': [0, 0], 'own_event': 1, 'own_t': 30.0, 'own_kind': 'bridge', 'intervened': False,
                    'cands': _cands([('bridge', 28.0, 0.2)] + [None] * 8)}, 'macro_end': 'stage_complete'}
    d = tmp_path / 'C' / 'ECC_CERT_LOOP_s20_urgent'
    _write(d / 'a_1.json', {'event': 'timeout', 'decisions': [no_opp]})
    _write(d / 'a_2.json', {'event': 'timeout', 'decisions': [own, acted]})
    _write(d / 'a_3.json', {'event': 'success', 'decisions': [own]})
    r = g.logs(tmp_path)['ECC_CERT_LOOP_s20_urgent']
    assert r['failures'] == 2
    assert r['failure_anatomy']['F0_no_opportunity'] == 1 and r['failure_anatomy']['F1_rule_acted'] == 1
    assert r['grid_coverage_own_event'] == {'n': 2, 'covered': 2, 'rate': 1.0}
    assert r['stage_event_calibration']['tp'] == 1 and r['capture_after_realized_switch'] == {'entry': {'n': 1, 'captured': 0}}
