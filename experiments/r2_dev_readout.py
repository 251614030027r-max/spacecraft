"""Readout of a development stage (``docs/final_method/ROUND2.md`` section 5; the feedback-mechanism development).

    python -B -m experiments.r2_dev_readout --root ROOT --output OUT_DIR [--stage dev]

Reads ROOT/C/<config>_s<seed>/<label>_<opening>.json (evaluation of the final models) and ROOT/B/TRAIN_<config>_s<seed>/
episodes.json (training record) for every configuration of the stage, and writes
* ``EPISODES.csv`` -- one row per evaluated episode (config, seed, opening, event, return, time, delta-v, corrections,
  long-range entries, deadline);
* ``SUMMARY.json`` -- per configuration and seed: captures, timeouts, unsafe, other outcomes, mean true return, and the
  training record in its first and last quarter (capture share, timeout share, mean true return; the hover signal of
  DEAD_ENDS D03 is a rising return with a rising timeout share). Missing or error files are listed and the
  configuration is marked incomplete. For the feedback-mechanism round, ``paired_mechanism`` compares the ECC_FB models
  with and without the ``nearest_arbitration`` switch, and ``verdict`` applies the pre-registered rules P and M of
  ``docs/final_method/FEEDBACK_DEV.md`` section 4.
"""
import argparse
import csv
import json
from pathlib import Path

from experiments.final import block, stage_of


def episodes(root: Path, config: str, seed: int, openings: list[int]) -> tuple[list[dict], list[int]]:
    rows, missing = [], []
    for o in openings:
        f = root / 'C' / f'{config}_s{seed}' / f'{config}_s{seed}_{o}.json'
        if not f.is_file():
            missing.append(o); continue
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('event') == 'error':
            missing.append(o); continue
        rows.append({'config': config, 'seed': seed, 'opening': o, 'event': r['event'], 'return': float(r['return']),
                     'time_s': float(r['time_s']), 'delta_v': float(r['delta_v']), 'corrections': int(r.get('corrections', 0)),
                     'long_entries': int(r.get('long_entries', 0)), 'arbitration_changed': int(r.get('arbitration_changed', 0)),
                     'deadline_s': float(r['mission']['deadline_s'])})
    return rows, missing


def outcome(rows: list[dict]) -> dict:
    n = len(rows)
    count = lambda e: sum(r['event'] == e for r in rows)
    return {'episodes': n, 'captures': count('success'), 'timeouts': count('timeout'), 'unsafe': count('unsafe'),
            'resource_exhausted': count('resource_exhausted'),
            'other': n - count('success') - count('timeout') - count('unsafe') - count('resource_exhausted'),
            'mean_true_return': sum(r['return'] for r in rows) / n if n else None,
            'corrections_per_episode': sum(r['corrections'] for r in rows) / n if n else None,
            'long_entries_per_episode': sum(r['long_entries'] for r in rows) / n if n else None}


def training(run: Path) -> dict | None:
    f = run / 'episodes.json'
    if not f.is_file():
        return None
    eps = json.loads(f.read_text(encoding='utf-8'))
    q = max(len(eps) // 4, 1)

    def part(xs):
        n = len(xs)
        return {'episodes': n, 'capture_share': sum(e['event'] == 'success' for e in xs) / n if n else None,
                'timeout_share': sum(e['event'] == 'timeout' for e in xs) / n if n else None,
                'mean_true_return': sum(e['return_true'] for e in xs) / n if n else None}
    first, last = part(eps[:q]), part(eps[-q:])
    hover = bool(first['episodes'] and last['mean_true_return'] > first['mean_true_return']
                 and last['timeout_share'] > first['timeout_share'])
    return {'episodes': len(eps), 'first_quarter': first, 'last_quarter': last, 'hover_signal': hover}


def readout(root: Path, stage: str = 'dev') -> tuple[dict, list[dict]]:
    st = stage_of(stage)
    ops = block(st['eval_openings'])
    summary, table = {'stage': stage, 'openings': st['eval_openings'], 'configs': {}}, []
    for config in st['trained']:
        per_seed, rows_all, incomplete = {}, [], False
        for seed in st['seeds']:
            rows, missing = episodes(root, config, seed, ops)
            per_seed[str(seed)] = {**outcome(rows), 'missing_or_error': missing,
                                   'training': training(root / 'B' / f'TRAIN_{config}_s{seed}')}
            incomplete |= bool(missing)
            rows_all += rows
        summary['configs'][config] = {'per_seed': per_seed, 'total': outcome(rows_all), 'complete': not incomplete,
                                      'arbitration_changed_per_episode': (sum(r['arbitration_changed'] for r in rows_all) / len(rows_all)
                                                                          if rows_all else None)}
        table += rows_all
    return summary, table


#: FEEDBACK_DEV.md section 4 (revised 2026-11-20): primary P and paired mechanism check M
P_CAPTURE_MARGIN, P_UNSAFE_TOLERANCE, P_SEEDS_NOT_WORSE = 10, 1, 2
M_CAPTURE_MARGIN, M_FLIP_RATIO, M_UNSAFE_TOLERANCE = 3, 2, 1


def paired(root: Path, stage: str = 'dev', config: str = 'ECC_FB', switch: str = 'nearest_arbitration') -> dict | None:
    """The same models evaluated with and without the switch: captures, unsafe and flips per opening and seed."""
    st = stage_of(stage)
    ops = block(st['eval_openings'])
    on_rows, off_rows, missing = [], [], 0
    for seed in st['seeds']:
        a, ma = episodes(root, config, seed, ops)
        b, mb = _switch_rows(root, config, seed, switch, ops)
        missing += len(ma) + len(mb)
        on_rows += a; off_rows += b
    if missing or not on_rows:
        return None
    key = lambda r: (r['seed'], r['opening'])
    off = {key(r): r for r in off_rows}
    both = [(r, off[key(r)]) for r in on_rows if key(r) in off]
    only_on = sum(r['event'] == 'success' and o['event'] != 'success' for r, o in both)
    only_off = sum(o['event'] == 'success' and r['event'] != 'success' for r, o in both)
    return {'pairs': len(both), 'captures_on': sum(r['event'] == 'success' for r, _ in both),
            'captures_off': sum(o['event'] == 'success' for _, o in both), 'only_on': only_on, 'only_off': only_off,
            'unsafe_on': sum(r['event'] == 'unsafe' for r, _ in both), 'unsafe_off': sum(o['event'] == 'unsafe' for _, o in both),
            'arbitration_changed_on': sum(r['arbitration_changed'] for r, _ in both)}


def _switch_rows(root: Path, config: str, seed: int, switch: str, ops: list[int]):
    rows, missing = [], []
    for o in ops:
        f = root / 'C' / f'{config}_s{seed}_{switch}' / f'{config}_s{seed}_{switch}_{o}.json'
        if not f.is_file():
            missing.append(o); continue
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('event') == 'error':
            missing.append(o); continue
        rows.append({'seed': seed, 'opening': o, 'event': r['event'], 'arbitration_changed': int(r.get('arbitration_changed', 0))})
    return rows, missing


def feedback_verdict(summary: dict, pair: dict | None) -> dict | None:
    """The pre-registered decision of the feedback-mechanism round, or None while results are incomplete."""
    cfg = summary['configs']
    base, fb = cfg.get('ECC'), cfg.get('ECC_FB')
    if base is None or fb is None or not base['complete'] or not fb['complete'] or pair is None:
        return None
    seeds_ok = sum(fb['per_seed'][s]['captures'] >= base['per_seed'][s]['captures'] for s in base['per_seed'])
    p = (fb['total']['captures'] >= base['total']['captures'] + P_CAPTURE_MARGIN
         and fb['total']['unsafe'] <= base['total']['unsafe'] + P_UNSAFE_TOLERANCE and seeds_ok >= P_SEEDS_NOT_WORSE)
    m = (pair['captures_on'] >= pair['captures_off'] + M_CAPTURE_MARGIN and pair['only_on'] >= M_FLIP_RATIO * pair['only_off']
         and pair['unsafe_on'] <= pair['unsafe_off'] + M_UNSAFE_TOLERANCE)
    return {'P': {'captures': fb['total']['captures'], 'baseline_captures': base['total']['captures'], 'unsafe': fb['total']['unsafe'],
                  'baseline_unsafe': base['total']['unsafe'], 'seeds_not_worse': seeds_ok, 'pass': p},
            'M': {**pair, 'pass': m},
            'conclusion': 'adopt ECC_FB' if p else ('report: mechanism effective, training bar not met' if m else 'keep ECC'),
            'formal_ecc': 'ECC_FB' if p else (None if m else 'ECC')}


EPISODE_FIELDS = ['config', 'seed', 'opening', 'event', 'return', 'time_s', 'delta_v', 'corrections', 'long_entries',
                  'arbitration_changed', 'deadline_s']


def write_episodes_csv(path: Path, table: list[dict]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader(); w.writerows(table)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--output', required=True); p.add_argument('--stage', default='dev')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    summary, table = readout(Path(a.root), a.stage)
    if a.stage == 'dev':
        summary['paired_mechanism'] = paired(Path(a.root), a.stage)
        summary['verdict'] = feedback_verdict(summary, summary['paired_mechanism'])
    else:
        summary['verdict'] = None
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    write_episodes_csv(out / 'EPISODES.csv', table)
    print(json.dumps({'totals': {c: v['total'] for c, v in summary['configs'].items()}, 'verdict': summary['verdict']}, indent=1))


if __name__ == '__main__':
    main()
