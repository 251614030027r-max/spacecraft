"""Readout of a development stage (``docs/final_method/ROUND2.md`` section 5; the feedback-mechanism development).

    python -B -m experiments.r2_dev_readout --root ROOT --output OUT_DIR [--stage dev]

Reads ROOT/C/<config>_s<seed>/<label>_<opening>.json (evaluation of the final models) and ROOT/B/TRAIN_<config>_s<seed>/
episodes.json (training record) for every configuration of the stage, and writes
* ``EPISODES.csv`` -- one row per evaluated episode (config, seed, opening, event, return, time, delta-v, corrections,
  long-range entries, deadline);
* ``SUMMARY.json`` -- per configuration and seed: captures, timeouts, unsafe, other outcomes, mean true return, and the
  training record in its first and last quarter (capture share, timeout share, mean true return; the hover signal of
  DEAD_ENDS D03 is a rising return with a rising timeout share). Missing or error files are listed and the
  configuration is marked incomplete. (The feedback-mechanism round's paired check and verdict, FEEDBACK_DEV.md
  section 4, are at science b47eb660; the formal round uses ``experiments/formal_r2_readout.py``.)
"""
import argparse
import csv
import json
from pathlib import Path

from experiments.final import Roots, block, stage_of


def episodes(root, config: str, seed: int, openings: list[int]) -> tuple[list[dict], list[int]]:
    rows, missing = [], []
    R = Roots(root)
    for o in openings:
        f = R.find('C', f'{config}_s{seed}', f'{config}_s{seed}_{o}.json')
        if not f.is_file():
            missing.append(o); continue
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('event') == 'error':
            missing.append(o); continue
        rows.append({'config': config, 'seed': seed, 'opening': o, 'event': r['event'], 'return': float(r['return']),
                     'time_s': float(r['time_s']), 'delta_v': float(r['delta_v']), 'corrections': int(r.get('corrections', 0)),
                     'long_entries': int(r.get('long_entries', 0)), 'deadline_s': float(r['mission']['deadline_s'])})
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


def readout(root, stage: str = 'dev') -> tuple[dict, list[dict]]:
    st = stage_of(stage)
    ops = block(st['eval_openings'])
    summary, table = {'stage': stage, 'openings': st['eval_openings'], 'configs': {}}, []
    for config in st['trained']:
        per_seed, rows_all, incomplete = {}, [], False
        for seed in st['seeds']:
            rows, missing = episodes(root, config, seed, ops)
            per_seed[str(seed)] = {**outcome(rows), 'missing_or_error': missing,
                                   'training': training(Roots(root).find('B', f'TRAIN_{config}_s{seed}'))}
            incomplete |= bool(missing)
            rows_all += rows
        summary['configs'][config] = {'per_seed': per_seed, 'total': outcome(rows_all), 'complete': not incomplete}
        table += rows_all
    return summary, table


#: FORMAL.md section 8: captures(switch) >= captures(base) + 5, only-switch >= 2 x only-base, unsafe(switch) <= unsafe(base) + 1
SW_CAPTURE_MARGIN, SW_FLIP_RATIO, SW_UNSAFE_TOLERANCE = 5, 2, 1


def paired_switch(root, stage: str = 'dev', config: str = 'ECC', switch: str = 'extended_lattice') -> dict | None:
    """The same models evaluated without and with an evaluation switch (paired per seed and opening), with the
    pre-registered decision of FORMAL.md section 8; None while results are incomplete."""
    st = stage_of(stage); ops = block(st['eval_openings']); R = Roots(root)
    base, on, missing = {}, {}, 0
    for seed in st['seeds']:
        for o in ops:
            for store, label in ((base, f'{config}_s{seed}'), (on, f'{config}_s{seed}_{switch}')):
                f = R.find('C', label, f'{label}_{o}.json')
                r = json.loads(f.read_text(encoding='utf-8')) if f.is_file() else None
                if r is None or r.get('event') == 'error':
                    missing += 1; continue
                store[(seed, o)] = r
    if missing:
        return None
    keys = sorted(base)
    cap = lambda d: sum(d[k]['event'] == 'success' for k in keys)
    uns = lambda d: sum(d[k]['event'] == 'unsafe' for k in keys)
    only_on = sum(on[k]['event'] == 'success' and base[k]['event'] != 'success' for k in keys)
    only_base = sum(base[k]['event'] == 'success' and on[k]['event'] != 'success' for k in keys)
    res = {'switch': switch, 'pairs': len(keys), 'captures_base': cap(base), 'captures_switch': cap(on), 'only_switch': only_on,
           'only_base': only_base, 'unsafe_base': uns(base), 'unsafe_switch': uns(on),
           'fine_corrections': sum(1 for r in on.values() for d in (r.get('decisions') or []) if (d.get('coord') or {}).get('lattice') == 'fine'),
           'per_seed': {str(s): {'base': sum(base[(s, o)]['event'] == 'success' for o in ops),
                                 'switch': sum(on[(s, o)]['event'] == 'success' for o in ops)} for s in st['seeds']}}
    res['pass'] = (res['captures_switch'] >= res['captures_base'] + SW_CAPTURE_MARGIN and only_on >= SW_FLIP_RATIO * only_base
                   and res['unsafe_switch'] <= res['unsafe_base'] + SW_UNSAFE_TOLERANCE)
    res['formal_ecc'] = 'ECC_FINE' if res['pass'] else 'ECC'
    return res


EPISODE_FIELDS = ['config', 'seed', 'opening', 'event', 'return', 'time_s', 'delta_v', 'corrections', 'long_entries',
                  'deadline_s']


def write_episodes_csv(path: Path, table: list[dict]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader(); w.writerows(table)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', action='append', required=True); p.add_argument('--output', required=True); p.add_argument('--stage', default='dev')
    p.add_argument('--paired-switch', help='evaluation switch to pair against the same models (FORMAL.md section 8: extended_lattice)')
    a = p.parse_args()
    if a.paired_switch:
        res = paired_switch(a.root, a.stage, switch=a.paired_switch)
        out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
        (out / f'PAIRED_{a.paired_switch}.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
        print(json.dumps(res, indent=1))
        return
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    summary, table = readout(a.root, a.stage)
    summary['verdict'] = None
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    write_episodes_csv(out / 'EPISODES.csv', table)
    print(json.dumps({'totals': {c: v['total'] for c, v in summary['configs'].items()}, 'verdict': summary['verdict']}, indent=1))


if __name__ == '__main__':
    main()
