"""Readout of a development stage (``docs/final_method/ROUND2.md`` section 5; the feedback-mechanism development).

    python -B -m experiments.r2_dev_readout --root ROOT --output OUT_DIR [--stage dev]

Reads ROOT/C/<config>_s<seed>/<label>_<opening>.json (evaluation of the final models) and ROOT/B/TRAIN_<config>_s<seed>/
episodes.json (training record) for every configuration of the stage, and writes
* ``EPISODES.csv`` -- one row per evaluated episode (config, seed, opening, event, return, time, delta-v, corrections,
  long-range entries, deadline);
* ``SUMMARY.json`` -- per configuration and seed: captures, timeouts, unsafe, other outcomes, mean true return, and the
  training record in its first and last quarter (capture share, timeout share, mean true return; the hover signal of
  DEAD_ENDS D03 is a rising return with a rising timeout share). Missing or error files are listed and the
  configuration is marked incomplete. For the feedback-mechanism round, ``verdict`` applies the pre-registered rule of
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


#: FEEDBACK_DEV.md section 4: captures(X) >= captures(ECC) + 5, unsafe(X) <= unsafe(ECC) + 1, and X >= ECC on >= 2 seeds
FB_CAPTURE_MARGIN, FB_UNSAFE_TOLERANCE, FB_SEEDS_NOT_WORSE = 5, 1, 2


def feedback_verdict(summary: dict) -> dict | None:
    """The pre-registered decision of the feedback-mechanism round, or None while the baseline is incomplete."""
    cfg = summary['configs']
    base = cfg.get('ECC')
    if base is None or not base['complete']:
        return None
    out = {'rule': f'captures(X) >= captures(ECC) + {FB_CAPTURE_MARGIN}, unsafe(X) <= unsafe(ECC) + {FB_UNSAFE_TOLERANCE}, '
                   f'captures(X, seed) >= captures(ECC, seed) on >= {FB_SEEDS_NOT_WORSE} seeds'}
    for x in ('ECC_FB', 'ECC_EQ'):
        c = cfg.get(x)
        if c is None or not c['complete'] or not c['total']['episodes']:
            out[x] = None; continue
        seeds_ok = sum(c['per_seed'][s]['captures'] >= base['per_seed'][s]['captures'] for s in base['per_seed'])
        passed = (c['total']['captures'] >= base['total']['captures'] + FB_CAPTURE_MARGIN
                  and c['total']['unsafe'] <= base['total']['unsafe'] + FB_UNSAFE_TOLERANCE and seeds_ok >= FB_SEEDS_NOT_WORSE)
        out[x] = {'captures': c['total']['captures'], 'baseline_captures': base['total']['captures'], 'unsafe': c['total']['unsafe'],
                  'baseline_unsafe': base['total']['unsafe'], 'seeds_not_worse': seeds_ok, 'pass': passed}
    fb, eq = out.get('ECC_FB'), out.get('ECC_EQ')
    if fb is None:
        out['formal_ecc'] = None
    elif fb['pass']:
        out['formal_ecc'] = 'ECC_FB'
    elif eq is None:
        out['formal_ecc'] = None                       # run ECC_EQ (conditional session)
        out['next'] = 'run ECC_EQ'
    else:
        out['formal_ecc'] = 'ECC_EQ' if eq['pass'] else 'ECC'
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--output', required=True); p.add_argument('--stage', default='dev')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    summary, table = readout(Path(a.root), a.stage)
    summary['verdict'] = feedback_verdict(summary) if a.stage == 'dev' else None
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    with open(out / 'EPISODES.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'seed', 'opening', 'event', 'return', 'time_s', 'delta_v', 'corrections',
                                          'long_entries', 'arbitration_changed', 'deadline_s'])
        w.writeheader(); w.writerows(table)
    print(json.dumps({'totals': {c: v['total'] for c, v in summary['configs'].items()}, 'verdict': summary['verdict']}, indent=1))


if __name__ == '__main__':
    main()
