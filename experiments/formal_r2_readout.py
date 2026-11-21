"""Readout of the round-2 formal experiment: every number and figure source of the paper (``docs/final_method/FORMAL.md``
sections 2-4).

    python -B -m experiments.formal_r2_readout --root LOCAL [--root CLOUD] --output OUT_DIR

Reads the formal stage of the registry: trained configurations (ROOT/C/<cfg>_s<seed>), the evaluation switch
(ROOT/C/<cfg>_s<seed>_<switch>), the fixed policies (ROOT/S/<policy>), and the training records
(ROOT/B/TRAIN_<cfg>_s<seed>/episodes.json). Writes READOUT.json, MAIN_TABLE.csv, PAIRED.csv, EPISODES_WIDE.csv,
TRAINING_CURVES.csv, COORDINATION_USAGE.csv, STRATIFIED.csv, COMPUTE.csv, SAFETY.json and CASES.json.

Paired statistics: per-seed rows, then the pooled paired difference with a bootstrap that resamples openings within
each seed (10,000 draws, numpy generator seeded 20261121); fixed policies are paired with every seed of a trained arm on the same
opening (they have no seed); the fixed-intent pair is one 'seed'. Time and delta-v differences use openings both arms
capture only.
"""
import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.final import Roots, block, registry, stage_of

DRAWS = 10_000
RNG_SEED = 20261121
FIXED_SEED = 'fixed'
#: FORMAL.md section 2: (a, b) paired comparisons, in reporting order
COMPARISONS = [('ECC', 'SAC_MPC'), ('ECC', 'TD3_MPC'), ('ECC', 'pure'), ('ECC', 'ECC_NO_CORE'), ('ECC_NO_CORE', 'SAC_MPC'),
               ('ECC', 'ECC+rigid_entry'), ('c_10_0', 'b_10')]
DEADLINE_BINS = [(140.0, 155.0), (155.0, 170.0), (170.0, 185.0), (185.0, 200.0001)]
KINDS = ('keep', 'correct', 'launch', 'none')


# ---------------------------------------------------------------------- loading
def _episode(path: Path) -> dict | None:
    r = json.loads(path.read_text(encoding='utf-8'))
    if r.get('event') == 'error':
        return None
    dec = r.get('decisions') or []
    coords = [d['coord'] for d in dec if d.get('coord')]
    omega = (r.get('initial') or {}).get('omega')
    return {'event': r['event'], 'return': float(r['return']), 'time_s': float(r['time_s']), 'delta_v': float(r['delta_v']),
            'deadline_s': float(r['mission']['deadline_s']),
            'spin_deg_s': float(np.degrees(np.linalg.norm(omega))) if omega is not None else None,
            'corrections': int(r.get('corrections', 0)), 'long_entries': int(r.get('long_entries', 0)),
            'kinds': Counter(c.get('kind') for c in coords), 'coord_wall_s': [float(c['wall_s']) for c in coords if 'wall_s' in c],
            'descent_correction': any(c.get('kind') == 'correct' and c.get('alt_kind') == 'descent' for c in coords)}


def load(root, stage: str = 'formal') -> tuple[dict, list[str], list[int]]:
    """rows[(arm, seed, opening)] -> episode; arms: trained configurations, '<cfg>+<switch>', fixed policies."""
    R, st_ = Roots(root), stage_of(stage)
    ops, seeds = block(st_['eval_openings']), [str(s) for s in st_['seeds']]
    reg = registry()
    rows, missing = {}, []

    def take(arm, seed, folder, label):
        for o in ops:
            f = R.find(folder, label, f'{label}_{o}.json')
            e = _episode(f) if f.is_file() else None
            if e is None:
                missing.append(f'{folder}/{label}/{label}_{o}.json')
            else:
                rows[(arm, seed, o)] = e

    for cfg in st_['trained']:
        for s in seeds:
            take(cfg, s, 'C', f'{cfg}_s{s}')
            for sw in st_.get('eval_switches', []):
                if cfg in reg['eval_switches'][sw]['applies_to']:
                    take(f'{cfg}+{sw}', s, 'C', f'{cfg}_s{s}_{sw}')
    for pol in st_.get('fixed', []):
        take(pol, FIXED_SEED, 'S', pol)
    return rows, missing, ops


def arms_of(rows) -> list[str]:
    return sorted({k[0] for k in rows}, key=lambda a: (a not in registry()['trained'], a))


# ---------------------------------------------------------------------- statistics
def outcome(rows, arm) -> dict:
    seeds = sorted({k[1] for k in rows if k[0] == arm})
    per, allr = {}, []
    for s in seeds:
        x = [r for k, r in rows.items() if k[0] == arm and k[1] == s]
        e = Counter(r['event'] for r in x)
        per[s] = {'n': len(x), 'capture': e['success'], 'timeout': e['timeout'], 'unsafe': e['unsafe'],
                  'resource_exhausted': e['resource_exhausted'],
                  'other': len(x) - e['success'] - e['timeout'] - e['unsafe'] - e['resource_exhausted'],
                  'mean_return': st.mean(r['return'] for r in x) if x else None}
        allr += x
    ok = [r for r in allr if r['event'] == 'success']
    pooled = {'n': len(allr), 'capture': len(ok), 'capture_rate': len(ok) / len(allr) if allr else None,
              'timeout': sum(r['event'] == 'timeout' for r in allr), 'unsafe': sum(r['event'] == 'unsafe' for r in allr),
              'resource_exhausted': sum(r['event'] == 'resource_exhausted' for r in allr),
              'mean_return': st.mean(r['return'] for r in allr) if allr else None,
              'success_time_median_s': st.median(r['time_s'] for r in ok) if ok else None,
              'success_dv_median_m_s': st.median(r['delta_v'] for r in ok) if ok else None}
    return {'per_seed': per, 'pooled': pooled}


def _pairs(rows, a, b):
    """{seed: [(row_a, row_b), ...]}; a fixed arm pairs with every seed of the other arm."""
    sa = sorted({k[1] for k in rows if k[0] == a}); sb = sorted({k[1] for k in rows if k[0] == b})
    seeds = sa if sa != [FIXED_SEED] else (sb if sb != [FIXED_SEED] else [FIXED_SEED])
    out = {}
    for s in seeds:
        ka, kb = (s if sa != [FIXED_SEED] else FIXED_SEED), (s if sb != [FIXED_SEED] else FIXED_SEED)
        out[s] = [(r, rows[(b, kb, k[2])]) for k, r in sorted(rows.items()) if k[0] == a and k[1] == ka and (b, kb, k[2]) in rows]
    return out


def paired(rows, a, b, metric) -> dict | None:
    per = {}
    for s, pairs in _pairs(rows, a, b).items():
        v = []
        for r, q in pairs:
            if metric == 'capture':
                v.append(float(r['event'] == 'success') - float(q['event'] == 'success'))
            elif metric == 'return':
                v.append(r['return'] - q['return'])
            elif r['event'] == 'success' and q['event'] == 'success':
                v.append(r[metric] - q[metric])
        per[s] = v
    n = sum(len(v) for v in per.values())
    if n == 0:
        return None
    rng = np.random.default_rng(RNG_SEED)
    tot = np.zeros(DRAWS)
    for v in per.values():                     # resample openings within each seed
        if v:
            x = np.asarray(v, float)
            tot += x[rng.integers(0, len(x), size=(DRAWS, len(x)))].sum(axis=1)
    draws = np.sort(tot / n)
    res = {'a': a, 'b': b, 'metric': metric, 'n': n, 'estimate': sum(sum(v) for v in per.values()) / n,
           'ci95': [float(draws[int(0.025 * DRAWS)]), float(draws[int(0.975 * DRAWS) - 1])],
           'per_seed': {s: (sum(v) if metric == 'capture' else (st.mean(v) if v else None)) for s, v in per.items()}}
    if metric == 'capture':
        res['flips'] = {'a_only': sum(x > 0 for v in per.values() for x in v), 'b_only': sum(x < 0 for v in per.values() for x in v)}
    else:
        res['median'] = st.median([x for v in per.values() for x in v])
    return res


# ---------------------------------------------------------------------- tables
def training_curves(root, stage='formal', window=100) -> list[dict]:
    R, st_ = Roots(root), stage_of(stage)
    out = []
    for cfg in st_['trained']:
        for s in st_['seeds']:
            f = R.find('B', f'TRAIN_{cfg}_s{s}', 'episodes.json')
            if not f.is_file():
                continue
            eps = sorted(json.loads(f.read_text(encoding='utf-8')), key=lambda e: e['steps'])
            for i in range(0, len(eps) - window + 1, window // 2):
                w = eps[i:i + window]
                out.append({'config': cfg, 'seed': s, 'episode_end': i + window, 'decisions': int(w[-1]['steps']),
                            'capture_rate': sum(e['event'] == 'success' for e in w) / window,
                            'mean_true_return': st.mean(e['return_true'] for e in w),
                            'corrections_per_episode': st.mean((e.get('exec_stats') or {}).get('corrections', 0) for e in w),
                            'long_entries_per_episode': st.mean((e.get('exec_stats') or {}).get('long_entries', 0) for e in w)})
    return out


def coordination_usage(rows) -> list[dict]:
    out = []
    for arm in arms_of(rows):
        for s in sorted({k[1] for k in rows if k[0] == arm}):
            x = [r for k, r in rows.items() if k[0] == arm and k[1] == s]
            tot = Counter()
            for r in x:
                tot.update(r['kinds'])
            n = sum(tot[k] for k in KINDS)
            if n == 0:
                continue
            out.append({'arm': arm, 'seed': s, 'episodes': len(x), 'decisions': n,
                        **{f'{k}_share': tot[k] / n for k in KINDS}, **{f'{k}_per_episode': tot[k] / len(x) for k in KINDS}})
    return out


def stratified(rows, ops) -> list[dict]:
    spins = sorted({r['spin_deg_s'] for r in rows.values() if r['spin_deg_s'] is not None})
    edges = list(np.quantile(spins, [1 / 3, 2 / 3])) if spins else []
    out = []
    for arm in arms_of(rows):
        x = [r for k, r in rows.items() if k[0] == arm]
        for lo, hi in DEADLINE_BINS:
            y = [r for r in x if lo <= r['deadline_s'] < hi]
            if y:
                out.append({'arm': arm, 'stratum': 'deadline_s', 'range': f'{lo:g}-{min(hi, 200):g}', 'n': len(y),
                            'capture_rate': sum(r['event'] == 'success' for r in y) / len(y)})
        if edges:
            for name, cond in (('low', lambda v: v < edges[0]), ('mid', lambda v: edges[0] <= v < edges[1]), ('high', lambda v: v >= edges[1])):
                y = [r for r in x if r['spin_deg_s'] is not None and cond(r['spin_deg_s'])]
                if y:
                    out.append({'arm': arm, 'stratum': 'spin_deg_s', 'range': name, 'n': len(y),
                                'capture_rate': sum(r['event'] == 'success' for r in y) / len(y),
                                'spin_min': min(r['spin_deg_s'] for r in y), 'spin_max': max(r['spin_deg_s'] for r in y)})
    return out


def compute(rows) -> list[dict]:
    out = []
    for arm in arms_of(rows):
        w = [t for k, r in rows.items() if k[0] == arm for t in r['coord_wall_s']]
        if w:
            out.append({'arm': arm, 'coordination_decisions': len(w), 'median_s': float(np.median(w)),
                        'p95_s': float(np.quantile(w, 0.95)), 'max_s': float(max(w)), 'decision_period_s': 16.0})
    return out


def safety(rows) -> list[dict]:
    return [{'arm': k[0], 'seed': k[1], 'opening': k[2], 'descent_correction_before': r['descent_correction']}
            for k, r in sorted(rows.items()) if r['event'] == 'unsafe']


def cases(rows, seed: str) -> dict:
    """FORMAL.md section 3: A = ECC captures without any correction and SAC-MPC captures; B = ECC captures with at least
    one correction and SAC-MPC times out; the smallest opening of each."""
    ops = sorted(k[2] for k in rows if k[0] == 'ECC' and k[1] == seed)
    a = b = None
    for o in ops:
        e, s = rows.get(('ECC', seed, o)), rows.get(('SAC_MPC', seed, o))
        if e is None or s is None:
            continue
        if a is None and e['event'] == 'success' and e['corrections'] == 0 and s['event'] == 'success':
            a = o
        if b is None and e['event'] == 'success' and e['corrections'] >= 1 and s['event'] == 'timeout':
            b = o
    return {'seed': int(seed), 'case_A_typical': a, 'case_B_correction': b}


def wide(rows, ops) -> list[dict]:
    arms = arms_of(rows)
    seeds = sorted({k[1] for k in rows if k[1] != FIXED_SEED})
    out = []
    for s in seeds:
        for o in ops:
            row = {'seed': s, 'opening': o}
            for arm in arms:
                r = rows.get((arm, s, o)) or rows.get((arm, FIXED_SEED, o))
                if r is None:
                    continue
                if 'deadline_s' not in row:
                    row['deadline_s'] = r['deadline_s']; row['spin_deg_s'] = r['spin_deg_s']
                for f in ('event', 'return', 'time_s', 'delta_v'):
                    row[f'{arm}_{f}'] = r[f]
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def write_all(root, out: Path, stage: str = 'formal') -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows, missing, ops = load(root, stage)
    arms = arms_of(rows)
    outcomes = {a: outcome(rows, a) for a in arms}
    comps = [c for c in COMPARISONS if c[0] in arms and c[1] in arms]
    pairs = {f'{a} vs {b}': {m: paired(rows, a, b, m) for m in ('capture', 'return', 'time_s', 'delta_v')} for a, b in comps}
    case = cases(rows, str(stage_of(stage)['seeds'][0]))
    main = [{'arm': a, 'seed': s, **v} for a in arms for s, v in outcomes[a]['per_seed'].items()] + \
           [{'arm': a, 'seed': 'pooled', **outcomes[a]['pooled']} for a in arms]
    write_csv(out / 'MAIN_TABLE.csv', main)
    write_csv(out / 'PAIRED.csv', [{'comparison': k, 'metric': m, **{f: (json.dumps(v[f]) if isinstance(v[f], (dict, list)) else v[f])
                                                                    for f in v if f not in ('a', 'b', 'metric')}}
                                   for k, d in pairs.items() for m, v in d.items() if v is not None])
    write_csv(out / 'EPISODES_WIDE.csv', wide(rows, ops))
    write_csv(out / 'TRAINING_CURVES.csv', training_curves(root, stage))
    write_csv(out / 'COORDINATION_USAGE.csv', coordination_usage(rows))
    write_csv(out / 'STRATIFIED.csv', stratified(rows, ops))
    write_csv(out / 'COMPUTE.csv', compute(rows))
    (out / 'SAFETY.json').write_text(json.dumps(safety(rows), indent=1), encoding='utf-8')
    (out / 'CASES.json').write_text(json.dumps(case, indent=1), encoding='utf-8')
    readout = {'stage': stage, 'openings': f'{ops[0]}-{ops[-1]}', 'missing_or_error': missing, 'outcomes': outcomes,
               'paired': pairs, 'cases': case}
    (out / 'READOUT.json').write_text(json.dumps(readout, indent=1), encoding='utf-8')
    return readout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', action='append', required=True); p.add_argument('--output', required=True)
    p.add_argument('--stage', default='formal')
    a = p.parse_args()
    r = write_all(a.root, Path(a.output), a.stage)
    print(json.dumps({'missing_or_error': len(r['missing_or_error']),
                      'capture': {k: v['pooled']['capture'] for k, v in r['outcomes'].items()}, 'cases': r['cases']}, indent=1))


if __name__ == '__main__':
    main()
