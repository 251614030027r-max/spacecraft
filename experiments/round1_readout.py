"""Readout of the first formal round (four trained groups) for the paper draft; development-stage references alongside.

    python -B -m experiments.round1_readout --output evidence/FORMAL/ROUND1_READOUT

Inputs (all in git): ``evidence/FORMAL/*_EPISODES.csv`` (deterministic evaluations, 3 seeds x 96 openings per
condition), ``evidence/FORMAL/*_VERIFIED.json`` (validation checkpoints), ``evidence/FORMAL/B/TRAIN_*/episodes.json``
(training episodes), ``evidence/ECC_FINAL_PRECHECK/screen/urgent/S/`` (development urgent screen, 48 openings) and
``evidence/FORMAL_PREP/PRESCREEN/budget_2.0/`` (development economy prescreen, 48 openings).
Outputs: ``READOUT.json`` (all numbers of facts/08), ``MAIN_TABLE.csv``, ``TRAINING_CURVES.csv``, ``VALIDATION_CURVES.csv``,
``DEV_REFERENCES.csv``. Paired statistics: per-seed rows, then pooled paired difference with a bootstrap interval that
resamples openings within each seed (10,000 draws), as registered in ``docs/paper/EXPERIMENTS.md`` section 5.
"""
import argparse
import csv
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'evidence' / 'FORMAL'
GROUPS = {  # paper name -> (file stem, config)
    'ECC': ('ECC_SESSION1', 'ECC_CERT_LOOP'),
    'SAC-MPC': ('SAC_SESSION2', 'SAC_C'),
    'ECC w/o screening': ('EACH_SESSION3', 'EACH_C'),
    'ECC w/o execution feedback': ('SAC_CERT_SESSION4', 'SAC_CERT_LOOP'),
}
SEEDS = ('20', '21', '22')
CONDITIONS = ('urgent', 'economy', 'natural')
DRAWS = 10_000


def load():
    rows = {}
    for name, (stem, _) in GROUPS.items():
        for r in csv.DictReader(open(FORMAL / f'{stem}_EPISODES.csv', encoding='utf-8')):
            if r['condition'] not in CONDITIONS:
                continue
            key = name if not r['switch'] else f"{name}+{r['switch']}"
            rows[(key, r['condition'], r['seed'], int(r['opening']))] = r
    return rows


def outcome(rows, arm, cond):
    out = {'per_seed': {}}
    allr = []
    for s in SEEDS:
        x = [r for k, r in rows.items() if k[0] == arm and k[1] == cond and k[2] == s]
        e = Counter(r['event'] for r in x)
        out['per_seed'][s] = {'n': len(x), 'capture': e['success'], 'timeout': e['timeout'],
                              'resource_exhausted': e['resource_exhausted'], 'unsafe': e['unsafe'],
                              'numerical_failure': e['numerical_failure'],
                              'mean_return': st.mean(float(r['return']) for r in x)}
        allr += x
    caps = [r['event'] == 'success' for r in allr]
    ok = [r for r in allr if r['event'] == 'success']
    out['pooled'] = {'n': len(allr), 'capture': sum(caps), 'capture_rate': sum(caps) / len(allr),
                     'unsafe': sum(r['event'] == 'unsafe' for r in allr),
                     'resource_exhausted': sum(r['event'] == 'resource_exhausted' for r in allr),
                     'mean_return': st.mean(float(r['return']) for r in allr),
                     'median_return': st.median(float(r['return']) for r in allr),
                     'success_time_median_s': st.median(float(r['time_s']) for r in ok) if ok else None,
                     'success_dv_median_m_s': st.median(float(r['delta_v']) for r in ok) if ok else None}
    return out


def paired(rows, a, b, cond, metric):
    per = {}
    for s in SEEDS:
        v = []
        for k, r in rows.items():
            if k[0] == a and k[1] == cond and k[2] == s:
                q = rows.get((b, cond, s, k[3]))
                if q is None:
                    continue
                if metric == 'capture':
                    v.append(float(r['event'] == 'success') - float(q['event'] == 'success'))
                elif metric == 'return':
                    v.append(float(r['return']) - float(q['return']))
                elif r['event'] == 'success' and q['event'] == 'success':      # time / delta-v: both captured only
                    v.append(float(r[metric]) - float(q[metric]))
        per[s] = v
    n = sum(len(v) for v in per.values())
    if n == 0:
        return None
    est = sum(sum(v) for v in per.values()) / n
    rnd = random.Random(20261118); draws = []
    for _ in range(DRAWS):
        tot = cnt = 0
        for v in per.values():
            if v:
                tot += sum(rnd.choice(v) for _ in v); cnt += len(v)
        draws.append(tot / cnt)
    draws.sort()
    res = {'n': n, 'estimate': est, 'ci95': [draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]],
           'per_seed': {s: (sum(v) if metric == 'capture' else (st.mean(v) if v else None)) for s, v in per.items()}}
    if metric == 'capture':
        res['flips'] = {'a_only': sum(x > 0 for v in per.values() for x in v), 'b_only': sum(x < 0 for v in per.values() for x in v)}
        res['seeds_positive'] = sum(sum(v) > 0 for v in per.values())
    else:
        res['median'] = st.median([x for v in per.values() for x in v])
    return res


def natural_split(rows, arm):
    out = {}
    for typ, budget in (('urgent', '3.0'), ('economy', '2.0')):
        x = [r for k, r in rows.items() if k[0] == arm and k[1] == 'natural' and r['budget_m_s'] == budget]
        out[typ] = {'n': len(x), 'capture': sum(r['event'] == 'success' for r in x),
                    'resource_exhausted': sum(r['event'] == 'resource_exhausted' for r in x)}
    return out


def training_curves(window=100):
    rows = []
    for name, (_, cfg) in GROUPS.items():
        for s in SEEDS:
            eps = json.loads((FORMAL / 'B' / f'TRAIN_{cfg}_s{s}' / 'episodes.json').read_text(encoding='utf-8'))
            cum, total = [], 0                        # cumulative stage decisions (approximates the decision budget)
            for e in eps:
                total += e['exec_stats']['stages']; cum.append(total)
            idx = [i for i, e in enumerate(eps) if e['mission']['urgent']]
            urgent = [eps[i] for i in idx]
            for i in range(0, len(urgent) - window + 1, window // 2):
                w = urgent[i:i + window]
                rows.append({'group': name, 'seed': s, 'urgent_episode_end': i + window,
                             'decisions_approx': cum[idx[i + window - 1]],
                             'capture_rate': sum(e['event'] == 'success' for e in w) / window,
                             'mean_true_return': st.mean(e['return_true'] for e in w),
                             'screen_interventions_per_episode': st.mean(e['exec_stats'].get('cert_interventions', 0) for e in w)})
    return rows


def validation_curves():
    rows = []
    for name, (stem, cfg) in GROUPS.items():
        g = json.loads((FORMAL / f'{stem}_VERIFIED.json').read_text(encoding='utf-8'))['groups']
        for s in SEEDS:
            for step in (1000, 2000, 3000, 4000, 5000):
                k = f'{cfg}_s{s}_step{step}'
                if k in g:
                    rows.append({'group': name, 'seed': s, 'decisions': step, 'n': g[k]['count'],
                                 'capture': g[k]['events'].get('success', 0), 'mean_true_return': g[k]['mean_true_return']})
    return rows


def dev_references():
    rows = []
    for label, folder, cond in (('urgent', ROOT / 'evidence/ECC_FINAL_PRECHECK/screen/urgent/S', 'urgent'),):
        for d in sorted(folder.iterdir()):
            fs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(d.glob('*.json'))]
            e = Counter(f['event'] for f in fs)
            rows.append({'condition': cond, 'policy': d.name, 'openings': f"{min(f['opening'] for f in fs)}-{max(f['opening'] for f in fs)}",
                         'n': len(fs), 'capture': e['success'], 'timeout': e['timeout'], 'resource_exhausted': e['resource_exhausted'],
                         'unsafe': e['unsafe'], 'mean_return': st.mean(f['return'] for f in fs)})
    pre = ROOT / 'evidence/FORMAL_PREP/PRESCREEN/budget_2.0'
    by = {}
    for p in sorted(pre.glob('*_29*.json')):
        f = json.loads(p.read_text(encoding='utf-8')); by.setdefault(p.name.rsplit('_', 1)[0], []).append(f)
    for pol, fs in sorted(by.items()):
        e = Counter(f['event'] for f in fs)
        rows.append({'condition': 'economy (2.0 m/s)', 'policy': pol, 'openings': f"{min(f['opening'] for f in fs)}-{max(f['opening'] for f in fs)}",
                     'n': len(fs), 'capture': e['success'], 'timeout': e['timeout'], 'resource_exhausted': e['resource_exhausted'],
                     'unsafe': e['unsafe'], 'mean_return': st.mean(f['return'] for f in fs)})
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); a = ap.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rows = load()
    arms = sorted({k[0] for k in rows})
    res = {'source': 'evidence/FORMAL (round 1, science e8474e21), deterministic evaluation, 3 seeds x 96 openings',
           'outcomes': {c: {arm: outcome(rows, arm, c) for arm in arms if any(k[0] == arm and k[1] == c for k in rows)} for c in CONDITIONS},
           'paired': {}, 'natural_split': {arm: natural_split(rows, arm) for arm in GROUPS}}
    comparisons = [('ECC', 'SAC-MPC'), ('ECC', 'ECC w/o screening'), ('ECC', 'ECC w/o execution feedback'),
                   ('ECC w/o screening', 'SAC-MPC'), ('ECC w/o execution feedback', 'SAC-MPC'), ('ECC', 'ECC+rigid')]
    for c in CONDITIONS:
        for x, y in comparisons:
            if not any(k[0] == y and k[1] == c for k in rows):
                continue
            res['paired'][f'{c}: {x} vs {y}'] = {m: paired(rows, x, y, c, m) for m in ('capture', 'return', 'time_s', 'delta_v')}
    main_rows = []
    for c in CONDITIONS:
        for arm in GROUPS:
            o = res['outcomes'][c][arm]
            for s in SEEDS:
                main_rows.append({'condition': c, 'group': arm, 'seed': s, **o['per_seed'][s]})
    write_csv(out / 'MAIN_TABLE.csv', main_rows)
    write_csv(out / 'TRAINING_CURVES.csv', training_curves())
    write_csv(out / 'VALIDATION_CURVES.csv', validation_curves())
    dev = dev_references(); write_csv(out / 'DEV_REFERENCES.csv', dev); res['dev_references'] = dev
    (out / 'READOUT.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in res['paired'].items() if k.startswith('urgent')}, indent=1)[:3000])


if __name__ == '__main__':
    main()
