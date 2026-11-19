"""Readout of the formal time-limited capture results for the paper (first draft).

    python -B -m experiments.paper_readout --output evidence/PAPER_READOUT

Groups (paper name -> configuration, evidence):
  ECC                        -> ECC_LE          (final method: screening with the long-range entry, critic inputs z)
  ECC w/o long-range entry   -> ECC_CERT_LOOP   (round-1 ECC: screening without the long-range entry)
  ECC w/o execution inputs   -> SAC_LE          (same coordination, critic on the policy view)
  ECC w/o coordination       -> EACH_C          (no screening, no long-range entry)
  SAC-MPC                    -> SAC_C
  TD3-MPC                    -> TD3_C           (cloud delivery, its own CSV format)
plus the entry re-timing switch of the round-1 ECC policy ('ECC w/o long-range entry+rigid').
All groups: seeds 20-22 x urgent openings 300000-300095, deterministic policy. Paired statistics: per-seed rows, then the
pooled paired difference with a bootstrap that resamples openings within each seed (10,000 draws).
Outputs: READOUT.json, MAIN_TABLE.csv, PAIRED.csv, URGENT_EPISODES_WIDE.csv, TRAINING_CURVES.csv, VALIDATION_CURVES.csv,
DEV_REFERENCES.csv (development urgent screen: pure MPC, fixed intents, direct entry; 48 other openings), SHA256SUMS.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'evidence' / 'FORMAL'
TD3 = FORMAL / 'TD3_C_CLOUD_20261118_7e55a65d'
SEEDS = ('20', '21', '22')
DRAWS = 10_000
GROUPS = {   # paper name -> (episodes CSV stem or None for TD3, configuration)
    'ECC': ('ECC_LE_SESSION6', 'ECC_LE'),
    'ECC w/o long-range entry': ('ECC_SESSION1', 'ECC_CERT_LOOP'),
    'ECC w/o execution inputs': ('SAC_LE_SESSION7', 'SAC_LE'),
    'ECC w/o coordination': ('EACH_SESSION3', 'EACH_C'),
    'SAC-MPC': ('SAC_SESSION2', 'SAC_C'),
    'TD3-MPC': (None, 'TD3_C'),
}
RIGID = 'ECC w/o long-range entry+rigid'


def _row(event, ret, t, dv):
    return {'event': event, 'return': float(ret), 'time_s': float(t), 'delta_v': float(dv)}


def load():
    rows = {}
    for name, (stem, cfg) in GROUPS.items():
        if stem is None:
            for s in SEEDS:
                for r in csv.DictReader(open(TD3 / f'eval_s{s}' / 'EPISODES.csv', encoding='utf-8')):
                    if r['condition'] == 'urgent':
                        rows[(name, s, int(r['opening']))] = _row(r['event'], r['return_true'], r['time_s'], r['delta_v'])
            continue
        for r in csv.DictReader(open(FORMAL / f'{stem}_EPISODES.csv', encoding='utf-8')):
            if r['condition'] != 'urgent':
                continue
            key = name if not r['switch'] else (RIGID if (name, r['switch']) == ('ECC w/o long-range entry', 'rigid') else None)
            if key:
                rows[(key, r['seed'], int(r['opening']))] = _row(r['event'], r['return'], r['time_s'], r['delta_v'])
    return rows


def outcome(rows, arm):
    per, allr = {}, []
    for s in SEEDS:
        x = [r for k, r in rows.items() if k[0] == arm and k[1] == s]
        e = Counter(r['event'] for r in x)
        per[s] = {'n': len(x), 'capture': e['success'], 'timeout': e['timeout'], 'resource_exhausted': e['resource_exhausted'],
                  'unsafe': e['unsafe'], 'mean_return': st.mean(r['return'] for r in x)}
        allr += x
    ok = [r for r in allr if r['event'] == 'success']
    return {'per_seed': per, 'pooled': {
        'n': len(allr), 'capture': len(ok), 'capture_rate': len(ok) / len(allr),
        'timeout': sum(r['event'] == 'timeout' for r in allr), 'unsafe': sum(r['event'] == 'unsafe' for r in allr),
        'resource_exhausted': sum(r['event'] == 'resource_exhausted' for r in allr),
        'mean_return': st.mean(r['return'] for r in allr), 'median_return': st.median(r['return'] for r in allr),
        'success_time_median_s': st.median(r['time_s'] for r in ok), 'success_dv_median_m_s': st.median(r['delta_v'] for r in ok)}}


def paired(rows, a, b, metric):
    per = {}
    for s in SEEDS:
        v = []
        for k, r in rows.items():
            if k[0] != a or k[1] != s or (b, s, k[2]) not in rows:
                continue
            q = rows[(b, s, k[2])]
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
    rnd = random.Random(20261119); draws = []
    for _ in range(DRAWS):
        tot = cnt = 0
        for v in per.values():
            if v:
                tot += sum(rnd.choice(v) for _ in v); cnt += len(v)
        draws.append(tot / cnt)
    draws.sort()
    res = {'n': n, 'estimate': sum(sum(v) for v in per.values()) / n,
           'ci95': [draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]],
           'per_seed': {s: (sum(v) if metric == 'capture' else (st.mean(v) if v else None)) for s, v in per.items()}}
    if metric == 'capture':
        res['flips'] = {'a_only': sum(x > 0 for v in per.values() for x in v), 'b_only': sum(x < 0 for v in per.values() for x in v)}
    else:
        res['median'] = st.median([x for v in per.values() for x in v])
    return res


def training_episodes(cfg, seed):
    p = FORMAL / 'B' / f'TRAIN_{cfg}_s{seed}' / 'episodes.json'
    if not p.exists():
        p = TD3 / f'train_s{seed}' / 'episodes.json'
    return json.loads(p.read_text(encoding='utf-8'))


def training_curves(window=100):
    rows = []
    for name, (_, cfg) in GROUPS.items():
        for s in SEEDS:
            eps = training_episodes(cfg, s)
            cum, total = [], 0
            for e in eps:
                total += (e.get('exec_stats') or {}).get('stages', 0); cum.append(total)
            idx = [i for i, e in enumerate(eps) if e['mission']['urgent']]
            for i in range(0, len(idx) - window + 1, window // 2):
                w = [eps[j] for j in idx[i:i + window]]
                rows.append({'group': name, 'seed': s, 'urgent_episode_end': i + window, 'decisions_approx': cum[idx[i + window - 1]],
                             'capture_rate': sum(e['event'] == 'success' for e in w) / window,
                             'mean_true_return': st.mean(e['return_true'] for e in w),
                             'screen_corrections_per_episode': st.mean((e.get('exec_stats') or {}).get('cert_interventions', 0) for e in w),
                             'long_range_entries_per_episode': st.mean((e.get('exec_stats') or {}).get('long_entries', 0) for e in w)})
    return rows


def validation_curves():
    rows = []
    for name, (stem, cfg) in GROUPS.items():
        if stem is None:
            for s in SEEDS:
                v = [r for r in csv.DictReader(open(TD3 / f'eval_s{s}' / 'EPISODES.csv', encoding='utf-8')) if r['condition'] == 'validation']
                for step in sorted({int(r['checkpoint_step']) for r in v}):
                    x = [r for r in v if int(r['checkpoint_step']) == step]
                    rows.append({'group': name, 'seed': s, 'decisions': step, 'n': len(x), 'capture': sum(r['event'] == 'success' for r in x),
                                 'mean_true_return': st.mean(float(r['return_true']) for r in x)})
            continue
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
    for d in sorted((ROOT / 'evidence/ECC_FINAL_PRECHECK/screen/urgent/S').iterdir()):
        fs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(d.glob('*.json'))]
        e = Counter(f['event'] for f in fs)
        rows.append({'policy': d.name, 'openings': f"{min(f['opening'] for f in fs)}-{max(f['opening'] for f in fs)}", 'n': len(fs),
                     'capture': e['success'], 'timeout': e['timeout'], 'resource_exhausted': e['resource_exhausted'],
                     'unsafe': e['unsafe'], 'mean_return': st.mean(f['return'] for f in fs)})
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); a = ap.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rows = load()
    arms = [*GROUPS, RIGID]
    res = {'source': 'evidence/FORMAL: deterministic urgent evaluation, seeds 20-22 x openings 300000-300095',
           'groups': {n: GROUPS[n][1] if n in GROUPS else 'ECC_CERT_LOOP (entry re-timing off)' for n in arms},
           'outcomes': {arm: outcome(rows, arm) for arm in arms}, 'paired': {}}
    comparisons = [('ECC', 'SAC-MPC'), ('ECC', 'TD3-MPC'), ('ECC', 'ECC w/o long-range entry'), ('ECC', 'ECC w/o execution inputs'),
                   ('ECC', 'ECC w/o coordination'), ('ECC w/o long-range entry', 'ECC w/o coordination'),
                   ('ECC w/o coordination', 'SAC-MPC'), ('ECC w/o long-range entry', RIGID)]
    prow = []
    for x, y in comparisons:
        res['paired'][f'{x} vs {y}'] = v = {m: paired(rows, x, y, m) for m in ('capture', 'return', 'time_s', 'delta_v')}
        for m, z in v.items():
            prow.append({'comparison': f'{x} vs {y}', 'metric': m, 'n': z['n'], 'estimate': z['estimate'], 'ci_low': z['ci95'][0],
                         'ci_high': z['ci95'][1], 'median': z.get('median', ''), 'seed20': z['per_seed']['20'],
                         'seed21': z['per_seed']['21'], 'seed22': z['per_seed']['22'],
                         'a_only': (z.get('flips') or {}).get('a_only', ''), 'b_only': (z.get('flips') or {}).get('b_only', '')})
    write_csv(out / 'PAIRED.csv', prow)
    write_csv(out / 'MAIN_TABLE.csv', [{'group': arm, 'seed': s, **res['outcomes'][arm]['per_seed'][s]} for arm in arms for s in SEEDS])
    wide = []
    for s in SEEDS:
        for o in range(300000, 300096):
            row = {'seed': s, 'opening': o}
            for arm in arms:
                r = rows.get((arm, s, o))
                if r:
                    t = arm.replace(' ', '_')
                    row.update({f'{t}_event': r['event'], f'{t}_return': r['return'], f'{t}_time_s': r['time_s'], f'{t}_delta_v': r['delta_v']})
            wide.append(row)
    write_csv(out / 'URGENT_EPISODES_WIDE.csv', wide)
    tc = training_curves(); write_csv(out / 'TRAINING_CURVES.csv', tc)
    write_csv(out / 'VALIDATION_CURVES.csv', validation_curves())
    dev = dev_references(); write_csv(out / 'DEV_REFERENCES.csv', dev); res['dev_references'] = dev
    # share of the last 200 training episodes of ECC in which the long-range entry was launched at least once
    res['long_range_entry_training_share'] = {s: sum((e.get('exec_stats') or {}).get('long_entries', 0) > 0
                                                     for e in training_episodes('ECC_LE', s)[-200:]) / 200 for s in SEEDS}
    (out / 'READOUT.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    names = sorted(p.name for p in out.iterdir() if p.name != 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n" for n in names), encoding='utf-8')
    for k, v in res['paired'].items():
        c = v['capture']
        print(f"{k:55s} {c['estimate']*100:+5.1f} [{c['ci95'][0]*100:+5.1f},{c['ci95'][1]*100:+5.1f}] seeds {c['per_seed']} flips {c['flips']}")


if __name__ == '__main__':
    main()
