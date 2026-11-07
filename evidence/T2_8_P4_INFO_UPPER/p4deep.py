import json, glob, itertools
from collections import defaultdict
import numpy as np
E = 'evidence/T2_8_P4_INFO'
info, o2 = {}, {}
for f in glob.glob(f'{E}/info/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: info[(r['opening'], round(d['t'], 3))] = d
for f in glob.glob(f'{E}/opp2/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: o2[(r['opening'], round(d['t'], 3))] = d
S = json.load(open(f'{E}/KEYSCORE_P4.json'))['scores']
def phase(d): return 'economy' if not d['urgent'] else ('window' if 40 <= d['left_s'] <= 140 else 'early')
def gapcls(g): return 'big>=50' if g >= 50 else ('mid20-50' if g >= 20 else 'small10-20')
# signal accuracy vs truth (no critic)
acc = defaultdict(lambda: [0, 0, 0])
for k, d in info.items():
    tv = [c['truth_track']['true'] for c in d['cands']]
    ov = [c['opp2']['value'] for c in o2[k]['cands']]
    for i, j in itertools.combinations(range(9), 2):
        g = abs(tv[i] - tv[j])
        if g < 10: continue
        x = ov[i] - ov[j]; t = tv[i] - tv[j]
        acc[(phase(d), gapcls(g))][0 if abs(x) < 5 else (1 if x * t > 0 else 2)] += 1
print('opp2 vs truth (tie / right / wrong) by phase and true gap:')
for k in sorted(acc): v = acc[k]; print('  ', k, v, 'right share of non-tie %.2f' % (v[1] / max(v[1] + v[2], 1)))
# critic accuracy by phase x gap, per arm at 1000-2000
ca = defaultdict(lambda: [0, 0])
for lab, s in S.items():
    arm, step = lab.split('_')[1], int(lab.split('_')[-1])
    if step < 1000: continue
    for row in s['rows']:
        k = (row['opening'], round(row['t'], 3)); d = info[k]; q = row['q']
        tv = [c['truth_track']['true'] for c in d['cands']]
        for i, j in itertools.combinations(range(9), 2):
            g = abs(tv[i] - tv[j])
            if g < 10: continue
            ok = (q[i] - q[j]) * (tv[i] - tv[j]) > 0 and abs(q[i] - q[j]) >= 1
            ca[(arm, phase(d), gapcls(g))][0 if ok else 1] += 1
print('critic right / wrong by arm, phase, gap (1000-2000):')
for k in sorted(ca): v = ca[k]; print('  ', k, v, 'acc %.2f' % (v[0] / sum(v)))
print()
for thr in (5, 20, 50):
    for half in ('selection', 'heldout'):
        prec = [0, 0]; cf = defaultdict(int)
        for k, d in info.items():
            if (k[0] < 2_910_020) != (half == 'selection'): continue
            tv = [c['truth_track']['true'] for c in d['cands']]; ov = [c['opp2']['value'] for c in o2[k]['cands']]
            for i, j in itertools.combinations(range(9), 2):
                if abs(ov[i] - ov[j]) < thr or abs(tv[i] - tv[j]) < 10: continue
                prec[0 if (ov[i] - ov[j]) * (tv[i] - tv[j]) > 0 else 1] += 1
        for lab, s in S.items():
            if int(lab.split('_')[-1]) < 1000: continue
            for row in s['rows']:
                k = (row['opening'], round(row['t'], 3))
                if (k[0] < 2_910_020) != (half == 'selection'): continue
                d = info[k]; q = row['q']; tv = [c['truth_track']['true'] for c in d['cands']]; ov = [c['opp2']['value'] for c in o2[k]['cands']]
                for i, j in itertools.combinations(range(9), 2):
                    if abs(tv[i] - tv[j]) < 10: continue
                    ok = (q[i] - q[j]) * (tv[i] - tv[j]) > 0 and abs(q[i] - q[j]) >= 1
                    cf['critic_wrong' if not ok else 'critic_right'] += 1
                    if abs(ov[i] - ov[j]) >= thr:
                        xr = (ov[i] - ov[j]) * (tv[i] - tv[j]) > 0
                        if not ok and xr: cf['fixed'] += 1
                        if ok and not xr: cf['broken'] += 1
        print(f'opp2 calls with |d| >= {thr:>2} {half:9s}: precision {prec[0]}/{sum(prec)} = {prec[0]/max(sum(prec),1):.2f}; fixed {cf["fixed"]} of {cf["critic_wrong"]} critic errors ({cf["fixed"]/cf["critic_wrong"]:.2f}), broken {cf["broken"]} of {cf["critic_right"]} ({cf["broken"]/cf["critic_right"]:.3f}), net {cf["fixed"]-cf["broken"]}')
