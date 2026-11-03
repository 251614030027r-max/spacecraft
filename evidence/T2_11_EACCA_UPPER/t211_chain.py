import json, glob, itertools
from collections import defaultdict
import numpy as np
info = {}
for f in glob.glob('evidence/T2_8_P4_INFO/info/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: info[(r['opening'], round(d['t'], 3))] = d
def phase(d): return 'economy' if not d['urgent'] else ('window' if 40 <= d['left_s'] <= 140 else 'early')
res = defaultdict(lambda: defaultdict(list))
for src, tag in (('evidence/T2_11_EACCA/KEYSCORE_T211.json', 'T211'), ('evidence/T2_8_P4_INFO/KEYSCORE_P4.json', 'night_engineered')):
    S = json.load(open(src))['scores']
    for lab, s in S.items():
        arm, step = lab.split('_')[1], int(lab.split('_')[-1])
        if step < 1000: continue
        key = f'{tag}:{arm}'
        for row in s['rows']:
            d = info[(row['opening'], round(row['t'], 3))]; ph = phase(d)
            tv = np.array([c['truth_track']['true'] for c in d['cands']]); q = np.array(row['q']); ev = [c['truth_track']['event'] for c in d['cands']]
            grid = np.array([c['a'] for c in d['cands']]); ka = int(np.argmin(np.linalg.norm(grid - np.asarray(row['actor']), axis=1)))
            ok = n = 0
            for i, j in itertools.combinations(range(9), 2):
                if abs(tv[i] - tv[j]) < 10: continue
                n += 1; ok += int((q[i] - q[j]) * (tv[i] - tv[j]) > 0 and abs(q[i] - q[j]) >= 1)
            if n: res[(key, ph)]['acc'].append(ok / n)
            res[(key, ph)]['regret'].append(tv.max() - tv[ka])
            res[(key, ph)]['lost'].append(float(ev[ka] != 'success' and 'success' in ev))
print('P4 key decisions, checkpoints 1000-2000, both seeds:  critic pair accuracy | actor regret | actor lost-capture share')
for k in sorted(res):
    v = res[k]; print(f'  {k[0]:26s} {k[1]:8s} {np.mean(v["acc"]):.2f} | {np.mean(v["regret"]):5.1f} | {np.mean(v["lost"]):.3f}')
