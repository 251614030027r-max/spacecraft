import json, glob
from collections import defaultdict
import numpy as np
E = 'evidence/T2_8_P4_INFO'
info = {}
for f in glob.glob(f'{E}/info/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: info[(r['opening'], round(d['t'], 3))] = d
S = json.load(open(f'{E}/KEYSCORE_P4.json'))['scores']
def phase(d): return 'economy' if not d['urgent'] else ('window' if 40 <= d['left_s'] <= 140 else 'early')
res = defaultdict(lambda: defaultdict(list))
for lab, s in S.items():
    arm, seed, step = lab.split('_')[1], lab.split('_')[2], int(lab.split('_')[-1])
    if step < 1000: continue
    for row in s['rows']:
        d = info[(row['opening'], round(row['t'], 3))]
        tv = np.array([c['truth_track']['true'] for c in d['cands']]); grid = np.array([c['a'] for c in d['cands']])
        best = tv.max()
        k_actor = int(np.argmin(np.linalg.norm(grid - np.asarray(row['actor']), axis=1)))
        k_crit = int(np.argmax(row['q']))
        p = phase(d)
        res[(arm, p)]['actor'].append(best - tv[k_actor]); res[(arm, p)]['critic_argmax'].append(best - tv[k_crit])
        res[(arm, p)]['random'].append(best - tv.mean())
print('mean regret vs best of 9 intents (truth_track points; lower = better), checkpoints 1000-2000, both seeds')
for k in sorted(res):
    v = res[k]; print(f'  {k[0]:4s} {k[1]:8s} actor {np.mean(v["actor"]):6.1f}   critic-argmax {np.mean(v["critic_argmax"]):6.1f}   uniform {np.mean(v["random"]):6.1f}   n={len(v["actor"])}')
