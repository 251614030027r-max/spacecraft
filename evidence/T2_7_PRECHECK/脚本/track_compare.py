import json, glob, sys
import numpy as np
T = {r['opening']: r for r in json.load(open(sys.argv[1]))}
from pathlib import Path as _P
root = str(_P(__file__).resolve().parents[2]) + '/'   # evidence/
runs = {'SAC s0': 'T2_6W_WIDE_140_200/C_eval/e_sac_s0', 'SAC s1': 'T2_6W_WIDE_140_200/C_eval/e_sac_s1',
        'EACC fixed s0': 'T2_6W1_CUTOFF/C_eval/e_mve_s0', 'EACC fixed s1': 'T2_6W1_CUTOFF/C_eval/e_mve_s1'}
tr = list(T.values())
print('plan-tracking rule (in action space): n %d, mean return %.1f, captures %d (urgent %d / %d, economy %d / %d), events %s'
      % (len(tr), np.mean([r['return'] for r in tr]), sum(r['event'] == 'success' for r in tr),
         sum(r['event'] == 'success' for r in tr if r['urgent']), sum(r['urgent'] for r in tr),
         sum(r['event'] == 'success' for r in tr if not r['urgent']), sum(not r['urgent'] for r in tr),
         {e: sum(r['event'] == e for r in tr) for e in set(r['event'] for r in tr)}))
print('\n%-14s %6s  %s' % ('run', 'step', 'mean return | captures (urgent) | rule - run paired mean, urgent, economy'))
for name, d in runs.items():
    for k in (1000, 2000, 3000, 4000):
        R = {}
        for f in glob.glob(root + d + f'/step_{k}/*.json'):
            if 'ERROR' not in f:
                r = json.load(open(f)); R[r['opening']] = r
        ops = sorted(set(R) & set(T))
        dd = [T[o]['return'] - R[o]['return'] for o in ops]
        du = [T[o]['return'] - R[o]['return'] for o in ops if T[o]['urgent']]
        de = [T[o]['return'] - R[o]['return'] for o in ops if not T[o]['urgent']]
        print('%-14s %6d  %6.1f | %2d (%2d) | %+6.1f %+6.1f %+6.1f' % (name, k, np.mean([R[o]['return'] for o in ops]),
              sum(R[o]['event'] == 'success' for o in ops), sum(R[o]['event'] == 'success' for o in ops if R[o]['mission']['urgent']),
              np.mean(dd), np.mean(du), np.mean(de)))
