"""Derive the plot data of the results figures (Section V) from evidence/PAPER_READOUT.

Run from figures/src:  python3 make_results_data.py   -> data/*.dat (read by fig_res_*.tex) and a summary on stdout.
"""
import csv
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[5] / 'evidence' / 'PAPER_READOUT'
OUT = Path(__file__).resolve().parent / 'data'; OUT.mkdir(exist_ok=True)
GROUPS = ['ECC', 'ECC w/o execution inputs', 'ECC w/o long-range entry', 'ECC w/o coordination', 'SAC-MPC', 'TD3-MPC']
KEY = {'ECC': 'ecc', 'ECC w/o execution inputs': 'noin', 'ECC w/o long-range entry': 'nole',
       'ECC w/o coordination': 'noco', 'SAC-MPC': 'sac', 'TD3-MPC': 'td3'}
rd = lambda f: list(csv.DictReader(open(SRC / f, encoding='utf-8')))
main = rd('MAIN_TABLE.csv'); wide = rd('URGENT_EPISODES_WIDE.csv'); paired = rd('PAIRED.csv')

# ---- outcome composition
with open(OUT / 'outcomes.dat', 'w') as f:
    f.write('y name capture timeout unsafe ncap\n')
    for i, g in enumerate(GROUPS):
        rows = [r for r in main if r['group'] == g]
        cap, to, un = (sum(int(r[k]) for r in rows) for k in ('capture', 'timeout', 'unsafe'))
        f.write(f"{i} {KEY[g]} {cap/288*100:.3f} {to/288*100:.3f} {un/288*100:.3f} {cap}\n")
        print(g, cap, [r['capture'] for r in rows], to, un)

# ---- paired contingency ECC vs SAC-MPC
e, s = 'ECC_event', 'SAC-MPC_event'
both = sum(r[e] == 'success' and r[s] == 'success' for r in wide)
eo = sum(r[e] == 'success' and r[s] != 'success' for r in wide)
so = sum(r[e] != 'success' and r[s] == 'success' for r in wide)
with open(OUT / 'contingency.dat', 'w') as f:
    f.write('both ecc sac neither\n'); f.write(f"{both} {eo} {so} {len(wide) - both - eo - so}\n")
print('contingency both/ecc-only/sac-only/neither', both, eo, so, len(wide) - both - eo - so)

# ---- paired comparisons (forest), top to bottom
ORDER = [('ECC vs SAC-MPC', 'ECC vs SAC--MPC'), ('ECC vs TD3-MPC', 'ECC vs TD3--MPC'),
         ('ECC w/o long-range entry vs ECC w/o coordination', '$+$ keep/correct'),
         ('ECC vs ECC w/o long-range entry', '$+$ long-range entry'),
         ('ECC vs ECC w/o execution inputs', '$+$ critic inputs $\\bz$'),
         ('ECC w/o long-range entry vs ECC w/o long-range entry+rigid', '$+$ entry re-timing')]
cap = [r for r in paired if r['metric'] == 'capture']
with open(OUT / 'forest.dat', 'w') as f:
    f.write('y est lo hi s20 s21 s22 a b\n')
    for i, (c, _) in enumerate(ORDER):
        r = next(r for r in cap if r['comparison'] == c)
        f.write(f"{len(ORDER)-1-i} {float(r['estimate'])*100:.3f} {float(r['ci_low'])*100:.3f} {float(r['ci_high'])*100:.3f} "
                f"{float(r['seed20'])/96*100:.3f} {float(r['seed21'])/96*100:.3f} {float(r['seed22'])/96*100:.3f} {r['a_only']} {r['b_only']}\n")
        print(c, r['estimate'], r['ci_low'], r['ci_high'], r['seed20'], r['seed21'], r['seed22'], r['a_only'], r['b_only'])

# ---- training curves on a common decision grid; mean and min--max over seeds
tr = rd('TRAINING_CURVES.csv'); grid = np.arange(1000, 4801, 100)
for g in GROUPS:
    caps, cors, les = [], [], []
    for sd in ('20', '21', '22'):
        rows = [r for r in tr if r['group'] == g and r['seed'] == sd]
        x = [float(r['decisions_approx']) for r in rows]
        caps.append(np.interp(grid, x, [float(r['capture_rate']) * 100 for r in rows]))
        cors.append(np.interp(grid, x, [float(r['screen_corrections_per_episode']) for r in rows]))
        les.append(np.interp(grid, x, [float(r['long_range_entries_per_episode']) for r in rows]))
    caps, cors, les = np.array(caps), np.array(cors), np.array(les)
    with open(OUT / f'train_{KEY[g]}.dat', 'w') as f:
        f.write('x cap caplo caphi int intlo inthi le lelo lehi\n')
        for j, xg in enumerate(grid):
            f.write(f"{xg} {caps[:, j].mean():.3f} {caps[:, j].min():.3f} {caps[:, j].max():.3f} {cors[:, j].mean():.3f} "
                    f"{cors[:, j].min():.3f} {cors[:, j].max():.3f} {les[:, j].mean():.3f} {les[:, j].min():.3f} {les[:, j].max():.3f}\n")

# ---- validation curves
va = rd('VALIDATION_CURVES.csv')
for g in ('ECC', 'ECC w/o execution inputs', 'ECC w/o long-range entry', 'SAC-MPC', 'TD3-MPC'):
    with open(OUT / f'val_{KEY[g]}.dat', 'w') as f:
        f.write('x mean s20 s21 s22\n')
        for d in (1000, 2000, 3000, 4000, 5000):
            v = {r['seed']: int(r['capture']) for r in va if r['group'] == g and int(r['decisions']) == d}
            f.write(f"{d} {np.mean(list(v.values())):.3f} {v['20']} {v['21']} {v['22']}\n")

# ---- reference lines (48 development openings) and ECC (288 test tasks)
dev = rd('DEV_REFERENCES.csv')
names = [('pure', 'pure'), ('b_10', 'unaware'), ('rally', 'rule'), ('c_10_0', 'fixed'), ('gstar', 'gstar')]
with open(OUT / 'reference.dat', 'w') as f:
    f.write('y name capture timeout fuel unsafe ncap\n')
    for i, (p, n) in enumerate(names):
        r = next(r for r in dev if r['policy'] == p); k = int(r['n'])
        f.write(f"{i} {n} {int(r['capture'])/k*100:.3f} {int(r['timeout'])/k*100:.3f} {int(r['resource_exhausted'])/k*100:.3f} "
                f"{int(r['unsafe'])/k*100:.3f} {r['capture']}/{k}\n")
    ecc = [r for r in main if r['group'] == 'ECC']
    c = sum(int(r['capture']) for r in ecc); to = sum(int(r['timeout']) for r in ecc); un = sum(int(r['unsafe']) for r in ecc)
    f.write(f"6 ecc {c/288*100:.3f} {to/288*100:.3f} 0.000 {un/288*100:.3f} {c}/288\n")
