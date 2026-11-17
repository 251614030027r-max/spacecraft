"""Derive the plot data of the results figures (Section V) from evidence/FORMAL/ROUND1_READOUT.

Run from figures/src:  python3 make_results_data.py   -> data/*.dat (read by fig_res_*.tex) and a summary on stdout.
"""
import csv, statistics as st
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[5] / 'evidence' / 'FORMAL' / 'ROUND1_READOUT'
OUT = Path(__file__).resolve().parent / 'data'; OUT.mkdir(exist_ok=True)
GROUPS = ['ECC', 'ECC w/o execution feedback', 'ECC w/o screening', 'SAC-MPC']
KEY = {'ECC': 'ecc', 'ECC w/o execution feedback': 'nofb', 'ECC w/o screening': 'noscr', 'SAC-MPC': 'sac'}
rd = lambda f: list(csv.DictReader(open(SRC / f, encoding='utf-8')))

# ---- main table (time-limited condition = 'urgent' rows)
main = [r for r in rd('MAIN_TABLE.csv') if r['condition'] == 'urgent']
wide = rd('URGENT_EPISODES_WIDE.csv')
col = lambda g: g.replace(' ', '_')
print('method & capture & seeds & timeout & unsafe & return & med time & med dv')
for g in GROUPS:
    rows = [r for r in main if r['group'] == g]
    cap = sum(int(r['capture']) for r in rows); to = sum(int(r['timeout']) for r in rows); un = sum(int(r['unsafe']) for r in rows)
    ret = np.mean([float(r['mean_return']) for r in rows])
    succ = [r for r in wide if r[f'{col(g)}_event'] == 'success']
    mt = st.median(float(r[f'{col(g)}_time_s']) for r in succ); mdv = st.median(float(r[f'{col(g)}_delta_v']) for r in succ)
    print(g, cap, [r['capture'] for r in rows], to, un, round(ret, 1), mt, round(mdv, 2))
with open(OUT / 'outcomes.dat', 'w') as f:
    f.write('y name capture timeout unsafe ncap\n')
    for i, g in enumerate(GROUPS):
        rows = [r for r in main if r['group'] == g]
        f.write(f"{i} {KEY[g]} {sum(int(r['capture']) for r in rows)/288*100:.3f} {sum(int(r['timeout']) for r in rows)/288*100:.3f} {sum(int(r['unsafe']) for r in rows)/288*100:.3f} {sum(int(r['capture']) for r in rows)}\n")
with open(OUT / 'seeds.dat', 'w') as f:
    f.write('y seed rate\n')
    for i, g in enumerate(GROUPS):
        for r in main:
            if r['group'] == g:
                f.write(f"{i} {r['seed']} {int(r['capture'])/96*100:.3f}\n")

# ---- paired contingency ECC vs SAC-MPC
both = sum(r['ECC_event'] == 'success' and r['SAC-MPC_event'] == 'success' for r in wide)
eo = sum(r['ECC_event'] == 'success' and r['SAC-MPC_event'] != 'success' for r in wide)
so = sum(r['ECC_event'] != 'success' and r['SAC-MPC_event'] == 'success' for r in wide)
print('contingency both/ecc-only/sac-only/neither', both, eo, so, len(wide) - both - eo - so)

# ---- paired comparisons (forest)
paired = [r for r in rd('PAIRED.csv') if r['condition'] == 'urgent' and r['metric'] == 'capture']
with open(OUT / 'forest.dat', 'w') as f:
    f.write('y est lo hi s20 s21 s22 a b\n')
    order = ['ECC vs SAC-MPC', 'ECC vs ECC w/o screening', 'ECC w/o execution feedback vs SAC-MPC',
             'ECC vs ECC w/o execution feedback', 'ECC w/o screening vs SAC-MPC', 'ECC vs ECC+rigid']
    for i, c in enumerate(order):
        r = next(r for r in paired if r['comparison'] == c)
        f.write(f"{len(order)-1-i} {float(r['estimate'])*100:.3f} {float(r['ci_low'])*100:.3f} {float(r['ci_high'])*100:.3f} "
                f"{float(r['seed20'])/96*100:.3f} {float(r['seed21'])/96*100:.3f} {float(r['seed22'])/96*100:.3f} {r['a_only']} {r['b_only']}\n")
        print(c, r['estimate'], r['ci_low'], r['ci_high'], r['seed20'], r['seed21'], r['seed22'], r['a_only'], r['b_only'])

# ---- training curves: interpolate each seed on a common decision grid; mean and min--max over seeds
tr = rd('TRAINING_CURVES.csv'); grid = np.arange(1000, 4801, 100)
for g in GROUPS:
    caps, ints = [], []
    for s in ('20', '21', '22'):
        rows = [r for r in tr if r['group'] == g and r['seed'] == s]
        x = [float(r['decisions_approx']) for r in rows]
        caps.append(np.interp(grid, x, [float(r['capture_rate']) * 100 for r in rows]))
        ints.append(np.interp(grid, x, [float(r['screen_interventions_per_episode']) for r in rows]))
    caps, ints = np.array(caps), np.array(ints)
    with open(OUT / f'train_{KEY[g]}.dat', 'w') as f:
        f.write('x cap caplo caphi int intlo inthi\n')
        for j, xg in enumerate(grid):
            f.write(f"{xg} {caps[:, j].mean():.3f} {caps[:, j].min():.3f} {caps[:, j].max():.3f} {ints[:, j].mean():.3f} {ints[:, j].min():.3f} {ints[:, j].max():.3f}\n")

# ---- validation curves
va = rd('VALIDATION_CURVES.csv')
for g in ('ECC', 'SAC-MPC'):
    with open(OUT / f'val_{KEY[g]}.dat', 'w') as f:
        f.write('x mean s20 s21 s22\n')
        for d in (1000, 2000, 3000, 4000, 5000):
            v = {r['seed']: int(r['capture']) for r in va if r['group'] == g and int(r['decisions']) == d}
            f.write(f"{d} {np.mean(list(v.values())):.3f} {v['20']} {v['21']} {v['22']}\n")

# ---- reference lines (48 openings) and ECC (288 test tasks)
dev = [r for r in rd('DEV_REFERENCES.csv') if r['condition'] == 'urgent']
names = [('pure', 'pure'), ('b_10', 'unaware'), ('rally', 'rule'), ('c_10_0', 'fixed'), ('gstar', 'gstar')]
with open(OUT / 'reference.dat', 'w') as f:
    f.write('y name capture timeout fuel unsafe ncap\n')
    for i, (p, n) in enumerate(names):
        r = next(r for r in dev if r['policy'] == p); k = int(r['n'])
        f.write(f"{i} {n} {int(r['capture'])/k*100:.3f} {int(r['timeout'])/k*100:.3f} {int(r['resource_exhausted'])/k*100:.3f} {int(r['unsafe'])/k*100:.3f} {r['capture']}/{k}\n")
        print(p, r['capture'], r['timeout'], r['resource_exhausted'], r['unsafe'], r['mean_return'])
    ecc = [r for r in main if r['group'] == 'ECC']
    c = sum(int(r['capture']) for r in ecc); to = sum(int(r['timeout']) for r in ecc); un = sum(int(r['unsafe']) for r in ecc)
    f.write(f"6 ecc {c/288*100:.3f} {to/288*100:.3f} 0.000 {un/288*100:.3f} {c}/288\n")
