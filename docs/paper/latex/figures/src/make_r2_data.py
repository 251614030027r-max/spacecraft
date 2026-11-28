"""Plot data of the Section V figures (Figs. 6-11) from the round-2 formal evidence and the supplementary assets.

Run from figures/src:  python3 make_r2_data.py   -> data/*.dat, data/*.tex (read by fig_res_*.tex) and a summary.
Sources (repository root):
  evidence/R2_FORMAL/  TRAINING_CURVES.csv (Fig. 6), MAIN_TABLE.csv and EPISODES_WIDE.csv (Fig. 10),
                       STRATIFIED.csv (Fig. 11 deadline and nominal spin bins), traj/ (2-s rows of the cases)
  evidence/R2_ASSETS/  traj_fine/ (0.1-s traces of the cases, Figs. 7-9), GEN_TABLE.csv (Fig. 11 conditions)
Mode intervals come from the 2-s rows: a row shows the mode before the check at its instant, so a switch is placed 2 s
before the first row that shows the new mode; this equals the event times of the decision records.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FORMAL = ROOT / 'evidence' / 'R2_FORMAL'
ASSETS = ROOT / 'evidence' / 'R2_ASSETS'
OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)
rd = lambda p: list(csv.DictReader(open(p, encoding='utf-8')))
T_R = 45.0  # terminal reserve
MODE = {'stage': 'mStage', 'bridge': 'mEntry', 'hold': 'mHold', 'descent': 'mDesc'}
NAN = 'nan'
fmt = lambda v, s=1.0, d=4: NAN if v is None else f'{v * s:.{d}f}'

# ------------------------------------------------------------------ Fig. 6: training capture rate
tc = rd(FORMAL / 'TRAINING_CURVES.csv')
GRID = np.arange(900, 5751, 50)
for cfg, key in (('ECC', 'ecc'), ('SAC_MPC', 'sac'), ('TD3_MPC', 'td3')):
    ys = []
    for seed in ('30', '31', '32'):
        r = [x for x in tc if x['config'] == cfg and x['seed'] == seed]
        xs = np.array([float(x['decisions']) for x in r]); y = np.array([float(x['capture_rate']) for x in r])
        ys.append(np.interp(GRID, xs, y) * 100)
    ys = np.array(ys)
    with open(OUT / f'r2_train_{key}.dat', 'w') as f:
        f.write('x cap caplo caphi\n')
        for i, x in enumerate(GRID):
            f.write(f'{x} {ys[:, i].mean():.3f} {ys[:, i].min():.3f} {ys[:, i].max():.3f}\n')
    print('train', cfg, 'mean capture first/last', round(ys[:, 0].mean(), 1), round(ys[:, -1].mean(), 1))

# ------------------------------------------------------------------ Fig. 10: outcomes, completion time, velocity increment
mt = rd(FORMAL / 'MAIN_TABLE.csv')
ARMS = (('ECC', 'ECC', 'pooled'), ('SAC_MPC', 'SAC_MPC', 'pooled'), ('TD3_MPC', 'TD3_MPC', 'pooled'), ('pure', 'pure', 'fixed'))
with open(OUT / 'r2_outcomes.dat', 'w') as f:
    f.write('x cap to unsafe fuel n\n')
    for i, (arm, _, row) in enumerate(ARMS):
        x = next(r for r in mt if r['arm'] == arm and r['seed'] == row)
        n = int(x['n'])
        v = [int(x[k]) for k in ('capture', 'timeout', 'unsafe', 'resource_exhausted')]
        f.write(f'{i} ' + ' '.join(f'{100 * u / n:.3f}' for u in v) + f' {n}\n')
        print('outcome', arm, v, n)
with open(OUT / 'r2_seedrate.dat', 'w') as f:
    f.write('x rate\n')
    for i, (arm, _, _) in enumerate(ARMS[:3]):
        for s in ('30', '31', '32'):
            x = next(r for r in mt if r['arm'] == arm and r['seed'] == s)
            f.write(f'{i} {100 * int(x["capture"]) / int(x["n"]):.3f}\n')
ew = rd(FORMAL / 'EPISODES_WIDE.csv')


def box(v):
    v = np.sort(np.asarray(v, float)); q1, med, q3 = np.percentile(v, [25, 50, 75]); iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min(); hi = v[v <= q3 + 1.5 * iqr].max()
    return med, q1, q3, lo, hi, v[(v < lo) | (v > hi)]


boxes = []
for i, (arm, col, _) in enumerate(ARMS):
    rows = [r for r in ew if r[f'{col}_event'] == 'success' and (col != 'pure' or r['seed'] == '30')]
    for q, unit in (('time_s', 'time'), ('delta_v', 'dv')):
        med, q1, q3, lo, hi, out = box([float(r[f'{col}_{q}']) for r in rows])
        boxes.append((unit, i, med, q1, q3, lo, hi, len(rows)))
        with open(OUT / f'r2_out_{unit}_{i}.dat', 'w') as f:
            f.write('x y\n')
            for u in out:
                f.write(f'{i} {u:.4f}\n')
    print('box', arm, 'n', len(rows), [round(b[2], 2) for b in boxes[-2:]])
frag = {'time': [], 'dv': []}
for unit, i, med, q1, q3, lo, hi, n in boxes:
    frag[unit].append(f'\\addplot+[boxplot prepared={{draw position={i},median={med:.4f},lower quartile={q1:.4f},'
                      f'upper quartile={q3:.4f},lower whisker={lo:.4f},upper whisker={hi:.4f}}},boxstyle{i}] coordinates {{}};')
    frag[unit].append(f'\\addplot[only marks,mark=*,mark size=0.5pt,gray] table {{data/r2_out_{unit}_{i}.dat}};')
for unit in frag:
    (OUT / f'r2_box_{unit}.tex').write_text('\n'.join(frag[unit]) + '\n')

# ------------------------------------------------------------------ Fig. 11: conditions
st = rd(FORMAL / 'STRATIFIED.csv')
gen = rd(ASSETS / 'GEN_TABLE.csv')
rate = lambda arm, stratum, rng: 100 * next(float(x['capture_rate']) for x in st if x['arm'] == arm and x['stratum'] == stratum and x['range'] == rng)
grate = lambda cond, arm: 100 * next(float(x['capture_rate']) for x in gen if x['condition'] == cond and x['method'] == arm and x['seed'] == 'all')
facets = {
    'deadline': [(b, rate('ECC', 'deadline_s', b), rate('SAC_MPC', 'deadline_s', b)) for b in ('140-155', '155-170', '170-185', '185-200')],
    'spin': [('1.5-2.1', grate('gen_spin_low', 'ECC'), grate('gen_spin_low', 'SAC_MPC'))]
            + [(lab, rate('ECC', 'spin_deg_s', b), rate('SAC_MPC', 'spin_deg_s', b))
               for lab, b in (('2.1-2.3', 'low'), ('2.3-2.5', 'mid'), ('2.5-2.7', 'high'))],
    'axis': [('10-20', grate('nominal', 'ECC'), grate('nominal', 'SAC_MPC')), ('20-40', grate('gen_nutation', 'ECC'), grate('gen_nutation', 'SAC_MPC'))],
}
for k, v in facets.items():
    with open(OUT / f'r2_cond_{k}.dat', 'w') as f:
        f.write('x bin ecc sac\n')
        for i, (lab, e, s) in enumerate(v):
            f.write(f'{i} {lab} {round(e, 1):.1f} {round(s, 1):.1f}\n')
    print('facet', k, [(lab, round(e, 1), round(s, 1)) for lab, e, s in v])
print('spin bin bounds', [(x['range'], round(float(x['spin_min']), 3), round(float(x['spin_max']), 3)) for x in st if x['arm'] == 'ECC' and x['stratum'] == 'spin_deg_s'])


# ------------------------------------------------------------------ cases
def load_rows(name):
    return json.loads((FORMAL / 'traj' / f'traj_{name}.json').read_text(encoding='utf-8'))


def fine(method, task):
    return json.loads((ASSETS / 'traj_fine' / f'traj_{method}_fine_{task}.json').read_text(encoding='utf-8'))


def intervals(rows, t_end):
    out, cur, t0 = [], rows[0]['mode'], rows[0]['t']
    for r in rows[1:]:
        if r['mode'] != cur:
            ts = r['t'] - 2.0
            out.append((cur, t0, ts)); cur, t0 = r['mode'], ts
    out.append((cur, t0, t_end))
    return [(m, a, b) for m, a, b in out if b > a]


def write_trace(path, tr):
    with open(path, 'w') as f:
        f.write('t perr spd att w clr corr fov rad ax tot fx fy fz tx ty tz dv\n')
        for x in tr:
            m, te = x['margins'], x['terminal']
            fov = m.get('fov_margin_deg')
            f.write(' '.join([f"{x['t']:.1f}", fmt(te['position_error_m']), fmt(te['speed_m_s'], d=5),
                              fmt(te['attitude_error_rad'], 180 / math.pi, 3), fmt(te['angular_velocity_error_rad_s'], d=5),
                              fmt(m.get('collision_clearance_m')), fmt(m.get('union_margin_m')),
                              NAN if fov is None else f'{50.0 - fov:.3f}',
                              fmt(m.get('radial_envelope_margin_m_s')), fmt(m.get('axial_envelope_margin_m_s')),
                              fmt(m.get('total_speed_margin_m_s'))]
                             + [fmt(u) for u in x['force_n']] + [fmt(u) for u in x['torque_nm']] + [fmt(x['delta_v'])]) + '\n')


# Task A: ECC on the first selected test task (complete capture)
dA = load_rows('ECC_301001'); fA = fine('ECC', 301001)
rows = dA['rows']; T = dA['time_s']; Td = dA['mission']['deadline_s']
assert abs(fA['time_s'] - T) < 1e-9 and fA['event'] == dA['event']
iv = intervals(rows, T)
with open(OUT / 'caseA_rows.dat', 'w') as f:
    f.write('t range psi lat\n')
    for r in rows:
        f.write(f"{r['t']:.1f} {r['range']:.4f} {r['port_los_deg']:.3f} {r['lat_err_deg']:.3f}\n")
write_trace(OUT / 'caseA_fine.dat', fA['trace'])
P = [np.asarray(r['rot']).T @ np.asarray(r['q']) for r in rows]
frag = []
for k, (m, a, b) in enumerate(iv):
    idx = [i for i, r in enumerate(rows) if a - 1e-6 <= r['t'] <= b + 1e-6]
    with open(OUT / f'caseA_seg{k}.dat', 'w') as f:
        f.write('x y z\n')
        for i in idx:
            f.write(f'{P[i][0]:.4f} {P[i][1]:.4f} {P[i][2]:.4f}\n')
    frag.append(f'\\addplot3[{MODE[m]},line width=1.1pt] table{{data/caseA_seg{k}.dat}};')
(OUT / 'caseA_traj.tex').write_text('\n'.join(frag) + '\n')


def at(t):
    i = min(range(len(rows)), key=lambda j: abs(rows[j]['t'] - t))
    return P[i]


marks = [('start', 0.0, 'north west'), ('entry', iv[1][1], 'west'), ('descent', iv[2][1], 'north east'), ('capture', T, 'south east')]
mk = []
for lab, t, anc in marks:
    p = at(t)
    mk.append(f'\\addplot3[only marks,mark=*,mark size=1.4pt,black] coordinates {{({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})}};')
    mk.append(f'\\node[anchor={anc},inner sep=1.5pt,font=\\scriptsize] at (axis cs:{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) {{{lab}}};')
(OUT / 'caseA_marks.tex').write_text('\n'.join(mk) + '\n')
# translucent operation sphere and approach cone as single patches (silhouettes for the view of Fig. 7(a))
AZ, EL = math.radians(-35), math.radians(38)
dview = np.array([math.sin(AZ) * math.cos(EL), -math.cos(AZ) * math.cos(EL), math.sin(EL)])
u = np.cross(dview, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(dview, u)
sil = [6 * (math.cos(s) * u + math.sin(s) * v) for s in np.linspace(0, 2 * math.pi, 73)]
apex, depth, tn = np.array([-1.5, 0, 0]), 9.0, math.tan(math.radians(35))
base = [apex + np.array([-depth, depth * tn * math.cos(s), depth * tn * math.sin(s)]) for s in np.linspace(0, 2 * math.pi, 145)]
scr = lambda p: np.array([np.dot(p, u), np.dot(p, v)])
pts = [apex] + base
S2 = np.array([scr(p) for p in pts])
# convex hull (monotone chain) in screen coordinates
order = sorted(range(len(pts)), key=lambda i: (S2[i][0], S2[i][1]))
cross2 = lambda o, a, b: (S2[a][0] - S2[o][0]) * (S2[b][1] - S2[o][1]) - (S2[a][1] - S2[o][1]) * (S2[b][0] - S2[o][0])
lower, upper = [], []
for i in order:
    while len(lower) >= 2 and cross2(lower[-2], lower[-1], i) <= 0: lower.pop()
    lower.append(i)
for i in reversed(order):
    while len(upper) >= 2 and cross2(upper[-2], upper[-1], i) <= 0: upper.pop()
    upper.append(i)
hull = lower[:-1] + upper[:-1]
c3 = lambda p: f'({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})'
(OUT / 'caseA_geom.tex').write_text(
    '\\addplot3[draw=gray!55,thin,fill=gray,fill opacity=0.07] coordinates {' + ' '.join(c3(p) for p in sil) + '};\n'
    + '\\addplot3[draw=none,fill=cDesc,fill opacity=0.13] coordinates {' + ' '.join(c3(pts[i]) for i in hull + [hull[0]]) + '};\n'
    + '\\addplot3[cDesc!70!black,thin] coordinates {' + ' '.join(c3(p) for p in base) + '};\n'
    + '\\addplot3[cDesc!70!black,thin] coordinates {' + c3(pts[hull[hull.index(0) - 1]]) + ' ' + c3(apex) + ' '
    + c3(pts[hull[(hull.index(0) + 1) % len(hull)]]) + '};\n')
(OUT / 'caseA_modes.tex').write_text(''.join(f'\\modeband{{{a:.1f}}}{{{b:.1f}}}{{{MODE[m]}}}\n' for m, a, b in iv))
rt = ' '.join(f'{x["t"]:.1f}' for x in dA['window_recoveries'])
(OUT / 'caseA_events.tex').write_text(f'\\def\\caseAT{{{T:.1f}}}\\def\\caseATd{{{Td:.1f}}}\\def\\caseAretime{{{rt}}}\n')
tr = fA['trace']
mins = {k: min(x['margins'][k] for x in tr if x['margins'].get(k) is not None) for k in
        ('collision_clearance_m', 'union_margin_m', 'radial_envelope_margin_m_s', 'axial_envelope_margin_m_s', 'total_speed_margin_m_s')}
print('case A', dA['event'], T, 'Td', Td, 'dv', round(dA['delta_v'], 3), 'modes', iv, 're-timing', rt)
print('  minima', {k: round(v, 4) for k, v in mins.items()}, 'max view angle', round(max(50 - x['margins']['fov_margin_deg'] for x in tr), 2))
print('  final errors', tr[-1]['terminal'])
print('  entry range', round(next(r['range'] for r in rows if r['t'] >= iv[1][1]), 2))

# Task B: ECC and SAC-MPC on the second selected test task (critical window)
E, S = load_rows('ECC_301025'), load_rows('SAC_MPC_301025')
for dd, key in ((E, 'ecc'), (S, 'sac')):
    with open(OUT / f'caseB_ts_{key}.dat', 'w') as f:
        f.write('t range lat psi\n')
        for r in dd['rows']:
            f.write(f"{r['t']:.1f} {r['range']:.4f} {r['lat_err_deg']:.3f} {r['port_los_deg']:.3f}\n")
bars = []
for dd, y in ((E, 1), (S, 0)):
    for m, a_, b_ in intervals(dd['rows'], dd['time_s']):
        bars.append(f'\\fill[{MODE[m]}!75] (axis cs:{a_:.1f},{y + 0.2:.2f}) rectangle (axis cs:{b_:.1f},{y + 0.8:.2f});')
TdB = E['mission']['deadline_s']
fE, fS = fine('ECC', 301025), fine('SAC_MPC', 301025)
(OUT / 'caseB_modes.tex').write_text('\\def\\caseBbars{' + ''.join(bars) + '}'
                                     + f'\n\\def\\caseBTd{{{TdB:.1f}}}\\def\\caseBTe{{{TdB - T_R:.1f}}}'
                                     + f'\\def\\caseBendE{{{fE["trace"][-1]["terminal"]["position_error_m"]:.2f}}}'
                                     + f'\\def\\caseBendS{{{fS["trace"][-1]["terminal"]["position_error_m"]:.2f}}}\n')
dec = next(x for x in E['decisions'] if x['coord']['kind'] == 'correct')
print('case B correction at', dec['t'], 'proposal', [round(v, 3) for v in dec['coord']['a']], 'exec', dec['coord']['a_exec'], 'pred', dec['coord']['alt_t'])
for dd, ff, n in ((E, fE, 'ECC'), (S, fS, 'SAC')):
    print(n, dd['event'], dd['time_s'], round(dd['delta_v'], 3), intervals(dd['rows'], dd['time_s']),
          'end distance', round(ff['trace'][-1]['terminal']['position_error_m'], 3))
print('Td', TdB, 'te', TdB - T_R)
