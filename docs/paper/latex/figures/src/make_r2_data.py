"""Plot data of the Section V figures from the round-2 formal evidence (evidence/R2_FORMAL).

Run from figures/src:  python3 make_r2_data.py   -> data/*.dat, data/*.tex (read by fig_res_*.tex) and a summary.
Sources: TRAINING_CURVES.csv (training), PAIRED.csv (paired comparisons), STRATIFIED.csv (deadline strata),
traj/traj_<method>_<task>.json (cases; rows every 2 s, decisions with the coordination record).
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[4] / 'evidence' / 'R2_FORMAL'
OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)
rd = lambda f: list(csv.DictReader(open(SRC / f, encoding='utf-8')))
T_R = 45.0  # terminal reserve of Table V
MODE = {'stage': 'mStage', 'bridge': 'mEntry', 'hold': 'mHold', 'descent': 'mDesc'}

# ------------------------------------------------------------------ training curves
tc = rd('TRAINING_CURVES.csv')
GRID = np.arange(900, 5751, 50)
for cfg, key in (('ECC', 'ecc'), ('SAC_MPC', 'sac'), ('TD3_MPC', 'td3')):
    cols = {}
    for name in ('capture_rate', 'corrections_per_episode', 'long_entries_per_episode'):
        ys = []
        for seed in ('30', '31', '32'):
            r = [x for x in tc if x['config'] == cfg and x['seed'] == seed]
            xs = np.array([float(x['decisions']) for x in r]); y = np.array([float(x[name]) for x in r])
            ys.append(np.interp(GRID, xs, y))
        ys = np.array(ys)
        cols[name] = (ys.mean(0), ys.min(0), ys.max(0))
    with open(OUT / f'r2_train_{key}.dat', 'w') as f:
        f.write('x cap caplo caphi cor corlo corhi lng lnglo lnghi\n')
        for i, x in enumerate(GRID):
            v = [cols['capture_rate'][j][i] * 100 for j in range(3)] + [cols['corrections_per_episode'][j][i] for j in range(3)] \
                + [cols['long_entries_per_episode'][j][i] for j in range(3)]
            f.write(f'{x} ' + ' '.join(f'{u:.4f}' for u in v) + '\n')
    print('train', cfg, 'first/last mean capture', round(cols['capture_rate'][0][0], 3), round(cols['capture_rate'][0][-1], 3))

# ------------------------------------------------------------------ paired comparisons (ECC vs baselines)
pr = rd('PAIRED.csv')
rows = [('ECC vs SAC_MPC', 'SAC--MPC'), ('ECC vs TD3_MPC', 'TD3--MPC'), ('ECC vs pure', 'MPC')]
with open(OUT / 'r2_forest.dat', 'w') as f:
    f.write('y est lo hi s30 s31 s32 a b\n')
    for i, (cmp_, name) in enumerate(rows):
        r = next(x for x in pr if x['comparison'] == cmp_ and x['metric'] == 'capture')
        lo, hi = json.loads(r['ci95']); ps = json.loads(r['per_seed']); fl = json.loads(r['flips'])
        f.write(f"{2 - i} {float(r['estimate'])*100:.3f} {lo*100:.3f} {hi*100:.3f} "
                f"{ps['30']/192*100:.3f} {ps['31']/192*100:.3f} {ps['32']/192*100:.3f} {fl['a_only']} {fl['b_only']}\n")
        print('paired', cmp_, round(float(r['estimate']) * 100, 1), [round(lo * 100, 1), round(hi * 100, 1)], ps, fl)

# ------------------------------------------------------------------ deadline strata
st = rd('STRATIFIED.csv')
bins = ['140-155', '155-170', '170-185', '185-200']
with open(OUT / 'r2_strata.dat', 'w') as f:
    f.write('x bin ecc sac td3 mpc\n')
    for i, b in enumerate(bins):
        v = [next(float(x['capture_rate']) for x in st if x['arm'] == a and x['stratum'] == 'deadline_s' and x['range'] == b) * 100
             for a in ('ECC', 'SAC_MPC', 'TD3_MPC', 'pure')]
        f.write(f'{i} {b} ' + ' '.join(f'{u:.2f}' for u in v) + '\n')
        print('stratum', b, [round(u, 1) for u in v])


# ------------------------------------------------------------------ cases
def load(name):
    return json.loads((SRC / 'traj' / f'traj_{name}.json').read_text(encoding='utf-8'))


def intervals(rows, t_end):
    """Mode intervals; a switch is placed at the first 2-s row that shows the new mode."""
    out, cur, t0 = [], rows[0]['mode'], rows[0]['t']
    for r in rows[1:]:
        if r['mode'] != cur:
            out.append((cur, t0, r['t'])); cur, t0 = r['mode'], r['t']
    out.append((cur, t0, t_end))
    return [(m, a, b) for m, a, b in out if b > a]


def passes(rows, gate=35.0):
    """Intervals in which the port--line-of-sight angle is within the entry gate (linear interpolation)."""
    out, t0 = [], None
    for a, b in zip(rows, rows[1:]):
        pa, pb = a['port_los_deg'] - gate, b['port_los_deg'] - gate
        if t0 is None and pa > 0 >= pb:
            t0 = a['t'] + (b['t'] - a['t']) * pa / (pa - pb)
        elif t0 is not None and pa <= 0 < pb:
            out.append((t0, a['t'] + (b['t'] - a['t']) * (-pa) / (pb - pa))); t0 = None
    if rows[0]['port_los_deg'] <= gate and not out and t0 is None:
        t0 = rows[0]['t']
    if t0 is not None:
        out.append((t0, rows[-1]['t']))
    return out


# Case A: ECC on task 301001 (complete capture)
d = load('ECC_301001'); rows = d['rows']; T = d['time_s']; Td = d['mission']['deadline_s']
iv = intervals(rows, T)
with open(OUT / 'caseA_ts.dat', 'w') as f:
    f.write('t range psi vmg att w dv\n')
    for r in rows:
        m = r.get('margins') or {}; te = r.get('terminal') or {}
        vmg = m.get('radial_envelope_margin_m_s'); att = te.get('attitude_error_rad'); w = te.get('angular_velocity_error_rad_s')
        f.write(f"{r['t']:.1f} {r['range']:.4f} {r['port_los_deg']:.3f} {vmg if vmg is not None else 'nan'} "
                f"{math.degrees(att) if att is not None else 'nan'} {w if w is not None else 'nan'} {r['dv']:.4f}\n")
P = [np.asarray(r['rot']).T @ np.asarray(r['q']) for r in rows]
frag = []
for k, (m, a, b) in enumerate(iv):
    idx = [i for i, r in enumerate(rows) if a <= r['t'] <= b]
    with open(OUT / f'caseA_seg{k}.dat', 'w') as f:
        f.write('x y z\n')
        for i in idx:
            f.write(f'{P[i][0]:.4f} {P[i][1]:.4f} {P[i][2]:.4f}\n')
    frag.append(f'\\addplot3[{MODE[m]},line width=1.0pt] table{{data/caseA_seg{k}.dat}};')
(OUT / 'caseA_traj.tex').write_text('\n'.join(frag) + '\n')
p0 = P[0]
(OUT / 'caseA_marks.tex').write_text(
    f'\\addplot3[only marks,mark=*,mark size=1.6pt,black] coordinates {{({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f})}};\n'
    f'\\node[anchor=north,inner sep=2pt,font=\\scriptsize] at (axis cs:{p0[0]:.3f},{p0[1]:.3f},{p0[2]-0.3:.3f}) {{start}};\n')
(OUT / 'caseA_modes.tex').write_text(''.join(f'\\modeband{{{a:.1f}}}{{{b:.1f}}}{{{MODE[m]}}}\n' for m, a, b in iv))
rt = ''.join(f'({x["t"]:.1f},YY) ' for x in d['window_recoveries'])
(OUT / 'caseA_events.tex').write_text(
    f'\\def\\caseAT{{{T:.1f}}}\\def\\caseATd{{{Td:.1f}}}\\def\\caseAretime{{{rt.strip()}}}\n')
print('case A', d['event'], T, 'Td', Td, 'dv', round(d['delta_v'], 3), 'modes', iv, 'gate intervals', [(round(a, 1), round(b, 1)) for a, b in passes(rows)])
print('  decisions', [(x['t'], x['coord']['kind'], x['coord'].get('own_t')) for x in d['decisions']], 're-timing', [(x['t'], x['slip_s']) for x in d['window_recoveries']])
print('  terminal', rows[-1]['terminal'])
pre = [r for r in rows if r['mode'] != 'descent' and (r.get('margins') or {}).get('radial_envelope_margin_m_s') is not None]
print('  min vmg before descent', round(min(r['margins']['radial_envelope_margin_m_s'] for r in pre), 4),
      'max psi in descent', round(max(r['port_los_deg'] for r in rows if r['mode'] == 'descent'), 2),
      'range at start', round(rows[0]['range'], 2), 'lat at start', round(rows[0]['lat_err_deg'], 1))
s = load('SAC_MPC_301001')
print('  SAC', s['event'], s['time_s'], round(s['delta_v'], 3), intervals(s['rows'], s['time_s']))

# Case B: ECC and SAC-MPC on task 301025 (critical window)
E, S = load('ECC_301025'), load('SAC_MPC_301025')
dec = next(x for x in E['decisions'] if x['coord']['kind'] == 'correct')
c = dec['coord']
with open(OUT / 'caseB_cands.dat', 'w') as f:
    f.write('ap as crit tau\n')
    for cd in c['cands']:
        f.write(f"{cd['b'][0]:.0f} {cd['b'][1]:.0f} {1 if cd['critical'] else 0} {cd['t']:.0f}\n")
(OUT / 'caseB_point.tex').write_text(
    f"\\def\\propA{{{c['a'][0]:.4f}}}\\def\\propB{{{c['a'][1]:.4f}}}\\def\\execA{{{c['a_exec'][0]:.0f}}}\\def\\execB{{{c['a_exec'][1]:.0f}}}\\def\\decT{{{dec['t']:.0f}}}\n")
print('case B correction at', dec['t'], 'proposal', [round(v, 3) for v in c['a']], 'exec', c['a_exec'], 'pred', c['alt_t'],
      'critical', [(cd['b'], cd['t']) for cd in c['cands'] if cd['critical']])


def timeline(dd, y, screened):
    rows = dd['rows']; T = dd['time_s']; s = []
    for m, a, b in intervals(rows, T):
        s.append(f'\\tlbar{{{a:.1f}}}{{{b:.1f}}}{{{y}}}{{{MODE[m]}}}')
    for x in dd['decisions']:
        cc = x.get('coord') or {}
        k = cc.get('kind')
        if screened and k == 'correct':
            s.append(f'\\tldecS{{{x["t"]:.1f}}}{{{y}}}')
            s.append(f'\\tlpred{{{x["t"]:.1f}}}{{{x["t"] + cc["alt_t"]:.1f}}}{{{y}}}')
        elif screened and k == 'keep':
            s.append(f'\\tldecK{{{x["t"]:.1f}}}{{{y}}}')
            s.append(f'\\tlpredK{{{x["t"]:.1f}}}{{{x["t"] + cc["own_t"]:.1f}}}{{{y}}}')
        else:
            s.append(f'\\tldec{{{x["t"]:.1f}}}{{{y}}}')
    for t0, t1 in passes(rows):
        s.append(f'\\tlwin{{{t0:.1f}}}{{{t1:.1f}}}{{{y}}}')
    s.append(('\\tlcap' if dd['event'] == 'success' else '\\tlfail') + f'{{{T:.1f}}}{{{y}}}')
    return s, intervals(rows, T), passes(rows)


sE, ivE, pE = timeline(E, 1, True)
sS, ivS, pS = timeline(S, 0, False)
Td = E['mission']['deadline_s']
(OUT / 'caseB_timeline.tex').write_text('\n'.join(sE + sS) + f'\n\\def\\tlTd{{{Td:.1f}}}\\def\\tlTe{{{Td - T_R:.1f}}}\n')
print('ECC 301025', E['event'], E['time_s'], round(E['delta_v'], 3), ivE, 'gate', [(round(a, 1), round(b, 1)) for a, b in pE])
print('  decisions', [(x['t'], x['coord']['kind'], x['coord'].get('own_t'), x['coord'].get('alt_t')) for x in E['decisions']],
      're-timing', E['window_recoveries'])
print('SAC 301025', S['event'], S['time_s'], round(S['delta_v'], 3), ivS, 'gate', [(round(a, 1), round(b, 1)) for a, b in pS])
print('  terminal', S['rows'][-1]['terminal'], 'range at 52 s ECC', [round(r['range'], 2) for r in E['rows'] if r['t'] == 52.0])
print('Td', Td, 'te', Td - T_R)
