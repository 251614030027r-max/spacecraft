"""Plot data of the Section V figures from the round-2 formal evidence (evidence/R2_FORMAL).

Run from figures/src:  python3 make_r2_data.py   -> data/*.dat, data/*.tex (read by fig_res_*.tex) and a summary.
Sources: TRAINING_CURVES.csv (training), STRATIFIED.csv (deadline strata),
traj/traj_<method>_<task>.json (cases; rows every 2 s with margins, errors and wrench; decisions with the coordination record).
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
            if name != 'capture_rate':  # per decision: 100 tasks per window, windows advance by 50 tasks
                win = np.array([xs[0]] + [np.nan] + [xs[i] - xs[i - 2] for i in range(2, len(xs))])
                keep = ~np.isnan(win); xs, y = xs[keep], (y * 100 / win)[keep]
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
    """Mode intervals. A row shows the mode before the check at its instant, so a switch is placed 2 s before the
    first row that shows the new mode (equal to the event times in the decision records)."""
    out, cur, t0 = [], rows[0]['mode'], rows[0]['t']
    for r in rows[1:]:
        if r['mode'] != cur:
            ts = r['t'] - 2.0
            out.append((cur, t0, ts)); cur, t0 = r['mode'], ts
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
NAN = 'nan'
fmt = lambda v, s=1.0, d=4: NAN if v is None else f'{v * s:.{d}f}'
with open(OUT / 'caseA_ts.dat', 'w') as f:
    f.write('t range psi dv perr spd att w clr corr fov rad ax tot fx fy fz tx ty tz\n')
    for r in rows:
        m = r.get('margins') or {}; te = r.get('terminal') or {}
        fov = m.get('fov_margin_deg')
        fF = r.get('force_n') or [None] * 3; tq = r.get('torque_nm') or [None] * 3
        f.write(' '.join([f"{r['t']:.1f}", f"{r['range']:.4f}", f"{r['port_los_deg']:.3f}", f"{r['dv']:.4f}",
                          fmt(te.get('position_error_m')), fmt(te.get('speed_m_s'), d=5),
                          fmt(te.get('attitude_error_rad'), 180 / math.pi, 3), fmt(te.get('angular_velocity_error_rad_s'), d=5),
                          fmt(m.get('collision_clearance_m')), fmt(m.get('union_margin_m')),
                          NAN if fov is None else f'{50.0 - fov:.3f}',
                          fmt(m.get('radial_envelope_margin_m_s')), fmt(m.get('axial_envelope_margin_m_s')),
                          fmt(m.get('total_speed_margin_m_s'))] + [fmt(x) for x in fF] + [fmt(x) for x in tq]) + '\n')
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

# Case B: ECC and SAC-MPC on task 301025 (critical window); rho, latitude offset and psi of both methods
E, S = load('ECC_301025'), load('SAC_MPC_301025')
for dd, key in ((E, 'ecc'), (S, 'sac')):
    with open(OUT / f'caseB_ts_{key}.dat', 'w') as f:
        f.write('t range lat psi\n')
        for r in dd['rows']:
            f.write(f"{r['t']:.1f} {r['range']:.4f} {r['lat_err_deg']:.3f} {r['port_los_deg']:.3f}\n")
bars = []
for dd, y in ((E, 1), (S, 0)):
    for m, a_, b_ in intervals(dd['rows'], dd['time_s']):
        bars.append(f'\\fill[{MODE[m]}!75] (axis cs:{a_:.1f},{y + 0.18:.2f}) rectangle (axis cs:{b_:.1f},{y + 0.82:.2f});')
Td = E['mission']['deadline_s']
(OUT / 'caseB_modes.tex').write_text('\\def\\caseBbars{' + ''.join(bars) + '}' + f'\n\\def\\caseBTd{{{Td:.1f}}}\\def\\caseBTe{{{Td - T_R:.1f}}}\n')
dec = next(x for x in E['decisions'] if x['coord']['kind'] == 'correct')
print('case B correction at', dec['t'], 'proposal', [round(v, 3) for v in dec['coord']['a']], 'exec', dec['coord']['a_exec'],
      'pred', dec['coord']['alt_t'])
for dd, n in ((E, 'ECC'), (S, 'SAC')):
    print(n, dd['event'], dd['time_s'], round(dd['delta_v'], 3), intervals(dd['rows'], dd['time_s']), 'terminal', dd['rows'][-1]['terminal'])
print('Td', Td, 'te', Td - T_R)
