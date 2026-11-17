"""Case-figure data from the replayed trajectories (evidence/PAPER_FIGS) -> figures/src/data/case_*.

Representative capture: ECC on task 300000.  Screening case: ECC and SAC-MPC on task 300005, decision at 48 s.
Rows are recorded every 2 s; position in the target body frame is R^T q with R the target attitude.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[4] / 'evidence' / 'PAPER_FIGS'
OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)
MODE = {'stage': 'mStage', 'bridge': 'mEntry', 'hold': 'mHold', 'descent': 'mDesc', 'pure': 'mStage'}
T_R = 45.0  # latest-entry reserve of Table IV


def load(name):
    return json.loads((SRC / f'traj_{name}.json').read_text(encoding='utf-8'))


def intervals(rows, t_end, decisions=()):
    """Mode intervals [t0, t1) from the 2-s rows; a mode switch is placed at the first row showing it."""
    out, cur, t0 = [], rows[0]['mode'], rows[0]['t']
    for r in rows[1:]:
        if r['mode'] != cur:
            out.append((cur, t0, r['t'])); cur, t0 = r['mode'], r['t']
    out.append((cur, t0, t_end))
    out = [list(x) for x in out]
    # place entry and descent onsets at the event times logged by the decisions (rows are 2 s apart)
    for i in range(1, len(out)):
        m, a = out[i][0], out[i][1]
        tev = [x['true_mode_event_t'] for x in decisions if x.get('true_mode_event_t') is not None
               and x.get('macro_end') == {'bridge': 'entry', 'descent': 'descent'}.get(m)]
        tev = [x for x in tev if a - 2.01 <= x <= a]
        if tev:
            out[i][1] = out[i - 1][2] = tev[0]
    return [(m, a, b) for m, a, b in out if b > a]


def passes(rows):
    """Port passes: the remaining phase wraps from small positive to near 360 deg."""
    ts = []
    for a, b in zip(rows, rows[1:]):
        if a['phase_deg'] < 60 and b['phase_deg'] > 300:
            pa, pb = a['phase_deg'], b['phase_deg'] - 360.0
            ts.append(a['t'] + (b['t'] - a['t']) * pa / (pa - pb))
    return ts


# ---------------------------------------------------------------- representative capture (ECC, 300000)
d = load('ECC_300000'); rows = d['rows']; T = d['time_s']; Td = d['mission']['deadline_s']
iv = intervals(rows, T, d['decisions'])
with open(OUT / 'case_ts.dat', 'w') as f:
    f.write('t range psi vmg att w dv\n')
    for r in rows:
        m = r.get('margins') or {}; te = r.get('terminal') or {}
        vmg = m.get('radial_envelope_margin_m_s'); att = te.get('attitude_error_rad'); w = te.get('angular_velocity_error_rad_s')
        nan = 'nan'
        f.write(f"{r['t']:.1f} {r['range']:.4f} {r['port_los_deg']:.3f} {vmg if vmg is not None else nan} "
                f"{math.degrees(att) if att is not None else nan} {w if w is not None else nan} {r['dv']:.4f}\n")
# trajectory segments in the body frame, one file per contiguous mode
P = [np.asarray(r['rot']).T @ np.asarray(r['q']) for r in rows]
frag = []
k = 0
for m, a, b in iv:
    idx = [i for i, r in enumerate(rows) if a <= r['t'] <= b]
    if idx and idx[-1] + 1 < len(rows) and rows[idx[-1]]['t'] < b:
        idx.append(idx[-1] + 1)
    with open(OUT / f'case_seg{k}.dat', 'w') as f:
        f.write('x y z\n')
        for i in idx:
            f.write(f'{P[i][0]:.4f} {P[i][1]:.4f} {P[i][2]:.4f}\n')
    frag.append(f'\\addplot3[{MODE[m]},line width=1.0pt] table{{data/case_seg{k}.dat}};')
    k += 1
(OUT / 'case_traj.tex').write_text('\n'.join(frag) + '\n')
p0, pT = P[0], P[-1]
ev = {d2['t']: d2 for d2 in d['decisions']}
entry_t = next(x['true_mode_event_t'] for x in d['decisions'] if x.get('macro_end') == 'entry')
desc_t = next(x['true_mode_event_t'] for x in d['decisions'] if x.get('macro_end') == 'descent')
(OUT / 'case_marks.tex').write_text(
    f'\\addplot3[only marks,mark=*,mark size=1.6pt,black] coordinates {{({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f})}};\n'
    f'\\node[anchor=north,inner sep=2pt,font=\\scriptsize] at (axis cs:{p0[0]:.3f},{p0[1]:.3f},{p0[2]-0.3:.3f}) {{start}};\n')
bands = ''.join(f'\\modeband{{{a:.1f}}}{{{b:.1f}}}{{{MODE[m]}}}\n' for m, a, b in iv)
(OUT / 'case_modes.tex').write_text(bands)
iv_ = [x for x in d['decisions'] if (x.get('cert') or {}).get('intervened')]
print('case 300000: event', d['event'], 'T', T, 'Td', Td, 'dv', round(d['delta_v'], 3), 'return', round(d['return'], 2))
print('  modes', [(m, a, b) for m, a, b in iv])
print('  entry event', entry_t, 'descent event', desc_t, 'passes', [round(x, 1) for x in passes(rows)],
      're-timing', d['window_recoveries'], 'interventions', [(x['t'], x['cert']['a_exec'], x['t'] + x['cert']['alt_t']) for x in iv_])
pre = [r for r in rows if r['mode'] != 'descent' and (r.get('margins') or {}).get('radial_envelope_margin_m_s') is not None]
print('  start range', round(rows[0]['range'], 2), 'lat', round(rows[0]['lat_err_deg'], 1),
      '| at entry: range', [round(r['range'], 2) for r in rows if r['t'] == 64.0], 'lat', [round(r['lat_err_deg'], 1) for r in rows if r['t'] == 64.0])
print('  min vmg before descent', round(min(r['margins']['radial_envelope_margin_m_s'] for r in pre), 4),
      '| max psi in descent', round(max(r['port_los_deg'] for r in rows if r['mode'] == 'descent'), 2))
print('  terminal', d['rows'][-1]['terminal'])

# ---------------------------------------------------------------- screening case (300005)
E, S = load('ECC_300005'), load('SAC_300005')
dec = next(x for x in E['decisions'] if x['t'] == 48.0)
c = dec['cert']
with open(OUT / 'screen_cands.dat', 'w') as f:
    f.write('ap as adm tau\n')
    for cd in c['cands']:
        f.write(f"{cd['b'][0]:.0f} {cd['b'][1]:.0f} {1 if cd['certified'] else 0} {cd['t']:.0f}\n")
(OUT / 'screen_point.tex').write_text(
    f"\\def\\propA{{{c['a'][0]:.4f}}}\\def\\propB{{{c['a'][1]:.4f}}}\\def\\execA{{{c['a_exec'][0]:.0f}}}\\def\\execB{{{c['a_exec'][1]:.0f}}}\n")
print('screen decision 48 s: proposal', [round(x, 3) for x in c['a']], 'exec', c['a_exec'], 'pred', c['alt_t'],
      'admissible', [(cd['b'], cd['t']) for cd in c['cands'] if cd['certified']])

def timeline(dd, y, screened):
    rows = dd['rows']; T = dd['time_s']; s = []
    for m, a, b in intervals(rows, T, dd['decisions']):
        s.append(f'\\tlbar{{{a:.1f}}}{{{b:.1f}}}{{{y}}}{{{MODE[m]}}}')
    for x in dd['decisions']:
        cc = x.get('cert') or {}
        if screened and cc.get('intervened'):
            s.append(f'\\tldecS{{{x["t"]:.1f}}}{{{y}}}')
            s.append(f'\\tlpred{{{x["t"]:.1f}}}{{{x["t"] + cc["alt_t"]:.1f}}}{{{y}}}')
        else:
            s.append(f'\\tldec{{{x["t"]:.1f}}}{{{y}}}')
    for tp in passes(rows):
        s.append(f'\\tlpass{{{tp:.1f}}}{{{y}}}')
    s.append(('\\tlcap' if dd['event'] == 'success' else '\\tlfail') + f'{{{T:.1f}}}{{{y}}}')
    return s, intervals(rows, T, dd['decisions']), passes(rows)

sE, ivE, pE = timeline(E, 1, True)
sS, ivS, pS = timeline(S, 0, False)
Td = E['mission']['deadline_s']
(OUT / 'screen_timeline.tex').write_text('\n'.join(sE + sS) + f'\n\\def\\tlTd{{{Td:.1f}}}\\def\\tlTe{{{Td - T_R:.1f}}}\n')
print('ECC 300005', E['event'], E['time_s'], round(E['delta_v'], 3), ivE, 'passes', [round(x, 1) for x in pE], 're-timing', E['window_recoveries'])
print('  interventions', [(x['t'], [round(v, 2) for v in x['cert']['a']], x['cert']['a_exec'], x['t'] + x['cert']['alt_t']) for x in E['decisions'] if x['cert'].get('intervened')])
print('  realized events', [(x['t'], x.get('macro_end'), x.get('true_mode_event_t')) for x in E['decisions']])
print('  terminal', E['rows'][-1]['terminal'])
print('SAC 300005', S['event'], S['time_s'], round(S['delta_v'], 3), ivS, 'passes', [round(x, 1) for x in pS])
print('  intents', [(x['t'], [round(v, 2) for v in x['a']]) for x in S['decisions']])
print('  terminal', S['rows'][-1]['terminal'])
for t in (48.0, 56.0, 64.0):
    print('  range at', t, 'ECC', [round(r['range'], 2) for r in E['rows'] if r['t'] == t], 'SAC', [round(r['range'], 2) for r in S['rows'] if r['t'] == t])
print('Td', Td, 'te', Td - T_R)
