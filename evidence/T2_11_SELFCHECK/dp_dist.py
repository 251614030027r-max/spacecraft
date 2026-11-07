import sys, json, glob
sys.path.insert(0, '/home/user/spacecraft')
import numpy as np
import train.t2_mve_sac as t
w = np.array(json.load(open(sys.argv[1]))['w'])
E = 'evidence/T2_8_P4_INFO'
S = json.load(open(f'{E}/KEYSCORE_P4.json'))['scores']
info = {}; o2 = {}
for f in glob.glob(f'{E}/info/*.json'):
    r = json.load(open(f)); o = json.load(open(f.replace('/info/', '/opp2/')))
    od = {round(d['t'], 3): d for d in o['decisions']}
    for d in r['decisions']: info[(r['opening'], round(d['t'], 3))] = (d, od[round(d['t'], 3)])
def P(d, oc):
    op = oc['opp2']
    f7 = {'remaining_s': d['left_s'] - 16.0, 'range_m': 15.0 * d['obs'][0], 'urgent': d['urgent'], 'feasible': op['feasible'], 'tau': op.get('tau'), 'p_success': op.get('p_success', 0.0)}
    return 1 / (1 + np.exp(-np.asarray(t.dec_features(f7)) @ w))
for lab in ('e_sac_s0_500', 'e_sac_s1_500', 'e_sac_s0_1000', 'e_mve_s1_500', 'e_mve_s0_2000'):
    dps = []; lost = []
    for row in S[lab]['rows']:
        d, od = info[(row['opening'], round(row['t'], 3))]
        p = np.array([P(d, oc) for oc in od['cands']]); grid = np.array([c['a'] for c in d['cands']])
        ka = int(np.argmin(np.linalg.norm(grid - np.asarray(row['actor']), axis=1)))
        tv = np.array([c['truth_track']['true'] for c in d['cands']]); ev = [c['truth_track']['event'] for c in d['cands']]
        dps.append(p.max() - p[ka]); lost.append(ev[ka] != 'success' and 'success' in ev)
    dps = np.array(dps); lost = np.array(lost)
    print(lab, 'decisions', len(dps), 'dP>=0.4', int((dps >= 0.4).sum()), 'dP>=0.2', int((dps >= 0.2).sum()), '| actor lost capture', int(lost.sum()),
          'of which dP>=0.2', int((lost & (dps >= 0.2)).sum()), 'dP>=0.4', int((lost & (dps >= 0.4)).sum()))
