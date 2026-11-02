"""Self-check 1: the training module's DecisionCalibration on P4 data (online on selection openings, frozen on held-out)."""
import sys, json, glob
sys.path.insert(0, '/home/user/spacecraft')
import numpy as np
import train.t2_mve_sac as t
E = 'evidence/T2_8_P4_INFO'
rows = []
for f in sorted(glob.glob(f'{E}/info/*.json')):
    r = json.load(open(f)); o = json.load(open(f.replace('/info/', '/opp2/')))
    od = {round(d['t'], 3): d for d in o['decisions']}
    for d in r['decisions']:
        dd = od[round(d['t'], 3)]
        for c, oc in zip(d['cands'], dd['cands']):
            op = oc['opp2']
            f7 = {'remaining_s': d['left_s'] - 16.0, 'range_m': 15.0 * d['obs'][0], 'urgent': d['urgent'], 'feasible': op['feasible'],
                  'tau': op.get('tau'), 'p_success': op.get('p_success', 0.0)}
            rows.append((r['opening'], t.dec_features(f7), c['truth_track']['event'] == 'success', bool(op['feasible'])))
rng = np.random.default_rng(0)
sel = [x for x in rows if x[0] < 2_910_020]; held = [x for x in rows if x[0] >= 2_910_020]
order = rng.permutation(len(sel))
cal = t.DecisionCalibration()
for i in order:
    cal.observe(sel[i][1], sel[i][2])
cal.fit()
w = cal.w
acc_h = np.mean([(float(np.asarray(x[1]) @ w) > 0) == x[2] for x in held])
raw_h = np.mean([x[3] == x[2] for x in held]); maj = max(np.mean([x[2] for x in held]), 1 - np.mean([x[2] for x in held]))
print(f'selection outcomes {len(sel)}: online predict-then-observe accuracy {cal.accuracy:.3f}, trusted {cal.trusted}')
print(f'held-out {len(held)}: calibrated {acc_h:.3f}  raw feasible {raw_h:.3f}  majority {maj:.3f}')
print('weights', np.round(w, 2).tolist())
json.dump({'w': w.tolist()}, open('/tmp/claude-0/-home-user-spacecraft/f6129d7c-5481-5ac5-9cc8-4c96d2c6d0ab/scratchpad/cal_w.json', 'w'))
