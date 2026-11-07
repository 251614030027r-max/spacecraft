"""P3 summary: critic ranking and actor choice of trained checkpoints on the fixed key states (exploratory).

    python keyscore_summary.py STATES_DIR KEYSCORE.json [OUT.json]

Truth per candidate: the probe's truth branch (the intent held for up to 4 stages, executor be596d9).
Decisive pair: truth returns >= 50 apart. Critic correct / wrong by the sign of min-twin-Q difference
(|dQ| < 1 counted as a tie). Actor: the grid intent nearest to the deterministic action; 'good' if its
truth return is within 50 of the decision's best. Labels are LABEL = {arm}_{seed}_{guide}_{step}."""
import json, glob, sys, itertools
import numpy as np
states, scores = sys.argv[1], json.load(open(sys.argv[2]))
dec = {}
for f in sorted(glob.glob(f'{states}/*.json')):
    r = json.load(open(f))
    for d in r.get('decisions', []):
        if d.get('obs') is not None and d.get('cands'):
            dec[(r['opening'], d['t'])] = d
out = {}
for label, s in scores['scores'].items():
    c = w = t = 0; good = n_act = 0; sep_dec = 0
    for row in s['rows']:
        d = dec[(row['opening'], row['t'])]
        tv = np.array([x['truth']['true'] for x in d['cands']]); q = np.array(row['q'])
        if tv.max() - tv.min() < 50:
            continue
        sep_dec += 1
        for i, j in itertools.combinations(range(len(tv)), 2):
            if abs(tv[i] - tv[j]) < 50: continue
            if abs(q[i] - q[j]) < 1.0: t += 1
            elif (tv[i] - tv[j]) * (q[i] - q[j]) > 0: c += 1
            else: w += 1
        grid = np.array([x['a'] for x in d['cands']])
        k = int(np.argmin(np.linalg.norm(grid - np.asarray(row['actor']), axis=1)))
        n_act += 1; good += int(tv[k] >= tv.max() - 50)
    out[label] = {'separating_decisions': sep_dec, 'pairs_correct': c, 'pairs_wrong': w, 'pairs_tie': t,
                  'pair_accuracy': c / max(c + w, 1), 'actor_good_share': good / max(n_act, 1)}
for k, v in out.items():
    print(k, v)
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], 'w'), indent=1)
