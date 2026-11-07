"""P1 / P2 summary of the key-state probe (sandbox, exploratory)."""
import json, glob, sys, itertools
import numpy as np
D = sys.argv[1]
dec = []
for f in sorted(glob.glob(f'{D}/*.json')):
    r = json.load(open(f))
    for d in r.get('decisions', []):
        dec.append((r['opening'], r['event'], d))
def mscore(c):
    v = None; ev = None
    for m in c['model']:
        if m is None: break
        if m.get('value') is not None: v = m['value']
        ev = m.get('event')
        if ev is not None: break
    return v, ev
n_sep = n_dec = 0; corr = wrong = tie = 0; unk = 0
spread = []; mind = []; rows = []
for op, epi, d in dec:
    cs = d['cands']
    if len(cs) < 2: continue
    n_dec += 1
    tv = np.array([c['truth']['true'] for c in cs]); ev = [c['truth']['event'] for c in cs]
    sep = (tv.max() - tv.min() >= 50) or (len(set(e == 'success' for e in ev)) > 1)
    n_sep += sep; spread.append(tv.max() - tv.min())
    q1 = np.array([c['truth']['stages'][0]['q'] for c in cs])
    dd = [np.linalg.norm(a - b) for a, b in itertools.combinations(q1, 2)]
    mind.append((min(dd), float(np.median(dd))))
    ms = [mscore(c) for c in cs]
    c_ = w_ = t_ = 0
    for i, j in itertools.combinations(range(len(cs)), 2):
        if abs(tv[i] - tv[j]) < 50: continue
        mi, mj = ms[i][0], ms[j][0]
        if mi is None or mj is None: unk += 1; continue
        s = np.sign(tv[i] - tv[j]) * np.sign(mi - mj)
        if abs(mi - mj) < 5: t_ += 1
        elif s > 0: c_ += 1
        else: w_ += 1
    corr += c_; wrong += w_; tie += t_
    rows.append((op, epi, d['t'], round(d['left_s']), len(cs), sep, round(tv.max() - tv.min()), ''.join('S' if e == 'success' else ('T' if e in ('timeout', 'resource_exhausted') else ('X' if e == 'unsafe' else '.')) for e in ev), c_, w_, t_, round(min(dd), 2)))
print('decisions', n_dec, 'truth separates', n_sep, f'({n_sep / max(n_dec, 1):.0%})')
print('decisive pairs: model correct', corr, 'wrong', wrong, 'tie(<5)', tie, 'model unknown', unk)
print('stage-end spread over 9 intents: min pairwise distance median %.2f m, P10 %.2f m; median pairwise %.2f m' % (np.median([m[0] for m in mind]), np.percentile([m[0] for m in mind], 10), np.median([m[1] for m in mind])))
print('opening episode t left n sep spread truth9 corr wrong tie min_d')
for r in rows: print(*r)
