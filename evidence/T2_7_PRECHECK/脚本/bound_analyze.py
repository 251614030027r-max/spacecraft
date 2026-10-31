import json, glob, sys, collections
import numpy as np
from itertools import combinations
decs = []
for f in sorted(glob.glob(sys.argv[1] + '/*.json')):
    d = json.load(open(f))
    if d.get('urgent'):
        for x in d['decisions']:
            decs.append((d['opening'], d['event'], x))
items = [it for *_, x in decs for it in x['items']]
print('urgent episodes', len({o for o, *_ in decs}), 'audited decisions', len(decs), 'items', len(items),
      '| executor candidate present', sum(bool(x['track_candidate']) for *_, x in decs))
cap = lambda it: it['truth']['event'] == 'success'
b = [it for it in items if it['L_true'] is not None]
print('bounded items %d of %d (candidates %d, executed %d)' % (len(b), len(items), sum(not i['executed'] for i in b), sum(i['executed'] for i in b)))
print('truth under the plan-tracking continuation: captures %d of %d' % (sum(cap(i) for i in items), len(items)))
if b:
    e = np.array([i['L_true'] - i['truth']['G'] for i in b])
    print('L - truth: median %.1f, p10 %.1f, p90 %.1f; optimistic by > 50: %d of %d (%.0f%%)' %
          (np.median(e), np.percentile(e, 10), np.percentile(e, 90), int((e > 50).sum()), len(e), 100 * (e > 50).mean()))
    print('bounded items captured in truth: %d of %d' % (sum(cap(i) for i in b), len(b)))
nb = [i for i in items if i['L_true'] is None]
print('unbounded items captured in truth: %d of %d' % (sum(cap(i) for i in nb), len(nb)))
right = wrong = und = 0; sep = 0
for *_, x in decs:
    it = x['items']
    if any(cap(i) for i in it) and not all(cap(i) for i in it):
        sep += 1
    for i, j in combinations(it, 2):
        gi, gj = i['truth']['G'], j['truth']['G']
        if (cap(i) != cap(j)) or abs(gi - gj) >= 50:
            hi, lo = (i, j) if gi > gj else (j, i)
            li, lj = hi['L_true'], lo['L_true']
            if li is not None and lj is not None:
                right += li > lj; wrong += li < lj; und += li == lj
            elif li is not None and lj is None: right += 1
            elif lj is not None and li is None: wrong += 1
            else: und += 1
print('decisions where truth separates captures: %d of %d; decisive pairs by the bounds: right %d wrong %d undecided %d' % (sep, len(decs), right, wrong, und))
spread = [max(i['L_true'] for i in x['items'] if i['L_true'] is not None) - min(i['L_true'] for i in x['items'] if i['L_true'] is not None)
          for *_, x in decs if sum(i['L_true'] is not None for i in x['items']) >= 2]
if spread:
    print('within-decision bound spread: median %.1f, p90 %.1f (n %d)' % (np.median(spread), np.percentile(spread, 90), len(spread)))
print('\nper decision: opening, episode(rule), t, left, bounded/n, truth captures/n')
for o, ev, x in decs:
    it = x['items']
    print('  ', o, ev, round(x['t']), round(x['left_s']), '%d/%d' % (sum(i['L_true'] is not None for i in it), len(it)), '%d/%d' % (sum(cap(i) for i in it), len(it)))
