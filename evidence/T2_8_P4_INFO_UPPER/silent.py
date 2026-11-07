import json, glob, itertools
from collections import Counter
E = 'evidence/T2_8_P4_INFO'
info, o2 = {}, {}
for f in glob.glob(f'{E}/info/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: info[(r['opening'], round(d['t'], 3))] = d
for f in glob.glob(f'{E}/opp2/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: o2[(r['opening'], round(d['t'], 3))] = d
c = Counter(); ex = []
for k, d in info.items():
    cs, os_ = d['cands'], o2[k]['cands']
    for i, j in itertools.combinations(range(9), 2):
        ti, tj = cs[i]['truth_track'], cs[j]['truth_track']
        if abs(ti['true'] - tj['true']) < 50: continue
        oi, oj = os_[i]['opp2'], os_[j]['opp2']
        if abs(oi['value'] - oj['value']) >= 5: continue
        hi, lo = (ti, tj) if ti['true'] > tj['true'] else (tj, ti)
        ohi = oi if ti['true'] > tj['true'] else oj
        key = ('both feasible' if oi['feasible'] and oj['feasible'] else ('both infeasible' if not oi['feasible'] and not oj['feasible'] else 'mixed'),
               'better:' + str(hi['event']), 'worse:' + str(lo['event']), 'better tracked stages %d' % min(hi['tracked'], 3))
        c[key] += 1
        if len(ex) < 6 and key[0] == 'both feasible': ex.append((k, round(ohi.get('p_success', -1), 2), round(hi['true']), round(lo['true']), ohi.get('tau')))
for k, v in c.most_common(): print(v, k)
print(ex)
