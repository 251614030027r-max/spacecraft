import json, glob, sys, collections
import numpy as np
P = Path = None
OPP, HZ = sys.argv[1], sys.argv[2]
H = {}
for f in glob.glob(HZ + '/*.json'):
    d = json.load(open(f))
    if d.get('urgent'):
        for x in d['decisions']:
            for c in x['cands']:
                H[(d['opening'], x['t'], tuple(round(v, 4) for v in c['a']))] = c
rows, decs = [], []
for f in sorted(glob.glob(OPP + '/*.json')):
    d = json.load(open(f))
    if not d.get('urgent'):
        continue
    for x in d['decisions']:
        items = []
        for c in x['cands']:
            h = H.get((d['opening'], x['t'], tuple(round(v, 4) for v in c['a'])))
            if h is None:
                continue
            t = h['truth']
            hold_cap = t['event'] == 'success'
            plan_cap = c.get('plan_event') == 'success'
            items.append({**c, 'hold_cap': hold_cap, 'hold_true': t['true'], 'plan_cap': plan_cap,
                          'any_cap': hold_cap or plan_cap, 'opening': d['opening'], 't': x['t']})
        if items:
            decs.append(items); rows += items
print('urgent decisions joined', len(decs), 'candidates', len(rows))
fe = collections.Counter(str(r['feasible']) for r in rows)
print('opportunity at the stage end (executor G*):', dict(fe))
ok = [r for r in rows if r['feasible'] is True]
print('feasible plans flown: success', sum(r['plan_cap'] for r in ok), 'of', len(ok),
      '| predicted p_success mean %.2f' % np.mean([r['p_success'] for r in ok]) if ok else '')
if ok:
    err = [r['L'] - r['plan_true'] for r in ok]
    print('L - plan truth: median %.1f, mean %.1f, p10 %.1f, p90 %.1f' % (np.median(err), np.mean(err), np.percentile(err, 10), np.percentile(err, 90)))
print('\ncandidates captured: by holding the intent', sum(r['hold_cap'] for r in rows), '| by the executor plan', sum(r['plan_cap'] for r in rows),
      '| either', sum(r['any_cap'] for r in rows))
print('captures only the plan finds (hold-intent fails):', sum(r['plan_cap'] and not r['hold_cap'] for r in rows))
sep_hold = sum(1 for it in decs if any(r['hold_cap'] for r in it) and not all(r['hold_cap'] for r in it))
sep_any = sum(1 for it in decs if any(r['any_cap'] for r in it) and not all(r['any_cap'] for r in it))
feas_sep = sum(1 for it in decs if any(r['feasible'] is True for r in it) and any(r['feasible'] is False for r in it))
print('\ndecisions where outcomes differ across candidates: hold-intent truth', sep_hold, '| best-achievable truth (hold or plan)', sep_any,
      '| the opportunity (feasible vs not) differs', feas_sep, 'of', len(decs))
# decisive pairs on the best-achievable truth: does the opportunity order them?
right = wrong = und = 0
for it in decs:
    for i in it:
        for j in it:
            if i['any_cap'] and not j['any_cap']:
                fi, fj = i['feasible'] in (True, 'entered'), j['feasible'] in (True, 'entered')
                if fi and not fj: right += 1
                elif fj and not fi: wrong += 1
                else:
                    li, lj = i.get('L'), j.get('L')
                    if li is not None and lj is not None and li != lj:
                        right += li > lj; wrong += li < lj
                    else:
                        und += 1
print('decisive pairs (best-achievable truth) ordered by the opportunity: right', right, 'wrong', wrong, 'undecided', und)
print('\nper opening: decisions, candidates feasible / captured by plan / by hold')
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for it in decs:
    o = it[0]['opening']; by[o][0] += 1
    for r in it:
        by[o][1] += r['feasible'] is True; by[o][2] += r['plan_cap']; by[o][3] += r['hold_cap']
for o, v in sorted(by.items()):
    print('  ', o, v)
