import json, glob, sys, collections
import numpy as np
decs = []
for f in sorted(glob.glob(sys.argv[1] + '/*.json')):
    d = json.load(open(f))
    if not d.get('urgent'):
        continue
    for x in d['decisions']:
        it = []
        for c in x['cands']:
            hold_cap = c['hold']['event'] == 'success'; plan_cap = c.get('plan_event') == 'success'
            it.append({**c, 'hold_cap': hold_cap, 'plan_cap': plan_cap, 'any_cap': hold_cap or plan_cap,
                       'feas': c['feasible'] in (True, 'entered'), 'opening': d['opening'], 'episode': d['event']})
        decs.append((d['opening'], x['t'], x['left_s'], d['event'], it))
rows = [r for *_, it in decs for r in it]
print('early decisions', len(decs), 'candidates', len(rows), '| episodes (rule behaviour):', dict(collections.Counter(e for _, t, _, e, _ in decs if t == 0.0)))
print('opportunity at the stage end:', dict(collections.Counter(str(r['feasible']) for r in rows)))
ok = [r for r in rows if r['feasible'] is True]
if ok:
    print('feasible plans flown: success %d of %d, predicted p_success %.2f' % (sum(r['plan_cap'] for r in ok), len(ok), np.mean([r['p_success'] for r in ok])))
print('captures: hold-intent', sum(r['hold_cap'] for r in rows), '| executor plan', sum(r['plan_cap'] for r in rows), '| either', sum(r['any_cap'] for r in rows))
sep = lambda it, k: any(r[k] for r in it) and not all(r[k] for r in it)
print('decisions where candidates differ: best-achievable truth', sum(sep(it, 'any_cap') for *_, it in decs),
      '| hold-intent truth', sum(sep(it, 'hold_cap') for *_, it in decs), '| opportunity feasibility', sum(sep(it, 'feas') for *_, it in decs), 'of', len(decs))
right = wrong = und = 0
for *_, it in decs:
    for i in it:
        for j in it:
            if i['any_cap'] and not j['any_cap']:
                if i['feas'] and not j['feas']: right += 1
                elif j['feas'] and not i['feas']: wrong += 1
                else: und += 1
print('decisive pairs ordered by opportunity feasibility: right', right, 'wrong', wrong, 'undecided', und)
print('\nper decision (opening, t, left, episode): candidates feasible / plan-captured / hold-captured')
for o, t, left, ev, it in decs:
    print('  ', o, t, round(left), ev, sum(r['feas'] for r in it), sum(r['plan_cap'] for r in it), sum(r['hold_cap'] for r in it), 'of', len(it))
