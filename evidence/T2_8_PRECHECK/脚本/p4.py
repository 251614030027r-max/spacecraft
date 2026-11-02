"""P4 summary (exploratory): against the 'intent, then the executor's projected plan' truth (info_probe.py), how well do
the trained critics rank the key-state intents, and which information would correct the pairs they get wrong?

    python -B evidence/T2_8_PRECHECK/脚本/p4.py INFO_DIR STATES_DIR KEYSCORE.json [OUT.json]

Decisive pair: truth_track returns >= 50 apart. Signals scored on a pair: right / wrong / tie (|difference| below the
signal's tie threshold: critic 1, label 5, opp 5, hold 5).
  label   the executor-model label of the key-state probe (stage-1 label + one chained stage; the production label)
  opp     the executor's earliest feasible entry from the stage goal (info_probe.py), no truth
  hold    the key-state probe's truth with the intent held up to 4 stages (reference only, a truth)
Pre-registered (before the full run): a signal 'carries the missing information' if, over the unguided checkpoints at
1000 and 2000 decisions, it ranks right >= 50 % of the pairs the critic gets wrong in the deadline window (40-140 s
before the deadline) and wrong <= 5 % of all decisive pairs."""
import json, glob, sys, itertools
from collections import defaultdict
import numpy as np

info_dir, states_dir, scores = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))['scores']
info, ks = {}, {}
for f in glob.glob(f'{info_dir}/*.json'):
    r = json.load(open(f))
    for d in r.get('decisions', []):
        info[(r['opening'], round(d['t'], 3))] = d
for f in glob.glob(f'{states_dir}/*.json'):
    r = json.load(open(f))
    for d in r.get('decisions', []):
        ks[(r['opening'], round(d['t'], 3))] = d


def label(c):
    v = None
    for m in c['model']:
        if m is None:
            break
        v = m.get('value', v)
        if m.get('event'):
            break
    return v


keys = sorted(k for k in info if k in ks)
mismatch = [k for k in keys if not info[k]['obs_match']]
sig = {}
for k in keys:
    I, K = info[k], ks[k]
    a_i = [tuple(c['a']) for c in I['cands']]; a_k = [tuple(c['a']) for c in K['cands']]
    idx = [a_k.index(a) if a in a_k else None for a in a_i]
    sig[k] = {'truth': [c['truth_track']['true'] for c in I['cands']], 'opp': [c['opp']['value'] for c in I['cands']],
              'label': [label(K['cands'][j]) if j is not None else None for j in idx],
              'hold': [K['cands'][j]['truth']['true'] if j is not None else None for j in idx],
              'kidx': idx, 'window': 40.0 <= I['left_s'] <= 140.0}
TIE = {'critic': 1.0, 'label': 5.0, 'opp': 5.0, 'hold': 5.0}


def judge(tv, x, i, j, tie):
    if x[i] is None or x[j] is None:
        return 'na'
    if abs(x[i] - x[j]) < tie:
        return 'tie'
    return 'right' if (tv[i] - tv[j]) * (x[i] - x[j]) > 0 else 'wrong'


out = {'decisions': len(keys), 'obs_mismatch': len(mismatch),
       'separating': sum(1 for k in keys if max(sig[k]['truth']) - min(sig[k]['truth']) >= 50), 'checkpoints': {}}
pool = defaultdict(lambda: defaultdict(int))
for lab_, s in sorted(scores.items()):
    if '_ng_' not in lab_:
        continue
    step = int(lab_.split('_')[-1]); arm = lab_.split('_')[1]
    rows = {(r['opening'], round(r['t'], 3)): r for r in s['rows']}
    c = defaultdict(int)
    for k in keys:
        if k not in rows:
            continue
        S = sig[k]; tv = S['truth']; q_all = rows[k]['q']
        q = [q_all[j] if j is not None else None for j in S['kidx']]
        phase = 'window' if S['window'] else 'early'
        for i, j in itertools.combinations(range(len(tv)), 2):
            if abs(tv[i] - tv[j]) < 50:
                continue
            cj = judge(tv, q, i, j, TIE['critic'])
            c[(phase, 'critic', cj)] += 1
            for name in ('label', 'opp', 'hold'):
                x = judge(tv, S[name], i, j, TIE[name])
                c[(phase, name, x)] += 1
                if cj == 'wrong':
                    c[(phase, f'{name}_on_critic_wrong', x)] += 1
                    if step in (1000, 2000):
                        pool[name][(phase, 'on_critic_wrong', x)] += 1
                if step in (1000, 2000):
                    pool[name][('all', x)] += 1
    out['checkpoints'][lab_] = {f'{p}/{n}/{x}': v for (p, n, x), v in sorted(c.items())}
verdict = {}
for name, c in pool.items():
    cw = sum(v for key, v in c.items() if len(key) == 3 and key[0] == 'window' and key[1] == 'on_critic_wrong')
    cw_right = c.get(('window', 'on_critic_wrong', 'right'), 0)
    allp = sum(v for key, v in c.items() if key[0] == 'all'); wrong = c.get(('all', 'wrong'), 0)
    verdict[name] = {'window_critic_wrong_pairs': cw, 'fixed_share': cw_right / max(cw, 1), 'decisive_pairs': allp,
                     'wrong_share': wrong / max(allp, 1),
                     'carries_missing_information': cw > 0 and cw_right / cw >= 0.5 and wrong / max(allp, 1) <= 0.05}
out['signals_unguided_1000_2000'] = verdict
print(json.dumps({k: v for k, v in out.items() if k != 'checkpoints'}, indent=1))
for k, v in out['checkpoints'].items():
    crit = {x: v.get(f'window/critic/{x}', 0) for x in ('right', 'wrong', 'tie')}
    print(k, 'window critic', crit, '| early critic', {x: v.get(f'early/critic/{x}', 0) for x in ('right', 'wrong', 'tie')})
if len(sys.argv) > 4:
    json.dump(out, open(sys.argv[4], 'w'), indent=1)
