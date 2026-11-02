"""P4 summary (exploratory; criteria written before the full run, revised 2026-11-02 after the lower window's review).

    python -B evidence/T2_8_PRECHECK/脚本/p4.py INFO_DIR STATES_DIR KEYSCORE.json OUT.json

Inputs: INFO_DIR from info_probe.py (truth_track, opp, observation per key decision; urgent and economy missions),
STATES_DIR the key-state probe (executor-model labels, urgent decisions only), KEYSCORE.json from
experiments.t2_keyscore on INFO_DIR with the T2.8 night checkpoints (fixed executor, labels {arm}_s{seed}_{step}).

Value reference. truth_track: the intent for one stage, then the executor's projected capture plan at every later
decision, to the end of the episode (terminal: capture, timeout, budget, unsafe). A branch that is not terminal
(MAX_STEPS) is excluded and counted. It is the reference, never a candidate signal.

Pairs. For each key decision, every pair of intents whose truth_track returns differ by >= 10 (one fuel / time
difference on a captured economy mission is ~5-15) and both terminal. Phases: urgent_window (40-140 s before the
deadline), urgent_early (first two decisions), economy (first four decisions).

Critic. Each night checkpoint separately (fixed critic): right / wrong / tie (|dQ| < 1, counted as wrong: it does
not rank). Reported per arm, step and phase; also per opening.

Signals (each a would-be feedback): 'label' (the production executor-model label; urgent decisions only) and 'opp' (the
executor's earliest feasible entry from the stage goal, no truth). Ties: |d| < 5. For signal X and critic C over the
pairs P:
  fix   = #(C wrong or tie, X right) / #(C wrong or tie)          X tie counts as not fixed
  break = #(C right, X wrong) / #(C right)                         X tie counts as not broken
  net   = combined ranking 'X where X is not tied, else C': #right(combined) - #right(C)
Pre-registered: X carries the missing information if, pooled over the night checkpoints at 1000, 1500 and 2000 of both
arms, fix >= 0.5, break <= 0.05 and net > 0, on the selection openings (2,910,000-2,910,019) AND the held-out
openings (2,910,020-2,910,039) separately, and net > 0 for at least two thirds of the individual checkpoints."""
import json, glob, sys, itertools
from collections import defaultdict
import numpy as np

info_dir, states_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[4]
scores = json.load(open(sys.argv[3]))['scores']
DECISIVE, TIE_Q, TIE_X = 10.0, 1.0, 5.0
STEPS = (1000, 1500, 2000)
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


def phase(d):
    if not d['urgent']:
        return 'economy'
    return 'urgent_window' if 40.0 <= d['left_s'] <= 140.0 else 'urgent_early'


dec = {}
nonterminal = 0
for k, d in sorted(info.items()):
    tv = [c['truth_track']['true'] for c in d['cands']]
    term = [c['truth_track'].get('terminal', c['truth_track']['event'] is not None) for c in d['cands']]
    nonterminal += sum(not t for t in term)
    lab = [None] * len(tv)
    if k in ks:
        a_k = [tuple(c['a']) for c in ks[k]['cands']]
        lab = [label(ks[k]['cands'][a_k.index(tuple(c['a']))]) if tuple(c['a']) in a_k else None for c in d['cands']]
    pairs = [(i, j) for i, j in itertools.combinations(range(len(tv)), 2) if term[i] and term[j] and abs(tv[i] - tv[j]) >= DECISIVE]
    dec[k] = {'phase': phase(d), 'truth': tv, 'opp': [c['opp']['value'] for c in d['cands']], 'label': lab, 'pairs': pairs,
              'half': 'selection' if k[0] < 2_910_020 else 'heldout'}


def sgn(x, i, j, tie):
    if x[i] is None or x[j] is None:
        return None
    return 0 if abs(x[i] - x[j]) < tie else (1 if x[i] > x[j] else -1)


crit = {}
sig = {n: defaultdict(lambda: defaultdict(int)) for n in ('label', 'opp')}
per_ck = {n: {} for n in ('label', 'opp')}
per_open = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for lab_, s in sorted(scores.items()):
    parts = lab_.split('_'); arm, step = parts[1], int(parts[-1])
    rows = {(r['opening'], round(r['t'], 3)): r['q'] for r in s['rows']}
    c = defaultdict(lambda: [0, 0, 0])
    net = {n: 0 for n in sig}
    for k, D in dec.items():
        if k not in rows:
            continue
        q = rows[k]
        for i, j in D['pairs']:
            t = 1 if D['truth'][i] > D['truth'][j] else -1
            cs = sgn(q, i, j, TIE_Q)
            ok = cs == t
            c[D['phase']][0 if ok else (2 if cs == 0 else 1)] += 1
            if step in STEPS:
                per_open[k[0]][arm][0 if ok else 1] += 1
            for n in sig:
                xs = sgn(D[n], i, j, TIE_X)
                if xs is None:
                    continue
                comb_ok = (xs == t) if xs != 0 else ok
                net[n] += int(comb_ok) - int(ok)
                if step not in STEPS:
                    continue
                g = sig[n][D['half']]
                if ok:
                    g['critic_right'] += 1; g['broken'] += int(xs == -t)
                else:
                    g['critic_wrong'] += 1; g['fixed'] += int(xs == t)
                g['net'] += int(comb_ok) - int(ok)
                g[f"pairs_{D['phase']}"] += 1
    crit[lab_] = {p: {'right': v[0], 'wrong': v[1], 'tie': v[2], 'accuracy': v[0] / max(sum(v), 1)} for p, v in c.items()}
    for n in sig:
        if step in STEPS:
            per_ck[n][lab_] = net[n]
verdict = {}
for n, halves in sig.items():
    res = {}
    for h, g in halves.items():
        res[h] = {**g, 'fix': g['fixed'] / max(g['critic_wrong'], 1), 'break': g['broken'] / max(g['critic_right'], 1)}
        res[h]['pass'] = g['critic_wrong'] > 0 and res[h]['fix'] >= 0.5 and res[h]['break'] <= 0.05 and g['net'] > 0
    nets = list(per_ck[n].values())
    res['checkpoints_net_positive'] = f'{sum(x > 0 for x in nets)}/{len(nets)}'
    res['carries_missing_information'] = (all(res.get(h, {}).get('pass', False) for h in ('selection', 'heldout'))
                                          and len(nets) > 0 and sum(x > 0 for x in nets) >= 2 * len(nets) / 3)
    verdict[n] = res
arm_summary = {}
for arm in ('mve', 'sac'):
    for ph in ('urgent_window', 'urgent_early', 'economy'):
        acc = [crit[l][ph]['accuracy'] for l in crit if l.split('_')[1] == arm and int(l.split('_')[-1]) in STEPS and ph in crit[l]]
        arm_summary[f'{arm}/{ph}'] = {'checkpoints': len(acc), 'accuracy_mean': float(np.mean(acc)) if acc else None,
                                      'accuracy_min': float(min(acc)) if acc else None}
out = {'decisions': len(dec), 'phases': {p: sum(1 for D in dec.values() if D['phase'] == p) for p in ('urgent_window', 'urgent_early', 'economy')},
       'decisive_pairs': {p: sum(len(D['pairs']) for D in dec.values() if D['phase'] == p) for p in ('urgent_window', 'urgent_early', 'economy')},
       'obs_mismatch': sum(1 for d in info.values() if d.get('obs_match') is False), 'nonterminal_branches': nonterminal,
       'critic_by_arm_phase': arm_summary, 'signals': verdict, 'critic_per_checkpoint': crit,
       'critic_per_opening': {o: {a: {'right': v[0], 'wrong': v[1]} for a, v in d.items()} for o, d in sorted(per_open.items())},
       'net_per_checkpoint': per_ck}
json.dump(out, open(out_path, 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('decisions', 'phases', 'decisive_pairs', 'obs_mismatch', 'nonterminal_branches', 'critic_by_arm_phase', 'signals')}, indent=1))
