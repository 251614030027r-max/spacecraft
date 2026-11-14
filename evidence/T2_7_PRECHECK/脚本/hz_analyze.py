import json, glob, sys, collections
import numpy as np
D = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + '/*.json'))]
U = [d for d in D if d.get('urgent')]
print('openings', len(D), 'urgent', len(U), 'urgent outcomes (rule behaviour)', dict(collections.Counter(d['event'] for d in U)))
decs = [(d, x) for d in U for x in d['decisions']]
cands = [(d, x, c) for d, x in decs for c in x['cands']]
print('critical decisions', len(decs), 'candidates', len(cands))

def model_by(c, H):
    for k in range(H):
        m = c['model'][k]
        if m is None: return 'alive'
        e = m.get('event')
        if e is None: continue
        if e == 'entry': return 'entry'
        if e.startswith('cutoff'): return 'cutoff'
        return 'blind'            # out_of_domain / infeasible / dropped: the model cannot follow
    return 'alive'

def truth_by(c, H):
    t = c['truth']; n = len(t['stages'])
    if t['event'] and n <= H:
        if t['event'] == 'success': return 'capture_entry' if t['entered_stage'] else 'capture_direct'
        return t['event']
    if t['entered_stage'] and t['entered_stage'] <= H: return 'entered_open'
    return 'alive'

tc = collections.Counter(truth_by(c, 4) for _, _, c in cands)
print('\ntruth within 4 decisions (holding the intent):', dict(tc))
for H in (1, 2, 3, 4):
    tab = collections.defaultdict(collections.Counter)
    for _, _, c in cands:
        tab[truth_by(c, 4)][model_by(c, H)] += 1
    print(f'\nH={H} ({16*H} s) model view of each truth class:')
    for k in ('capture_entry', 'capture_direct', 'timeout', 'unsafe', 'entered_open', 'alive'):
        if tab[k]: print('  ', k, dict(tab[k]))
# decisions where the truth separates captures from non-captures, and what the model separates
print()
for H in (1, 2, 3, 4):
    t_sep = m_sep = good = bad = und = 0
    for d, x in decs:
        tg = [truth_by(c, 4).startswith('capture') for c in x['cands']]
        me = [model_by(c, H) == 'entry' for c in x['cands']]
        if any(tg) and not all(tg): t_sep += 1
        if any(me) and not all(me): m_sep += 1
        for i, ci in enumerate(x['cands']):
            for j, cj in enumerate(x['cands']):
                if tg[i] and not tg[j]:
                    if me[i] and not me[j]: good += 1
                    elif me[j] and not me[i]: bad += 1
                    else: und += 1
    print(f'H={H}: decisions truth separates {t_sep}/{len(decs)}, model separates by entry {m_sep}/{len(decs)}; decisive pairs right {good} wrong {bad} undecided {und}')
# predicted vs true position, both still in a stage
print()
for k in range(4):
    e = []
    for _, _, c in cands:
        m = c['model'][k]; st = c['truth']['stages']
        if m is not None and m.get('event') is None and 'q' in m and len(st) > k and not st[k]['held'] and abs(st[k]['stage_s'] - 16) < 1e-6 and not c['truth']['entered_stage']:
            e.append(float(np.linalg.norm(np.asarray(m['q']) - np.asarray(st[k]['q']))))
    if e: print(f'stage {k+1}: position error n={len(e)} median {np.median(e):.3f} m p90 {np.percentile(e, 90):.3f} m')

print('\nundecided decisive pairs at H=2 by (good truth, good model | bad truth, bad model):')
und = collections.Counter(); ops = collections.defaultdict(set)
for d, x in decs:
    for ci in x['cands']:
        for cj in x['cands']:
            ti, tj = truth_by(ci, 4), truth_by(cj, 4)
            if ti.startswith('capture') and not tj.startswith('capture'):
                mi, mj = model_by(ci, 2), model_by(cj, 2)
                if not (mi == 'entry' and mj != 'entry') and not (mj == 'entry' and mi != 'entry'):
                    key = (ti, mi, tj, mj); und[key] += 1; ops[key].add(d['opening'])
for k, n in und.most_common():
    print('  ', n, k, 'openings', sorted(ops[k]))
