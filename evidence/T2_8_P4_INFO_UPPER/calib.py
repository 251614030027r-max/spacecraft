"""Offline check: decision-level execution calibration of opp2 (fit on selection openings, test on held-out)."""
import json, glob, itertools
import numpy as np
E = 'evidence/T2_8_P4_INFO'
info, o2 = {}, {}
for f in glob.glob(f'{E}/info/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: info[(r['opening'], round(d['t'], 3))] = d
for f in glob.glob(f'{E}/opp2/*.json'):
    r = json.load(open(f))
    for d in r['decisions']: o2[(r['opening'], round(d['t'], 3))] = d
S = json.load(open(f'{E}/KEYSCORE_P4.json'))['scores']
def feats(d, oc):
    o = oc['opp2']; left = d['left_s'] - 16.0
    tau = o.get('tau', 300.0) if o['feasible'] else 300.0
    obs = d['obs']
    return [1.0, float(o['feasible']), np.clip((left - tau) / 100.0, -3, 3), np.clip((left - tau - 90.0) / 50.0, -3, 3),
            o.get('p_success', 0.0) if o['feasible'] else 0.0, obs[0], float(d['urgent'])]
X, y, keys = [], [], []
for k, d in info.items():
    for j, c in enumerate(d['cands']):
        X.append(feats(d, o2[k]['cands'][j])); y.append(float(c['truth_track']['event'] == 'success')); keys.append((k, j))
X, y = np.array(X), np.array(y)
sel = np.array([k[0][0] < 2_910_020 for k in keys])
def fit(X, y, l2=1e-2, it=3000, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) / len(y) + l2 * w; w -= lr * g
    return w
w = fit(X[sel], y[sel])
p = 1 / (1 + np.exp(-X @ w))
for name, m in (('selection', sel), ('heldout', ~sel)):
    pr = p[m] > 0.5; acc = (pr == (y[m] > 0.5)).mean(); base = max(y[m].mean(), 1 - y[m].mean())
    print(f'{name}: capture-prediction accuracy {acc:.3f} (majority {base:.3f}); raw opp2 feasible accuracy {((X[m,1] > 0.5) == (y[m] > 0.5)).mean():.3f}')
P = {kk: p[i] for i, kk in enumerate(keys)}
# calibrated value: capture reward vs timeout via probability; ranking signal
def val(k, j): return P[(k, j)]
for thr in (0.2, 0.4):
    for half in ('selection', 'heldout'):
        cf = dict(cw=0, cr=0, fixed=0, broken=0); prec = [0, 0]
        for lab, s in S.items():
            if int(lab.split('_')[-1]) < 1000: continue
            for row in s['rows']:
                k = (row['opening'], round(row['t'], 3))
                if (k[0] < 2_910_020) != (half == 'selection'): continue
                d = info[k]; q = row['q']; tv = [c['truth_track']['true'] for c in d['cands']]
                for i, j in itertools.combinations(range(9), 2):
                    if abs(tv[i] - tv[j]) < 10: continue
                    ok = (q[i] - q[j]) * (tv[i] - tv[j]) > 0 and abs(q[i] - q[j]) >= 1
                    cf['cr' if ok else 'cw'] += 1
                    dx = val(k, i) - val(k, j)
                    if abs(dx) >= thr:
                        xr = dx * (tv[i] - tv[j]) > 0
                        cf['fixed'] += int(not ok and xr); cf['broken'] += int(ok and not xr)
        print(f'calibrated P(capture) gap >= {thr} {half:9s}: fix {cf["fixed"]/cf["cw"]:.2f}  break {cf["broken"]/cf["cr"]:.3f}  net {cf["fixed"]-cf["broken"]}')
print('weights', np.round(w, 2))
print()
from collections import defaultdict
def phase(d): return 'economy' if not d['urgent'] else ('window' if 40 <= d['left_s'] <= 140 else 'early')
reg = defaultdict(lambda: defaultdict(list)); bigfix = defaultdict(lambda: [0, 0, 0])
for lab, s in S.items():
    arm = lab.split('_')[1]
    if int(lab.split('_')[-1]) < 1000: continue
    for row in s['rows']:
        k = (row['opening'], round(row['t'], 3)); d = info[k]
        half = 'sel' if k[0] < 2_910_020 else 'held'
        tv = np.array([c['truth_track']['true'] for c in d['cands']]); q = np.array(row['q']); pc = np.array([P[(k, j)] for j in range(9)])
        grid = np.array([c['a'] for c in d['cands']]); ka = int(np.argmin(np.linalg.norm(grid - np.asarray(row['actor']), axis=1)))
        # advantage rule: if some candidate's calibrated capture prob exceeds the actor's by >= 0.4, take the best such by critic among them
        better = np.where(pc >= pc[ka] + 0.4)[0]
        kn = ka if len(better) == 0 else better[np.argmax(q[better])]
        ph = phase(d)
        reg[(half, arm, ph)]['actor'].append(tv.max() - tv[ka]); reg[(half, arm, ph)]['actor+cal_adv'].append(tv.max() - tv[kn])
        reg[(half, arm, ph)]['switched'].append(float(kn != ka))
        for i, j in itertools.combinations(range(9), 2):
            if abs(tv[i] - tv[j]) < 50: continue
            ok = (q[i] - q[j]) * (tv[i] - tv[j]) > 0 and abs(q[i] - q[j]) >= 1
            if ok: continue
            dx = pc[i] - pc[j]
            bigfix[(half, arm)][0 if abs(dx) < 0.4 else (1 if dx * (tv[i] - tv[j]) > 0 else 2)] += 1
print('regret (points, lower better): actor vs actor redirected to a candidate with calibrated capture prob +0.4 (best by critic)')
for k in sorted(reg):
    v = reg[k]; print(f'  {k[0]:4s} {k[1]:3s} {k[2]:8s} actor {np.mean(v["actor"]):5.1f} -> {np.mean(v["actor+cal_adv"]):5.1f}  switched {np.mean(v["switched"]):.2f}  n={len(v["actor"])}')
print('critic errors on capture-vs-timeout pairs (gap>=50): calibrated signal tie / fixes / wrong')
for k in sorted(bigfix): print('  ', k, bigfix[k])
print()
comp = defaultdict(lambda: defaultdict(float)); n = defaultdict(int)
for lab, s in S.items():
    arm = lab.split('_')[1]
    if int(lab.split('_')[-1]) < 1000: continue
    for row in s['rows']:
        k = (row['opening'], round(row['t'], 3)); d = info[k]; ph = phase(d)
        tv = np.array([c['truth_track']['true'] for c in d['cands']]); ev = [c['truth_track']['event'] for c in d['cands']]
        grid = np.array([c['a'] for c in d['cands']]); ka = int(np.argmin(np.linalg.norm(grid - np.asarray(row['actor']), axis=1)))
        b = int(np.argmax(tv)); r = tv[b] - tv[ka]; n[(arm, ph)] += 1
        kind = 'lost capture' if (ev[ka] != 'success' and ev[b] == 'success') else ('both fail' if ev[b] != 'success' else 'cost only')
        comp[(arm, ph)][kind] += r; comp[(arm, ph)][kind + ' n'] += 1
        comp[(arm, ph)]['pcap actor'] += P[(k, ka)]
for k in sorted(comp):
    v = comp[k]; N = n[k]
    print(k, 'mean regret by cause:', {c: round(v[c] / N, 1) for c in ('lost capture', 'cost only', 'both fail')},
          'share of decisions:', {c: round(v[c + ' n'] / N, 2) for c in ('lost capture', 'cost only', 'both fail')})
