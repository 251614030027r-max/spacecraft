"""T1-B V8 pre-check (upper, offline, no new truth simulation): why ECPG needs a deployment-time
verification trigger, and why a learned risk trigger is the learnable part.

Data: every truth candidate scan in git (V1 audit 292140-147, V2 pilot 292120-127, V7 side bench
confirm 292500-531 and independent 293100-131; continuation = V1 teacher) and the G* plan features
of each first-decision candidate recomputed from the deterministic reset (``candidate_features.jsonl``,
made by ``candidate_features.py``; observation identical to the evaluated initial observation).

    python -B evidence/T1B_V8_PRECHECK/v8_precheck_offline.py evidence/T1B_V8_PRECHECK/candidate_features.jsonl OUT.json

Sections:
 A  structure of the learning space (oracle gain decomposition, capture / delta-v headroom);
 B  upper bound of ECPG preference distillation without verification: a masked categorical head
    trained by CE to the event-aware *truth* preference (p_M with truth instead of model labels),
    leave-one-block-out, 52-D observation (+ G* plan features), earliest prior bias 0 / 1, 5 seeds;
    and a random-split learning curve (16..64 training states);
 C  learnability of the trigger signal: earliest-candidate failure from the same inputs (AUC);
 D  first-decision recombination with model labels (V7 side bench) and the C-classifier trigger.
"""
import glob, json, sys

import numpy as np
import pandas as pd
import torch as th


def load_truth():
    t = {}
    for pattern, key in (('evidence/T1B_V1_AUDIT/candidate_truth_scan/*.json', 'j0'), ('evidence/T1B_V2_PILOT/scan_292120/*.json', 'first_index'),
                         ('evidence/T1B_V7_SIDE_BENCH/final/**/scan_*.json', 'first_index')):
        for f in sorted(glob.glob(pattern, recursive=True)):
            d = json.load(open(f, encoding='utf-8-sig'))
            if d.get('skipped') or 'event' not in d: continue
            t.setdefault(int(d['opening']), {})[int(d[key])] = d
    return t


def phi(r, j):
    c = r['cands'][j]; m = r['mission']; tau = c['tau']; a = np.radians(c['ang'])
    return np.r_[tau / 200., (m['deadline_s'] - tau) / 200., c['dv_lb'] / max(m['budget_m_s'], 1e-6), np.cos(a), np.sin(a), j / 2.]


def rows_from(features, truth):
    rows = []
    for l in open(features):
        r = json.loads(l); op = r['opening']; t = truth.get(op); k = len(r['cands'])
        if not t or k < 2 or any(j not in t for j in range(k)): continue
        X = np.stack([np.r_[r['obs'], phi(r, j)] for j in range(k)] + [np.zeros(68)] * (3 - k)).astype(np.float32)
        rows.append({'op': op, 'k': k, 'D': r['mission']['deadline_s'], 'B': r['mission']['budget_m_s'], 'X': X,
                     'ret': [float(t[j]['discounted_return']) for j in range(k)], 'event': [t[j]['event'] for j in range(k)],
                     'dv': [float(t[j]['delta_v']) for j in range(k)], 'time': [float(t[j]['time_s']) for j in range(k)]})
    return sorted(rows, key=lambda r: r['op'])


def block(op):
    return 'dev' if op < 292200 else ('confirm' if op < 292600 else 'indep')


def section_a(rows):
    out = {}
    for name in ('all', 'dev', 'confirm', 'indep'):
        R = [r for r in rows if name == 'all' or block(r['op']) == name]
        best = [int(np.argmax(r['ret'])) for r in R]; gain = [r['ret'][b] - r['ret'][0] for r, b in zip(R, best)]
        ok0 = [r for r, b in zip(R, best) if r['event'][0] == 'success']; b0 = [b for r, b in zip(R, best) if r['event'][0] == 'success']
        alt = [(r['event'][0], r['event'][j]) for r in R for j in range(1, r['k'])]
        out[name] = {'states': len(R), 'earliest_return': float(np.mean([r['ret'][0] for r in R])), 'oracle_return': float(np.mean([r['ret'][b] for r, b in zip(R, best)])),
                     'oracle_gain_from_rescues_gt50': float(sum(g for g in gain if g > 50) / len(R)), 'oracle_gain_rest': float(sum(g for g in gain if g <= 50) / len(R)),
                     'rescue_states': [r['op'] for r, g in zip(R, gain) if g > 50],
                     'capture_earliest': float(np.mean([e[0] == 'success' for e in (r['event'] for r in R)])), 'capture_oracle': float(np.mean([r['event'][b] == 'success' for r, b in zip(R, best)])),
                     'earliest_fail_all_fail': int(sum(all(e != 'success' for e in r['event']) for r in R)),
                     'alternatives_catastrophic_when_earliest_succeeds': f"{sum(a == 'success' and b != 'success' for a, b in alt)}/{sum(a == 'success' for a, _ in alt)}",
                     'earliest_success_states_dv_earliest_vs_oracle': [float(np.mean([r['dv'][0] for r in ok0])), float(np.mean([r['dv'][b] for r, b in zip(ok0, b0)]))],
                     'earliest_success_states_time_earliest_vs_oracle': [float(np.mean([r['time'][0] for r in ok0])), float(np.mean([r['time'][b] for r, b in zip(ok0, b0)]))]}
    return out


def p_truth(r, T=2.0):
    g = np.array(r['ret']); ok = np.array([e == 'success' for e in r['event']])
    if not ok.any(): return None
    lg = np.where(ok, g / T, -1e9); p = np.exp(lg - lg.max()) * ok; return np.r_[p / p.sum(), [0.] * (3 - r['k'])]


class Head(th.nn.Module):
    def __init__(self, features):
        super().__init__(); self.features = features
        self.f = th.nn.Sequential(th.nn.Linear(68 if features else 52, 64), th.nn.ReLU(), th.nn.Linear(64, 64), th.nn.ReLU(), th.nn.Linear(64, 1 if features else 3))

    def forward(self, X):   # X: (B, 3, 68)
        return self.f(X)[..., 0] if self.features else self.f(X[:, 0, :52])


def distill(train, test, features, prior, seed, epochs=300):
    th.manual_seed(seed); net = Head(features); opt = th.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    tr = [r for r in train if p_truth(r) is not None]
    X = th.tensor(np.stack([r['X'] for r in tr])); M = th.tensor([[1.] * r['k'] + [0.] * (3 - r['k']) for r in tr]); P = th.tensor(np.stack([p_truth(r) for r in tr]).astype(np.float32))
    for _ in range(epochs):
        loss = -(P * th.log_softmax(net(X).masked_fill(M == 0, -1e9), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        lg = net(th.tensor(np.stack([r['X'] for r in test]))).numpy()
    picks = []
    for r, l in zip(test, lg):
        l = l[:r['k']].copy(); l[0] += prior; picks.append(int(np.argmax(l)))
    return (float(np.mean([r['ret'][j] for r, j in zip(test, picks)])), int(sum(r['event'][0] == 'success' and r['event'][j] != 'success' for r, j in zip(test, picks))),
            int(sum(j != 0 for j in picks)))


def section_b(rows):
    out = {'leave_one_block_out': {}, 'learning_curve_random_split_16_test': {}}
    for name in ('dev', 'confirm', 'indep'):
        test = [r for r in rows if block(r['op']) == name]; train = [r for r in rows if block(r['op']) != name]
        res = {'n_train': len(train), 'n_test': len(test), 'earliest': float(np.mean([r['ret'][0] for r in test])), 'oracle': float(np.mean([max(r['ret']) for r in test]))}
        for features in (False, True):
            for prior in (0., 1.):
                runs = [distill(train, test, features, prior, s) for s in range(5)]
                res[f"{'obs+gstar_features' if features else 'obs'}_prior{prior:g}"] = {'return_mean': float(np.mean([x[0] for x in runs])), 'per_seed': [x[0] for x in runs],
                                                                                    'catastrophic_picks': [x[1] for x in runs], 'deviations': [x[2] for x in runs]}
        out['leave_one_block_out'][name] = res
    rng = np.random.default_rng(0)
    for n in (16, 32, 48, 64):
        v = []
        for rep in range(12):
            p = rng.permutation(len(rows)); te = [rows[i] for i in p[:16]]; tr = [rows[i] for i in p[16:16 + n]]
            e = np.mean([r['ret'][0] for r in te]); o = np.mean([max(r['ret']) for r in te]); x = distill(tr, te, True, 1., rep)
            v.append((x[0] - e, o - e, x[1]))
        v = np.array(v); out['learning_curve_random_split_16_test'][n] = {'gain_vs_earliest': float(v[:, 0].mean()), 'oracle_gain': float(v[:, 1].mean()), 'catastrophic_per_16': float(v[:, 2].mean())}
    return out


def risk_scores(train, test, seeds=5):
    X = np.stack([r['X'][0] for r in train]); Y = np.array([r['event'][0] != 'success' for r in train], np.float32); mu, sd = X.mean(0), X.std(0) + 1e-6; out = []
    for seed in range(seeds):
        th.manual_seed(seed); net = th.nn.Sequential(th.nn.Linear(68, 32), th.nn.ReLU(), th.nn.Linear(32, 1)); opt = th.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-2)
        for _ in range(400):
            l = th.nn.functional.binary_cross_entropy_with_logits(net(th.tensor((X - mu) / sd))[:, 0], th.tensor(Y)); opt.zero_grad(); l.backward(); opt.step()
        with th.no_grad(): out.append(th.sigmoid(net(th.tensor((np.stack([r['X'][0] for r in test]) - mu) / sd))[:, 0]).numpy())
    return np.mean(out, 0)


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y, bool); pos, neg = s[y], s[~y]
    return float(np.mean([(p > n) + .5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else None


def section_c_d(rows):
    D = 'evidence/T1B_V7_SIDE_BENCH/final'
    dep = pd.read_csv(f'{D}/SIDE_DEPLOYMENT.csv', encoding='utf-8-sig'); dep = dep[dep.checkpoint.astype(str) == '10000']
    c, d = {}, {}
    for name in ('dev', 'confirm', 'indep'):
        test = [r for r in rows if block(r['op']) == name]; train = [r for r in rows if block(r['op']) != name]
        s = risk_scores(train, test); y = [r['event'][0] != 'success' for r in test]; order = list(np.argsort(-s))
        c[name] = {'n_train': len(train), 'n_test': len(test), 'earliest_failures': int(sum(y)), 'auc': auc(s, y),
                   'rescue_rank_by_risk': {r['op']: order.index(i) + 1 for i, r in enumerate(test) if r['event'][0] != 'success' and 'success' in r['event']}}
        if name == 'dev': continue
        mp = pd.read_csv(f"{D}/{'MODEL_PAIR_ACCURACY.csv' if name == 'confirm' else 'INDEPENDENT_MODEL_PAIRS.csv'}", encoding='utf-8-sig')
        tc = dep[(dep.block == ('confirm_292500' if name == 'confirm' else 'indep_293100')) & (dep.arm == 'e_tcpf')].set_index('opening')
        def greedy(r):
            m0 = mp[(mp.opening == r['op']) & (mp.i == 0)]; bj, bd = 0, 0.
            for _, x in m0.iterrows():
                if x.model_delta > bd: bj, bd = int(x.j), float(x.model_delta)
            return bj
        g = [greedy(r) for r in test]; prop = [int(tc.loc[r['op'], 'first_index']) if r['op'] in tc.index else 0 for r in test]
        e = float(np.mean([r['ret'][0] for r in test])); va = float(np.mean([r['ret'][j] for r, j in zip(test, g)]))
        res = {'earliest': e, 'verify_all_model_greedy': va, 'oracle': float(np.mean([max(r['ret']) for r in test])), 'triggers': {}}
        for theta in (0.5, 0.3, 0.1):
            for use_prop in (False, True):
                trig = [(si > theta) or (use_prop and p != 0) for si, p in zip(s, prop)]; k = int(sum(trig))
                res['triggers'][f'risk>{theta}' + ('+e_tcpf_proposal' if use_prop else '')] = {
                    'triggered': k, 'return': float(np.mean([r['ret'][j if t else 0] for r, j, t in zip(test, g, trig)])),
                    'capture': float(np.mean([r['event'][j if t else 0] == 'success' for r, j, t in zip(test, g, trig)])),
                    'random_trigger_same_rate_expectation': e + k / len(test) * (va - e)}
        d[name] = res
    return c, d


if __name__ == '__main__':
    truth = load_truth(); rows = rows_from(sys.argv[1], truth)
    c, d = section_c_d(rows)
    out = {'states': len(rows), 'A_structure': section_a(rows), 'B_preference_distillation_without_verification': section_b(rows),
           'C_earliest_failure_learnability': c, 'D_first_decision_recombination_with_classifier_trigger': d}
    json.dump(out, open(sys.argv[2], 'w'), indent=1, default=float); print(json.dumps(out['A_structure']['all'], indent=1))
