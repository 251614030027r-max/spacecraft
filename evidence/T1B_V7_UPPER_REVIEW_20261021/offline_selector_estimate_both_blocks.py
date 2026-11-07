"""Offline recombination on both independent blocks from the lower's final V7 side-bench files
(evidence/T1B_V7_SIDE_BENCH/final, 685f7a5); no new simulation. Same rules as
offline_selector_estimate.py: E-TCPF + MPC verification (veto a non-earliest first intent whose
real-controller model branch is predicted worse than the earliest; the vetoed episode is the matched
E-SAC one) and the model-greedy selector (truth of the best model-predicted candidate under the
fixed teacher continuation)."""
import json, sys
import pandas as pd
D = sys.argv[1]; out = {}
dep = pd.read_csv(f'{D}/SIDE_DEPLOYMENT.csv', encoding='utf-8-sig'); dep = dep[dep.checkpoint.astype(str) == '10000']
for block, pairs, space in (('confirm_292500', 'MODEL_PAIR_ACCURACY.csv', 'CONFIRM_LEARNING_SPACE.csv'), ('indep_293100', 'INDEPENDENT_MODEL_PAIRS.csv', 'INDEPENDENT_LEARNING_SPACE.csv')):
    d = dep[dep.block == block]; mp = pd.read_csv(f'{D}/{pairs}', encoding='utf-8-sig'); ls = pd.read_csv(f'{D}/{space}', encoding='utf-8-sig').set_index('opening')
    sac, tc = d[d.arm == 'e_sac'].set_index('opening'), d[d.arm == 'e_tcpf'].set_index('opening'); ver = gr = 0.; vetoes = deviations = 0
    for op in sac.index:
        c = int(tc.loc[op, 'first_index']); m0 = mp[(mp.opening == op) & (mp.i == 0)]; md = None
        if c:
            deviations += 1; m = m0[m0.j == c]; md = float(m.model_delta.iloc[0]) if len(m) else None
        veto = md is not None and md < 0; vetoes += int(veto)
        ver += float(sac.loc[op, 'return']) if veto else float(tc.loc[op, 'return'])
        bj, bd = 0, 0.
        for _, r in m0.iterrows():
            if r.model_delta > bd: bj, bd = int(r.j), float(r.model_delta)
        gr += float(ls.loc[op, 'earliest_return']) + (0. if bj == 0 else float(m0[m0.j == bj].truth_delta.iloc[0]))
    n = len(sac)
    out[block] = {'n': n, 'e_sac': float(sac['return'].mean()), 'e_tcpf': float(tc['return'].mean()), 'e_tcpf_mpc_verified': ver / n,
                  'tcpf_deviations': deviations, 'vetoed': vetoes, 'model_greedy': gr / n,
                  'always_earliest_truth': float(ls.earliest_return.mean()), 'best_first_truth': float(ls.best_return.mean())}
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[2], 'w'), indent=1)
