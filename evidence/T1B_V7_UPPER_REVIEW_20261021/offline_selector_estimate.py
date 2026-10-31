"""Offline estimate on the confirmation block 292500-292531 from the lower's published V7 side-bench
files (no new simulation): E-SAC 10k, E-TCPF 10k, E-TCPF with an MPC verification of its proposed
first intent against the earliest candidate (veto if the real-controller model branch predicts it
worse; the vetoed episode is replaced by the matched E-SAC episode, which chose the earliest), and the
pure model-greedy selector (best model-predicted first candidate; return = truth of that candidate
under the fixed teacher continuation)."""
import json, sys
import pandas as pd
D = sys.argv[1]
dep = pd.read_csv(f'{D}/SIDE_DEPLOYMENT.csv', encoding='utf-8-sig'); dep = dep[(dep.checkpoint.astype(str) == '10000') & (dep.block == 'confirm_292500')]
mp = pd.read_csv(f'{D}/MODEL_PAIR_ACCURACY.csv', encoding='utf-8-sig'); ls = pd.read_csv(f'{D}/CONFIRM_LEARNING_SPACE.csv', encoding='utf-8-sig').set_index('opening')
sac = dep[dep.arm == 'e_sac'].set_index('opening'); tc = dep[dep.arm == 'e_tcpf'].set_index('opening')
rows, ver, greedy = [], 0., 0.
for op in sac.index:
    c = int(tc.loc[op, 'first_index']); rt, rs = float(tc.loc[op, 'return']), float(sac.loc[op, 'return'])
    md = None
    if c:
        m = mp[(mp.opening == op) & (mp.i == 0) & (mp.j == c)]; md = float(m.model_delta.iloc[0]) if len(m) else None
    rv = rs if (md is not None and md < 0) else rt; ver += rv
    m0 = mp[(mp.opening == op) & (mp.i == 0)]; bj, bd = 0, 0.
    for _, r in m0.iterrows():
        if r.model_delta > bd: bj, bd = int(r.j), float(r.model_delta)
    greedy += float(ls.loc[op, 'earliest_return']) + (0. if bj == 0 else float(m0[m0.j == bj].truth_delta.iloc[0]))
    rows.append({'opening': int(op), 'tcpf_first': c, 'model_delta_vs_earliest': md, 'sac_return': rs, 'tcpf_return': rt, 'verified_return': rv, 'model_greedy_first': bj})
n = len(sac)
out = {'block': 'confirm_292500 (32)', 'e_sac_10k': float(sac['return'].mean()), 'e_tcpf_10k': float(tc['return'].mean()), 'e_tcpf_10k_mpc_verified': ver / n,
       'model_greedy_teacher_truth': greedy / n, 'always_earliest_truth': float(ls.earliest_return.mean()), 'best_first_truth': float(ls.best_return.mean()),
       'caveat': 'verified/greedy are offline recombinations: teacher-continuation truth for greedy, matched E-SAC episode for vetoes; not deployed runs', 'rows': rows}
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1)); json.dump(out, open(sys.argv[2], 'w'), indent=1)
