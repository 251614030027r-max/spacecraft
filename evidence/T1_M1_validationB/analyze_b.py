import json, glob, sys, numpy as np
from collections import Counter
from env.t1_gym import OBSERVATION_NAMES
R=[json.loads(l) for f in sorted(glob.glob(sys.argv[1]+'/*.jsonl')) for l in open(f)]
print('states', len(R), Counter(r['source'] for r in R), 'shadow==real closed loop', sum(r['real_matches_shadow_closed_loop'] for r in R), '/', len(R))
print('kinds', Counter(r['kind'] for r in R))
A=[r for r in R if 'e_b' in r]
print('with alternative', len(A), ' structural gate open', sum(r['w_structural'] for r in A))
print('gate reasons', Counter(x.split(':')[0] for r in A for x in r['reasons']))
def q(v): v=np.asarray(v); return f"med {np.median(v):.4f} p90 {np.quantile(v,.9):.4f} max {v.max():.4f}"
for gated in (False, True):
    S=[r for r in A if (r['w_structural'] or not gated)]
    if not S: continue
    print('--- subset', 'gate open' if gated else 'all with alternative', len(S))
    for key in ('e_r','e_p','e_o'):
        ea=[r['e_a'][key] for r in S]; eb=[r['e_b'][key] for r in S]
        ratio=np.array(eb)/np.maximum(np.array(ea),1e-9)
        print(f" {key}: executed {q(ea)} | counterfactual {q(eb)} | cf/exec ratio med {np.median(ratio):.2f} p90 {np.quantile(ratio,.9):.2f}; corr {np.corrcoef(ea,eb)[0,1]:.2f}")
    da=np.abs(np.array([r['dobs_a'] for r in S])); db=np.abs(np.array([r['dobs_b'] for r in S]))
    top=np.argsort(-db.mean(0))[:6]
    print(' largest cf obs-dim errors (mean |dz|):', [(OBSERVATION_NAMES[i], round(db[:,i].mean(),4), round(da[:,i].mean(),4)) for i in top])
    sep=[r['dQ_obs'] for r in S]; print(' |z^(a~) - z^(a)| (signal size):', q(sep))
    print(' events cf:', Counter(r['event_b'] for r in S), ' sat_b', q([r['sat_b'] for r in S]))
print('pred wall a', q([r['wall_pred_a'] for r in R]), ' b', q([r['wall_pred_b'] for r in A]))
