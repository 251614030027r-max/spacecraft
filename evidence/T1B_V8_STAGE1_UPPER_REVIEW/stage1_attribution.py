"""Upper review of the V8 stage-I closed loop (evidence/T1B_V8_STAGE1, 9c55ddc); reads only the
delivered deploy JSONs, ACTUAL_EPISODES.csv and offline_e_sac_10000.json; no simulation.

    python -B evidence/T1B_V8_STAGE1_UPPER_REVIEW/stage1_attribution.py evidence/T1B_V8_STAGE1 OUT.json

Per paired opening: risk / random return minus the delivered no-verification E-SAC 10k return,
what each triggered decision verified, and where the risk trigger spent its branches (truth class
of the opening's first decision from the offline readout: rescue / all_fail / better_alternative /
earliest_best)."""
import csv, glob, json, sys

D = sys.argv[1]
load = lambda arm: {int(x['opening']): x for x in (json.load(open(f, encoding='utf-8-sig')) for f in glob.glob(f'{D}/deploy_{arm}/*.json'))}
R, X = load('risk'), load('random')
base = {int(r['opening']): float(r['baseline_return']) for r in csv.DictReader(open(f'{D}/ACTUAL_EPISODES.csv', encoding='utf-8-sig')) if r['arm'] == 'risk'}
off = json.load(open(f'{D}/offline_e_sac_10000.json', encoding='utf-8'))


def cls(truth):
    ok = [t[1] == 'success' for t in truth]; ret = [t[0] for t in truth]
    return ('rescue' if any(ok) else 'all_fail') if not ok[0] else ('better_alternative' if max(ret) - ret[0] > 1. else 'earliest_best')


C = {r['opening']: cls(r['truth']) for r in off['rows']}
both = sorted(set(R) & set(X) & set(base))
rows, spend = [], {'risk': {}, 'random': {}}
for o in both:
    rows.append({'opening': o, 'class': C.get(o), 'risk_minus_off': R[o]['discounted_return'] - base[o], 'random_minus_off': X[o]['discounted_return'] - base[o],
                 'risk_branches': R[o]['model_branches'], 'random_branches': X[o]['model_branches'],
                 'risk_changed': sum(d['triggered'] and d['executed'] != d['proposal'] for d in R[o]['decisions']),
                 'random_changed': sum(d['triggered'] and d['executed'] != d['proposal'] for d in X[o]['decisions'])})
    for arm, E in (('risk', R), ('random', X)):
        spend[arm][C.get(o)] = spend[arm].get(C.get(o), 0) + E[o]['model_branches']
mean = lambda k, rs: sum(r[k] for r in rs) / len(rs)
rest = [r for r in rows if r['opening'] != 293117]
out = {'paired_openings': len(rows), 'rescue_states_in_pairing': [r['opening'] for r in rows if r['class'] == 'rescue'],
       'mean_minus_off': {'risk': mean('risk_minus_off', rows), 'random': mean('random_minus_off', rows)},
       'mean_minus_off_without_293117': {'risk': mean('risk_minus_off', rest), 'random': mean('random_minus_off', rest)},
       'branches_by_first_decision_class': spend,
       'risk_triggered_decisions_by_class': {c: sum(d['triggered'] for o in both if C.get(o) == c for d in R[o]['decisions']) for c in set(C.values())},
       'openings_with_nonzero_difference': [r for r in rows if abs(r['risk_minus_off']) > 1e-6 or abs(r['random_minus_off']) > 1e-6]}
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'openings_with_nonzero_difference'}, indent=1))
