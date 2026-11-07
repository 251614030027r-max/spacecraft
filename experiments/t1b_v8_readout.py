"""T1-B V8 stage-I readout (reads the outputs of ``experiments.t1b_v8`` at 181f5b4; no simulation).

    python -B -m experiments.t1b_v8_readout --offline OFF.json --risk DIR --random DIR --baseline DIR[,DIR]
                                            --serial-model DIR[,DIR] --smoke DIR --output OUT.json

Three items the research layer required (2026-10-21) on top of the frozen gates:

* **trigger behaviour (G1)**: which first decisions the fixed threshold min(Q1, Q2)(proposal) < 0
  actually triggers, split by reason (deviation / risk) and by truth class (rescue: earliest fails,
  a later candidate captures; all-fail; earliest captures with a > 1 better candidate; earliest
  captures, no better candidate), recall of the earliest failures, the Q of each class, and the
  rate/recall the same critic would give at other thresholds (reported only; the threshold stays 0);
* **real verification latency**: wall time of every triggered decision (its branches run in
  parallel, under the stage-I load) and the serial cost of one branch (side bench serial scans),
  each against the lead time of the executed intent (time from the decision to its planned arrival);
* **equal-compute control**: risk trigger vs random trigger (rho = the offline first-decision
  trigger rate, fixed before the closed loop), paired on the same openings, with the model branches
  each arm spent.
"""
import argparse, json
from pathlib import Path

import numpy as np

from experiments.t1b_v2 import default, dump

THRESHOLDS = (-40., -20., 0., 20., 40.)


def _q(x, qs=(50, 95, 100)):
    x = np.asarray(x, float)
    return {f'p{q}': float(np.percentile(x, q)) for q in qs} if x.size else None


def truth_class(truth):
    ret = [t[0] for t in truth]; ok = [t[1] == 'success' for t in truth]
    if not ok[0]:
        return 'rescue' if any(ok) else 'all_fail'
    return 'better_alternative' if max(ret) - ret[0] > 1. else 'earliest_best'


def trigger_behaviour(rows, q_theta=0.):
    out = {'states': len(rows), 'threshold': q_theta, 'classes': {}, 'triggered_openings': []}
    for r in rows:
        r['class'] = truth_class(r['truth']); r['reason'] = 'deviation' if r['proposal'] != 0 else ('risk' if r['q_min'][0] < q_theta else None)
    for c in ('rescue', 'all_fail', 'better_alternative', 'earliest_best'):
        rs = [r for r in rows if r['class'] == c]
        out['classes'][c] = {'states': len(rs), 'triggered': sum(r['reason'] is not None for r in rs),
                             'by_deviation': sum(r['reason'] == 'deviation' for r in rs), 'by_risk': sum(r['reason'] == 'risk' for r in rs),
                             'q_min_earliest': _q([r['q_min'][0] for r in rs], (0, 50, 100)), 'openings': [r['opening'] for r in rs]}
    fails = [r for r in rows if r['class'] in ('rescue', 'all_fail')]
    trig = [r for r in rows if r['reason'] is not None]
    out.update(triggered=len(trig), trigger_rate=len(trig) / max(len(rows), 1),
               earliest_failure_recall=sum(r['reason'] is not None for r in fails) / max(len(fails), 1),
               triggered_share_on_earliest_failures=sum(r['class'] in ('rescue', 'all_fail') for r in trig) / max(len(trig), 1),
               rescue_triggered={r['opening']: r['reason'] is not None for r in rows if r['class'] == 'rescue'})
    for r in trig:
        out['triggered_openings'].append({'opening': r['opening'], 'reason': r['reason'], 'class': r['class'], 'proposal': r['proposal'],
                                          'q_min': [round(v, 2) for v in r['q_min']], 'truth_events': [t[1] for t in r['truth']],
                                          'truth_returns': [round(t[0], 2) for t in r['truth']]})
    out['other_thresholds_report_only'] = {}
    for th in THRESHOLDS:
        t = [r['proposal'] != 0 or r['q_min'][r['proposal']] < th for r in rows]
        out['other_thresholds_report_only'][str(th)] = {'trigger_rate': float(np.mean(t)),
                                                        'earliest_failure_recall': float(np.mean([x for x, r in zip(t, rows) if r['class'] in ('rescue', 'all_fail')])) if fails else None}
    return out


def load_dir(d):
    return {int(x['opening']): x for x in (json.loads(f.read_text(encoding='utf-8-sig')) for f in sorted(Path(d).glob('*.json'))) if 'event' in x}


def episodes(dirs):
    out = {}
    for d in dirs.split(','):
        out.update(load_dir(d))
    return out


def arm_stats(eps, base):
    ops = sorted(o for o in eps if o in base)
    dec = [d for o in ops for d in eps[o]['decisions']]; trig = [d for d in dec if d['triggered']]
    lead = [d['candidates_s'][d['executed']] for d in trig]
    return {'openings': len(ops), 'return': float(np.mean([eps[o]['discounted_return'] for o in ops])) if ops else None,
            'baseline_off_return': float(np.mean([base[o]['discounted_return'] for o in ops])) if ops else None,
            'paired_delta_vs_off': float(np.mean([eps[o]['discounted_return'] - base[o]['discounted_return'] for o in ops])) if ops else None,
            'capture': sum(eps[o]['event'] == 'success' for o in ops), 'baseline_capture': sum(base[o]['event'] == 'success' for o in ops),
            'unsafe': sum(eps[o]['event'] == 'unsafe' for o in ops), 'baseline_unsafe': sum(base[o]['event'] == 'unsafe' for o in ops),
            'learning_decisions': len(dec), 'triggered_decisions': len(trig), 'trigger_rate': len(trig) / max(len(dec), 1),
            'trigger_reasons': {k: sum(d['reason'] == k for d in trig) for k in ('deviation', 'risk', 'random', 'all')},
            'second_or_later_decisions_triggered': sum(1 for o in ops for k, d in enumerate(eps[o]['decisions']) if k and d['triggered']),
            'model_branches_per_episode': sum(d['branches'] for d in dec) / max(len(ops), 1),
            'changed_choice_after_verification': sum(d['executed'] != d['proposal'] for d in trig),
            'latency_s_triggered_decision_parallel_branches': _q([d['wall_s'] for d in trig]),
            'lead_time_s_executed_intent': _q(lead, (0, 50, 100)),
            'latency_over_lead': _q([d['wall_s'] / max(l, 1e-6) for d, l in zip(trig, lead)]),
            'events': {o: eps[o]['event'] for o in ops}}


def serial_branch_cost(dirs):
    w = [r['wall_s'] for d in dirs.split(',') for f in sorted(Path(d).glob('model_*.json'))
         for r in json.loads(f.read_text(encoding='utf-8-sig'))['rows'] if 'wall_s' in r]
    return {'branches': len(w), 'wall_s_one_branch_serial': _q(w)}


def smoke(d):
    d = Path(d); fin = json.loads((d / 'finished.json').read_text(encoding='utf-8')); aud = json.loads((d / 'learning_audit.json').read_text(encoding='utf-8'))
    ce = [x['actor_ce'] for x in aud if x.get('actor_ce') is not None]
    return {'preference_complete': (fin.get('preference_counts') or {}).get('complete', 0), 'ce_probes': len(ce), 'ce_finite': bool(ce) and bool(np.all(np.isfinite(ce)))}


def gates(g1, risk, rand, sm):
    g = {'G0_smoke': sm is not None and sm['preference_complete'] >= 1 and sm['ce_finite'],
         'G1_auc': g1['auc'] is not None and g1['auc'] >= .80, 'G1_rescue_triggered': all(g1['behaviour']['rescue_triggered'].values()) and bool(g1['behaviour']['rescue_triggered']),
         'G1_failure_recall': g1['behaviour']['earliest_failure_recall'] >= .80, 'G1_rate': g1['behaviour']['trigger_rate'] <= .60,
         'G2_return': risk['paired_delta_vs_off'] is not None and risk['paired_delta_vs_off'] >= 2.0,
         'G2_capture_unsafe': risk['capture'] >= risk['baseline_capture'] and risk['unsafe'] <= risk['baseline_unsafe'], 'G2_rate': risk['trigger_rate'] <= .60}
    both = sorted(set(risk['events']) & set(rand['events']))
    g['G3_paired_openings'] = len(both)
    g['G3_branch_ratio_random_over_risk'] = rand['model_branches_per_episode'] / max(risk['model_branches_per_episode'], 1e-9)
    g['G3_equal_compute'] = bool(both) and 0.75 <= g['G3_branch_ratio_random_over_risk'] <= 1.25
    g['G3_learned_beats_random'] = rand.get('paired_delta_risk_minus_random') is not None and rand['paired_delta_risk_minus_random'] >= 1.0
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('offline', 'risk', 'random', 'baseline', 'serial-model', 'smoke', 'output'): p.add_argument('--' + k, required=k in ('offline', 'risk', 'baseline', 'output'))
    a = p.parse_args()
    off = json.loads(Path(a.offline).read_text(encoding='utf-8')); base = episodes(a.baseline)
    g1 = {'auc': off['summary']['auc_minus_q_earliest_vs_earliest_failure'], 'behaviour': trigger_behaviour(off['rows'], off['summary']['q_theta']),
          'offline_rho_for_random_arm': off['summary']['first_decision_recombination_truth_continuation_V1_teacher']['verify_risk']['triggered_fraction']}
    risk = arm_stats(episodes(a.risk), base); rand = arm_stats(episodes(a.random), base) if a.random else None
    if rand:
        both = sorted(set(risk['events']) & set(rand['events'])); er, ed = episodes(a.risk), episodes(a.random)
        rand['paired_delta_risk_minus_random'] = float(np.mean([er[o]['discounted_return'] - ed[o]['discounted_return'] for o in both])) if both else None
    sm = smoke(a.smoke) if a.smoke else None
    out = {'G1_trigger': g1, 'risk_arm': risk, 'random_arm': rand, 'serial_branch_cost': serial_branch_cost(a.serial_model) if a.serial_model else None,
           'smoke': sm, 'gates': gates(g1, risk, rand, sm) if rand else None}
    dump(Path(a.output), out); print(json.dumps({k: out[k] for k in ('gates', 'serial_branch_cost', 'smoke')}, indent=1, default=default))


if __name__ == '__main__':
    main()
