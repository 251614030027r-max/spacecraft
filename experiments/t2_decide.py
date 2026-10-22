"""T2.2 autonomous decision (no simulation): apply the pre-registered promotion criteria (research layer,
2026-10-22) to a ``t2_readout curve`` result and name the next branch of the autonomous run order.

    python -B -m experiments.t2_decide --readout VB_READOUT.json --output DECISION.json [--final 3000]

Arms are ``e_mve_sN`` / ``e_sac_sN`` (same seeds); checkpoints ``step_0 .. step_<final>``.

  C0 complete     every arm / checkpoint has all openings and no ERROR file (``all_complete``)
  C1 MVE level    every seed: E-MVE mean return at <final> > 23.4217 (T2_STAGE_B E-MVE at 3000)
  C2 coupling     every seed: paired MVE - SAC > 0 at 2000 and at <final>; seed mean at <final> >= 10
  C3 urgent       seed mean of the paired urgent MVE - SAC at <final> > 0
  C4 spread       seed mean of the paired MVE - SAC without the two largest gains at <final> > 0
  C5 safety       every seed at <final>: MVE captures >= SAC captures - 1 and MVE unsafe <= SAC unsafe + 1
  learning        both arms: seed mean of (<final> - step_0) mean return < 60  ->  learning failure

Verdict: ``incomplete`` (C0 fails: re-run the missing evaluations, then decide again), ``pass`` (C1..C5),
``learning_failure`` (branch L), otherwise ``coupling_failure`` (branch K).
"""
import argparse, json
from pathlib import Path

import numpy as np

T2_MVE_3000 = 23.421727291070482          # evidence/T2_STAGE_B/B_READOUT.json, e_mve step 3000


def decide(r: dict, final: int = 3000) -> dict:
    arms = r['arms']; paired = r.get('paired', {})
    seeds = sorted({a.split('_s')[-1] for a in arms if a.startswith('e_mve_s')} & {a.split('_s')[-1] for a in arms if a.startswith('e_sac_s')})
    F, M = str(final), str(2000)
    get = lambda arm, k: (arms.get(arm) or {}).get(k)
    pr = lambda s, k: paired.get(f's{s}/step_{k}')
    out = {'seeds': seeds, 'final': final, 'criteria': {}, 'values': {}}
    c = out['criteria']; v = out['values']
    c['C0_complete'] = bool(r.get('all_complete')) and bool(seeds) and all(get(f'e_{a}_s{s}', F) for a in ('mve', 'sac') for s in seeds)
    if not c['C0_complete']:
        out['verdict'] = 'incomplete'; return out
    v['mve_final'] = {s: get(f'e_mve_s{s}', F)['mean_return'] for s in seeds}
    v['sac_final'] = {s: get(f'e_sac_s{s}', F)['mean_return'] for s in seeds}
    v['paired_2000'] = {s: (pr(s, M) or {}).get('mean') for s in seeds}
    v['paired_final'] = {s: pr(s, F)['mean'] for s in seeds}
    v['paired_final_urgent'] = {s: pr(s, F)['urgent'] for s in seeds}
    v['paired_final_without_top2'] = {s: pr(s, F)['without_top2'] for s in seeds}
    v['learning_gain'] = {a: float(np.mean([get(f'e_{a}_s{s}', F)['mean_return'] - get(f'e_{a}_s{s}', '0')['mean_return'] for s in seeds]))
                          for a in ('mve', 'sac')}
    c['C1_mve_level'] = all(x > T2_MVE_3000 for x in v['mve_final'].values())
    c['C2_coupling'] = (all(x is not None and x > 0 for x in v['paired_2000'].values()) and all(x > 0 for x in v['paired_final'].values())
                        and float(np.mean(list(v['paired_final'].values()))) >= 10.0)
    c['C3_urgent'] = float(np.mean(list(v['paired_final_urgent'].values()))) > 0
    c['C4_spread'] = float(np.mean(list(v['paired_final_without_top2'].values()))) > 0
    c['C5_safety'] = all(pr(s, F)['capture_mve'] >= pr(s, F)['capture_sac'] - 1 and pr(s, F)['unsafe_mve'] <= pr(s, F)['unsafe_sac'] + 1 for s in seeds)
    if all(c[k] for k in ('C1_mve_level', 'C2_coupling', 'C3_urgent', 'C4_spread', 'C5_safety')):
        out['verdict'] = 'pass'
    elif v['learning_gain']['mve'] < 60 and v['learning_gain']['sac'] < 60:
        out['verdict'] = 'learning_failure'
    else:
        out['verdict'] = 'coupling_failure'
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readout', required=True); p.add_argument('--output', required=True); p.add_argument('--final', type=int, default=3000)
    a = p.parse_args()
    res = decide(json.loads(Path(a.readout).read_text(encoding='utf-8')), a.final)
    Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
