"""Autonomous decisions (no simulation): apply pre-registered criteria to a ``t2_readout curve`` result.

    python -B -m experiments.t2_decide --readout VB_READOUT.json --output DECISION.json [--final 3000]             (T2.2)
    python -B -m experiments.t2_decide --criteria t23 --readout READOUT.json --output DECISION.json [--final 4000]  (T2.3)
    python -B -m experiments.t2_decide --criteria t24 --readout READOUT.json --output DECISION.json                (T2.4)
    python -B -m experiments.t2_decide --criteria t25 --readout READOUT.json --output DECISION.json                (T2.5)

T2.2 (research layer, 2026-10-22):

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
T22_SAC_3000 = 54.32                       # evidence/T2_2_TEST/seed1_final: e_sac_s1 step 3000, the best arm of T2.2
T23_EACC_4000 = 42.36499930600538         # evidence/T2_3_TEST/B_seed0: e_mve_s0 step 4000 (T2.3 coupled arm)


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


def decide_t23(r: dict, final: int = 4000, checkpoints=(1000, 2000, 3000, 4000)) -> dict:
    """T2.3 (pre-registered with the run order): the coupled arm (EACC, ``e_mve``) against SAC on the same
    executor (pass-window commitment in both arms), 2 seeds.

      C0 complete     every arm / checkpoint has all openings and no ERROR file
      C1 beyond T2.2  every seed: EACC mean return at <final> > 54.32 (T2.2's best arm, SAC seed 1 at 3000)
      C2 coupling     every seed: paired EACC - SAC averaged over the checkpoints 1000..<final> > 0 (learning-curve area)
      C3 end level    seed mean of the paired EACC - SAC at <final> >= -3 (not worse at convergence)
      C4 converged    every arm and seed: |R(<final>) - R(<final> - 1000)| <= 10
      C5 safety       every seed at <final>: EACC captures >= SAC captures - 1 and EACC unsafe <= SAC unsafe + 1
    Verdict: incomplete | pass (C1..C5) | not_converged (C4 fails) | learning_failure (both arms gain < 60 from 0k) |
    coupling_failure."""
    arms = r['arms']; paired = r.get('paired', {})
    seeds = sorted({a.split('_s')[-1] for a in arms if a.startswith('e_mve_s')} & {a.split('_s')[-1] for a in arms if a.startswith('e_sac_s')})
    get = lambda arm, k: (arms.get(arm) or {}).get(str(k))
    pr = lambda s, k: paired.get(f's{s}/step_{k}')
    out = {'seeds': seeds, 'final': final, 'criteria': {}, 'values': {}}
    c = out['criteria']; v = out['values']
    cks = [k for k in checkpoints if k <= final]
    c['C0_complete'] = bool(r.get('all_complete')) and bool(seeds) and all(get(f'e_{a}_s{s}', k) for a in ('mve', 'sac') for s in seeds for k in (0, *cks)) \
        and all(pr(s, k) for s in seeds for k in cks)
    if not c['C0_complete']:
        out['verdict'] = 'incomplete'; return out
    v['eacc_final'] = {s: get(f'e_mve_s{s}', final)['mean_return'] for s in seeds}
    v['sac_final'] = {s: get(f'e_sac_s{s}', final)['mean_return'] for s in seeds}
    v['paired_auc'] = {s: float(np.mean([pr(s, k)['mean'] for k in cks])) for s in seeds}
    v['paired_final'] = {s: pr(s, final)['mean'] for s in seeds}
    v['paired_final_urgent'] = {s: pr(s, final)['urgent'] for s in seeds}
    v['paired_final_without_top2'] = {s: pr(s, final)['without_top2'] for s in seeds}
    v['last_step_change'] = {f'e_{a}_s{s}': get(f'e_{a}_s{s}', final)['mean_return'] - get(f'e_{a}_s{s}', final - 1000)['mean_return']
                             for a in ('mve', 'sac') for s in seeds}
    v['learning_gain'] = {a: float(np.mean([get(f'e_{a}_s{s}', final)['mean_return'] - get(f'e_{a}_s{s}', 0)['mean_return'] for s in seeds]))
                          for a in ('mve', 'sac')}
    c['C1_beyond_t22'] = all(x > T22_SAC_3000 for x in v['eacc_final'].values())
    c['C2_coupling_auc'] = all(x > 0 for x in v['paired_auc'].values())
    c['C3_end_level'] = float(np.mean(list(v['paired_final'].values()))) >= -3.0
    c['C4_converged'] = all(abs(x) <= 10.0 for x in v['last_step_change'].values())
    c['C5_safety'] = all(pr(s, final)['capture_mve'] >= pr(s, final)['capture_sac'] - 1 and pr(s, final)['unsafe_mve'] <= pr(s, final)['unsafe_sac'] + 1 for s in seeds)
    if all(c[k] for k in ('C1_beyond_t22', 'C2_coupling_auc', 'C3_end_level', 'C4_converged', 'C5_safety')):
        out['verdict'] = 'pass'
    elif not c['C4_converged']:
        out['verdict'] = 'not_converged'
    elif v['learning_gain']['mve'] < 60 and v['learning_gain']['sac'] < 60:
        out['verdict'] = 'learning_failure'
    else:
        out['verdict'] = 'coupling_failure'
    return out


def decide_t24(r: dict) -> dict:
    """T2.4 (pre-registered with its run order). Seed 0: EACC trained to 4000 against the T2.3 SAC seed 0 (same
    executor; the T2.4 change touches only the coupled labels). Seed 1: EACC to 4000 against the T2.3 SAC
    seed 1, which was stopped at 3000: compared over 1000..3000.

      C0 complete     seed 0: both arms at 0..4000; seed 1: both arms at 1000..3000; all 24/24, no ERROR file
      C1 beyond T2.3  EACC seed 0 at 4000 > 42.365 (the T2.3 coupled arm)
      C2 coupling     paired EACC - SAC averaged over the checkpoints > 0: seed 0 over 1000..4000, seed 1 over 1000..3000
      C3 end level    paired at the last common checkpoint >= -3 (seed 0 at 4000, seed 1 at 3000)
      C4 converged    seed 0, both arms: |R(4000) - R(3000)| <= 10
      C5 safety       seed 0 at 4000: EACC captures >= SAC captures - 1, EACC unsafe <= SAC unsafe + 1
    Verdict: incomplete | pass (all) | seed0_only (seed 0 passes C1..C5, seed 1 fails C2 or C3) | not_converged |
    coupling_failure."""
    arms = r['arms']; paired = r.get('paired', {})
    get = lambda arm, k: (arms.get(arm) or {}).get(str(k))
    pr = lambda s, k: paired.get(f's{s}/step_{k}')
    plan = {'0': (1000, 2000, 3000, 4000), '1': (1000, 2000, 3000)}
    out = {'criteria': {}, 'values': {}}; c = out['criteria']; v = out['values']
    c['C0_complete'] = bool(r.get('all_complete')) and all(get(f'e_{a}_s0', k) for a in ('mve', 'sac') for k in (0, 1000, 2000, 3000, 4000)) \
        and all(pr(s, k) for s, ks in plan.items() for k in ks)
    if not c['C0_complete']:
        out['verdict'] = 'incomplete'; return out
    v['eacc_s0_4000'] = get('e_mve_s0', 4000)['mean_return']; v['sac_s0_4000'] = get('e_sac_s0', 4000)['mean_return']
    v['paired_auc'] = {s: float(np.mean([pr(s, k)['mean'] for k in ks])) for s, ks in plan.items()}
    v['paired_end'] = {s: pr(s, ks[-1])['mean'] for s, ks in plan.items()}
    v['paired_end_urgent'] = {s: pr(s, ks[-1])['urgent'] for s, ks in plan.items()}
    v['last_step_change_s0'] = {a: get(f'e_{a}_s0', 4000)['mean_return'] - get(f'e_{a}_s0', 3000)['mean_return'] for a in ('mve', 'sac')}
    c['C1_beyond_t23'] = v['eacc_s0_4000'] > T23_EACC_4000
    c['C2_coupling_s0'] = v['paired_auc']['0'] > 0; c['C2_coupling_s1'] = v['paired_auc']['1'] > 0
    c['C3_end_s0'] = v['paired_end']['0'] >= -3.0; c['C3_end_s1'] = v['paired_end']['1'] >= -3.0
    c['C4_converged'] = all(abs(x) <= 10.0 for x in v['last_step_change_s0'].values())
    p = pr('0', 4000)
    c['C5_safety'] = p['capture_mve'] >= p['capture_sac'] - 1 and p['unsafe_mve'] <= p['unsafe_sac'] + 1
    seed0 = all(c[k] for k in ('C1_beyond_t23', 'C2_coupling_s0', 'C3_end_s0', 'C4_converged', 'C5_safety'))
    if seed0 and c['C2_coupling_s1'] and c['C3_end_s1']:
        out['verdict'] = 'pass'
    elif seed0:
        out['verdict'] = 'seed0_only'
    elif not c['C4_converged']:
        out['verdict'] = 'not_converged'
    else:
        out['verdict'] = 'coupling_failure'
    return out


T25_CHECKPOINTS = (500, 1000, 1500, 2000, 2500, 3000)


def decide_t25(r: dict, checkpoints=T25_CHECKPOINTS) -> dict:
    """T2.5 formal development test (pre-registered with its run order): SAC and EACC trained together from zero,
    seeds 0, 1, 2, 3000 decisions, checkpoints 0 and 500..3000 on the 24 development openings.

      C0 complete     every arm, seed and checkpoint 24/24 and no ERROR file
      P1 efficiency   paired EACC - SAC averaged over 500..3000 (learning-curve area): seed mean > 0 and >= 2 of 3 seeds > 0
      P2 urgent       the same area on urgent missions: seed mean > 0
      P3 final        paired at 3000: seed mean >= -3 (not worse at the end)
      S  safety       every seed at 3000: EACC captures >= SAC - 1; total EACC unsafe <= total SAC unsafe + 1
      C  converged    each arm: seed mean |R(3000) - R(2500)| <= 10
    Verdict: incomplete | pass (all) | partial (P1, P3, S hold; P2 or C fails) | no_gain (P1 fails; P3 and S hold) |
    harmful (P3 or S fails)."""
    arms = r['arms']; paired = r.get('paired', {})
    seeds = sorted({a.split('_s')[-1] for a in arms if a.startswith('e_mve_s')} & {a.split('_s')[-1] for a in arms if a.startswith('e_sac_s')})
    get = lambda arm, k: (arms.get(arm) or {}).get(str(k))
    pr = lambda s, k: paired.get(f's{s}/step_{k}')
    out = {'seeds': seeds, 'criteria': {}, 'values': {}}; c = out['criteria']; v = out['values']
    c['C0_complete'] = bool(r.get('all_complete')) and len(seeds) >= 3 and all(get(f'e_{a}_s{s}', k) for a in ('mve', 'sac') for s in seeds for k in (0, *checkpoints)) \
        and all(pr(s, k) for s in seeds for k in checkpoints)
    if not c['C0_complete']:
        out['verdict'] = 'incomplete'; return out
    last, prev = checkpoints[-1], checkpoints[-2]
    v['auc'] = {s: float(np.mean([pr(s, k)['mean'] for k in checkpoints])) for s in seeds}
    v['auc_urgent'] = {s: float(np.mean([pr(s, k)['urgent'] for k in checkpoints])) for s in seeds}
    v['final'] = {s: pr(s, last)['mean'] for s in seeds}
    v['final_urgent'] = {s: pr(s, last)['urgent'] for s in seeds}
    v['eacc_final'] = {s: get(f'e_mve_s{s}', last)['mean_return'] for s in seeds}
    v['sac_final'] = {s: get(f'e_sac_s{s}', last)['mean_return'] for s in seeds}
    v['last_change'] = {a: float(np.mean([abs(get(f'e_{a}_s{s}', last)['mean_return'] - get(f'e_{a}_s{s}', prev)['mean_return']) for s in seeds]))
                        for a in ('mve', 'sac')}
    c['P1_efficiency'] = float(np.mean(list(v['auc'].values()))) > 0 and sum(x > 0 for x in v['auc'].values()) >= 2
    c['P2_urgent'] = float(np.mean(list(v['auc_urgent'].values()))) > 0
    c['P3_final'] = float(np.mean(list(v['final'].values()))) >= -3.0
    c['S_safety'] = all(pr(s, last)['capture_mve'] >= pr(s, last)['capture_sac'] - 1 for s in seeds) \
        and sum(pr(s, last)['unsafe_mve'] for s in seeds) <= sum(pr(s, last)['unsafe_sac'] for s in seeds) + 1
    c['C_converged'] = all(x <= 10.0 for x in v['last_change'].values())
    if not (c['P3_final'] and c['S_safety']):
        out['verdict'] = 'harmful'
    elif not c['P1_efficiency']:
        out['verdict'] = 'no_gain'
    elif c['P2_urgent'] and c['C_converged']:
        out['verdict'] = 'pass'
    else:
        out['verdict'] = 'partial'
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readout', required=True); p.add_argument('--output', required=True); p.add_argument('--final', type=int, default=None)
    p.add_argument('--criteria', choices=['t22', 't23', 't24', 't25'], default='t22')
    a = p.parse_args()
    r = json.loads(Path(a.readout).read_text(encoding='utf-8'))
    res = decide_t25(r) if a.criteria == 't25' else decide_t24(r) if a.criteria == 't24' else decide_t23(r, a.final or 4000) if a.criteria == 't23' else decide(r, a.final or 3000)
    Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
