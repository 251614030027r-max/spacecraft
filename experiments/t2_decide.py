"""Autonomous decisions (no simulation): apply pre-registered criteria to a ``t2_readout curve`` result.

    python -B -m experiments.t2_decide --readout VB_READOUT.json --output DECISION.json [--final 3000]             (T2.2)
    python -B -m experiments.t2_decide --criteria t23 --readout READOUT.json --output DECISION.json [--final 4000]  (T2.3)
    python -B -m experiments.t2_decide --criteria t24 --readout READOUT.json --output DECISION.json                (T2.4)
    python -B -m experiments.t2_decide --criteria t25 --readout READOUT.json --output DECISION.json                (T2.5 formal)
    python -B -m experiments.t2_decide --criteria v25 --readout READOUT.json --train-root DIR --output DECISION.json  (T2.5 validation)
    python -B -m experiments.t2_decide --criteria v251 --readout READOUT.json --train-root DIR --output DECISION.json (T2.5.1 validation)
    python -B -m experiments.t2_decide --criteria t26 --readout READOUT.json --train-root DIR --output DECISION.json  (T2.6 time pressure)

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
T24_EACC_4000 = 53.79                      # evidence/T2_4_TEST: e_mve_s0 step 4000 (T2.4 coupled arm, RESULT_FINAL)


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


def decide_v25(r: dict, mech: dict | None = None) -> dict:
    """T2.5 validation (pre-registered with its run order): EACC seeds 0 and 1 trained to 4000; SAC reused from T2.3
    (seed 0 to 4000, seed 1 to 3000; T2.5 changes only the coupled labels, the SAC arm generates none).

      C0 complete     EACC seed 0 at 0..4000 and seed 1 at 1000..3000 (and 4000), SAC as above, all 24/24, no ERROR file
      M  mechanism    training logs of both EACC seeds (mean over probes from 1000 on): two_stage_share >= 0.5,
                      decisive_share > 0, c_mean >= 0.5, grad_cos > 0
      E1 efficiency   paired EACC - SAC averaged over the checkpoints: seed 0 over 1000..4000 > 0 and seed 1 over 1000..3000 > 0
      E2 final        seed 0 at 4000: paired >= -3 and EACC > 53.79 (the T2.4 coupled arm)
      S  safety       seed 0 at 4000: EACC captures >= SAC - 1, EACC unsafe <= SAC unsafe + 1
    Verdict: incomplete | mechanism_failure (M fails) | pass (M, E1, E2, S) | mixed (M, S hold; E1 holds on one seed or
    E2 fails) | no_gain (M, S hold; E1 fails on both seeds) | harmful (S fails)."""
    out = decide_t24(r) if r else {'criteria': {}, 'values': {}, 'verdict': 'incomplete'}
    if out['verdict'] == 'incomplete':
        return out
    c, v = out['criteria'], out['values']
    mech = mech or {}
    v['mechanism'] = mech
    c['M_mechanism'] = bool(mech) and all(
        (x.get('two_stage_share') or 0) >= 0.5 and (x.get('decisive_share') or 0) > 0 and (x.get('c_mean') or 0) >= 0.5
        and (x.get('grad_cos') or 0) > 0 for x in mech.values())
    c['E1_s0'] = v['paired_auc']['0'] > 0; c['E1_s1'] = v['paired_auc']['1'] > 0
    c['E2_final'] = v['paired_end']['0'] >= -3.0 and v['eacc_s0_4000'] > T24_EACC_4000
    if not c['M_mechanism']:
        out['verdict'] = 'mechanism_failure'
    elif not c['C5_safety']:
        out['verdict'] = 'harmful'
    elif c['E1_s0'] and c['E1_s1'] and c['E2_final']:
        out['verdict'] = 'pass'
    elif not (c['E1_s0'] or c['E1_s1']):
        out['verdict'] = 'no_gain'
    else:
        out['verdict'] = 'mixed'
    return out


def decide_v251(r: dict, mech: dict | None = None) -> dict:
    """T2.5.1 validation (pre-registered with its run order): as v25, with the mechanism criterion of the one-step
    stage-2 rows in place of the Retrace trace (T2.5 validation measured c ~ 0.15: the trace collapses under policy drift).

      M  mechanism    training logs of both EACC seeds (mean over probes from 1000 on): two_stage_share >= 0.5,
                      decisive_share > 0, w2_mean >= 0.5 (stage-2 rows are not discounted away),
                      pi_gap_sigma_median <= 0.25 (the label's action lies within a quarter sigma of the current policy's
                      in critic value), grad_cos > 0; c_mean is reported only
      C0, E1, E2, S and the verdicts as v25."""
    out = decide_v25(r, {k: dict(x, c_mean=1.0) for k, x in (mech or {}).items()})
    if out['verdict'] == 'incomplete':
        return out
    c, v = out['criteria'], out['values']
    mech = mech or {}
    v['mechanism'] = mech
    c['M_mechanism'] = bool(mech) and all(
        (x.get('two_stage_share') or 0) >= 0.5 and (x.get('decisive_share') or 0) > 0 and (x.get('w2_mean') or 0) >= 0.5
        and x.get('pi_gap_sigma_median') is not None and x['pi_gap_sigma_median'] <= 0.25
        and (x.get('grad_cos') or 0) > 0 for x in mech.values())
    if not c['M_mechanism']:
        out['verdict'] = 'mechanism_failure'
    return out


T26_CHECKPOINTS = (500, 1000, 1500, 2000, 2500, 3000)


def decide_t26(r: dict, mech: dict | None = None, checkpoints=T26_CHECKPOINTS) -> dict:
    """T2.6 time-pressure development test (pre-registered with its run order): SAC and EACC trained together from zero on
    the selected profile, seeds 0 and 1, 3000 decisions, checkpoints 500..3000 on the 24 development openings.

      C0 complete     both arms, both seeds, every checkpoint 24/24, no ERROR file
      M  mechanism    as v251 (both EACC runs, probes from 1000 on)
      E1 efficiency   paired EACC - SAC averaged over the checkpoints: > 0 on both seeds and >= +5 averaged over the seeds
      U  urgent       urgent captures EACC - SAC summed over the checkpoints: >= +4 over both seeds, neither seed < 0
      E2 final        paired at the last checkpoint, seed mean >= -3
      S  safety       each seed: EACC unsafe over the checkpoints <= SAC + 1; EACC captures at the last checkpoint >= SAC - 1
    Verdict: incomplete | mechanism_failure | harmful (E2 or S fails) | pass (all) | partial (E1 > 0 on both seeds but below
    +5, or U fails with E1 passing, or U passes with E1 > 0 on one seed) | no_gain."""
    paired = r.get('paired', {}); seeds = ('0', '1'); last = checkpoints[-1]
    pr = lambda s, k: paired.get(f's{s}/step_{k}')
    out = {'criteria': {}, 'values': {}}; c = out['criteria']; v = out['values']
    c['C0_complete'] = bool(r.get('all_complete')) and all(pr(s, k) and pr(s, k)['n'] == 24 for s in seeds for k in checkpoints)
    if not c['C0_complete']:
        out['verdict'] = 'incomplete'; return out
    v['paired_auc'] = {s: float(np.mean([pr(s, k)['mean'] for k in checkpoints])) for s in seeds}
    v['paired_auc_urgent'] = {s: float(np.mean([pr(s, k)['urgent'] for k in checkpoints])) for s in seeds}
    v['urgent_capture_gain'] = {s: int(sum(pr(s, k)['urgent_capture_mve'] - pr(s, k)['urgent_capture_sac'] for k in checkpoints)) for s in seeds}
    v['paired_end'] = {s: pr(s, last)['mean'] for s in seeds}
    v['unsafe'] = {s: {'mve': int(sum(pr(s, k)['unsafe_mve'] for k in checkpoints)), 'sac': int(sum(pr(s, k)['unsafe_sac'] for k in checkpoints))} for s in seeds}
    mech = mech or {}; v['mechanism'] = mech
    c['M_mechanism'] = bool(mech) and all(
        (x.get('two_stage_share') or 0) >= 0.5 and (x.get('decisive_share') or 0) > 0 and (x.get('w2_mean') or 0) >= 0.5
        and x.get('pi_gap_sigma_median') is not None and x['pi_gap_sigma_median'] <= 0.25 and (x.get('grad_cos') or 0) > 0 for x in mech.values())
    pos = [v['paired_auc'][s] > 0 for s in seeds]
    c['E1_efficiency'] = all(pos) and float(np.mean(list(v['paired_auc'].values()))) >= 5.0
    c['U_urgent'] = sum(v['urgent_capture_gain'].values()) >= 4 and min(v['urgent_capture_gain'].values()) >= 0
    c['E2_final'] = float(np.mean(list(v['paired_end'].values()))) >= -3.0
    c['S_safety'] = all(v['unsafe'][s]['mve'] <= v['unsafe'][s]['sac'] + 1 and pr(s, last)['capture_mve'] >= pr(s, last)['capture_sac'] - 1 for s in seeds)
    if not c['M_mechanism']:
        out['verdict'] = 'mechanism_failure'
    elif not (c['E2_final'] and c['S_safety']):
        out['verdict'] = 'harmful'
    elif c['E1_efficiency'] and c['U_urgent']:
        out['verdict'] = 'pass'
    elif all(pos) or (c['U_urgent'] and any(pos)):
        out['verdict'] = 'partial'
    else:
        out['verdict'] = 'no_gain'
    return out


def mechanism_readings(train_root, arms=('TRAIN_e_mve_s0', 'TRAIN_e_mve_s1'), start=1000) -> dict:
    """Mean of the T2.5 mechanism probes of each EACC run from ``start`` decisions on (model_stats.json)."""
    out = {}
    for arm in arms:
        f = Path(train_root) / arm / 'model_stats.json'
        if not f.exists():
            continue
        ms = [m for m in json.loads(f.read_text(encoding='utf-8')) if m['steps'] >= start]
        out[arm] = {k: (float(np.mean([m[k] for m in ms if m.get(k) is not None])) if any(m.get(k) is not None for m in ms) else None)
                    for k in ('two_stage_share', 'decisive_share', 'c_mean', 'grad_cos', 'loss_share', 'increment_abs_mean', 'w_mean',
                              'w2_mean', 'pi_gap_sigma_median', 'stage2_rows', 'stage2_entry_share')}
    return out


T25_CHECKPOINTS = (500, 1000, 1500, 2000, 2500, 3000)


def decide_t25(r: dict, checkpoints=T25_CHECKPOINTS, min_seeds: int = 3, require_converged: bool = True) -> dict:
    """T2.5 formal development test (pre-registered with its run order): SAC and EACC trained together from zero,
    seeds 0, 1, 2, 3000 decisions, checkpoints 0 and 500..3000 on the 24 development openings.

      C0 complete     every arm, seed and checkpoint 24/24 and no ERROR file
      P1 efficiency   paired EACC - SAC averaged over 500..3000 (learning-curve area): seed mean > 0 and >= 2 of 3 seeds > 0
      P2 urgent       the same area on urgent missions: seed mean > 0
      P3 final        paired at 3000: seed mean >= -3 (not worse at the end)
      S  safety       every seed at 3000: EACC captures >= SAC - 1; total EACC unsafe <= total SAC unsafe + 1
      C  converged    each arm: seed mean |R(3000) - R(2500)| <= 10
    Verdict: incomplete | pass (all) | partial (P1, P3, S hold; P2 or C fails) | no_gain (P1 fails; P3 and S hold) |
    harmful (P3 or S fails).

    Validation use (``--checkpoints 1000,2000 --min-seeds 2 --no-converged``): two seeds, P1 needs the seed mean > 0
    and at least half of the seeds > 0; C is reported, not required. The 0-step anchor is required for EACC only
    (both arms share the initial networks of a seed)."""
    arms = r['arms']; paired = r.get('paired', {})
    seeds = sorted({a.split('_s')[-1] for a in arms if a.startswith('e_mve_s')} & {a.split('_s')[-1] for a in arms if a.startswith('e_sac_s')})
    get = lambda arm, k: (arms.get(arm) or {}).get(str(k))
    pr = lambda s, k: paired.get(f's{s}/step_{k}')
    out = {'seeds': seeds, 'criteria': {}, 'values': {}}; c = out['criteria']; v = out['values']
    c['C0_complete'] = bool(r.get('all_complete')) and len(seeds) >= min_seeds and all(get(f'e_mve_s{s}', 0) for s in seeds) \
        and all(get(f'e_{a}_s{s}', k) for a in ('mve', 'sac') for s in seeds for k in checkpoints) and all(pr(s, k) for s in seeds for k in checkpoints)
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
    c['P1_efficiency'] = float(np.mean(list(v['auc'].values()))) > 0 and sum(x > 0 for x in v['auc'].values()) >= (len(seeds) + 1) // 2
    c['P2_urgent'] = float(np.mean(list(v['auc_urgent'].values()))) > 0
    c['P3_final'] = float(np.mean(list(v['final'].values()))) >= -3.0
    c['S_safety'] = all(pr(s, last)['capture_mve'] >= pr(s, last)['capture_sac'] - 1 for s in seeds) \
        and sum(pr(s, last)['unsafe_mve'] for s in seeds) <= sum(pr(s, last)['unsafe_sac'] for s in seeds) + 1
    c['C_converged'] = all(x <= 10.0 for x in v['last_change'].values())
    if not (c['P3_final'] and c['S_safety']):
        out['verdict'] = 'harmful'
    elif not c['P1_efficiency']:
        out['verdict'] = 'no_gain'
    elif c['P2_urgent'] and (c['C_converged'] or not require_converged):
        out['verdict'] = 'pass'
    else:
        out['verdict'] = 'partial'
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readout', required=True); p.add_argument('--output', required=True); p.add_argument('--final', type=int, default=None)
    p.add_argument('--criteria', choices=['t22', 't23', 't24', 't25', 'v25', 'v251', 't26'], default='t22')
    p.add_argument('--train-root', default=None, help='v25 / v251: the directory holding TRAIN_e_mve_s0 / _s1 (mechanism readings)')
    p.add_argument('--checkpoints', default=None, help='t25: comparison checkpoints, e.g. 1000,2000')
    p.add_argument('--min-seeds', type=int, default=3); p.add_argument('--no-converged', action='store_true')
    a = p.parse_args()
    r = json.loads(Path(a.readout).read_text(encoding='utf-8'))
    ck = tuple(int(x) for x in a.checkpoints.split(',')) if a.checkpoints else T25_CHECKPOINTS
    if a.criteria in ('v25', 'v251', 't26'):
        if not a.train_root:
            p.error('--criteria v25 / v251 / t26 needs --train-root (the mechanism readings)')
        mech = mechanism_readings(a.train_root)
        res = {'v25': decide_v25, 'v251': decide_v251, 't26': decide_t26}[a.criteria](r, mech)
    else:
        res = decide_t25(r, ck, a.min_seeds, not a.no_converged) if a.criteria == 't25' else decide_t24(r) if a.criteria == 't24' else decide_t23(r, a.final or 4000) if a.criteria == 't23' else decide(r, a.final or 3000)
    Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
