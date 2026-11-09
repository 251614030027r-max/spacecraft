"""ECC final round readouts (spec ``docs/final_method/ECC_FINAL_SPEC.md``; registered before the data they read).

    python -B -m experiments.gov_readout screen     --root DIR --output SCREEN_DECISION.json
    python -B -m experiments.gov_readout fixedcheck --root DIR --output FIXED_CHECK.json
    python -B -m experiments.gov_readout a          --root DIR --output A_DECISION.json
    python -B -m experiments.gov_readout b          --root DIR --output B_DECISION.json [--coordination gov|cert] [--a-decision A.json]

Condition (section 6): urgent missions (deadline U(140, 200) s) are the primary endpoint; economy missions are the no-harm
check; the natural 1:1 distribution is the external validity. Deadline bins (< 160, 160-180, >= 180 s, economy) are reported
for the dose relation only (the bins come from the exploratory RUN_ECC observation). ``screen``: gates L1-L3 on ``DIR/S/<policy>``
(urgent, non-learning policies). ``fixedcheck``: section 6B (``DIR/S/<policy>``, ``_cert``, ``_gov``). ``a``: criteria M1-M5 on
``DIR/C/{EACH_C,SAC_C}_s{10,11}_{monitor,gov,cert}[_economy|_natural]``; M2 or M3 failing rejects the governor. ``b``: the selection
order of section 7 (G1-G5 of ``experiments.u2_readout.gates`` on the urgent pairs; natural 1:1 not inferior).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.u2_readout import MARGIN, SEEDS, arm_summary, gates, load_eval, paired

STAGE_PAIRED_FLIP = 200.0                  # one capture / timeout flip (capture +100 against timeout -100)
FIXED_STAGE_POLICIES = ('rally', 'c_10_0', 'c_10_-5', 'c_6_5', 'b_10')
GATE_L1, GATE_L2, GATE_L3 = 0.80, 0.15, 0.15          # ECC_FINAL_SPEC section 6 (user 2026-11-08)
SHORTEST = 6
N_COND = {'urgent': 96, 'economy': 48, 'natural': 48}
SUFFIX = {'urgent': '', 'economy': '_economy', 'natural': '_natural'}
ECONOMY_REPLACE_PER_EPISODE = 0.05
BINS = ('<160', '160-180', '>=180', 'economy')


def bin_of(r: dict) -> str:
    m = r['mission']
    if not m['urgent']:
        return 'economy'
    d = float(m['deadline_s'])
    return '<160' if d < 160 else ('160-180' if d < 180 else '>=180')


def by_bin(ev: dict, b: str) -> dict:
    return {o: r for o, r in ev.items() if bin_of(r) == b}


def cap(rs) -> float:
    rs = list(rs)
    return float(np.mean([r['event'] == 'success' for r in rs])) if rs else float('nan')


# ---------------------------------------------------------------------- step 0: screen gates
def screen(root: Path) -> dict:
    pols = {p.name: load_eval(p) for p in sorted((root / 'S').iterdir()) if p.is_dir() and not p.name.endswith(('_cert', '_gov'))}
    out = {'policies': {k: {'n': len(v), 'capture_rate': cap(v.values()), 'unsafe': sum(r['event'] == 'unsafe' for r in v.values()),
                            'mean_return': float(np.mean([r['return'] for r in v.values()])) if v else None} for k, v in pols.items()}}
    g = out['policies']['gstar']['capture_rate']
    fixed = {k: out['policies'][k]['capture_rate'] for k in FIXED_STAGE_POLICIES if k in pols}
    best = max(fixed.values())
    ops = sorted(set.intersection(*[set(pols[k]) for k in fixed]))
    oracle = float(np.mean([any(pols[k][o]['event'] == 'success' for k in fixed) for o in ops])) if ops else float('nan')
    gates_ = {'L1': g >= GATE_L1, 'L2': oracle >= g - GATE_L2, 'L3': best <= g - GATE_L3}
    short = sorted(pols['gstar'].values(), key=lambda r: r['mission']['deadline_s'])[:SHORTEST]
    out.update(gstar=g, best_fixed=best, best_fixed_policy=max(fixed, key=fixed.get), oracle_fixed=oracle, fixed_policies_screened=sorted(fixed),
               shortest_gstar=[{'opening': r['opening'], 'deadline_s': r['mission']['deadline_s'], 'event': r['event']} for r in short],
               only_gstar=[o for o in ops if pols['gstar'][o]['event'] == 'success' and not any(pols[k][o]['event'] == 'success' for k in fixed)],
               by_bin={b: {k: cap(by_bin(v, b).values()) for k, v in pols.items()} for b in BINS[:3]},
               gates={k: bool(v) for k, v in gates_.items()})
    out['verdict'] = 'pass' if all(gates_.values()) else ('L2_low_only' if gates_['L1'] and gates_['L3'] else 'fail')
    return out


def paired_fixed(root: Path) -> dict:
    """Section 6B sandbox paired check: ``DIR/S/<policy>`` (uncoordinated = the screen), ``DIR/S/<policy>_cert`` and
    ``DIR/S/<policy>_gov`` on the screen openings. Requirement: the governor loses no capture, introduces no unsafe and
    rescues at least as many as the old certification. Reported: interventions, realization, persistence."""
    out = {'policies': {}}
    tot = {'gov_rescued': 0, 'cert_rescued': 0, 'gov_lost': 0, 'gov_unsafe_introduced': 0, 'cert_lost': 0}
    for d in sorted((root / 'S').iterdir()):
        if not d.is_dir() or d.name.endswith(('_cert', '_gov')):
            continue
        base, cert, gov = load_eval(d), load_eval(root / 'S' / f'{d.name}_cert'), load_eval(root / 'S' / f'{d.name}_gov')
        if not cert or not gov:
            continue
        gm, cm = same_model(gov, base), same_model(cert, base)
        out['policies'][d.name] = {'gov_vs_off': gm, 'cert_vs_off': cm, 'gov_replacements': sum(r.get('gov_replacements', 0) or 0 for r in gov.values()),
                                   'cert_interventions': sum(r.get('cert_interventions', 0) or 0 for r in cert.values()),
                                   **realization_persistence(gov), 'gov_wall': gov_wall(gov)}
        tot['gov_rescued'] += len(gm['rescued']); tot['cert_rescued'] += len(cm['rescued']); tot['gov_lost'] += len(gm['lost'])
        tot['gov_unsafe_introduced'] += len(gm['unsafe_introduced']); tot['cert_lost'] += len(cm['lost'])
    allgov = {(p, o): r for p in out['policies'] for o, r in load_eval(root / 'S' / f'{p}_gov').items()}
    out.update(totals=tot, **{f'pooled_{k}': v for k, v in realization_persistence(allgov).items()},
               requirement={'no_lost': tot['gov_lost'] == 0, 'no_unsafe_introduced': tot['gov_unsafe_introduced'] == 0,
                            'rescued_ge_cert': tot['gov_rescued'] >= tot['cert_rescued']})
    out['pass'] = all(out['requirement'].values())
    return out


# ---------------------------------------------------------------------- governor records
def gov_records(ev: dict) -> list:
    """(opening, decision index, decision record, episode) for every decision with a governor record."""
    out = []
    for o, r in ev.items():
        for i, d in enumerate(r.get('decisions', [])):
            if d.get('gov') is not None:
                out.append((o, i, d, r))
    return out


def realization_persistence(ev: dict) -> dict:
    """Over 'replace' decisions actually flown (act mode): realization -- the episode enters an entry / descent mode in a
    macro step that starts no later than the predicted event time + 16 s; persistence -- the event is entered in the
    replaced step itself, or the next decision still has an admissible candidate (status other than 'none')."""
    real, pers, n = 0, 0, 0
    for o, i, d, r in gov_records(ev):
        g = d['gov']
        if g.get('status') != 'replace' or not g.get('act'):
            continue
        n += 1
        ds = r['decisions']; t_pred = float(g['t']) + float(g['alt_t']) + 16.0
        starts = [0.0] + [float(x['t']) for x in ds[:-1]]
        real += any(ds[j]['mode'] in ('bridge', 'descent') and starts[j] <= t_pred + 1e-6 for j in range(i, len(ds)))
        nxt = ds[i + 1].get('gov') if i + 1 < len(ds) else None
        pers += ds[i]['mode'] in ('bridge', 'descent') or (nxt is not None and nxt.get('status') != 'none')
    return {'replacements': n, 'realization': real / n if n else None, 'persistence': pers / n if n else None}


def destroy_rate(ev: dict) -> float | None:
    """Share of episodes with at least one 'replace' judgement (monitor mode: the uncoordinated proposal would destroy the
    critical opportunity while an admissible alternative exists)."""
    if not ev:
        return None
    return float(np.mean([any((d.get('gov') or {}).get('status') == 'replace' for d in r.get('decisions', [])) for r in ev.values()]))


def replace_per_episode(ev: dict) -> float | None:
    if not ev:
        return None
    return float(np.mean([sum((d.get('gov') or {}).get('status') == 'replace' for d in r.get('decisions', [])) for r in ev.values()]))


def gov_wall(ev: dict) -> dict:
    w = [float(d['gov']['wall_s']) for _, _, d, _ in gov_records(ev)]
    return {'decisions': len(w), 'mean_s': float(np.mean(w)) if w else None, 'p95_s': float(np.percentile(w, 95)) if w else None,
            'max_s': float(np.max(w)) if w else None}


def same_model(a: dict, b: dict) -> dict:
    """a - b on the same openings (one model): mean, captures, lost captures, unsafe introduced."""
    ops = sorted(set(a) & set(b))
    d = [a[o]['return'] - b[o]['return'] for o in ops]
    return {'n': len(ops), 'mean': float(np.mean(d)) if d else None, 'capture': [sum(a[o]['event'] == 'success' for o in ops),
                                                                               sum(b[o]['event'] == 'success' for o in ops)],
            'lost': [o for o in ops if b[o]['event'] == 'success' and a[o]['event'] != 'success'],
            'rescued': [o for o in ops if a[o]['event'] == 'success' and b[o]['event'] != 'success'],
            'unsafe_introduced': [o for o in ops if a[o]['event'] == 'unsafe' and b[o]['event'] != 'unsafe']}


def cert_mechanism(C: dict, M: dict, n_pairs: int) -> dict:
    """Section 3A (report only): the old certification's same-model mechanism readings -- episodes with an intervention,
    realization of the certified probe's predicted switch (an entry / descent mode entered in a macro step starting no later
    than the predicted time + 16 s), the proposition-4 prediction share x realization x 200 against the observed gain."""
    n_iv, real, eps_iv = 0, 0, 0
    for o, r in C.items():
        ds = r.get('decisions', []); starts = [0.0] + [float(x['t']) for x in ds[:-1]]
        hit = False
        for i, d in enumerate(ds):
            c = d.get('cert') or {}
            if not c.get('intervened'):
                continue
            hit = True; n_iv += 1
            t_pred = starts[i] + float(c.get('alt_t', 0.0)) + 16.0
            real += any(ds[j]['mode'] in ('bridge', 'descent') and starts[j] <= t_pred + 1e-6 for j in range(i, len(ds)))
        eps_iv += hit
    share = eps_iv / len(C) if C else None
    rr = real / n_iv if n_iv else None
    ops = sorted(set(C) & set(M))
    obs = float(np.mean([C[o]['return'] - M[o]['return'] for o in ops])) if ops else None
    pred = share * rr * STAGE_PAIRED_FLIP if (share is not None and rr is not None) else 0.0
    band = STAGE_PAIRED_FLIP / max(n_pairs, 1)
    sign = lambda x: 0 if x is None or abs(x) < band else (1 if x > 0 else -1)
    return {'pairs': len(ops), 'interventions': n_iv, 'episodes_intervened_share': share, 'realization': rr, 'prediction': pred,
            'observed_mean_gain': obs, 'direction_consistent': sign(pred) == sign(obs),
            'captures': [sum(C[o]['event'] == 'success' for o in ops), sum(M[o]['event'] == 'success' for o in ops)],
            'lost': [o for o in ops if M[o]['event'] == 'success' and C[o]['event'] != 'success'],
            'rescued': [o for o in ops if C[o]['event'] == 'success' and M[o]['event'] != 'success'],
            'unsafe_introduced': [o for o in ops if C[o]['event'] == 'unsafe' and M[o]['event'] != 'unsafe']}


# ---------------------------------------------------------------------- step A
def critical_stats(ev: dict) -> dict:
    """Per-decision criticality (proposition 4): share of governed decisions judged critical, and replacements on
    non-critical decisions (must be 0 by construction) and on critical ones."""
    recs = [d['gov'] for _, _, d, _ in gov_records(ev)]
    crit = [g for g in recs if g.get('critical')]
    return {'decisions': len(recs), 'critical_share': len(crit) / len(recs) if recs else None,
            'replace_non_critical': sum(g.get('status') == 'replace' and not g.get('critical') for g in recs),
            'replace_critical_share': (sum(g.get('status') == 'replace' for g in crit) / len(crit)) if crit else None}


def _cmp(G: dict, M: dict, n_pairs: int) -> dict:
    rp = realization_persistence(G); dr = destroy_rate(M)
    pred = (dr * rp['realization'] * STAGE_PAIRED_FLIP) if (dr is not None and rp['realization'] is not None) else 0.0
    ops = sorted(set(G) & set(M))
    obs = float(np.mean([G[o]['return'] - M[o]['return'] for o in ops])) if ops else None
    band = STAGE_PAIRED_FLIP / max(n_pairs, 1)
    sign = lambda x: 0 if x is None or abs(x) < band else (1 if x > 0 else -1)
    return {'pairs': len(ops), 'observed_mean_gain': obs, 'prediction': pred, 'destroy_rate': dr, **rp, 'zero_band': band,
            'direction_consistent': sign(pred) == sign(obs), 'captures': [sum(G[o]['event'] == 'success' for o in ops), sum(M[o]['event'] == 'success' for o in ops)],
            'replace_per_episode': replace_per_episode(G), 'critical_monitor': critical_stats(M), 'critical_gov': critical_stats(G)}


def readout_a(root: Path) -> dict:
    models = [f'{c}_s{s}' for c in ('EACH_C', 'SAC_C') for s in SEEDS]
    ev = {(m, k, c): load_eval(root / 'C' / f'{m}_{k}{SUFFIX[c]}') for m in models for k in ('monitor', 'gov', 'cert') for c in N_COND}
    out = {'evaluations': {f'{m}_{k}{SUFFIX[c]}': len(v) for (m, k, c), v in ev.items()}}
    out['complete'] = all(len(v) == N_COND[c] for (m, k, c), v in ev.items())
    lost, unsafe, conds = [], [], {}
    for c in N_COND:
        per = {}
        for m in models:
            G, M, C = ev[(m, 'gov', c)], ev[(m, 'monitor', c)], ev[(m, 'cert', c)]
            gm = same_model(G, M)
            per[m] = {'gov_vs_monitor': gm, 'cert_vs_monitor': same_model(C, M), 'gov_vs_cert': same_model(G, C), 'gov_wall': gov_wall(G)}
            lost += [(m, c, o) for o in gm['lost']]; unsafe += [(m, c, o) for o in gm['unsafe_introduced']]
        Gall = {(m, o): r for m in models for o, r in ev[(m, 'gov', c)].items()}
        Mall = {(m, o): r for m in models for o, r in ev[(m, 'monitor', c)].items()}
        Call = {(m, o): r for m in models for o, r in ev[(m, 'cert', c)].items()}
        conds[c] = {'models': per, **_cmp(Gall, Mall, len(set(Gall) & set(Mall))), 'cert_mechanism': cert_mechanism(Call, Mall, len(set(Call) & set(Mall))),
                    'cert_mean_gain': same_model(Call, Mall)['mean'], 'cert_captures': same_model(Call, Mall)['capture']}
        if c in ('urgent', 'economy'):
            conds[c]['dose'] = {b: _cmp(by_bin(Gall, b), by_bin(Mall, b), len(by_bin(Gall, b))) for b in BINS if by_bin(Gall, b)}
    out['conditions'] = conds
    u, e = conds['urgent'], conds['economy']
    noncrit = sum(conds[c]['critical_gov']['replace_non_critical'] for c in N_COND)
    crit = {'M1': (u['observed_mean_gain'] or 0) > 0 and u['captures'][0] > u['captures'][1],
            'M2': not lost, 'M3': not unsafe,
            'M4': (e['replace_per_episode'] or 0) <= ECONOMY_REPLACE_PER_EPISODE and noncrit == 0,
            'M5': all(conds[c]['direction_consistent'] for c in N_COND)}
    out.update(criteria={k: bool(v) for k, v in crit.items()}, lost=lost, unsafe_introduced=unsafe,
               mechanism_pass=all(crit.values()), governor_rejected=not (crit['M2'] and crit['M3']),
               b_coordination='cert' if not (crit['M2'] and crit['M3']) else 'gov')
    return out


# ---------------------------------------------------------------------- step B
def readout_b(root: Path, coordination: str = 'gov') -> dict:
    loop, dep = ('ECC_GOV', 'EACH_C_{s}_gov') if coordination == 'gov' else ('ECC_CERT_LOOP', 'EACH_C_{s}_cert')
    names = {'SAC_C': 'SAC_C_{s}', 'EACH_C': 'EACH_C_{s}_monitor', 'DEPLOY': dep, 'LOOP': loop + '_{s}',
             'LOOP_MONITOR': loop + '_{s}_monitor', 'ECC_DEPLOY_OLD': 'EACH_C_{s}_cert'}
    ev = {c: {} for c in N_COND}
    for c in N_COND:
        for k, pat in names.items():
            for s in SEEDS:
                ev[c][f'{k}_s{s}'] = load_eval(root / 'C' / (pat.format(s=f's{s}') + SUFFIX[c]))
    out = {'coordination': coordination, 'evaluations': {c: {k: len(v) for k, v in ev[c].items()} for c in N_COND}}
    out['complete'] = all(len(v) == N_COND[c] for c in N_COND for v in ev[c].values())

    def rung(cand):
        p = paired(ev['urgent'], cand, 'SAC_C'); g = gates(p)
        pn = paired(ev['natural'], cand, 'SAC_C')
        natural_ok = pn['mean'] is not None and pn['mean'] >= -MARGIN and pn['unsafe_first'] == 0
        return {'urgent': {**p, 'gates': g}, 'natural': {**pn, 'not_inferior': bool(natural_ok)},
                'economy_report': paired(ev['economy'], cand, 'SAC_C'), 'pass': bool(all(g.values()) and natural_ok)}
    r1, r2 = rung('LOOP'), rung('DEPLOY')
    final = 'in_loop' if r1['pass'] else ('deploy' if r2['pass'] else 'ECC_deploy')
    out.update(rung1_in_loop=r1, rung2_deploy=r2, final_method=final,
               final_config={'in_loop': loop, 'deploy': f'EACH_C + {coordination} at deployment', 'ECC_deploy': 'EACH_C + old certification at deployment (RUN_ECC)'}[final])
    # reported only
    dose = {}
    allu = {k: {**ev['urgent'][k], **{('e', o): r for o, r in ev['economy'][k].items()}} for k in ev['urgent']}
    for b in BINS:
        t = {k: by_bin(v, b) for k, v in allu.items()}
        dose[b] = {k: paired(t, k, 'SAC_C') for k in ('EACH_C', 'DEPLOY', 'LOOP', 'ECC_DEPLOY_OLD')}
        dose[b]['summary'] = {k: arm_summary([r for s in SEEDS for r in t[f'{k}_s{s}'].values()]) for k in ('SAC_C', 'EACH_C', 'DEPLOY', 'LOOP')}
        dose[b]['same_model_deploy_vs_monitor'] = paired(t, 'DEPLOY', 'EACH_C')
        dose[b]['loop_vs_loop_monitor'] = paired(t, 'LOOP', 'LOOP_MONITOR')
    out['dose'] = dose
    mech = {}
    for c in N_COND:
        for k, mon in (('DEPLOY', 'EACH_C'), ('LOOP', 'LOOP_MONITOR')):
            G = {(s, o): r for s in SEEDS for o, r in ev[c][f'{k}_s{s}'].items()}
            M = {(s, o): r for s in SEEDS for o, r in ev[c][f'{mon}_s{s}'].items()}
            mech[f'{k}_{c}'] = _cmp(G, M, len(set(G) & set(M)))
    out['mechanism'] = mech
    cm = {}
    for c in N_COND:                                # section 3A, report only: the certification's mechanism readings
        M = {(s, o): r for s in SEEDS for o, r in ev[c][f'EACH_C_s{s}'].items()}
        cm[f'EACH_C_cert_{c}'] = cert_mechanism({(s, o): r for s in SEEDS for o, r in ev[c][f'ECC_DEPLOY_OLD_s{s}'].items()}, M, len(M))
        if coordination == 'cert':
            LM = {(s, o): r for s in SEEDS for o, r in ev[c][f'LOOP_MONITOR_s{s}'].items()}
            cm[f'LOOP_{c}'] = cert_mechanism({(s, o): r for s in SEEDS for o, r in ev[c][f'LOOP_s{s}'].items()}, LM, len(LM))
    out['cert_mechanism'] = cm
    out['policy_behaviour'] = {c: {k: critical_stats({(s, o): r for s in SEEDS for o, r in ev[c][f'{k}_s{s}'].items()}) | {
        'destroy_rate': destroy_rate({(s, o): r for s in SEEDS for o, r in ev[c][f'{k}_s{s}'].items()})} for k in ('EACH_C', 'LOOP_MONITOR')} for c in N_COND}
    out['gov_wall'] = {k: gov_wall({(s, o): r for s in SEEDS for o, r in ev['urgent'][f'{k}_s{s}'].items()}) for k in ('DEPLOY', 'LOOP')}
    curves = {}
    try:
        from experiments.t2_curves import run_curve
        for c in ('SAC_C', 'EACH_C', loop):
            for s in SEEDS:
                f = root / 'B' / f'TRAIN_{c}_s{s}' / 'episodes.json'
                if f.exists():
                    eps = json.loads(f.read_text(encoding='utf-8'))
                    cur = run_curve(eps, 50, 3000, grid_from=250)
                    bins = {}
                    for e in eps:
                        b = min(int(e['steps']) // 500, 5)
                        x = bins.setdefault(b, {'episodes': 0, 'timeouts': 0, 'unsafe': 0, 'replacements': 0})
                        x['episodes'] += 1; x['timeouts'] += e['event'] in ('timeout', 'resource_exhausted'); x['unsafe'] += e['event'] == 'unsafe'
                        st = e.get('exec_stats') or {}
                        x['replacements'] += int(st.get('gov_replacements', 0) or st.get('cert_interventions', 0) or 0)
                    curves[f'{c}_s{s}'] = {**{k: cur[k] for k in ('curve_auc', 'late_return', 'late_urgent_capture', 'episodes')}, 'per_500': bins}
    except Exception as exc:                       # reported, never gated
        curves['error'] = repr(exc)
    out['curves'] = curves
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd', choices=('screen', 'fixedcheck', 'a', 'b'))
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--coordination', choices=('gov', 'cert')); p.add_argument('--a-decision')
    a = p.parse_args()
    root = Path(a.root)
    if a.cmd == 'screen':
        res = screen(root)
    elif a.cmd == 'fixedcheck':
        res = paired_fixed(root)
    elif a.cmd == 'a':
        res = readout_a(root)
    else:
        coord = a.coordination
        if coord is None and a.a_decision:
            coord = json.loads(Path(a.a_decision).read_text(encoding='utf-8'))['b_coordination']
        res = readout_b(root, coord or 'gov')
    Path(a.output).write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
    keys = {'screen': ('verdict', 'gates', 'gstar', 'best_fixed', 'best_fixed_policy', 'oracle_fixed', 'shortest_gstar'),
            'fixedcheck': ('pass', 'requirement', 'totals', 'pooled_realization', 'pooled_persistence'),
            'a': ('complete', 'criteria', 'mechanism_pass', 'governor_rejected', 'b_coordination'),
            'b': ('complete', 'final_method', 'final_config')}[a.cmd]
    print(json.dumps({k: res.get(k) for k in keys}, indent=1, default=str))


if __name__ == '__main__':
    main()
