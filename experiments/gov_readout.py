"""ECC final round readouts (spec ``docs/final_method/ECC_FINAL_SPEC.md``; registered with the spec, before any data).

    python -B -m experiments.gov_readout screen --root DIR --output SCREEN_DECISION.json
    python -B -m experiments.gov_readout a      --root DIR --output A_DECISION.json
    python -B -m experiments.gov_readout b      --root DIR --output B_DECISION.json [--coordination gov|cert] [--a-decision A.json]

``screen`` (step 0): ``DIR/S/<policy>/`` of the stage ``gov_screen`` (opportunity-scarcity condition, non-learning policies);
gates F, D1, D2, M (section 6). ``a`` (step A): ``DIR/C/{EACH_C,SAC_C}_s{10,11}_{monitor,gov,cert}`` -- same model, monitor
(= uncoordinated) against governor; criteria M1-M5 (section 7); M2 or M3 failing rejects the governor (step B then uses the old
certification). ``b`` (step B): ``DIR/C/...`` of the stage ``gov_b`` (and ``*_natural``); the selection order of section 7:
1 in-loop method, 2 EACH_C + deployment coordination, 3 the frozen ``ECC_deploy``. G1-G5 are ``experiments.u2_readout.gates``
on the N1 tier (seed-matched, opening-paired). Everything else is reported, not gated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.u2_readout import MARGIN, SEEDS, arm_summary, gates, load_eval, paired

TIERS = (1, 2, 3)
STAGE_PAIRED_FLIP = 200.0                  # one capture / timeout flip (capture +100 against timeout -100)
FIXED_STAGE_POLICIES = ('rally', 'c_10_0', 'c_10_-5', 'c_6_5', 'b_10')
GATE_F, GATE_D1, GATE_D2, GATE_M = 0.80, 0.15, 0.10, 0.80
N3_REPLACE_PER_EPISODE = 0.05


def tier_of(r: dict) -> int:
    return int(r['mission']['tier'])


def by_tier(ev: dict, n: int) -> dict:
    return {o: r for o, r in ev.items() if tier_of(r) == n}


def cap(rs) -> float:
    rs = list(rs)
    return float(np.mean([r['event'] == 'success' for r in rs])) if rs else float('nan')


# ---------------------------------------------------------------------- step 0: screen gates
def screen(root: Path) -> dict:
    pols = {p.name: load_eval(p) for p in sorted((root / 'S').iterdir()) if p.is_dir()}
    out = {'policies': sorted(pols), 'tiers': {}}
    for n in TIERS:
        t = {k: by_tier(v, n) for k, v in pols.items()}
        out['tiers'][n] = {k: {'n': len(v), 'capture_rate': cap(v.values()), 'unsafe': sum(r['event'] == 'unsafe' for r in v.values()),
                               'mean_return': float(np.mean([r['return'] for r in v.values()])) if v else None} for k, v in t.items()}
    t1 = {k: by_tier(v, 1) for k, v in pols.items()}
    g = out['tiers'][1]['gstar']['capture_rate']
    fixed = {k: out['tiers'][1][k]['capture_rate'] for k in FIXED_STAGE_POLICIES if k in t1}
    best = max(fixed.values())
    ops = sorted(set.intersection(*[set(t1[k]) for k in fixed]))
    oracle = float(np.mean([any(t1[k][o]['event'] == 'success' for k in fixed) for o in ops])) if ops else float('nan')
    g3 = out['tiers'][3]['gstar']['capture_rate']
    gates_ = {'F': g >= GATE_F, 'D1': best <= g - GATE_D1, 'D2': oracle >= best + GATE_D2, 'M': g3 >= GATE_M}
    out.update(gstar_n1=g, best_fixed_n1=best, best_fixed_policy=max(fixed, key=fixed.get), oracle_fixed_n1=oracle, gstar_n3=g3,
               gates={k: bool(v) for k, v in gates_.items()})
    if all(gates_.values()):
        out['verdict'] = 'pass'
    elif not gates_['F'] or not gates_['M']:
        out['verdict'] = 'too_hard'; out['alternative'] = 'scarcity:strict'
    else:
        out['verdict'] = 'too_easy'; out['alternative'] = 'scarcity:inside'
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


# ---------------------------------------------------------------------- step A
def readout_a(root: Path) -> dict:
    models = [f'{c}_s{s}' for c in ('EACH_C', 'SAC_C') for s in SEEDS]
    ev = {(m, k): load_eval(root / 'C' / f'{m}_{k}') for m in models for k in ('monitor', 'gov', 'cert')}
    out = {'evaluations': {f'{m}_{k}': len(v) for (m, k), v in ev.items()}, 'tiers': {}}
    out['complete'] = all(v == 96 for v in out['evaluations'].values())
    lost, unsafe = [], []
    for n in TIERS:
        rows = {}
        pooled = {'gov_vs_monitor': [], 'cert_vs_monitor': [], 'gov_vs_cert': []}
        caps = {'gov': 0, 'monitor': 0, 'cert': 0}
        for m in models:
            G, M, C = (by_tier(ev[(m, k)], n) for k in ('gov', 'monitor', 'cert'))
            gm, cm, gc = same_model(G, M), same_model(C, M), same_model(G, C)
            rows[m] = {'gov_vs_monitor': gm, 'cert_vs_monitor': cm, 'gov_vs_cert': gc,
                       'destroy_rate_monitor': destroy_rate(M), 'gov_replace_per_episode': replace_per_episode(G),
                       'gov_realization_persistence': realization_persistence(G), 'gov_wall': gov_wall(G)}
            for key, p in (('gov_vs_monitor', gm), ('cert_vs_monitor', cm), ('gov_vs_cert', gc)):
                if p['mean'] is not None:
                    pooled[key] += [p['mean']] * p['n']
            caps['gov'] += gm['capture'][0]; caps['monitor'] += gm['capture'][1]; caps['cert'] += cm['capture'][0]
            lost += [(m, n, o) for o in gm['lost']]; unsafe += [(m, n, o) for o in gm['unsafe_introduced']]
        n_pairs = sum(r['gov_vs_monitor']['n'] for r in rows.values())
        G_all = {(m, o): r for m in models for o, r in by_tier(ev[(m, 'gov')], n).items()}
        M_all = {(m, o): r for m in models for o, r in by_tier(ev[(m, 'monitor')], n).items()}
        rp = realization_persistence(G_all)
        dr = destroy_rate(M_all)
        pred = (dr * rp['realization'] * STAGE_PAIRED_FLIP) if (dr is not None and rp['realization'] is not None) else 0.0
        obs = float(np.mean(pooled['gov_vs_monitor'])) if pooled['gov_vs_monitor'] else None
        band = STAGE_PAIRED_FLIP / max(n_pairs, 1)
        sign = lambda x: 0 if x is None or abs(x) < band else (1 if x > 0 else -1)
        out['tiers'][n] = {'models': rows, 'pairs': n_pairs, 'captures': caps,
                           'gov_vs_monitor_mean': obs,
                           'cert_vs_monitor_mean': float(np.mean(pooled['cert_vs_monitor'])) if pooled['cert_vs_monitor'] else None,
                           'gov_vs_cert_mean': float(np.mean(pooled['gov_vs_cert'])) if pooled['gov_vs_cert'] else None,
                           'destroy_rate': dr, 'realization': rp['realization'], 'persistence': rp['persistence'], 'replacements': rp['replacements'],
                           'prediction_return': pred, 'zero_band': band, 'direction_consistent': sign(pred) == sign(obs),
                           'gov_replace_per_episode': replace_per_episode(G_all)}
    t1, t3 = out['tiers'][1], out['tiers'][3]
    crit = {'M1': (t1['gov_vs_monitor_mean'] or 0) > 0 and t1['captures']['gov'] > t1['captures']['monitor'],
            'M2': not lost, 'M3': not unsafe,
            'M4': (t3['gov_replace_per_episode'] or 0) <= N3_REPLACE_PER_EPISODE,
            'M5': all(out['tiers'][n]['direction_consistent'] for n in TIERS)}
    out.update(criteria={k: bool(v) for k, v in crit.items()}, lost=lost, unsafe_introduced=unsafe,
               mechanism_pass=all(crit.values()), governor_rejected=not (crit['M2'] and crit['M3']),
               b_coordination='cert' if not (crit['M2'] and crit['M3']) else 'gov')
    return out


# ---------------------------------------------------------------------- step B
def tiered(evals: dict, n: int | None) -> dict:
    return evals if n is None else {k: by_tier(v, n) for k, v in evals.items()}


def readout_b(root: Path, coordination: str = 'gov') -> dict:
    loop, dep = ('ECC_GOV', 'EACH_C_{s}_gov') if coordination == 'gov' else ('ECC_CERT_LOOP', 'EACH_C_{s}_cert')
    names = {'SAC_C': 'SAC_C_{s}', 'EACH_C': 'EACH_C_{s}_monitor', 'DEPLOY': dep, 'LOOP': loop + '_{s}',
             'LOOP_MONITOR': loop + '_{s}_monitor', 'ECC_DEPLOY_OLD': 'EACH_C_{s}_cert'}
    ev, nat = {}, {}
    for k, pat in names.items():
        for s in SEEDS:
            d = pat.format(s=f's{s}')
            ev[f'{k}_s{s}'] = load_eval(root / 'C' / d)
            nat[f'{k}_s{s}'] = load_eval(root / 'C' / f'{d}_natural')
    out = {'coordination': coordination, 'evaluations': {k: len(v) for k, v in ev.items()},
           'natural_evaluations': {k: len(v) for k, v in nat.items()}}
    out['complete'] = all(v == 96 for v in out['evaluations'].values()) and all(v == 96 for v in out['natural_evaluations'].values())

    def rung(cand):
        p = paired(tiered(ev, 1), cand, 'SAC_C'); g = gates(p)
        pn = paired(nat, cand, 'SAC_C')
        natural_ok = pn['mean'] is not None and pn['mean'] >= -MARGIN and pn['unsafe_first'] == 0
        return {'n1': {**p, 'gates': g}, 'natural': {**pn, 'not_inferior': bool(natural_ok)}, 'pass': bool(all(g.values()) and natural_ok)}
    r1, r2 = rung('LOOP'), rung('DEPLOY')
    final = 'in_loop' if r1['pass'] else ('deploy' if r2['pass'] else 'ECC_deploy')
    out.update(rung1_in_loop=r1, rung2_deploy=r2, final_method=final,
               final_config={'in_loop': loop, 'deploy': f'EACH_C + {coordination} at deployment', 'ECC_deploy': 'EACH_C + old certification at deployment (RUN_ECC)'}[final])
    # reported only
    dose = {}
    for n in TIERS:
        t = tiered(ev, n)
        dose[n] = {k: {'vs_SAC_C': paired(t, k, 'SAC_C'), 'summary': arm_summary([r for s in SEEDS for r in t[f'{k}_s{s}'].values()])}
                   for k in ('EACH_C', 'DEPLOY', 'LOOP', 'ECC_DEPLOY_OLD')}
        dose[n]['SAC_C'] = {'summary': arm_summary([r for s in SEEDS for r in t[f'SAC_C_s{s}'].values()])}
        dose[n]['same_model_deploy_vs_monitor'] = paired(t, 'DEPLOY', 'EACH_C')
        dose[n]['loop_vs_loop_monitor'] = paired(t, 'LOOP', 'LOOP_MONITOR')
        for k in ('DEPLOY', 'LOOP'):
            merged = {(s, o): r for s in SEEDS for o, r in t[f'{k}_s{s}'].items()}
            dose[n][f'{k}_realization_persistence'] = realization_persistence(merged)
            dose[n][f'{k}_replace_per_episode'] = replace_per_episode(merged)
        for k in ('EACH_C', 'LOOP_MONITOR'):
            merged = {(s, o): r for s in SEEDS for o, r in t[f'{k}_s{s}'].items()}
            dose[n][f'{k}_destroy_rate'] = destroy_rate(merged)
    out['dose'] = dose
    out['natural_report'] = {k: paired(nat, k, 'SAC_C') for k in ('EACH_C', 'DEPLOY', 'LOOP', 'ECC_DEPLOY_OLD')}
    out['gov_wall'] = {k: gov_wall({(s, o): r for s in SEEDS for o, r in ev[f'{k}_s{s}'].items()}) for k in ('DEPLOY', 'LOOP')}
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
                        x['replacements'] += int((e.get('exec_stats') or {}).get('gov_replacements', 0) or (e.get('exec_stats') or {}).get('cert_interventions', 0) or 0)
                    curves[f'{c}_s{s}'] = {**{k: cur[k] for k in ('curve_auc', 'late_return', 'late_urgent_capture', 'episodes')},
                                           'per_500': bins}
    except Exception as exc:                       # reported, never gated
        curves['error'] = repr(exc)
    out['curves'] = curves
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd', choices=('screen', 'a', 'b'))
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--coordination', choices=('gov', 'cert')); p.add_argument('--a-decision')
    a = p.parse_args()
    root = Path(a.root)
    if a.cmd == 'screen':
        res = screen(root)
    elif a.cmd == 'a':
        res = readout_a(root)
    else:
        coord = a.coordination
        if coord is None and a.a_decision:
            coord = json.loads(Path(a.a_decision).read_text(encoding='utf-8'))['b_coordination']
        res = readout_b(root, coord or 'gov')
    Path(a.output).write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
    keys = {'screen': ('verdict', 'gates', 'gstar_n1', 'best_fixed_n1', 'oracle_fixed_n1', 'gstar_n3'),
            'a': ('complete', 'criteria', 'mechanism_pass', 'governor_rejected', 'b_coordination'),
            'b': ('complete', 'final_method', 'final_config')}[a.cmd]
    print(json.dumps({k: res.get(k) for k in keys}, indent=1, default=str))


if __name__ == '__main__':
    main()
