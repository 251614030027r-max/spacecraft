"""T2 readouts (no simulation).

    python -B -m experiments.t2_readout screen --dir SCREEN_DIR --output A.json
    python -B -m experiments.t2_readout curve  --eval-root EVAL_ROOT --train-root TRAIN_ROOT --screen SCREEN_DIR --output B.json

Screen (stage A, pre-registered gates):
  A1 mechanics: every episode finished; held + rejected stages <= 10 % of stages.
  A2 state dependence: per-opening best fixed policy minus the best *constant-action* policy
     >= 10 (mean) and >= 5 after dropping the two largest per-opening gains.
  A3 spread: at least three policies are each the best on >= 15 % of the openings.
  Reported (not gated): margin over the deadline-aware ``rally`` rule (mean and without the top two),
  the openings where a non-rally policy beats it by >= 2, and the old G* reference row.
Curve (stage B): per arm and checkpoint, mean return / capture / delta-v / time on the same openings,
the 0k anchor, the rule and the per-opening best fixed policy from stage A; training-time model use.
"""
import argparse, glob, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(dir_):
    out = defaultdict(dict)
    for f in sorted(glob.glob(str(Path(dir_) / '*.json'))):
        r = json.loads(Path(f).read_text(encoding='utf-8'))
        if 'policy' in r and 'event' in r:
            out[r['policy']][int(r['opening'])] = r
    return out


def stats(rs):
    rs = list(rs)
    return {'n': len(rs), 'mean_return': float(np.mean([r['return'] for r in rs])) if rs else None,
            'capture': int(sum(r['event'] == 'success' for r in rs)), 'unsafe': int(sum(r['event'] == 'unsafe' for r in rs)),
            'mean_delta_v_success': float(np.mean([r['delta_v'] for r in rs if r['event'] == 'success'])) if any(r['event'] == 'success' for r in rs) else None,
            'mean_time_success': float(np.mean([r['time_s'] for r in rs if r['event'] == 'success'])) if any(r['event'] == 'success' for r in rs) else None,
            'events': dict(Counter(r['event'] for r in rs))}


def screen(dir_, out):
    P = load(dir_)
    pols = sorted(p for p in P if p not in ('gstar', 'pure') and not p.startswith('b_'))
    ops = sorted(set.intersection(*[set(P[p]) for p in pols]))
    const = [p for p in pols if p.startswith('c_')]
    table = {p: stats(P[p][o] for o in ops) for p in pols}
    if 'gstar' in P:
        table['gstar_reference_outside_action_space'] = stats(P['gstar'][o] for o in ops if o in P['gstar'])
    if 'pure' in P:
        table['pure_mpc_baseline'] = stats(P['pure'][o] for o in ops if o in P['pure'])
    for p in sorted(x for x in P if x.startswith('b_')):
        table[f'basic_layering_{p}'] = stats(P[p][o] for o in ops if o in P[p])
    best = {o: max(pols, key=lambda p: P[p][o]['return']) for o in ops}
    best_ret = {o: P[best[o]][o]['return'] for o in ops}
    bc = max(const, key=lambda p: table[p]['mean_return'])
    g_const = sorted([best_ret[o] - P[bc][o]['return'] for o in ops], reverse=True)
    res = {'openings': ops, 'policies': table, 'best_constant_policy': bc, 'per_opening_best_mean': float(np.mean(list(best_ret.values()))),
           'gain_vs_best_constant': {'mean': float(np.mean(g_const)), 'without_top2': float(np.mean(g_const[2:])), 'sorted': g_const},
           'best_policy_counts': dict(Counter(best.values()))}
    if 'rally' in pols:
        g_rule = {o: best_ret[o] - P['rally'][o]['return'] for o in ops}
        srt = sorted(g_rule.values(), reverse=True)
        res['gain_vs_rally_rule_report_only'] = {'mean': float(np.mean(srt)), 'without_top2': float(np.mean(srt[2:])),
                                                 'openings_beaten_by_2': [o for o in ops if g_rule[o] >= 2.0]}
    stages = sum(P[p][o].get('stages', 0) for p in pols for o in ops)
    bad = sum(P[p][o].get('holds', 0) + P[p][o].get('rejected', 0) for p in pols for o in ops)
    gates = {'A1_mechanics': len(ops) > 0 and bad <= 0.10 * max(stages, 1),
             'A2_state_dependence': res['gain_vs_best_constant']['mean'] >= 10.0 and res['gain_vs_best_constant']['without_top2'] >= 5.0,
             'A3_spread': sum(c >= 0.15 * len(ops) for c in res['best_policy_counts'].values()) >= 3}
    errors = sorted(Path(f).name for f in glob.glob(str(Path(dir_) / 'ERROR_*.json')))
    gates['A1_mechanics'] = gates['A1_mechanics'] and not errors
    res.update(held_or_rejected_fraction=bad / max(stages, 1), errors=errors, gates=gates, passed=all(gates.values()))
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps({k: res[k] for k in ('gates', 'passed', 'gain_vs_best_constant', 'best_policy_counts')}, indent=1))
    if 'gain_vs_rally_rule_report_only' in res: print('vs rally (report):', res['gain_vs_rally_rule_report_only'])


def paired(eval_root):
    """E-MVE minus E-SAC per seed and checkpoint, paired by opening: mean, mean without the two largest
    per-opening gains, urgent / economy means, positive / negative counts (arms named e_mve_sN / e_sac_sN)."""
    root = Path(eval_root); out = {}
    for mve in sorted(root.glob('e_mve_s*')):
        sac = root / mve.name.replace('e_mve', 'e_sac')
        if not sac.is_dir():
            continue
        for ck in sorted(mve.glob('step_*'), key=lambda p: int(p.name.split('_')[1])):
            load_ = lambda d: {r['opening']: r for r in (json.loads(Path(f).read_text(encoding='utf-8')) for f in d.glob('*.json')
                                                       if not Path(f).name.startswith('ERROR_'))}
            a, b = load_(ck), load_(sac / ck.name)
            ops = sorted(set(a) & set(b))
            if not ops:
                continue
            d = {o: a[o]['return'] - b[o]['return'] for o in ops}
            srt = sorted(d.values(), reverse=True)
            mean = lambda xs: float(np.mean(xs)) if xs else None
            out[f'{mve.name[len("e_mve_"):]}/{ck.name}'] = {
                'n': len(ops), 'mean': mean(srt), 'without_top2': mean(srt[2:]),
                'urgent': mean([d[o] for o in ops if a[o]['mission']['urgent']]), 'economy': mean([d[o] for o in ops if not a[o]['mission']['urgent']]),
                'positive': sum(x > 0 for x in srt), 'negative': sum(x < 0 for x in srt),
                'capture_mve': sum(a[o]['event'] == 'success' for o in ops), 'capture_sac': sum(b[o]['event'] == 'success' for o in ops),
                'unsafe_mve': sum(a[o]['event'] == 'unsafe' for o in ops), 'unsafe_sac': sum(b[o]['event'] == 'unsafe' for o in ops)}
    return out


def curve(eval_root, train_root, screen_dir, out, expected=24):
    S = load(screen_dir); res = {'arms': {}}
    for arm_dir in sorted(Path(eval_root).glob('*')):
        if not arm_dir.is_dir(): continue
        per = {}
        for ck_dir in sorted(arm_dir.glob('step_*'), key=lambda p: int(p.name.split('_')[1])):
            rs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in sorted(ck_dir.glob('*.json')) if not Path(f).name.startswith('ERROR_')]
            per[int(ck_dir.name.split('_')[1])] = {**stats(rs), 'urgent': stats(r for r in rs if r['mission']['urgent']),
                                                   'economy': stats(r for r in rs if not r['mission']['urgent'])}
        res['arms'][arm_dir.name] = per
    pols = [p for p in S if p not in ('gstar', 'pure') and not p.startswith('b_')]
    if pols:
        ops = sorted(set.intersection(*[set(S[p]) for p in pols]))
        res['reference'] = {'rally_rule': stats(S['rally'][o] for o in ops) if 'rally' in S else None,
                            'basic_layering_best': max((stats(S[p][o] for o in ops if o in S[p]) for p in S if p.startswith('b_')),
                                                       key=lambda x: x['mean_return'], default=None),
                            'neutral_c_0_0': stats(S['c_0_0'][o] for o in ops) if 'c_0_0' in S else None,
                            'pure_mpc': stats(S['pure'][o] for o in ops if o in S['pure']) if 'pure' in S else None,
                            'best_constant': max((stats(S[p][o] for o in ops) for p in pols if p.startswith('c_')), key=lambda x: x['mean_return']),
                            'per_opening_best_fixed': float(np.mean([max(S[p][o]['return'] for p in pols) for o in ops]))}
    res['completeness'] = {}
    for arm, per in res['arms'].items():
        for k, st in per.items():
            ck_dir = Path(eval_root) / arm / f'step_{k}'
            errors = sorted(Path(f).name for f in ck_dir.glob('ERROR_*.json'))
            res['completeness'][f'{arm}/step_{k}'] = {'n': st['n'], 'errors': errors, 'complete': st['n'] == expected and not errors}
    res['all_complete'] = bool(res['completeness']) and all(c['complete'] for c in res['completeness'].values())
    res['paired'] = paired(eval_root)
    res['training'] = {}
    for t in sorted(Path(train_root).glob('*')):
        f = t / 'model_stats.json'
        if f.exists():
            ms = json.loads(f.read_text(encoding='utf-8')); fin = json.loads((t / 'finished.json').read_text(encoding='utf-8')) if (t / 'finished.json').exists() else {}
            used = [m for m in ms if m['model_rows']]
            res['training'][t.name] = {'finished': fin, 'probes': len(ms), 'probes_with_model_rows': len(used),
                                       'lambda_eff_mean': float(np.mean([m['lambda_eff'] for m in used])) if used else 0.0,
                                       'model_td_last': used[-1]['model_td'] if used else None}
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1)[:4000])


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['screen', 'curve'])
    p.add_argument('--dir'); p.add_argument('--eval-root'); p.add_argument('--train-root'); p.add_argument('--screen'); p.add_argument('--output', required=True)
    p.add_argument('--expected', type=int, default=24, help='openings every arm / checkpoint must have (completeness)')
    a = p.parse_args()
    if a.mode == 'screen': screen(a.dir, a.output)
    else: curve(a.eval_root, a.train_root, a.screen, a.output, a.expected)


if __name__ == '__main__':
    main()
