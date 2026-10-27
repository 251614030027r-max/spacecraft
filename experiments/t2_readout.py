"""T2 readouts (no simulation).

    python -B -m experiments.t2_readout screen --dir SCREEN_DIR --output A.json
    python -B -m experiments.t2_readout curve  --eval-root EVAL_ROOT --train-root TRAIN_ROOT --screen SCREEN_DIR --output B.json
    python -B -m experiments.t2_readout window --rigid SCREEN_RIGID --window SCREEN_WINDOW --output W.json
    python -B -m experiments.t2_readout pressure --dir ROOT --tags p170,p155,p140 --output P.json   (T2.6 time-pressure screen)

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


#: T2.6W analysis bins of the urgent deadline (s): high / mid / low time pressure (reported, never gated)
DEADLINE_BINS = (('high', 0.0, 160.0), ('mid', 160.0, 180.0), ('low', 180.0, 1e9))


def _deadline_bins(a, b, d, ops):
    out = {}
    for name, lo, hi in DEADLINE_BINS:
        xs = [o for o in ops if a[o]['mission']['urgent'] and lo <= a[o]['mission']['deadline_s'] < hi]
        out[name] = {'n': len(xs), 'mean': float(np.mean([d[o] for o in xs])) if xs else None,
                     'capture_mve': sum(a[o]['event'] == 'success' for o in xs), 'capture_sac': sum(b[o]['event'] == 'success' for o in xs)}
    return out


def _both_success(a, b, ops):
    """Delta-v and time EACC - SAC on the openings both arms capture (no survivor bias), all / economy / urgent."""
    out = {}
    for name, sel in (('all', lambda o: True), ('economy', lambda o: not a[o]['mission']['urgent']), ('urgent', lambda o: a[o]['mission']['urgent'])):
        xs = [o for o in ops if sel(o) and a[o]['event'] == 'success' and b[o]['event'] == 'success']
        out[name] = {'n': len(xs), 'delta_v': float(np.mean([a[o]['delta_v'] - b[o]['delta_v'] for o in xs])) if xs else None,
                     'time_s': float(np.mean([a[o]['time_s'] - b[o]['time_s'] for o in xs])) if xs else None}
    return out


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
                'unsafe_mve': sum(a[o]['event'] == 'unsafe' for o in ops), 'unsafe_sac': sum(b[o]['event'] == 'unsafe' for o in ops),
                'urgent_capture_mve': sum(a[o]['event'] == 'success' for o in ops if a[o]['mission']['urgent']),
                'urgent_capture_sac': sum(b[o]['event'] == 'success' for o in ops if b[o]['mission']['urgent']),
                'urgent_bins': _deadline_bins(a, b, d, ops), 'both_success': _both_success(a, b, ops)}
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
            used = [m for m in ms if m.get('model_rows')]
            q_pi = [m['q_pi_mean'] for m in ms]
            res['training'][t.name] = {'finished': {k: v for k, v in fin.items() if k != 'worker_order'}, 'probes': len(ms),
                                       'probes_with_model_rows': len(used), 'first_model_step': used[0]['steps'] if used else None,
                                       'w_mean': float(np.mean([m['w_mean'] for m in used])) if used else None,
                                       'abs_err_median': float(np.median([m['abs_err_median'] for m in used])) if used else None,
                                       'sigma_last': ms[-1].get('sigma') if ms else None,
                                       'model_td_last': used[-1]['model_td'] if used else None,
                                       'q_pi_last_decile': float(np.mean(q_pi[-max(1, len(q_pi) // 10):])) if q_pi else None,
                                       # T2.4/T2.5 mechanism readings (mean over probes that report them)
                                       **{k: (float(np.mean([m[k] for m in used if m.get(k) is not None])) if any(m.get(k) is not None for m in used) else None)
                                          for k in ('loss_share', 'grad_cos', 'two_stage_share', 'decisive_share', 'c_mean', 'c_low_share',
                                                    'increment_abs_mean', 'age_capped_share', 'label_age_mean', 'entry_bias_now', 'entry_share',
                                                    'stage2_rows', 'w2_mean', 'pi_gap_sigma_median', 'stage2_entry_share')}}
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1)[:4000])


def window(rigid_dir, window_dir, out):
    """Sub-mechanism screen: the same fixed policies and openings with the rigid (T2.2) and the pass-window entry
    commitment; paired by policy and opening: return, capture, time, delta-v, lost entries."""
    R, W = load(rigid_dir), load(window_dir)
    rows = [(p, o) for p in sorted(set(R) & set(W)) for o in sorted(set(R[p]) & set(W[p]))]
    d = [W[p][o]['return'] - R[p][o]['return'] for p, o in rows]
    srt = sorted(d, reverse=True)
    lost = lambda X: int(sum(X[p][o].get('entries_lost', 0) for p, o in rows))
    res = {'pairs': len(rows), 'rigid': stats(R[p][o] for p, o in rows), 'window': stats(W[p][o] for p, o in rows),
           'paired_return': {'mean': float(np.mean(d)) if d else None, 'without_top2': float(np.mean(srt[2:])) if len(srt) > 2 else None,
                             'positive': sum(x > 0 for x in d), 'negative': sum(x < 0 for x in d)},
           'entries_lost': {'rigid': lost(R), 'window': lost(W)},
           'per_policy': {p: float(np.mean([W[p][o]['return'] - R[p][o]['return'] for q, o in rows if q == p])) for p in sorted({p for p, _ in rows})}}
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


IN_SPACE = ('c_', 'rally')     # fixed policies inside the frozen 2-D stage action space


def _profile(dir_):
    """One time-pressure profile: ROOT/<tag>/screen (fixed policies, G*, pure) and ROOT/<tag>/probe (learned checkpoints)."""
    P = load(Path(dir_) / 'screen'); L = load(Path(dir_) / 'probe')
    ins = sorted(p for p in P if p.startswith(IN_SPACE))
    ops = sorted(set.intersection(*[set(P[p]) for p in ins + ['gstar']]))
    urgent = [o for o in ops if P['gstar'][o]['mission']['urgent']]
    cap = lambda r: r['event'] == 'success'
    pol = {p: {**stats(P[p][o] for o in ops), 'urgent_capture': int(sum(cap(P[p][o]) for o in urgent))} for p in ins + ['gstar', 'pure'] if p in P}
    lrn = {p: {**stats(L[p][o] for o in ops if o in L[p]), 'urgent_capture': int(sum(cap(L[p][o]) for o in urgent if o in L[p]))} for p in L}
    best = max(ins, key=lambda p: pol[p]['mean_return'])
    reach_in = [o for o in ops if any(cap(P[p][o]) for p in ins)]
    reach_any = [o for o in ops if o in reach_in or any(o in L[p] and cap(L[p][o]) for p in L)]
    gain = sorted((max(P[p][o]['return'] for p in ins) - P[best][o]['return'] for o in ops), reverse=True)
    return {'openings': len(ops), 'urgent_openings': len(urgent), 'deadlines_urgent': [P['gstar'][o]['mission']['deadline_s'] for o in urgent],
            'policies': pol, 'learned': lrn, 'uniform_best': best,
            'reach_in_space': len(reach_in), 'reach_any': len(reach_any), 'urgent_reach_any': sum(o in reach_any for o in urgent),
            'unreachable': [o for o in ops if o not in reach_any],
            'headroom_capture': len(reach_in) - pol[best]['capture'],
            'decision_value': float(np.mean(gain)), 'decision_value_without_top1': float(np.mean(gain[1:])) if len(gain) > 1 else None,
            'unsafe_in_space': int(sum(pol[p]['unsafe'] for p in ins)), 'unsafe_learned': int(sum(v['unsafe'] for v in lrn.values())),
            'learned_urgent_capture': int(sum(v['urgent_capture'] for v in lrn.values())),
            'gstar_unreached_in_time': [o for o in ops if not cap(P['gstar'][o])]}


def pressure(root, tags, out):
    """T2.6 time-pressure screen and its pre-registered selection (run order T2_6_SCREEN). tags[0] is the standard task.

      F1 reference   G* captures >= 22/24 with no unsafe
      F2 reachable   >= 21/24 openings captured by a policy inside the stage action space (a fixed in-space policy or a
                     learned checkpoint), and every urgent opening but at most two
      F3 safe        unsafe of the in-space policies <= standard + 2; pure unsafe <= standard + 1; learned unsafe <= standard + 1
      D1 pressure    the learned checkpoints (trained on the standard task) lose >= 2 urgent captures in total against the
                     standard task, and the decision value without its largest opening stays > 0
    Selection: the tightest profile passing F1-F3 and D1; else none (the tightest one passing F1-F3 is reported)."""
    R = {t: _profile(Path(root) / t) for t in tags}
    base = R[tags[0]]; sel = None; feas = None
    for t in tags[1:]:
        r = R[t]
        c = {'F1_reference': r['policies']['gstar']['capture'] >= 22 and r['policies']['gstar']['unsafe'] == 0,
             'F2_reachable': r['reach_any'] >= 21 and r['urgent_reach_any'] >= r['urgent_openings'] - 2,
             'F3_safe': r['unsafe_in_space'] <= base['unsafe_in_space'] + 2 and r['policies'].get('pure', {}).get('unsafe', 0) <= base['policies'].get('pure', {}).get('unsafe', 0) + 1
                        and r['unsafe_learned'] <= base['unsafe_learned'] + 1,
             'D1_pressure': base['learned_urgent_capture'] - r['learned_urgent_capture'] >= 2 and (r['decision_value_without_top1'] or 0) > 0}
        r['criteria'] = c
        if c['F1_reference'] and c['F2_reachable'] and c['F3_safe']:
            feas = t
            if c['D1_pressure']:
                sel = t
    res = {'profiles': R, 'selected': sel, 'tightest_feasible': feas, 'order': tags}
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({'selected': sel, 'tightest_feasible': feas, **{t: R[t].get('criteria') for t in tags[1:]}}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['screen', 'curve', 'window', 'pressure'])
    p.add_argument('--rigid'); p.add_argument('--window')
    p.add_argument('--dir'); p.add_argument('--eval-root'); p.add_argument('--train-root'); p.add_argument('--screen'); p.add_argument('--output', required=True)
    p.add_argument('--tags', default='p170,p155,p140', help='pressure: profile directories under --dir, the standard task first')
    p.add_argument('--expected', type=int, default=24, help='openings every arm / checkpoint must have (completeness)')
    a = p.parse_args()
    if a.mode == 'screen': screen(a.dir, a.output)
    elif a.mode == 'window': window(a.rigid, a.window, a.output)
    elif a.mode == 'pressure': pressure(a.dir, a.tags.split(','), a.output)
    else: curve(a.eval_root, a.train_root, a.screen, a.output, a.expected)


if __name__ == '__main__':
    main()
