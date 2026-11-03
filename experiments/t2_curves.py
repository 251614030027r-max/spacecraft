"""T2.12 training curves, final-model readout and the pre-registered T2.12 decision (no simulation).

    python -B -m experiments.t2_curves curve  --runs LABEL=TRAIN_DIR ... --output CURVES.json [--window 50] [--steps 2000]
    python -B -m experiments.t2_curves final  --runs LABEL=EVAL_DIR ... --output FINAL.json [--expected 48]
    python -B -m experiments.t2_curves plot   --curves CURVES.json [--final FINAL.json] --output FIG.png
    python -B -m experiments.t2_curves decide --curves CURVES.json --final FINAL.json --output DECISION.json

Labels are ``<arm>_s<seed>`` (arm e_rel, e_priv, e_acc, e_sac, ...). A training directory holds ``episodes.json``; an
evaluation directory holds the per-opening JSON files of one checkpoint (``experiments.t2 eval``).

Training curve (research layer 2026-11-03): one point per finished training episode, x = stored training decisions when
it ended, y = its true task return (time and delta-v costs, capture +100 / timeout -100 / resource exhausted -100 /
unsafe -150; no shaping). Runs from learner 4.0 record it (``return_true``); older runs are rebuilt from the logged
event, duration, delta-v and mission costs (identical up to rounding). Smoothing: the mean of the last ``window``
episodes, the same for every run; the band of an arm is the range of its seeds (not a confidence interval). These are
behaviour-policy (stochastic, training missions) statistics, distinct from the deterministic final-model evaluation.

Summary per run: curve_auc = mean of the smoothed curve on the decision grid 250, 300, ..., steps; late_return / late
urgent capture = mean over the episodes that ended in (steps - 500, steps].
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np

TERMINAL = {'success': 100.0, 'timeout': -100.0, 'resource_exhausted': -100.0, 'unsafe': -150.0}


def true_return(ep: dict) -> float:
    if ep.get('return_true') is not None:
        return float(ep['return_true'])
    m = ep['mission']
    return float(-m['time_cost_per_s'] * float(ep['time_s']) - m['fuel_cost_per_m_s'] * float(ep['delta_v']) + TERMINAL.get(ep['event'], 0.0))


def _runs(items):
    out = {}
    for x in items:
        k, _, v = x.partition('=')
        out[k] = Path(v)
    return out


#: decisions after a run's last finished episode for which its rolling value still counts (the last episodes of a run end
#: up to a few dozen decisions before its last stored decision)
END_TOLERANCE = 100


def run_curve(episodes: list, window: int, steps: int, grid_from: int = 250, grid_step: int = 50) -> dict:
    eps = sorted(episodes, key=lambda e: int(e['steps']))
    x = np.array([int(e['steps']) for e in eps]); y = np.array([true_return(e) for e in eps])
    urg = np.array([bool(e['mission']['urgent']) for e in eps]); cap = np.array([e['event'] == 'success' for e in eps])
    roll = np.array([y[max(0, i - window + 1):i + 1].mean() for i in range(len(y))])
    # urgent capture: rolling over the last ``window`` urgent episodes
    ui = np.flatnonzero(urg)
    uroll = np.array([cap[ui[max(0, j - window + 1):j + 1]].mean() for j in range(len(ui))])
    grid = np.arange(grid_from, steps + 1, grid_step)
    # a grid point after the run's last episode (a shorter run plotted on a longer axis) is left empty, not carried flat
    at = lambda xs, vs, g: (float(vs[np.searchsorted(xs, g, side='right') - 1])
                            if np.searchsorted(xs, g, side='right') > 0 and len(xs) and g <= xs[-1] + END_TOLERANCE else float('nan'))
    smooth = [at(x, roll, g) for g in grid]
    auc = float(np.nanmean(smooth)) if not all(np.isnan(smooth)) else None
    late = (x > steps - 500) & (x <= steps)          # a longer run is read on the same budget
    return {'episodes': int(len(eps)), 'x': x.tolist(), 'return_true': y.round(3).tolist(), 'rolling': roll.round(3).tolist(),
            'urgent_x': x[ui].tolist(), 'urgent_capture_rolling': uroll.round(4).tolist(),
            'grid': grid.tolist(), 'grid_rolling': _clean(smooth),
            'curve_auc': auc, 'late_return': float(y[late].mean()) if late.any() else None,
            'late_urgent_capture': float(cap[late & urg].mean()) if (late & urg).any() else None,
            'late_episodes': int(late.sum()), 'recorded_returns': bool(all(e.get('return_true') is not None for e in eps))}


def _clean(xs):
    """Standard JSON (no NaN): empty grid points are null (PowerShell's ConvertFrom-Json rejects NaN)."""
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in xs]


def _mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None      # None: no run of the arm reaches the late window (a shorter run on a longer axis)


def curves(runs: dict, out: Path, window: int = 50, steps: int = 2000) -> dict:
    res = {'window': window, 'steps': steps, 'runs': {}}
    for label, d in sorted(runs.items()):
        eps = json.loads((d / 'episodes.json').read_text(encoding='utf-8'))
        res['runs'][label] = run_curve(eps, window, steps)
    res['arms'] = {}
    for arm in sorted({k.rsplit('_s', 1)[0] for k in res['runs']}):
        ks = [k for k in res['runs'] if k.rsplit('_s', 1)[0] == arm]
        g = np.array([[np.nan if v is None else v for v in res['runs'][k]['grid_rolling']] for k in ks], dtype=float)
        with warnings.catch_warnings():              # all-empty grid columns (no run of the arm reaches them) stay empty
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_, lo_, hi_ = np.nanmean(g, 0), np.nanmin(g, 0), np.nanmax(g, 0)
        res['arms'][arm] = {'runs': ks, 'grid': res['runs'][ks[0]]['grid'], 'mean': _clean(mean_.tolist()), 'lo': _clean(lo_.tolist()), 'hi': _clean(hi_.tolist()),
                            'curve_auc': float(np.mean([res['runs'][k]['curve_auc'] for k in ks])),
                            'late_return': _mean_or_none([res['runs'][k]['late_return'] for k in ks]),
                            'late_urgent_capture': _mean_or_none([res['runs'][k]['late_urgent_capture'] for k in ks])}
    out.write_text(json.dumps(res), encoding='utf-8')
    print(json.dumps({a: {k: v for k, v in r.items() if k in ('runs', 'curve_auc', 'late_return', 'late_urgent_capture')} for a, r in res['arms'].items()}, indent=1))
    return res


def _stats(rs):
    rs = list(rs)
    ok = [r for r in rs if r['event'] == 'success']
    return {'n': len(rs), 'mean_return': float(np.mean([r['return'] for r in rs])) if rs else None, 'capture': len(ok),
            'unsafe': sum(r['event'] == 'unsafe' for r in rs),
            'mean_delta_v_success': float(np.mean([r['delta_v'] for r in ok])) if ok else None,
            'mean_time_success': float(np.mean([r['time_s'] for r in ok])) if ok else None}


def final(runs: dict, out: Path, expected: int = 48) -> dict:
    res = {'runs': {}, 'expected': expected}
    for label, d in sorted(runs.items()):
        rs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(d.glob('*.json')) if not f.name.startswith('ERROR_')]
        res['runs'][label] = {**_stats(rs), 'urgent': _stats(r for r in rs if r['mission']['urgent']),
                              'economy': _stats(r for r in rs if not r['mission']['urgent']),
                              'errors': sorted(f.name for f in d.glob('ERROR_*.json')),
                              'per_opening': {int(r['opening']): {'return': r['return'], 'event': r['event'], 'urgent': r['mission']['urgent'],
                                                                  'time_s': r['time_s'], 'delta_v': r['delta_v']} for r in rs}}
        res['runs'][label]['complete'] = res['runs'][label]['n'] == expected and not res['runs'][label]['errors']
    res['arms'] = {}
    for arm in sorted({k.rsplit('_s', 1)[0] for k in res['runs']}):
        ks = [k for k in res['runs'] if k.rsplit('_s', 1)[0] == arm]
        res['arms'][arm] = {'runs': ks, 'mean_return': float(np.mean([res['runs'][k]['mean_return'] for k in ks])),
                            'urgent_capture': [res['runs'][k]['urgent']['capture'] for k in ks],
                            'economy_capture': [res['runs'][k]['economy']['capture'] for k in ks],
                            'unsafe': int(sum(res['runs'][k]['unsafe'] for k in ks))}
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res['arms'], indent=1))
    return res


#: T2.12 pre-registered decision (fixed 2026-11-03 before any T2.12 training). One opening flip on 48 openings moves a
#: run's mean return by ~4.2 (200 / 48): MARGIN = two flips.
MARGIN = 8.3


def decide_t212(c: dict, f: dict, method='e_rel', ablation='e_priv', reference='e_acc', sac='e_sac') -> dict:
    """S safety: no unsafe final episode of the method. N not below T2.11: the method's arm-mean final return and arm-mean
    curve AUC each >= the reference's - MARGIN. I mechanism increment over the same critic without counterfactual feedback:
    arm-mean curve AUC above the ablation's, both method seeds above the ablation's arm mean, and arm-mean final return not
    below the ablation's. G above SAC on the same interface: arm-mean final return and curve AUC above, and both method
    seeds above SAC's arm mean on both.
    Verdict: unsafe (S fails) > revert (N fails: keep T2.11) > no_increment (I fails) > no_gain_over_sac (G fails) > adopt."""
    A, F = c['arms'], f['arms']
    runs = lambda arm: [k for k in c['runs'] if k.rsplit('_s', 1)[0] == arm]
    auc = lambda arm: A[arm]['curve_auc']; fin = lambda arm: F[arm]['mean_return']
    m_runs = runs(method)
    crit = {
        'S_safety': F[method]['unsafe'] == 0,
        'N_not_below_T2_11': fin(method) >= fin(reference) - MARGIN and auc(method) >= auc(reference) - MARGIN,
        'I_increment_over_ablation': auc(method) > auc(ablation) and all(c['runs'][k]['curve_auc'] > auc(ablation) for k in m_runs)
                                     and fin(method) >= fin(ablation),
        'G_above_sac': fin(method) > fin(sac) and auc(method) > auc(sac)
                       and all(c['runs'][k]['curve_auc'] > auc(sac) and f['runs'][k]['mean_return'] > fin(sac) for k in m_runs),
        'complete': all(f['runs'][k]['complete'] for k in f['runs']),
    }
    verdict = ('incomplete' if not crit['complete'] else 'unsafe' if not crit['S_safety'] else 'revert' if not crit['N_not_below_T2_11']
               else 'no_increment' if not crit['I_increment_over_ablation'] else 'no_gain_over_sac' if not crit['G_above_sac'] else 'adopt')
    arms = [x for x in (method, ablation, reference, sac) if x in A and x in F]
    return {'criteria': crit, 'verdict': verdict, 'margin': MARGIN,
            'values': {arm: {'curve_auc': auc(arm), 'final_mean_return': fin(arm), 'late_return': A[arm]['late_return'],
                             'late_urgent_capture': A[arm]['late_urgent_capture'], 'final_urgent_capture': F[arm]['urgent_capture'],
                             'final_economy_capture': F[arm]['economy_capture'], 'unsafe': F[arm]['unsafe'],
                             'runs': {k: {'curve_auc': c['runs'][k]['curve_auc'], 'final_mean_return': f['runs'][k]['mean_return']} for k in A[arm]['runs']}}
                       for arm in arms},
            'reported': {'reference_minus_ablation': {'curve_auc': auc(reference) - auc(ablation), 'final': fin(reference) - fin(ablation)}
                         if reference in A and ablation in A else None}}


COL = {'e_rel': '#2a78d6', 'e_acc': '#7aa6d9', 'e_priv': '#3a9e6a', 'e_sac': '#eb6834'}
NAME = {'e_rel': 'EACC', 'e_acc': 'EACC-A (T2.11)', 'e_priv': '特权 critic，无反事实反馈', 'e_sac': 'SAC（无反馈）'}


def plot(cpath: Path, out: Path, fpath: Path | None = None):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams['axes.unicode_minus'] = False
    for name in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Source Han Sans SC', 'Microsoft YaHei', 'SimHei'):
        if any(name in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = name; break
    c = json.loads(cpath.read_text(encoding='utf-8'))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for arm, r in c['arms'].items():
        col = COL.get(arm, '#555'); g = np.array(r['grid'])
        arr = lambda xs: np.array([np.nan if x is None else x for x in xs], dtype=float)
        axs[0].fill_between(g, arr(r['lo']), arr(r['hi']), color=col, alpha=0.15, lw=0)
        axs[0].plot(g, arr(r['mean']), color=col, lw=2, label=f"{NAME.get(arm, arm)}（{len(r['runs'])} 个种子）")
        # urgent capture: arm mean of the per-run rolling rate on the same grid
        ys = []
        for k in r['runs']:
            ux, uy = np.array(c['runs'][k]['urgent_x']), np.array(c['runs'][k]['urgent_capture_rolling'])
            idx = np.searchsorted(ux, g, side='right') - 1
            ys.append(np.where((idx >= 0) & (g <= ux[-1] + END_TOLERANCE), uy[np.clip(idx, 0, None)], np.nan))
        ys = np.array(ys)
        warnings.simplefilter('ignore', RuntimeWarning)
        axs[1].fill_between(g, np.nanmin(ys, 0), np.nanmax(ys, 0), color=col, alpha=0.15, lw=0)
        axs[1].plot(g, np.nanmean(ys, 0), color=col, lw=2)
    axs[0].set_ylabel(f"训练回合真实任务回报（近 {c['window']} 回合均值）"); axs[1].set_ylabel(f"紧急任务捕获率（近 {c['window']} 个紧急回合）")
    for ax in axs:
        ax.set_xlabel('累计训练决策数'); ax.grid(axis='y', color='#e5e5e5'); ax.spines[['top', 'right']].set_visible(False)
    axs[0].legend(frameon=False, loc='lower right', fontsize=9)
    fig.suptitle('阴影为各种子的范围（不是置信区间）；训练期为带探索的行为策略', fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=160); print('saved', out)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['curve', 'final', 'plot', 'decide'])
    p.add_argument('--runs', nargs='*', default=[]); p.add_argument('--output', required=True)
    p.add_argument('--window', type=int, default=50); p.add_argument('--steps', type=int, default=2000); p.add_argument('--expected', type=int, default=48)
    p.add_argument('--curves'); p.add_argument('--final')
    a = p.parse_args(); out = Path(a.output)
    if a.mode == 'curve':
        curves(_runs(a.runs), out, a.window, a.steps)
    elif a.mode == 'final':
        final(_runs(a.runs), out, a.expected)
    elif a.mode == 'plot':
        plot(Path(a.curves), out, Path(a.final) if a.final else None)
    else:
        res = decide_t212(json.loads(Path(a.curves).read_text(encoding='utf-8')), json.loads(Path(a.final).read_text(encoding='utf-8')))
        out.write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
