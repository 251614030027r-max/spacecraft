"""Figures and tables of the EACH final experiments, rebuilt from raw logs only (no simulation, no hand-entered numbers).

    python -B -m experiments.final_figures --stage {dryrun,pretest,formal} --root DIR --output DIR [--geoscan DIR] [--traj DIR]

Inputs under ``--root`` (the layout of ``experiments.final``): ``B/TRAIN_<CONFIG>_s<seed>/`` training runs, ``C/<CONFIG>_s<seed>``
and ``C/<CONFIG>_s<seed>_rigid`` deterministic evaluations, ``S/<policy>`` and ``S/<policy>_rigid`` fixed-policy screens.
Legends, colours and the stage's openings, seeds and budget come from ``docs/final_method/METHOD_REGISTRY.json``;
``experiments.final.resolve`` is applied first, so mixed science SHAs, budgets or configurations stop the script.

Outputs (PNG 300 dpi + PDF, and the JSON behind each figure):
  fig_training     true return of training episodes (rolling 50, per seed and arm mean, seed range) and urgent capture
  fig_final        final models on the stage's evaluation openings: capture, urgent capture, return (seed points)
  fig_window       C: pass-window vs rigid commitment, paired by opening (entries lost, capture, paired return)
  fig_geometry     A: sweep-cone entry geometry and conditional G* feasibility (from ``experiments.final geoscan``)
  fig_trajectory   B: trajectories with stage / hold / bridge / descent modes (from ``experiments.final traj``)
  TABLES.json      every number shown, with n and the openings used
Statistics: openings are the pairing unit, seeds the training replicates; time and delta-v only on openings where both
compared entries capture; every panel states n, seeds, budget and the stage.
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

from experiments.final import block, registry, resolve, stage_of

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#9a9993', '#e4e3df'
MODE_COLORS = {'stage': '#dce9f9', 'hold': '#f2f1ee', 'bridge': '#fde3d6', 'descent': '#d5f0e5'}
EVENT_SHORT = {'success': 'capture', 'timeout': 'timeout', 'resource_exhausted': 'fuel out', 'unsafe': 'unsafe'}


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['DejaVu Sans'], 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                         'grid.linewidth': 0.7, 'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.8,
                         'savefig.dpi': 300, 'figure.dpi': 100})
    return plt


def save(fig, out: Path, name: str, stage: str, data: dict):
    fig.text(0.995, 0.005, f'{stage} stage', ha='right', va='bottom', fontsize=8, color=MUTED, alpha=0.9)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    (out / f'{name}.json').write_text(json.dumps(data, indent=1, default=float), encoding='utf-8')


def load_eval(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob('*.json')):
        if f.name.startswith('ERROR_'):
            continue
        r = json.loads(f.read_text(encoding='utf-8'))
        if 'opening' in r and 'event' in r:
            out[int(r['opening'])] = r
    return out


def summary(rs: list) -> dict:
    ok = [r for r in rs if r['event'] == 'success']; urg = [r for r in rs if r['mission']['urgent']]
    return {'n': len(rs), 'capture': len(ok), 'urgent_n': len(urg), 'urgent_capture': sum(r['event'] == 'success' for r in urg),
            'unsafe': sum(r['event'] == 'unsafe' for r in rs), 'timeout': sum(r['event'] == 'timeout' for r in rs),
            'resource_exhausted': sum(r['event'] == 'resource_exhausted' for r in rs),
            'mean_return': float(np.mean([r['return'] for r in rs])) if rs else None,
            'median_return': float(np.median([r['return'] for r in rs])) if rs else None,
            'entries_lost': int(sum(r.get('entries_lost', 0) or 0 for r in rs))}


def both_success(a: dict, b: dict) -> dict:
    ops = [o for o in a if o in b and a[o]['event'] == 'success' and b[o]['event'] == 'success']
    return {'n': len(ops), 'time_s': [float(np.mean([a[o]['time_s'] for o in ops])) if ops else None, float(np.mean([b[o]['time_s'] for o in ops])) if ops else None],
            'delta_v': [float(np.mean([a[o]['delta_v'] for o in ops])) if ops else None, float(np.mean([b[o]['delta_v'] for o in ops])) if ops else None],
            'only_first_captures': sum(1 for o in a if o in b and a[o]['event'] == 'success' and b[o]['event'] != 'success'),
            'only_second_captures': sum(1 for o in a if o in b and a[o]['event'] != 'success' and b[o]['event'] == 'success')}


# ---------------------------------------------------------------------- training curves
def fig_training(plt, root: Path, res: dict, stage: str, out: Path, tables: dict):
    from experiments.t2_curves import run_curve
    reg = registry(); p = reg['protocol']; steps = int(res['steps']); win = int(p['smoothing_window_episodes'])
    by = {}
    for name, r in res['runs'].items():
        eps = json.loads((root / r['episodes']).read_text(encoding='utf-8'))
        by.setdefault(r['config'], []).append((r['seed'], run_curve(eps, win, steps, grid_from=int(p['curve_grid_from']))))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    data = {'window_episodes': win, 'steps': steps, 'arms': {}}
    for cfg, rows in by.items():
        c = reg['trained'][cfg]; col = c['color']
        grid = np.array(rows[0][1]['grid'])
        G = np.array([[np.nan if v is None else v for v in cur['grid_rolling']] for _, cur in rows], float)
        for seed, cur in rows:
            ax[0].plot(cur['x'], cur['rolling'], color=col, alpha=0.28, linewidth=0.9)
            ax[1].plot(cur['urgent_x'], cur['urgent_capture_rolling'], color=col, alpha=0.28, linewidth=0.9)
        with np.errstate(all='ignore'):
            import warnings
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nanmean(G, axis=0); lo, hi = np.nanmin(G, axis=0), np.nanmax(G, axis=0)
        ax[0].plot(grid, mean, color=col, linewidth=2.2, label=f"{c['display']} ({len(rows)} seeds)")
        ax[0].fill_between(grid, lo, hi, color=col, alpha=0.10, linewidth=0)
        data['arms'][cfg] = {'display': c['display'], 'seeds': [s for s, _ in rows], 'curve_auc': [cur['curve_auc'] for _, cur in rows],
                             'curve_auc_mean': (float(np.mean(aucs)) if (aucs := [cur['curve_auc'] for _, cur in rows if cur['curve_auc'] is not None]) else None),
                             'late_return': [cur['late_return'] for _, cur in rows], 'late_urgent_capture': [cur['late_urgent_capture'] for _, cur in rows],
                             'grid': grid.tolist(), 'mean': np.where(np.isnan(mean), None, mean).tolist()}
    ax[0].set_xlabel('stored decisions'); ax[0].set_ylabel(f'true return (rolling {win} episodes)')
    ax[0].set_title('(a) training episodes: true task return', loc='left'); ax[0].legend(fontsize=8, loc='lower right')
    ax[1].set_xlabel('stored decisions'); ax[1].set_ylabel(f'urgent capture (rolling {win} urgent episodes)')
    ax[1].set_title('(b) training episodes: urgent capture', loc='left'); ax[1].set_ylim(-0.02, 1.02)
    fig.suptitle(f'Behaviour-policy statistics (stochastic actions, training openings); budget {steps} decisions', fontsize=9, color=INK2, y=1.02)
    save(fig, out, 'fig_training', stage, data); plt.close(fig)
    tables['training'] = data


# ---------------------------------------------------------------------- final models
def fig_final(plt, root: Path, res: dict, stage: str, out: Path, tables: dict):
    reg = registry(); st = stage_of(stage); ops = block(st['eval_openings'])
    entries = []
    for cfg in st['trained']:
        seeds = {}
        for s in st['seeds']:
            d = root / 'C' / f'{cfg}_s{s}'
            if d.exists():
                seeds[s] = load_eval(d)
        if seeds:
            entries.append((cfg, reg['trained'][cfg]['display'], reg['trained'][cfg]['color'], seeds))
    for pol in st['fixed']:
        d = root / 'S' / pol
        if d.exists():
            entries.append((pol, reg['fixed'][pol]['display'], MUTED, {None: load_eval(d)}))
    if not entries:
        return
    data = {'openings': st['eval_openings'], 'entries': {}}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    for i, (key, disp, col, seeds) in enumerate(entries):
        rows = {s: summary([r for o, r in ev.items() if o in ops]) for s, ev in seeds.items()}
        data['entries'][key] = {'display': disp, 'per_seed': {str(s): v for s, v in rows.items()}}
        for j, (metric, den) in enumerate((('capture', 'n'), ('urgent_capture', 'urgent_n'))):
            vals = [v[metric] / v[den] for v in rows.values() if v[den] > 0]      # no urgent mission in the set: no bar
            if not vals:
                continue
            ax[j].bar(i, np.mean(vals), color=col, alpha=0.55, width=0.7)
            ax[j].scatter([i] * len(vals), vals, color=col if col != MUTED else INK2, s=18, zorder=3)
        vals = [v['mean_return'] for v in rows.values()]
        ax[2].bar(i, np.mean(vals), color=col, alpha=0.55, width=0.7); ax[2].scatter([i] * len(vals), vals, color=col if col != MUTED else INK2, s=18, zorder=3)
    names = [e[1] for e in entries]
    for j, t in enumerate(('(a) capture rate', '(b) urgent capture rate', '(c) mean true return')):
        ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(names, rotation=35, ha='right', fontsize=8); ax[j].set_title(t, loc='left')
    ax[0].set_ylim(0, 1.02); ax[1].set_ylim(0, 1.02)
    # paired against the main method (opening by opening, seed-averaged where seeds exist)
    main = next((e for e in entries if e[0] == 'EACH'), None)
    if main is not None:
        def per_opening(seeds):
            return {o: float(np.mean([ev[o]['return'] for ev in seeds.values() if o in ev])) for o in ops if any(o in ev for ev in seeds.values())}
        m = per_opening(main[3]); data['paired_vs_EACH'] = {}
        for key, disp, _, seeds in entries:
            if key == 'EACH':
                continue
            o2 = per_opening(seeds); common = [o for o in m if o in o2]
            d = np.array([m[o] - o2[o] for o in common])
            first = {o: r for o, r in next(iter(main[3].values())).items()}; second = {o: r for o, r in next(iter(seeds.values())).items()}
            data['paired_vs_EACH'][key] = {'n': len(common), 'mean_diff': float(d.mean()) if len(d) else None,
                                           'median_diff': float(np.median(d)) if len(d) else None,
                                           'without_top2': float(np.sort(d)[:-2].mean()) if len(d) > 2 else None,
                                           'both_success_first_seed': both_success(first, second)}
    fig.suptitle(f"Final models, deterministic, evaluation openings {st['eval_openings']} (n = {len(ops)} per seed); points: seeds",
                 fontsize=9, color=INK2, y=1.03)
    save(fig, out, 'fig_final', stage, data); plt.close(fig)
    tables['final'] = data


# ---------------------------------------------------------------------- C: window commitment
def fig_window(plt, root: Path, stage: str, out: Path, tables: dict):
    reg = registry(); st = stage_of(stage); ops = block(st['eval_openings'])
    pairs = []
    for cfg in st.get('rigid_pair_trained', []):
        for s in st['seeds']:
            a, b = root / 'C' / f'{cfg}_s{s}', root / 'C' / f'{cfg}_s{s}_rigid'
            if a.exists() and b.exists():
                pairs.append((f"{reg['trained'][cfg]['display']} s{s}", load_eval(a), load_eval(b)))
    for pol in st['window_pairs']:
        a, b = root / 'S' / pol, root / 'S' / f'{pol}_rigid'
        if a.exists() and b.exists():
            pairs.append((reg['fixed'][pol]['display'], load_eval(a), load_eval(b)))
    if not pairs:
        return
    data = {'openings': st['eval_openings'], 'pairs': {}}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={'width_ratios': [1, 1, 1.3]})
    for i, (name, w, r) in enumerate(pairs):
        common = [o for o in ops if o in w and o in r]
        sw, sr = summary([w[o] for o in common]), summary([r[o] for o in common])
        d = np.array([w[o]['return'] - r[o]['return'] for o in common])
        data['pairs'][name] = {'n': len(common), 'window': sw, 'rigid': sr, 'paired_return_mean': float(d.mean()) if len(d) else None,
                               'paired_return_median': float(np.median(d)) if len(d) else None,
                               'paired_positive': int((d > 1).sum()), 'paired_negative': int((d < -1).sum()),
                               'both_success': both_success({o: w[o] for o in common}, {o: r[o] for o in common})}
        ax[0].bar([i - 0.18, i + 0.18], [sr['entries_lost'], sw['entries_lost']], width=0.34, color=['#c94f1a', '#2a78d6'])
        ax[1].bar([i - 0.18, i + 0.18], [sr['capture'] / max(sr['n'], 1), sw['capture'] / max(sw['n'], 1)], width=0.34, color=['#c94f1a', '#2a78d6'])
        ax[2].scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, s=10, color=INK2, alpha=0.7)
        if len(d):
            ax[2].plot([i - 0.25, i + 0.25], [d.mean()] * 2, color='#2a78d6', linewidth=2.5)
    names = [p[0] for p in pairs]
    for j, t in enumerate(('(a) entries lost (rigid | window)', '(b) capture rate (rigid | window)', '(c) paired return: window - rigid')):
        ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(names, rotation=30, ha='right', fontsize=8); ax[j].set_title(t, loc='left')
    ax[1].set_ylim(0, 1.02); ax[2].axhline(0, color=MUTED, linewidth=1)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color='#c94f1a', label='rigid commitment'), Patch(color='#2a78d6', label='pass-window commitment')], fontsize=8, loc='upper left')
    fig.suptitle(f"Same openings, same policy, same lower control; only the entry commitment differs (n = {len(ops)} openings)",
                 fontsize=9, color=INK2, y=1.03)
    save(fig, out, 'fig_window', stage, data); plt.close(fig)
    tables['window'] = data


# ---------------------------------------------------------------------- A: sweep-cone geometry
def fig_geometry(plt, gdir: Path, stage: str, out: Path, tables: dict):
    files = sorted(gdir.glob('geoscan_*.json'))
    if not files:
        return
    g = json.loads(files[0].read_text(encoding='utf-8'))
    ranges = list(g['ranges'])
    fig, ax = plt.subplots(1, len(ranges), figsize=(4.3 * len(ranges), 3.9), sharey=True, squeeze=False); ax = ax[0]
    lat = np.array(g['lat_err_deg']); ph = np.array(g['phase_deg']); taus = g['taus_s']
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap('viridis_r', len(taus)); norm = mcolors.BoundaryNorm(list(taus) + [taus[-1] + 10], cmap.N)
    data = {'opening': g['opening'], 'beta_deg': g['beta_deg'], 'rate_deg_s': g['rate_deg_s'], 'ranges': {}}
    for k, r in enumerate(ranges):
        R = g['ranges'][r]
        E = np.array([[np.nan if v is None else v for v in row] for row in R['earliest_feasible_tau_s']], float)
        A = np.array([[np.nan if v is None else v for v in row] for row in R['min_angle_deg']], float)
        im = ax[k].pcolormesh(ph, lat, E, cmap=cmap, norm=norm, shading='nearest')
        cs = ax[k].contour(ph, lat, A, levels=[g['gate_deg']], colors=INK, linewidths=1.2, linestyles='--')
        ax[k].set_title(f'(range {r} m)', loc='left'); ax[k].set_xlabel('port phase to line-of-sight meridian (deg)')
        for arr in g['intent_arrows']:
            if f"{arr['range']:g}" != r:
                continue
            (y0, x0), (y1, x1) = arr['from'], arr['to']
            dx = ((x1 - x0 + 180) % 360) - 180
            colr = {'wait': MUTED, 'intercept': '#c94f1a', 'co-rotate': '#2a78d6'}[arr['intent']]
            ax[k].annotate('', xy=(x0 + dx, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=colr, lw=1.6))
        data['ranges'][r] = {'gate_share': float(np.mean(R['gate_pass'])), 'feasible_share': float(np.mean(~np.isnan(E))),
                             'construction_error_deg': R['construction_error_deg']}
    ax[0].set_ylabel('latitude error  lambda - beta (deg)')
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01); cb.set_label('earliest feasible G* arrival (s); blank: none')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=INK, ls='--', label=f"{g['gate_deg']:g} deg pre-screen (nominal geometry)"),
                        Line2D([], [], color=MUTED, label='one stage of wait (a = 0, 0)'),
                        Line2D([], [], color='#c94f1a', label='one stage of intercept (a_z = +1)'),
                        Line2D([], [], color='#2a78d6', label='one stage of co-rotate (a_z = -1)')],
               fontsize=8, loc='lower center', ncol=4, bbox_to_anchor=(0.45, -0.08))
    fig.suptitle(f"Sweep-cone entry geometry, opening {g['opening']} (beta = {g['beta_deg']:.1f} deg, spin {g['rate_deg_s']:.2f} deg/s, "
                 f"zero relative velocity, constant-rate forecast, estimated inertia)", fontsize=9, color=INK2, y=1.03)
    save(fig, out, 'fig_geometry', stage, data); plt.close(fig)
    tables['geometry'] = data


# ---------------------------------------------------------------------- B: trajectories
def fig_trajectory(plt, tdir: Path, stage: str, out: Path, tables: dict):
    reg = registry(); files = sorted(tdir.glob('traj_*.json'))
    if not files:
        return
    runs = [json.loads(f.read_text(encoding='utf-8')) for f in files]
    openings = sorted({r['opening'] for r in runs})
    fig = plt.figure(figsize=(13, 3.6 * len(openings)))
    gs = fig.add_gridspec(len(openings), 3, width_ratios=[1.1, 1.2, 1.0], hspace=0.6, wspace=0.38)
    data = {'openings': openings, 'runs': []}
    colour = lambda e: (reg['trained'].get(e) or {}).get('color') or {'pure': '#eb6834', 'rally': '#1baf7a'}.get(e, MUTED)
    for i, o in enumerate(openings):
        rs = [r for r in runs if r['opening'] == o]
        a3 = fig.add_subplot(gs[i, 0], projection='3d'); a1 = fig.add_subplot(gs[i, 1]); a2 = fig.add_subplot(gs[i, 2])
        main = next((r for r in rs if r['entry'] == 'EACH'), next((r for r in rs if r['entry'] != 'pure'), rs[0]))
        main_disp = (reg['trained'].get(main['entry']) or reg['fixed'].get(main['entry']) or {}).get('display', main['entry'])
        t = [x['t'] for x in main['rows']]; modes = [x['mode'] for x in main['rows']]
        for j in range(1, len(t)):
            c = MODE_COLORS.get(modes[j])
            if c:
                a1.axvspan(t[j - 1], t[j], color=c, linewidth=0, zorder=0)
        for r in rs:
            col = colour(r['entry']); disp = (reg['trained'].get(r['entry']) or reg['fixed'].get(r['entry']) or {}).get('display', r['entry'])
            P = np.array([np.array(x['rot']).T @ np.array(x['q']) for x in r['rows']])
            a3.plot(P[:, 0], P[:, 1], P[:, 2], color=col, linewidth=1.5, label=f"{disp}: {EVENT_SHORT.get(r['event'], r['event'])}")
            a3.scatter(*P[-1], color=col, s=25)
            a1.plot([x['t'] for x in r['rows']], [x['range'] for x in r['rows']], color=col)
            a2.plot([x['phase_deg'] for x in r['rows']], [x['lat_err_deg'] for x in r['rows']], color=col, linewidth=1.2)
            a2.scatter(r['rows'][0]['phase_deg'], r['rows'][0]['lat_err_deg'], color=col, s=14, marker='o')
            a2.scatter(r['rows'][-1]['phase_deg'], r['rows'][-1]['lat_err_deg'], color=col, s=34, marker='X' if r['event'] != 'success' else '*')
            data['runs'].append({'opening': o, 'entry': r['entry'], 'event': r['event'], 'return': r['return'], 'time_s': r['time_s'],
                                 'delta_v': r['delta_v'], 'entries_lost': r.get('entries_lost'),
                                 'modes': [m for k, m in enumerate(modes) if k == 0 or m != modes[k - 1]] if r is main else None})
        if main.get('approach_axis') is not None:
            ax_b = np.asarray(main['approach_axis'], float)
            a3.plot(*np.c_[np.zeros(3), 6.3 * ax_b], color='#c94f1a', linewidth=2.5); a3.text(*(8.0 * ax_b), 'port axis', fontsize=7.5, color=INK2)
        a3.scatter(0, 0, 0, color=INK, s=12)
        a3.set_xlabel('x (m)', fontsize=8); a3.set_ylabel('y (m)', fontsize=8)
        a3.set_title(f'opening {o}: target body frame (m)', loc='left', fontsize=9.5); a3.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.02))
        a1.axhline(6.3, color=MUTED, ls=':', lw=1); a1.text(1, 6.45, 'mouth radius 6.3 m', fontsize=7.5, color=INK2)
        a1.set_xlabel('time (s)'); a1.set_ylabel('range (m)')
        a1.set_title(f'range; shading: {main_disp} modes', loc='left', fontsize=9)
        a2.axhline(0, color=MUTED, lw=1); a2.set_xlabel('port phase to line-of-sight meridian (deg)'); a2.set_ylabel('latitude error (deg)')
        a2.set_title('path on the sweep-cone plane (fig. A axes); o start, * capture', loc='left', fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=m) for m, c in MODE_COLORS.items()], loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.13 / len(openings)))
    save(fig, out, 'fig_trajectory', stage, data); plt.close(fig)
    tables['trajectory'] = data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--geoscan'); p.add_argument('--traj')
    a = p.parse_args()
    root, out = Path(a.root), Path(a.output)
    res = resolve(a.stage, root)
    plt = style(); tables = {'stage': a.stage, 'science_sha': res['science_sha'], 'steps': res['steps'], 'registry_sha256': res['registry_sha256']}
    fig_training(plt, root, res, a.stage, out, tables)
    fig_final(plt, root, res, a.stage, out, tables)
    fig_window(plt, root, a.stage, out, tables)
    if a.geoscan:
        fig_geometry(plt, Path(a.geoscan), a.stage, out, tables)
    if a.traj:
        fig_trajectory(plt, Path(a.traj), a.stage, out, tables)
    (out / 'TABLES.json').write_text(json.dumps(tables, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in tables.items()}))


if __name__ == '__main__':
    main()
