"""Presentation / paper figures of EACH from published evidence (no simulation except the geometry scan's planner calls).

    python -B -m experiments.ppt_figures dev     --output DIR [--geoscan GEOSCAN_DIR]
    python -B -m experiments.ppt_figures pretest --root PRETEST_DIR --output DIR

``dev``: development-stage evidence already in git (openings 294000-294047, urgent deadlines U(140, 200) s, two seeds x
2000 decisions), labelled as development validation:
  P1 training curves        EACH (T2.12 e_priv), SAC (T2.11 e_sac), SAC + planner features (T2.8 e_sac, engineered)
  P2 final models           capture, urgent capture, mean true return; pure MPC, direct G*, deadline rule (T2.6W screen)
  P3 return distribution    per-opening true return (box plot), same entries
  P4 window commitment      T2.3 A2: rigid vs pass-window commitment, 120 paired episodes
  P5 sweep-cone geometry    from ``experiments.final geoscan`` output (``--geoscan``)
  P6 how executor knowledge enters learning   2000-step models and curve AUC of every development variant
``pretest``: the same P1-P3 plus P7 trajectories and P8 ablations from a pretest directory (``experiments.final`` layout).
Legends follow ``docs/final_method/METHOD_REGISTRY.json``; colours: EACH blue, SAC orange, references grey.
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parents[1] / 'evidence'
BLUE, ORANGE, GREY, GREEN, PURPLE, RED, INK2, GRID = '#2a78d6', '#eb6834', '#9a9993', '#1baf7a', '#7b3fb5', '#c94f1a', '#52514e', '#e4e3df'

DEV_TRAINED = {   # display: (colour, [(episodes dir, eval dir)])
    'EACH (U1+C1)': (BLUE, [(EV / 'T2_12_EACC/B' / f'TRAIN_e_priv_s{s}', EV / 'T2_12_EACC/C' / f'e_priv_s{s}' / 'step_2000') for s in (0, 1)]),
    'SAC': (ORANGE, [(EV / 'T2_11_EACCA/B' / f'TRAIN_e_sac_s{s}', EV / 'T2_11_EACCA/C' / f'e_sac_s{s}' / 'step_2000') for s in (0, 1)]),
    'SAC + 规划特征': (GREY, [(EV / 'T2_8_NIGHT/B' / f'TRAIN_e_sac_s{s}', EV / 'T2_8_NIGHT/C' / f'e_sac_s{s}' / 'step_2000') for s in (0, 1)]),
}
DEV_FIXED = {'纯 MPC': 'pure', '直接 G★': 'gstar', '截止规则': 'rally'}
DEV_MECH = [   # (group, display, round, arm)
    ('无执行器知识', 'SAC', 'T2_11_EACCA', 'e_sac'),
    ('作为 critic 输入（U1）', 'EACH', 'T2_12_EACC', 'e_priv'),
    ('作为 critic 回归目标', 'EACC-A 标签', 'T2_11_EACCA', 'e_acc'),
    ('作为 critic 回归目标', '相对标签', 'T2_12_EACC', 'e_rel'),
    ('作为 critic 回归目标', '终局标签', 'T2_14_TERM', 'e_term'),
    ('作为策略修正', '局部修正', 'T2_17_PULL', 'e_pi'),
    ('作为策略修正', '局部 + 非局部', 'T2_17_PULL', 'e_pull'),
    ('历史校准作为 critic 输入', '机会图', 'T2_19_CERT', 'e_cert'),
]
GROUP_COLOUR = {'无执行器知识': ORANGE, '作为 critic 输入（U1）': BLUE, '作为 critic 回归目标': '#8e2a1e', '作为策略修正': GREEN,
                '历史校准作为 critic 输入': PURPLE}


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                         'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 2.2, 'savefig.dpi': 300})
    return plt


def save(fig, out: Path, name: str, note: str, data: dict):
    fig.text(0.01, -0.02, note, ha='left', va='top', fontsize=8.5, color=GREY, transform=fig.transFigure)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', facecolor='white')
    (out / f'{name}.json').write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def load_eval(d: Path) -> dict:
    out = {}
    for f in sorted(Path(d).glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if 'opening' in r and 'event' in r and not f.name.startswith('ERROR_'):
            out[int(r['opening'])] = r
    return out


def rate(rs, urgent=None):
    rs = [r for r in rs if urgent is None or r['mission']['urgent'] == urgent]
    return (sum(r['event'] == 'success' for r in rs) / len(rs)) if rs else np.nan


# ---------------------------------------------------------------------- P1 training curves
def p1_curves(plt, entries: dict, out: Path, note: str, steps: int = 2000):
    from experiments.t2_curves import run_curve
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); data = {}
    for name, (col, runs) in entries.items():
        cs = [run_curve(json.loads((b / 'episodes.json').read_text(encoding='utf-8')), 50, steps) for b, _ in runs]
        grid = np.array(cs[0]['grid']); G = np.array([[np.nan if v is None else v for v in c['grid_rolling']] for c in cs], float)
        ls = '--' if col == GREY else '-'
        ax.plot(grid, np.nanmean(G, 0), color=col, ls=ls, label=f'{name}（{len(cs)} 个种子）')
        ax.fill_between(grid, np.nanmin(G, 0), np.nanmax(G, 0), color=col, alpha=0.15, linewidth=0)
        data[name] = {'curve_auc': [c['curve_auc'] for c in cs], 'grid': grid.tolist(), 'mean': np.nanmean(G, 0).tolist()}
    ax.set_xlabel('训练决策数'); ax.set_ylabel('训练回合真实回报（50 回合滑动平均）')
    ax.set_title('训练曲线（阴影：种子范围）', loc='left'); ax.legend(loc='lower right')
    save(fig, out, 'P1_training_curves', note, data); plt.close(fig)


# ---------------------------------------------------------------------- P2 / P3 final models
def p2_p3_final(plt, trained: dict, fixed: dict, out: Path, note: str):
    """trained: display -> (colour, [eval dirs]); fixed: display -> eval dir."""
    rows = [(n, c, [load_eval(d) for d in ds]) for n, (c, ds) in trained.items()] + [(n, GREY, [load_eval(d)]) for n, d in fixed.items()]
    order = [r for r in rows if r[1] == GREY and r[0] in fixed] + [r for r in rows if r[0] not in fixed]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2)); data = {}
    for i, (name, col, seeds) in enumerate(order):
        per = [list(s.values()) for s in seeds]
        vals = {'总体捕获率': [rate(p) for p in per], '紧急任务捕获率': [rate(p, True) for p in per], '平均真实回报': [float(np.mean([r['return'] for r in p])) for p in per]}
        data[name] = {**vals, 'n_openings': [len(p) for p in per], 'unsafe': [sum(r['event'] == 'unsafe' for r in p) for p in per]}
        for j, v in enumerate(vals.values()):
            ax[j].bar(i, np.mean(v), color=col, alpha=0.85 if col != GREY else 0.45, width=0.68, hatch='//' if name == 'SAC + 规划特征' else None,
                      edgecolor='white')
            if len(v) > 1:
                ax[j].scatter([i] * len(v), v, color='#0b0b0b', s=16, zorder=3)
    names = [o[0] for o in order]
    for j, t in enumerate(('总体捕获率', '紧急任务捕获率', '平均真实回报')):
        ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(names, rotation=30, ha='right'); ax[j].set_title(t, loc='left')
    ax[0].set_ylim(0, 1.02); ax[1].set_ylim(0, 1.02); ax[2].axhline(0, color=GREY, lw=1)
    save(fig, out, 'P2_final_models', note + '；点：种子', data); plt.close(fig)
    fig, axb = plt.subplots(figsize=(8, 4.2))
    box = [[r['return'] for s in seeds for r in s.values()] for _, _, seeds in order]
    bp = axb.boxplot(box, patch_artist=True, widths=0.6, medianprops=dict(color='#0b0b0b'), flierprops=dict(markersize=3))
    for patch, (_, col, _) in zip(bp['boxes'], order):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    axb.set_xticks(range(1, len(names) + 1)); axb.set_xticklabels(names, rotation=30, ha='right'); axb.axhline(0, color=GREY, lw=1)
    axb.set_ylabel('单个开局的真实回报'); axb.set_title('逐开局回报分布（训练方法合并两个种子）', loc='left')
    save(fig, out, 'P3_return_distribution', note, {n: {'median': float(np.median(b)), 'n': len(b)} for n, b in zip(names, box)}); plt.close(fig)


# ---------------------------------------------------------------------- P4 window commitment
def p4_window(plt, out: Path, note: str):
    A = json.loads((EV / 'T2_3_TEST/A/A2_WINDOW.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
    labs, cols = ['刚性时刻', '窗口承诺'], [RED, BLUE]
    ax[0].bar(labs, [A['entries_lost']['rigid'], A['entries_lost']['window']], color=cols); ax[0].set_title('入口丢失次数', loc='left')
    ax[1].bar(labs, [A['rigid']['capture'] / A['rigid']['n'], A['window']['capture'] / A['window']['n']], color=cols); ax[1].set_ylim(0, 1)
    ax[1].set_title('捕获率', loc='left')
    ax[2].bar(labs, [A['rigid']['mean_return'], A['window']['mean_return']], color=cols); ax[2].set_title('平均真实回报', loc='left')
    for a in ax:
        for p in a.patches:
            a.annotate(f'{p.get_height():.2f}' if p.get_height() < 2 else f'{p.get_height():.0f}' if a is ax[0] else f'{p.get_height():.1f}',
                       (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=10)
    fig.suptitle(f"同一开局、同一策略、同一底层控制，只改入口承诺方式（{A['pairs']} 组配对：5 种固定策略 × 24 个开局）", fontsize=10.5, color=INK2, y=1.03)
    save(fig, out, 'P4_window_commitment', note, A); plt.close(fig)


# ---------------------------------------------------------------------- P5 sweep-cone geometry
def p5_geometry(plt, gdir: Path, out: Path, note: str):
    files = sorted(Path(gdir).glob('geoscan_*.json'))
    if not files:
        return
    g = json.loads(files[0].read_text(encoding='utf-8'))
    import matplotlib.colors as mcolors
    taus = g['taus_s']; cmap = plt.get_cmap('viridis_r', len(taus)); norm = mcolors.BoundaryNorm(list(taus) + [taus[-1] + 10], cmap.N)
    ranges = list(g['ranges'])
    fig, ax = plt.subplots(1, len(ranges), figsize=(4.4 * len(ranges), 4.0), sharey=True, squeeze=False); ax = ax[0]
    lat, ph = np.array(g['lat_err_deg']), np.array(g['phase_deg'])
    for k, r in enumerate(ranges):
        R = g['ranges'][r]
        E = np.array([[np.nan if v is None else v for v in row] for row in R['earliest_feasible_tau_s']], float)
        A = np.array([[np.nan if v is None else v for v in row] for row in R['min_angle_deg']], float)
        im = ax[k].pcolormesh(ph, lat, E, cmap=cmap, norm=norm, shading='nearest')
        ax[k].contour(ph, lat, A, levels=[g['gate_deg']], colors='#0b0b0b', linewidths=1.3, linestyles='--')
        for arr in g['intent_arrows']:
            if f"{arr['range']:g}" == r and arr['from'][1] == 180.0:
                (y0, x0), (y1, x1) = arr['from'], arr['to']; dx = ((x1 - x0 + 180) % 360) - 180
                c = {'wait': GREY, 'intercept': RED, 'co-rotate': BLUE}[arr['intent']]
                ax[k].annotate('', xy=(x0 + dx, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=c, lw=2))
        ax[k].set_title(f'距离 {r} m', loc='left'); ax[k].set_xlabel('端口到达视线子午面的相位（°）'); ax[k].set_xlim(0, 360)
    ax[0].set_ylabel('纬向误差 Δλ（°）')
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01); cb.set_label('最早可行入口到达时间（s）；空白：不可行')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color='#0b0b0b', ls='--', label='35° 几何预筛边界'), Line2D([], [], color=GREY, label='等待一个阶段'),
                        Line2D([], [], color=RED, label='迎击一个阶段'), Line2D([], [], color=BLUE, label='伴飞一个阶段')],
               loc='lower center', ncol=4, bbox_to_anchor=(0.45, -0.16))
    fig.suptitle(f"扫掠锥入口机会集（开局 {g['opening']}，β = {g['beta_deg']:.0f}°，自旋 {g['rate_deg_s']:.2f}°/s，零相对速度）", fontsize=11, y=1.02)
    save(fig, out, 'P5_sweep_cone_geometry', note, {k: {'gate_share': float(np.mean(v['gate_pass']))} for k, v in g['ranges'].items()}); plt.close(fig)


# ---------------------------------------------------------------------- P6 how executor knowledge enters learning
def p6_mechanism(plt, out: Path, note: str):
    from experiments.t2_curves import run_curve
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4)); data = {}
    for i, (grp, name, rnd, arm) in enumerate(DEV_MECH):
        aucs, rets = [], []
        for s in (0, 1):
            eps = json.loads((EV / rnd / 'B' / f'TRAIN_{arm}_s{s}' / 'episodes.json').read_text(encoding='utf-8'))
            aucs.append(run_curve(eps, 50, 2000)['curve_auc'])
            rets.append(float(np.mean([r['return'] for r in load_eval(EV / rnd / 'C' / f'{arm}_s{s}' / 'step_2000').values()])))
        data[name] = {'group': grp, 'round': rnd, 'arm': arm, 'curve_auc': aucs, 'final_return': rets}
        for j, v in enumerate((aucs, rets)):
            ax[j].bar(i, np.mean(v), color=GROUP_COLOUR[grp], width=0.68, alpha=0.85); ax[j].scatter([i, i], v, color='#0b0b0b', s=16, zorder=3)
    names = [m[1] for m in DEV_MECH]
    for j, t in enumerate(('训练曲线 AUC（250–2000 决策）', '2000 决策模型的平均真实回报（48 个开局）')):
        ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(names, rotation=30, ha='right'); ax[j].set_title(t, loc='left'); ax[j].axhline(0, color=GREY, lw=1)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COLOUR.items()], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle('执行器知识以何种方式进入学习：作为 critic 输入（U1）最好；回归目标、策略修正、历史校准输入均不及', fontsize=11.5, y=1.03)
    save(fig, out, 'P6_how_executor_knowledge_enters', note + '；点：种子', data); plt.close(fig)


# ---------------------------------------------------------------------- pretest-stage figures
def pretest(plt, root: Path, out: Path):
    from experiments.final import registry, stage_of, resolve
    reg = registry(); st = stage_of('pretest'); res = resolve('pretest', root)
    note = f"预实验：2 个种子 × 2000 决策，评估开局 {st['eval_openings']}，科学 {str(res['science_sha'])[:7]}"
    disp = {'EACH': 'EACH (U1+C1)', 'SAC': 'SAC', 'SAC_ENG': 'SAC + 规划特征', 'DUP': '无 U1（等宽 critic）', 'NO_GEOM': '无 C1 几何'}
    col = {'EACH': BLUE, 'SAC': ORANGE, 'SAC_ENG': GREY, 'DUP': PURPLE, 'NO_GEOM': GREEN}
    main = ('EACH', 'SAC', 'SAC_ENG')
    p1_curves(plt, {disp[c]: (col[c], [(root / 'B' / f'TRAIN_{c}_s{s}', None) for s in st['seeds']]) for c in main}, out, note)
    p2_p3_final(plt, {disp[c]: (col[c], [root / 'C' / f'{c}_s{s}' for s in st['seeds']]) for c in main},
                {'纯 MPC': root / 'S' / 'pure', '直接 G★': root / 'S' / 'gstar', '截止规则': root / 'S' / 'rally'}, out, note)
    abl = ('EACH', 'DUP', 'NO_GEOM', 'SAC')
    p1_curves(plt, {disp[c]: (col[c] if c != 'SAC_ENG' else GREY, [(root / 'B' / f'TRAIN_{c}_s{s}', None) for s in st['seeds']]) for c in abl}, out / 'ablation', note)
    p2_p3_final(plt, {disp[c]: (col[c], [root / 'C' / f'{c}_s{s}' for s in st['seeds']]) for c in abl}, {}, out / 'ablation', note)
    # window pairing on the trained EACH models and fixed policies
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); data = {}
    pairs = [(f'EACH 种子 {s}', root / 'C' / f'EACH_s{s}', root / 'C' / f'EACH_s{s}_rigid') for s in st['seeds']] + \
            [({'gstar': '直接 G★', 'rally': '截止规则', 'c_10_0': '固定意图'}[p], root / 'S' / p, root / 'S' / f'{p}_rigid') for p in st['window_pairs']]
    for i, (n, w, r) in enumerate(pairs):
        W, R = load_eval(w), load_eval(r)
        lost = [sum(x.get('entries_lost', 0) or 0 for x in R.values()), sum(x.get('entries_lost', 0) or 0 for x in W.values())]
        cap = [rate(list(R.values())), rate(list(W.values()))]
        data[n] = {'entries_lost_rigid_window': lost, 'capture_rigid_window': cap}
        ax[0].bar([i - 0.18, i + 0.18], lost, width=0.34, color=[RED, BLUE]); ax[1].bar([i - 0.18, i + 0.18], cap, width=0.34, color=[RED, BLUE])
    for a, t in zip(ax, ('入口丢失次数', '捕获率')):
        a.set_xticks(range(len(pairs))); a.set_xticklabels([p[0] for p in pairs], rotation=25, ha='right'); a.set_title(t, loc='left')
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=RED, label='刚性时刻'), Patch(color=BLUE, label='窗口承诺')])
    save(fig, out, 'P4_window_commitment_pretest', note, data); plt.close(fig)
    if (root / 'G').exists():
        p5_geometry(plt, root / 'G', out, note)


def main():
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='cmd', required=True)
    d = s.add_parser('dev'); d.add_argument('--output', required=True); d.add_argument('--geoscan')
    t = s.add_parser('pretest'); t.add_argument('--root', required=True); t.add_argument('--output', required=True)
    a = p.parse_args(); plt = style(); out = Path(a.output)
    if a.cmd == 'dev':
        note = '开发验证：开局 294000–294047，紧急期限 U(140, 200) s，2 个种子 × 2000 决策；SAC + 规划特征取自 T2.8（较早代码版本），纯 MPC / G★ / 截止规则取自 T2.6W 同开局筛选'
        p1_curves(plt, {k: (c, [(b, None) for b, _ in runs]) for k, (c, runs) in DEV_TRAINED.items()}, out, note)
        fixed_dirs = {}
        for k, v in DEV_FIXED.items():          # the T2.6W reference screen keeps all policies in one directory
            tmp = out / '_ref' / v; tmp.mkdir(parents=True, exist_ok=True)
            for f in (EV / 'T2_6W_WIDE_140_200/A_ref').glob(f'{v}_[0-9]*.json'):
                (tmp / f.name).write_text(f.read_text(encoding='utf-8'), encoding='utf-8')
            fixed_dirs[k] = tmp
        p2_p3_final(plt, {k: (c, [e for _, e in runs]) for k, (c, runs) in DEV_TRAINED.items()}, fixed_dirs, out, note)
        p4_window(plt, out, '开发验证：T2.3 A2，开发开局 294000–294023')
        if a.geoscan:
            p5_geometry(plt, Path(a.geoscan), out, '冻结执行器的几何与 G★ 可行性扫描（不读任何评估回报）')
        p6_mechanism(plt, out, note)
    else:
        pretest(plt, Path(a.root), out)


if __name__ == '__main__':
    main()
