"""T2 presentation figures from published assets (no training).

    python -B -m experiments.t2_figures traj --policy ours=CKPT.zip --policy pure --policy basic=b_10 --openings 294022,294015 --output DIR
    python -B -m experiments.t2_figures plot --assets ASSETS.json --output DIR

``traj`` re-flies single episodes serially (one process, so its wall times are serial timings) and records
the truth every 2 s: relative position / velocity, target attitude, the port axis, mode, delta-v and the wall
time of the step. Policies: ``NAME=PATH.zip`` (a trained SAC checkpoint, deterministic), ``pure`` (the
shared MPC alone), ``NAME=b_10`` / ``NAME=c_10_0`` / ``rally`` (fixed policies of ``experiments.t2``).
``--window`` flies the T2.3 pass-window commitment; the default is T2.2's rigid commitment (the executor
the published T2.2 checkpoints and evaluations used).

``plot`` draws five figures (PNG 300 dpi + PDF) from the paths listed in ASSETS.json:
  fig1_training   training-episode return (behaviour policy, moving mean) and evaluation return per checkpoint
  fig2_convergence critic TD error and Q(s, pi(s)) over updates
  fig3_trajectory  one opening: trajectory around the target (target frame) + range and port angle over time
  fig4_performance capture rate, mean return, time and delta-v of successful episodes, serial compute
  fig5_mechanism   (a) rotation geometry at equal closing speed, (b) entry commitment, (c) label accuracy
Panels whose inputs are absent are left out; nothing is invented.
"""
import argparse, glob, json, time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------- style (validated reference palette)
OURS, BASE1, BASE2, MUTED, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993', '#0b0b0b', '#52514e', '#e4e3df'
MODE_COLORS = {'stage': '#dce9f9', 'hold': '#f2f1ee', 'bridge': '#fde3d6', 'descent': '#d5f0e5'}


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 2.0, 'savefig.dpi': 300, 'figure.dpi': 110})
    return plt


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', facecolor='white')


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(str(Path(d) / '*.json'))):
        if Path(f).name.startswith('ERROR_'):
            continue
        r = json.loads(Path(f).read_text(encoding='utf-8'))
        if 'opening' in r and 'event' in r:
            out[int(r['opening'])] = r
    return out


def stats(rs):
    rs = list(rs); ok = [r for r in rs if r['event'] == 'success']
    return {'n': len(rs), 'return': float(np.mean([r['return'] for r in rs])), 'capture': len(ok) / max(len(rs), 1),
            'unsafe': sum(r['event'] == 'unsafe' for r in rs),
            'time': float(np.mean([r['time_s'] for r in ok])) if ok else np.nan,
            'dv': float(np.mean([r['delta_v'] for r in ok])) if ok else np.nan}


def episode_return(e):
    """True (unshaped) return of a training episode from its log: time and fuel cost of its mission, terminal reward."""
    m = e['mission']
    terminal = {'success': 100.0, 'unsafe': -150.0, 'timeout': -100.0, 'resource_exhausted': -100.0}.get(e['event'], -100.0)
    return -m['time_cost_per_s'] * e['time_s'] - m['fuel_cost_per_m_s'] * e['delta_v'] + terminal


# ---------------------------------------------------------------------- trajectories
def fly(opening: int, spec: str, window: bool) -> dict:
    import torch; torch.set_num_threads(1)
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import T2StageEnv, angle_deg
    from experiments.t2 import policy_action
    name, _, src = spec.partition('=')
    src = src or name
    env = T2StageEnv(openings=[opening], shaping=False, geometry=not src.startswith('b_'), window=window)
    obs, _ = env.reset(options={'opening': opening})
    rows, inference = [], []
    axis = np.asarray(env.env.config.task.approach_axis)

    def record(wall):
        q, v = relative_inertial(env.env); t = env.env.target_state
        rows.append({'t': env.env.time_s, 'q': q.tolist(), 'v': v.tolist(), 'dv': env.env._delta_v, 'rot': t.rotation.tolist(),
                     'omega': t.omega.tolist(), 'port': (t.rotation @ axis).tolist(), 'range': float(np.linalg.norm(q)),
                     'port_los_deg': angle_deg(t.rotation @ axis, q), 'mode': env.generator.mode if env.generator else 'pure', 'wall_s': wall})
    record(0.0)
    step = env.env.step
    def timed(*a, **k):
        t0 = time.perf_counter(); out = step(*a, **k); record(time.perf_counter() - t0); return out
    env.env.step = timed
    t0 = time.perf_counter(); ret = 0.0; event = None
    if src == 'pure':
        while event is None:
            r, term, trunc, i = env.step_pure(); ret += r; event = i.get('event')
            if event is None and env.env._delta_v >= env.mission['budget_m_s']:
                event = 'resource_exhausted'; ret -= env.env.config.reward.timeout_penalty
    else:
        if src.endswith('.zip'):
            from train.t2_mve_sac import T2SAC
            model = T2SAC.load(src, device='cpu')
            def act(o):
                s = time.perf_counter(); a = model.predict(o, deterministic=True)[0]; inference.append(time.perf_counter() - s); return a
        else:
            act = lambda o: policy_action(src, env, o)
        done = False
        while not done:
            obs, r, term, trunc, i = env.step(act(obs)); done = term or trunc; ret += i['true_reward']; event = i['event']
    return {'opening': opening, 'policy': name, 'source': src, 'window': window, 'event': event, 'return': ret,
            'time_s': env.env.time_s, 'delta_v': env.env._delta_v, 'mission': env.mission, 'wall_s': time.perf_counter() - t0,
            'inference_s': inference, 'beta_axis': None, 'rows': rows}


def traj(policies, openings, out: Path, window: bool):
    out.mkdir(parents=True, exist_ok=True)
    for o in openings:
        for spec in policies:
            name = spec.partition('=')[0]
            f = out / f'traj_{name}_{o}.json'
            if f.exists():
                continue
            res = fly(o, spec, window)
            f.write_text(json.dumps(res), encoding='utf-8')
            print(json.dumps({k: res[k] for k in ('opening', 'policy', 'event', 'return', 'time_s', 'delta_v', 'wall_s')}), flush=True)


# ---------------------------------------------------------------------- figures
LABELS = {'ours': '本文框架（SAC–G★–MPC）', 'pure': '纯 MPC', 'basic': '基本分层（不感知旋转）', 'coupled_t22': 'T2.2 耦合版'}
COLORS = {'ours': OURS, 'pure': BASE1, 'basic': BASE2, 'coupled_t22': '#6250d6'}


def fig_training(plt, A, out):
    runs = A.get('training', {})
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, path in runs.items():
        eps = json.loads((Path(path) / 'episodes.json').read_text(encoding='utf-8'))
        x = np.array([e['steps'] for e in eps]); y = np.array([episode_return(e) for e in eps])
        k = 50
        if len(y) >= k:
            m = np.convolve(y, np.ones(k) / k, mode='valid')
            ax[0].plot(x[k - 1:], m, color=COLORS[key], label=LABELS[key])
    ax[0].set_title('(a) 训练回合回报（行为策略，50 回合滑动平均）', loc='left')
    ax[0].set_xlabel('训练决策数'); ax[0].set_ylabel('回合真实回报')
    for key, root in A.get('evaluation', {}).items():
        ks, ys = [], []
        for d in sorted(Path(root).glob('step_*'), key=lambda p: int(p.name.split('_')[1])):
            rs = load_dir(d)
            if rs:
                ks.append(int(d.name.split('_')[1])); ys.append(stats(rs.values())['return'])
        ax[1].plot(ks, ys, color=COLORS[key], marker='o', markersize=7, markeredgecolor='white', markeredgewidth=1.5, label=LABELS[key])
        if ys:
            ax[1].annotate(f'{ys[-1]:.1f}', (ks[-1], ys[-1]), textcoords='offset points', xytext=(6, -3), color=INK2, fontsize=9)
    refs = sorted(reference_returns(A).items(), key=lambda kv: -kv[1])
    for j, (key, val) in enumerate(refs):
        ax[1].axhline(val, color=COLORS[key], linestyle='--', linewidth=1.3)
        ax[1].annotate(f'{LABELS[key]} {val:.1f}', (1.0, val), xycoords=('axes fraction', 'data'), xytext=(-4, 4 if j == 0 else -12),
                       textcoords='offset points', ha='right', color=INK2, fontsize=8.5)
    ax[1].set_title('(b) 评估回报（24 个开发开局，确定性策略）', loc='left')
    ax[1].set_xlabel('训练决策数'); ax[1].set_ylabel('平均回报')
    ax[0].legend(loc='lower right', fontsize=9); ax[1].legend(loc='center right', fontsize=9)
    fig.tight_layout(); save(fig, out, 'fig1_training'); plt.close(fig)


def reference_returns(A):
    S = A.get('screen')
    if not S:
        return {}
    out = {}
    pure = load_dir_policy(S, 'pure')
    if pure:
        out['pure'] = stats(pure.values())['return']
    basic = [load_dir_policy(S, p) for p in ('b_0', 'b_5', 'b_10')]
    basic = [b for b in basic if b]
    if basic:
        out['basic'] = max(stats(b.values())['return'] for b in basic)
    return out


def load_dir_policy(screen, policy):
    out = {}
    for f in glob.glob(str(Path(screen) / f'{policy}_*.json')):
        r = json.loads(Path(f).read_text(encoding='utf-8'))
        if r.get('policy') == policy:
            out[int(r['opening'])] = r
    return out


def fig_convergence(plt, A, out):
    runs = A.get('training', {})
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for key, path in runs.items():
        f = Path(path) / 'model_stats.json'
        if not f.exists():
            continue
        ms = json.loads(f.read_text(encoding='utf-8'))
        x = np.array([m['update'] for m in ms]); td = np.array([m['real_td'] for m in ms]); q = np.array([m['q_pi_mean'] for m in ms])
        k = 20
        sm = lambda y: np.convolve(y, np.ones(k) / k, mode='valid') if len(y) >= k else y
        xs = x[k - 1:] if len(x) >= k else x
        ax[0].plot(x, td, color=COLORS[key], alpha=0.18, linewidth=1)
        ax[0].plot(xs, sm(td), color=COLORS[key], label=LABELS[key])
        ax[1].plot(xs, sm(q), color=COLORS[key], label=LABELS[key])
    ax[0].set_yscale('log'); ax[0].set_title('(a) Critic 时序差分误差', loc='left'); ax[0].set_xlabel('梯度更新次数'); ax[0].set_ylabel('TD 误差（对数）')
    ax[1].set_title('(b) 策略价值估计 Q(s, π(s))', loc='left'); ax[1].set_xlabel('梯度更新次数'); ax[1].set_ylabel('批平均')
    for a in ax:
        a.legend(loc='best', fontsize=9)
    fig.tight_layout(); save(fig, out, 'fig2_convergence'); plt.close(fig)


def fig_trajectory(plt, A, out):
    T = A.get('trajectory')
    if not T:
        return
    files = sorted(glob.glob(str(Path(T['dir']) / f"traj_*_{T['opening']}.json")))
    runs = {json.loads(Path(f).read_text(encoding='utf-8'))['policy']: json.loads(Path(f).read_text(encoding='utf-8')) for f in files}
    order = [k for k in ('ours', 'basic', 'pure') if k in runs]
    if not order:
        return
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.45, wspace=0.22)
    ax3 = fig.add_subplot(gs[:, 0], projection='3d')
    ref = runs[order[0]]['rows'][0]
    R0 = np.array(ref['rot']); w = np.array(ref['omega']); s_hat = R0 @ w / np.linalg.norm(w)
    end_mark = {'success': ('o', '捕获'), 'unsafe': ('X', '不安全'), 'timeout': ('s', '超时'), 'resource_exhausted': ('s', '燃料耗尽')}
    for key in order:
        r = runs[key]
        # target body frame: the port is fixed there, the chaser's path shows how it met the rotating port
        P = np.array([np.array(x['rot']).T @ np.array(x['q']) for x in r['rows']])
        ax3.plot(P[:, 0], P[:, 1], P[:, 2], color=COLORS[key], linewidth=1.8, label=f"{LABELS[key]}：{end_mark.get(r['event'], ('s', r['event']))[1]}")
        m, _ = end_mark.get(r['event'], ('s', ''))
        ax3.scatter(*P[-1], color=COLORS[key], marker=m, s=60, edgecolor='white', linewidth=1.2, depthshade=False)
        ax3.scatter(*P[0], color=COLORS[key], marker='.', s=30, depthshade=False)
    axis = np.array(runs[order[0]]['rows'][0]['rot']).T @ np.array(runs[order[0]]['rows'][0]['port'])
    import itertools
    h = np.array([1.0, 1.25, 1.0])
    corners = [np.array(c) * h for c in itertools.product((-1, 1), repeat=3)]
    for c1, c2 in itertools.combinations(corners, 2):
        if np.count_nonzero(c1 != c2) == 1:          # the 12 edges of the box
            ax3.plot(*np.c_[c1, c2], color=MUTED, linewidth=1)
    ax3.plot(*np.c_[np.zeros(3), 6.3 * axis], color='#c94f1a', linewidth=2.5)
    ax3.text(*(9.0 * axis), '对接口轴', color=INK2, fontsize=9)
    from matplotlib.ticker import MaxNLocator
    for g in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        g.set_major_locator(MaxNLocator(4))
    ax3.set_title(f"(a) 目标本体系中的逼近轨迹（开局 {T['opening']}）", loc='left')
    ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)'); ax3.set_zlabel('z (m)')
    ax3.legend(loc='upper left', fontsize=8.5, bbox_to_anchor=(0.0, 0.98))
    ax3.view_init(elev=22, azim=-55)
    a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    rows = runs[order[0]]['rows']
    t = [x['t'] for x in rows]; modes = [x['mode'] for x in rows]
    for i in range(1, len(rows)):
        c = MODE_COLORS.get(modes[i])
        if c:
            for a in (a1, a2):
                a.axvspan(t[i - 1], t[i], color=c, linewidth=0, zorder=0)
    for key in order:
        rr = runs[key]['rows']
        a1.plot([x['t'] for x in rr], [x['range'] for x in rr], color=COLORS[key])
        a2.plot([x['t'] for x in rr], [x['port_los_deg'] for x in rr], color=COLORS[key])
    a1.axhline(6.3, color=MUTED, linestyle=':', linewidth=1.2); a1.text(1, 6.5, '入口半径 6.3 m', color=INK2, fontsize=8.5)
    a2.axhline(35, color=MUTED, linestyle=':', linewidth=1.2); a2.text(1, 38, '入口窗口 35°', color=INK2, fontsize=8.5)
    a1.set_ylabel('距离 (m)'); a1.set_title('(b) 距离与对接口–视线夹角（底色：本文方法的阶段 / 桥接 / 下降）', loc='left', fontsize=10)
    a2.set_ylabel('夹角 (°)'); a2.set_xlabel('时间 (s)')
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=MODE_COLORS[m], label=l) for m, l in (('stage', '阶段'), ('bridge', '桥接'), ('descent', '下降'))],
              loc='upper right', fontsize=8, ncol=3)
    save(fig, out, 'fig3_trajectory'); plt.close(fig)


def fig_performance(plt, A, out):
    rows = {}
    S = A.get('screen')
    if S:
        pure = load_dir_policy(S, 'pure')
        if pure:
            rows['pure'] = stats(pure.values())
        basic = {p: load_dir_policy(S, p) for p in ('b_0', 'b_5', 'b_10')}
        basic = {p: b for p, b in basic.items() if b}
        if basic:
            best = max(basic, key=lambda p: stats(basic[p].values())['return'])
            rows['basic'] = stats(basic[best].values())
    if A.get('final'):
        rows['ours'] = stats(load_dir(A['final']).values())
    compute = {}
    T = A.get('trajectory')
    if T:
        for f in glob.glob(str(Path(T['dir']) / 'traj_*.json')):
            r = json.loads(Path(f).read_text(encoding='utf-8'))
            walls = [x['wall_s'] for x in r['rows'][1:]]
            compute.setdefault(r['policy'], []).extend(walls)
    keys = [k for k in ('pure', 'basic', 'ours') if k in rows]
    if not keys:
        return
    panels = [('capture', '捕获率', lambda v: f'{100 * v:.0f}%'), ('return', '平均回报', lambda v: f'{v:.1f}'),
              ('time', '成功回合用时 (s)', lambda v: f'{v:.0f}'), ('dv', '成功回合 Δv (m/s)', lambda v: f'{v:.2f}')]
    if compute:
        panels.append(('compute', '每 2 s 控制周期墙钟 (ms，串行)', lambda v: f'{v:.0f}'))
    fig, ax = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.4))
    for a, (m, title, fmt) in zip(ax, panels):
        vals = [(1000 * float(np.mean(compute[k])) if compute.get(k) else np.nan) if m == 'compute' else rows[k][m] for k in keys]
        bars = a.bar(range(len(keys)), vals, color=[COLORS[k] for k in keys], width=0.62, edgecolor='white', linewidth=2)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                a.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v), textcoords='offset points', xytext=(0, 3 if v >= 0 else -12),
                           ha='center', fontsize=9, color=INK)
        a.axhline(0, color=MUTED, linewidth=1)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.grid(axis='x', visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORS[k], label=LABELS[k]) for k in keys], loc='lower center', ncol=len(keys), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('性能对比（24 个开发开局）', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); save(fig, out, 'fig4_performance'); plt.close(fig)


def fig_mechanism(plt, A, out):
    panels = []
    S = A.get('screen')
    if S and all(load_dir_policy(S, p) for p in ('b_0', 'c_0_0')):
        panels.append('geometry')
    if A.get('window') and Path(A['window']).exists():
        panels.append('window')
    if A.get('audit') and Path(A['audit']).exists():
        panels.append('audit')
    if not panels:
        return
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    ax = np.atleast_1d(ax)
    letters = 'abc'
    for i, (a, p) in enumerate(zip(ax, panels)):
        L = f'({letters[i]})'
        if p == 'geometry':
            speeds = [(0, '0.05'), (5, '0.125'), (10, '0.20')]
            pairs = [(stats(load_dir_policy(S, f'b_{k}').values())['return'], stats(load_dir_policy(S, f'c_{k}_0').values())['return'])
                     for k, _ in speeds if load_dir_policy(S, f'b_{k}') and load_dir_policy(S, f'c_{k}_0')]
            x = np.arange(len(pairs)); wdt = 0.36
            b1 = a.bar(x - wdt / 2, [u for u, _ in pairs], wdt, color=BASE2, label='不感知旋转', edgecolor='white', linewidth=2)
            b2 = a.bar(x + wdt / 2, [g for _, g in pairs], wdt, color=OURS, label='旋转感知锥面几何', edgecolor='white', linewidth=2)
            for bars in (b1, b2):
                for b in bars:
                    v = b.get_height(); a.annotate(f'{v:.0f}', (b.get_x() + b.get_width() / 2, v), textcoords='offset points',
                                                   xytext=(0, 3 if v >= 0 else -12), ha='center', fontsize=8.5)
            a.set_xticks(x, [f'接近速度 {s} m/s' for _, s in speeds[:len(pairs)]], fontsize=8.5)
            a.axhline(0, color=MUTED, linewidth=1); a.set_ylabel('平均回报'); a.legend(fontsize=8.5, loc='lower right')
            a.set_title(f'{L} 旋转几何：同一接近速度', loc='left')
        elif p == 'window':
            w = json.loads(Path(A['window']).read_text(encoding='utf-8'))
            lost = [w['entries_lost']['rigid'], w['entries_lost']['window']]
            bars = a.bar([0, 1], lost, color=[BASE1, OURS], width=0.55, edgecolor='white', linewidth=2)
            for b, v in zip(bars, lost):
                a.annotate(str(v), (b.get_x() + b.get_width() / 2, v), textcoords='offset points', xytext=(0, 3), ha='center')
            a.set_xticks([0, 1], ['时刻承诺', '窗口承诺']); a.set_ylabel('丢失的入口承诺（次）')
            pr = w['paired_return']
            a.set_title(f"{L} 入口承诺（{w['pairs']} 对回合）", loc='left')
            a.text(0.97, 0.55, f"配对回报 {pr['mean']:+.1f}/回合", transform=a.transAxes, ha='right', va='top', color=INK2, fontsize=9.5)
        else:
            s = json.loads(Path(A['audit']).read_text(encoding='utf-8'))
            names = [('entry_label_abs_err_median', '入口结局标签\n|误差| 中位数'), ('stage_pos_err_p90_m', '阶段末位置\n误差 p90 (m)')]
            x = np.arange(len(names)); wdt = 0.36
            for off, kind, col, lab in ((-wdt / 2, 'raw', MUTED, '仅模型预测'), (wdt / 2, 'anchored', OURS, '真实执行校正')):
                vals = [s[kind][k] for k, _ in names]
                bars = a.bar(x + off, vals, wdt, color=col, label=lab, edgecolor='white', linewidth=2)
                for b, v in zip(bars, vals):
                    a.annotate(f'{v:.2g}', (b.get_x() + b.get_width() / 2, v), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8.5)
            a.set_yscale('log'); a.set_xticks(x, [n for _, n in names], fontsize=8.5); a.legend(fontsize=8.5, loc='upper right', bbox_to_anchor=(1.0, 0.9))
            a.set_title(f"{L} 上行反馈精度（{s['openings']} 个开局，{s['anchored']['candidates']} 个候选）", loc='left', fontsize=10)
            a.text(0.97, 0.66, f"入口事件一致率\n{100 * s['raw']['event_agreement']:.0f}% → {100 * s['anchored']['event_agreement']:.0f}%",
                   transform=a.transAxes, ha='right', va='top', color=INK2, fontsize=9)
    fig.tight_layout(); save(fig, out, 'fig5_mechanism'); plt.close(fig)


def plot(assets: Path, out: Path):
    A = json.loads(assets.read_text(encoding='utf-8'))
    plt = style()
    for f in (fig_training, fig_convergence, fig_trajectory, fig_performance, fig_mechanism):
        f(plt, A, out)
        print(f.__name__, 'done', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['traj', 'plot'])
    p.add_argument('--output', required=True); p.add_argument('--policy', action='append', default=[]); p.add_argument('--openings')
    p.add_argument('--window', action='store_true'); p.add_argument('--assets')
    a = p.parse_args()
    if a.mode == 'traj':
        traj(a.policy, [int(x) for x in a.openings.split(',')], Path(a.output), a.window)
    else:
        plot(Path(a.assets), Path(a.output))


if __name__ == '__main__':
    main()
