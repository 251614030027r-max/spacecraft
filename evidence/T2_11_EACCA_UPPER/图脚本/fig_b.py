"""Figure B: why the feedback helps (or not) -- critic ranking vs truth on the shared key states.

    python fig_b.py STATES_DIR KEYSCORE.json OUT.png [EXAMPLE_STEP]

Left: share of decisive pairs (truth returns >= 50 apart) the critic orders correctly, by training step, per arm
(label prefix e_mve / e_sac; guided and unguided runs pooled unless the label carries 'ng' only). Middle: share of
separating decisions where the deterministic actor's nearest grid intent is within 50 of the best truth.
Right: one example decision (pre-declared rule: the separating decision with the most decisive pairs; ties by opening,
time): 3 x 3 truth, executor-model label, and the min-twin-Q of each arm's runs at EXAMPLE_STEP (mean over runs)."""
import json, glob, sys, itertools
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
from matplotlib import font_manager
for f in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Source Han Sans SC'):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f; break
states, scores, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
ex_step = int(sys.argv[4]) if len(sys.argv) > 4 else None
dec = {}
for f in sorted(glob.glob(f'{states}/*.json')):
    r = json.load(open(f))
    for d in r.get('decisions', []):
        if d.get('obs') is not None and d.get('cands'):
            dec[(r['opening'], d['t'])] = d
def decisive(d):
    tv = [c['truth']['true'] for c in d['cands']]
    return [(i, j) for i, j in itertools.combinations(range(len(tv)), 2) if abs(tv[i] - tv[j]) >= 50]
acc, good = defaultdict(list), defaultdict(list)          # (arm, step) -> per-run values
qmap = defaultdict(list)
for label, s in scores['scores'].items():
    if '_ng_' not in label: continue
    arm, step = label.split('_')[1], int(label.split('_')[-1])
    c = w = 0; g = n = 0
    for row in s['rows']:
        d = dec[(row['opening'], row['t'])]; tv = np.array([x['truth']['true'] for x in d['cands']]); q = np.array(row['q'])
        pairs = decisive(d)
        if not pairs: continue
        for i, j in pairs:
            if abs(q[i] - q[j]) < 1.0: continue
            c += (tv[i] - tv[j]) * (q[i] - q[j]) > 0; w += (tv[i] - tv[j]) * (q[i] - q[j]) < 0
        grid = np.array([x['a'] for x in d['cands']]); k = int(np.argmin(np.linalg.norm(grid - np.asarray(row['actor']), axis=1)))
        n += 1; g += tv[k] >= tv.max() - 50
        qmap[(arm, step, row['opening'], row['t'])].append(q)
    acc[(arm, step)].append(c / max(c + w, 1)); good[(arm, step)].append(g / max(n, 1))
COL = {'mve': '#2a78d6', 'sac': '#eb6834'}; NAME = {'mve': 'EACC', 'sac': 'SAC'}
fig = plt.figure(figsize=(13, 4.4)); gs = fig.add_gridspec(2, 6)
for col, (data, ylab) in enumerate(((acc, 'critic 对可分候选对的判对率'), (good, '策略选中真值最优档的比例'))):
    ax = fig.add_subplot(gs[:, col * 2:col * 2 + 2])
    for a in ('mve', 'sac'):
        steps = sorted(s for (x, s) in data if x == a)
        for st in steps:
            ax.scatter([st] * len(data[(a, st)]), data[(a, st)], color=COL[a], s=14, alpha=0.4)
        ax.plot(steps, [np.mean(data[(a, st)]) for st in steps], color=COL[a], lw=2.5, marker='o', label=NAME[a])
    ax.set_xlabel('训练决策步数'); ax.set_ylabel(ylab); ax.set_ylim(0, 1); ax.grid(axis='y', color='#e5e5e5')
    ax.spines[['top', 'right']].set_visible(False)
    if col == 0: ax.legend(frameon=False)
key = max(dec, key=lambda k: (len(decisive(dec[k])), -k[0], -k[1])); d = dec[key]
ex_step = ex_step or max(s for (_, s) in acc)
def grid_of(vals):
    m = np.full((3, 3), np.nan)
    for c, v in zip(d['cands'], vals):
        p, z = c['a']; m[2 - [-1.0, 0.0, 1.0].index(z), [-0.5, 0.25, 1.0].index(p)] = v
    return m
def label_val(c):
    v = None
    for m in c['model']:
        if m is None: break
        v = m.get('value', v)
        if m.get('event'): break
    return v
panels = [('真值回报', [c['truth']['true'] for c in d['cands']]), ('执行器模型标签', [label_val(c) for c in d['cands']])]
for a in ('mve', 'sac'):
    qs = qmap.get((a, ex_step, key[0], key[1]))
    if qs: panels.append((f'{NAME[a]} critic（{ex_step} 步）', list(np.mean(qs, axis=0))))
for i, (t, vals) in enumerate(panels[:4]):
    ax = fig.add_subplot(gs[i // 2, 4 + i % 2]); m = grid_of(vals)
    im = ax.imshow(m, cmap='Blues', aspect='equal'); ax.set_title(t, fontsize=8)
    lo, hi = np.nanmin(m), np.nanmax(m)
    for (yy, xx), v in np.ndenumerate(m):
        dark = hi > lo and (v - lo) / (hi - lo) > 0.6
        ax.text(xx, yy, '' if np.isnan(v) else f'{v:.0f}', ha='center', va='center', fontsize=7, color='white' if dark else '#222')
    ax.set_xticks(range(3), ['慢', '中', '快'], fontsize=7); ax.set_yticks(range(3), ['+', '0', '-'], fontsize=7)
fig.suptitle(f'图 B：关键状态上 critic 的排序与真值（{len(dec)} 个关键决策；示例 {key[0]} t={key[1]:.0f} s，按事先规则选取）', fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=150); print('saved', out, 'example', key)
