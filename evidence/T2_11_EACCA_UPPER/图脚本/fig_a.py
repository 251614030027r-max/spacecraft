"""Figure A: learning curves by arm from a t2_readout curve READOUT.json (deployment, 48 dev openings)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
from matplotlib import font_manager
for f in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Source Han Sans SC'):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f; break
r = json.load(open(sys.argv[1])); out = sys.argv[2]; title = sys.argv[3] if len(sys.argv) > 3 else ''
COL = {'mve': '#2a78d6', 'sac': '#eb6834'}; NAME = {'mve': 'EACC 反事实标签', 'sac': 'SAC（无标签）'}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, lab in ((axs[0], 'ret', '部署平均回报（48 个开发开局）'), (axs[1], 'urg', '紧急任务捕获数')):
    for a in ('mve', 'sac'):
        runs = sorted(k for k in r['arms'] if k.startswith(f'e_{a}_s'))
        cks = sorted({int(c) for k in runs for c in r['arms'][k]})
        val = lambda k, c: (r['arms'][k].get(str(c)) or {})
        ys = []
        for k in runs:
            y = [val(k, c).get('mean_return') if key == 'ret' else (val(k, c).get('urgent') or {}).get('capture') for c in cks]
            ax.plot(cks, y, color=COL[a], lw=1, alpha=0.45); ys.append(y)
        m = np.nanmean(np.array(ys, dtype=float), axis=0)
        ax.plot(cks, m, color=COL[a], lw=2.5, marker='o', ms=7, label=f'{NAME[a]}（{len(runs)} 条）')
        ax.annotate(f'{m[-1]:.1f}', (cks[-1], m[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color='#333')
    ax.set_xlabel('训练决策步数'); ax.set_ylabel(lab); ax.grid(axis='y', color='#e5e5e5'); ax.spines[['top', 'right']].set_visible(False)
axs[0].legend(frameon=False, loc='lower right')
fig.suptitle(title + '\n细线为单条训练，粗线为该臂均值', fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=150)
print('saved', out)
