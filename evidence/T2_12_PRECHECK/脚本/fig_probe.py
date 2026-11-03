"""Figure of the T2.12 pre-training probe: label error against truth, T2.11 absolute label vs T2.12 rule, per outcome-class
pair (candidate | executed intent). Input: the probe directory (rel_*.json). Output: PNG.

    python -B evidence/T2_12_PRECHECK/脚本/fig_probe.py PROBE_DIR OUT.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams['axes.unicode_minus'] = False
for name in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Source Han Sans SC', 'Microsoft YaHei', 'SimHei'):
    if any(name in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = name; break
rows = []
for f in sorted(Path(sys.argv[1]).glob('rel_*.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    for dec in d.get('decisions', []):
        F = dec['fact']; tv = lambda tr: tr['total_true'] if tr['cls'] != 'stage' else tr['stage_true']
        if F['same']['cls'] != F['truth']['cls']:
            continue
        ef = F['same']['true'] - tv(F['truth'])
        for c in dec['cands']:
            tr = c['truth']
            if c['anch']['cls'] != tr['cls']:
                continue
            ea = c['anch']['true'] - tv(tr)
            same = c['anch']['cls'] == F['same']['cls']
            rows.append((f"{tr['cls']}|{F['truth']['cls']}", ea, ea - ef if same else ea))
NAME = {'entry|entry': '候选入口 | 事实入口', 'entry|stage': '候选入口 | 事实阶段末', 'stage|entry': '候选阶段末 | 事实入口', 'stage|stage': '候选阶段末 | 事实阶段末'}
keys = [k for k in NAME if any(r[0] == k for r in rows)]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1.3, 1]})
x = np.arange(len(keys)); wd = 0.36
for j, (lab, col, idx) in enumerate((('T2.11 绝对标签', '#9aa5b1', 1), ('T2.12 执行锚定相对标签', '#2a78d6', 2))):
    rms = [np.sqrt(np.mean([r[idx] ** 2 for r in rows if r[0] == k])) for k in keys]
    axs[0].bar(x + (j - 0.5) * wd, rms, wd, color=col, label=lab)
    for xi, v in zip(x, rms):
        axs[0].annotate(f'{v:.2f}', (xi + (j - 0.5) * wd, v), ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
axs[0].set_xticks(x); axs[0].set_xticklabels([f"{NAME[k]}\n(n={sum(r[0] == k for r in rows)})" for k in keys], fontsize=8.5)
axs[0].set_ylabel('标签误差均方根（回报单位）'); axs[0].legend(frameon=False, fontsize=9)
ee = [r for r in rows if r[0] == 'entry|entry']
axs[1].scatter([abs(r[1]) for r in ee], [abs(r[2]) for r in ee], s=22, color='#2a78d6', alpha=0.8)
m = max(max(abs(r[1]) for r in ee), max(abs(r[2]) for r in ee)) * 1.05
axs[1].plot([0, m], [0, m], color='#bbb', lw=1, ls='--')
axs[1].set_xlabel('T2.11 绝对标签误差 |·|'); axs[1].set_ylabel('T2.12 相对标签误差 |·|')
axs[1].set_title('同类入口对（每点一个候选）', fontsize=10)
for ax in axs:
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#eee')
fig.suptitle('训练前真值检验：19 个紧急审计开局、143 个决策、1144 个候选（沙箱，探索性；阶段末的价值部分未计入）', fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=160); print('saved', sys.argv[2])
