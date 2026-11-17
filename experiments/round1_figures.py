"""Result figures of the paper draft from ``evidence/FORMAL/ROUND1_READOUT`` (no simulation).

    python -B -m experiments.round1_figures --readout evidence/FORMAL/ROUND1_READOUT --output DIR

Vector PDFs at IEEE single-column width: fig_res_urgent, fig_res_effects, fig_res_learning, fig_res_economy,
fig_res_reference.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 8,
                     'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
W = 3.5
GROUPS = ['ECC', 'ECC w/o execution feedback', 'ECC w/o screening', 'SAC-MPC']
LABEL = {'ECC': 'ECC', 'ECC w/o execution feedback': 'ECC w/o exec. feedback', 'ECC w/o screening': 'ECC w/o screening',
         'SAC-MPC': 'SAC–MPC'}
COLOR = {'ECC': '#0072B2', 'ECC w/o execution feedback': '#009E73', 'ECC w/o screening': '#CC79A7', 'SAC-MPC': '#D55E00'}
OUT_COLOR = {'capture': '#4C72B0', 'timeout': '#C8C8C8', 'resource_exhausted': '#E69F00', 'unsafe': '#C44E52'}
SEEDS = ('20', '21', '22')


def rows(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02); plt.close(fig)


def fig_urgent(r, out):
    fig, ax = plt.subplots(figsize=(W, 1.9))
    y = np.arange(len(GROUPS))[::-1]
    for yi, g in zip(y, GROUPS):
        p = r['outcomes']['urgent'][g]['per_seed']
        tot = {k: sum(p[s][k] for s in SEEDS) for k in ('capture', 'timeout', 'resource_exhausted', 'unsafe')}
        left = 0
        for k in ('capture', 'timeout', 'resource_exhausted', 'unsafe'):
            v = tot[k] / 288 * 100
            if v > 0:
                ax.barh(yi, v, left=left, color=OUT_COLOR[k], height=0.62, edgecolor='white', linewidth=0.4)
            left += v
        ax.text(tot['capture'] / 288 * 100 - 1.5, yi, f"{tot['capture'] / 288 * 100:.1f}%", va='center', ha='right', color='white', fontsize=7)
        for s, mk in zip(SEEDS, ('o', 's', '^')):
            ax.plot(p[s]['capture'] / 96 * 100, yi + 0.42, mk, ms=2.8, color='k', mfc='none', mew=0.6)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[g] for g in GROUPS]); ax.set_xlim(0, 100)
    ax.set_xlabel('Share of urgent episodes (%)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUT_COLOR[k]) for k in ('capture', 'timeout', 'unsafe')]
    handles += [plt.Line2D([], [], marker=m, ls='', color='k', mfc='none', ms=3) for m in ('o', 's', '^')]
    ax.legend(handles, ['Capture', 'Timeout', 'Unsafe', 'Seed 20', 'Seed 21', 'Seed 22'], ncol=6, loc='lower center',
              bbox_to_anchor=(0.45, 1.0), frameon=False, handlelength=0.9, columnspacing=0.7)
    save(fig, out, 'fig_res_urgent')


def fig_effects(r, out):
    items = [('urgent: ECC vs SAC-MPC', 'ECC vs SAC–MPC'),
             ('urgent: ECC vs ECC w/o screening', 'Screening | with feedback'),
             ('urgent: ECC w/o execution feedback vs SAC-MPC', 'Screening | without feedback'),
             ('urgent: ECC vs ECC w/o execution feedback', 'Feedback | with screening'),
             ('urgent: ECC w/o screening vs SAC-MPC', 'Feedback | without screening'),
             ('urgent: ECC vs ECC+rigid', 'Entry re-timing')]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    for i, (key, lab) in enumerate(items):
        c = r['paired'][key]['capture']; yi = len(items) - 1 - i
        est, (lo, hi) = c['estimate'] * 100, (c['ci95'][0] * 100, c['ci95'][1] * 100)
        ax.plot([lo, hi], [yi, yi], color='#0072B2' if i == 0 else '#555555', lw=1.4)
        ax.plot(est, yi, 'o', color='#0072B2' if i == 0 else '#555555', ms=4)
        for s, mk in zip(SEEDS, ('o', 's', '^')):
            ax.plot(c['per_seed'][s] / 96 * 100, yi + 0.25, mk, ms=2.4, color='k', mfc='none', mew=0.5)
        ax.text(1.01, yi, f"{est:+.1f} [{lo:+.1f}, {hi:+.1f}]", va='center', fontsize=6.5,
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_yticks(range(len(items))); ax.set_yticklabels([lab for _, lab in items][::-1])
    ax.set_xlim(-8, 24); ax.set_xticks([-5, 0, 5, 10, 15, 20])
    ax.set_xlabel('Paired capture-rate difference (percentage points)')
    save(fig, out, 'fig_res_effects')


def fig_learning(readout, out):
    tr = rows(readout / 'TRAINING_CURVES.csv'); va = rows(readout / 'VALIDATION_CURVES.csv')
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.7), gridspec_kw={'width_ratios': [1.6, 1]})
    grid = np.linspace(900, 4700, 20)
    for g in GROUPS:
        curves = []
        for s in SEEDS:
            x = [float(t['decisions_approx']) for t in tr if t['group'] == g and t['seed'] == s]
            y = [float(t['capture_rate']) * 100 for t in tr if t['group'] == g and t['seed'] == s]
            curves.append(np.interp(grid, x, y))
        c = np.array(curves)
        a.plot(grid, c.mean(0), color=COLOR[g], label=LABEL[g])
        a.fill_between(grid, c.min(0), c.max(0), color=COLOR[g], alpha=0.12, lw=0)
    a.set_xlabel('Training decisions'); a.set_ylabel('Training capture, urgent (%)')
    a.set_ylim(40, 90)
    fig.legend(*a.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               fontsize=6, handlelength=1.2, columnspacing=0.8)
    for g in ('ECC', 'SAC-MPC'):
        steps = [1000, 2000, 3000, 4000, 5000]
        c = np.array([[float(next(v['capture'] for v in va if v['group'] == g and v['seed'] == s and int(v['decisions']) == st)) / 24 * 100
                       for st in steps] for s in SEEDS])
        b.plot(steps, c.mean(0), 'o-', ms=2.5, color=COLOR[g], label=LABEL[g])
        b.fill_between(steps, c.min(0), c.max(0), color=COLOR[g], alpha=0.12, lw=0)
    b.set_xlabel('Checkpoint (decisions)'); b.set_ylabel('Validation capture (%)'); b.set_ylim(40, 90)
    b.set_xticks([1000, 3000, 5000]); b.set_xticklabels(['1k', '3k', '5k'])
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.93))
    save(fig, out, 'fig_res_learning')


def fig_economy(r, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.6))
    x = np.arange(len(GROUPS))
    cap = [r['outcomes']['economy'][g]['pooled']['capture'] / 288 * 100 for g in GROUPS]
    rx = [r['outcomes']['economy'][g]['pooled']['resource_exhausted'] / 288 * 100 for g in GROUPS]
    a.bar(x, cap, color=[COLOR[g] for g in GROUPS], width=0.65)
    a.set_ylim(80, 95); a.set_ylabel('Economy capture (%)')
    a.set_xticks(x); a.set_xticklabels(['ECC', 'w/o feedback', 'w/o screening', 'SAC–MPC'], fontsize=6, rotation=30, ha='right')
    b.bar(x, rx, color=[COLOR[g] for g in GROUPS], width=0.65)
    b.set_ylabel('Propellant exhaustion (%)'); b.set_xticks(x); b.set_xticklabels(['ECC', 'w/o feedback', 'w/o screening', 'SAC–MPC'], fontsize=6, rotation=30, ha='right')
    fig.tight_layout(w_pad=0.8)
    save(fig, out, 'fig_res_economy')


def fig_reference(readout, out):
    ref = [d for d in rows(readout / 'DEV_REFERENCES.csv') if d['condition'] == 'urgent']
    order = [('pure', 'MPC only'), ('b_10', 'Closing w/o RWG'), ('c_10_0', 'Fixed intent (1, 0)'),
             ('rally', 'Deadline rule'), ('gstar', 'Direct entry planning')]
    by = {d['policy']: d for d in ref}
    fig, ax = plt.subplots(figsize=(W, 1.6))
    y = np.arange(len(order))[::-1]
    for yi, (p, lab) in zip(y, order):
        d = by[p]; n = int(d['n']); left = 0
        for k in ('capture', 'timeout', 'resource_exhausted', 'unsafe'):
            v = int(d[k]) / n * 100
            if v > 0:
                ax.barh(yi, v, left=left, color=OUT_COLOR[k], height=0.62, edgecolor='white', linewidth=0.4)
            left += v
        ax.text(int(d['capture']) / n * 100 - 1.5, yi, f"{int(d['capture'])}/{n}", va='center', ha='right', color='white', fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in order]); ax.set_xlim(0, 100)
    ax.set_xlabel('Share of urgent episodes (%)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUT_COLOR[k]) for k in ('capture', 'timeout', 'resource_exhausted', 'unsafe')]
    ax.legend(handles, ['Capture', 'Timeout', 'Propellant exhausted', 'Unsafe'], ncol=4, loc='lower center',
              bbox_to_anchor=(0.4, 1.0), frameon=False, handlelength=0.9, columnspacing=0.7)
    save(fig, out, 'fig_res_reference')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--readout', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args()
    readout, out = Path(a.readout), Path(a.output); out.mkdir(parents=True, exist_ok=True)
    r = json.loads((readout / 'READOUT.json').read_text(encoding='utf-8'))
    fig_urgent(r, out); fig_effects(r, out); fig_learning(readout, out); fig_economy(r, out); fig_reference(readout, out)


if __name__ == '__main__':
    main()
