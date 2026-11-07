"""U2 decision-time certification: readout and decision (stage u2c; spec ``docs/final_method/U2_CERT_SPEC.md``).

    python -B -m experiments.cert_readout --root U2C_DIR --output U2C_DECISION.json

``U2C_DIR/C``: ``{EACH,SAC}_s{10,11}`` (certification off) and ``{EACH,SAC}_s{10,11}_cert`` (on), each on the 48 new
openings 294220-294267; the models are the pretest's (same checkpoints with and without certification), so a
difference between a model with and without certification is caused by the certification alone (deterministic
evaluation, no training noise). Comparisons are seed-matched and opening-paired (96 pairs).

Main (the paper's claim, feedback against no feedback): EACH + certification vs SAC, gates G1-G5 of
``experiments.u2_readout`` -> 'pass' / 'no_gain'.
U2 contribution (same model, paired): EACH + certification vs EACH; also SAC + certification vs SAC. Tiers:
  'pass'      G1-G5;
  'improved'  mean > 0, each seed >= 0, urgent captures >= +1, total captures not fewer, unsafe 0, economy mean >= -2;
  'no_effect' otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.u2_readout import SEEDS, arm_summary, gates, load_eval, paired

ECONOMY_TOLERANCE = 2.0


def _class_diff(evals: dict, a: str, b: str, urgent: bool) -> dict:
    d, cap_a, cap_b = [], 0, 0
    for s in SEEDS:
        A, B = evals[f'{a}_s{s}'], evals[f'{b}_s{s}']
        for o in sorted(set(A) & set(B)):
            if bool(A[o]['mission']['urgent']) != urgent:
                continue
            d.append(A[o]['return'] - B[o]['return'])
            cap_a += A[o]['event'] == 'success'; cap_b += B[o]['event'] == 'success'
    return {'n': len(d), 'mean': float(np.mean(d)) if d else None, 'capture': [int(cap_a), int(cap_b)]}


def tier(evals: dict, a: str, b: str) -> dict:
    p = paired(evals, a, b); g = gates(p)
    urg, eco = _class_diff(evals, a, b, True), _class_diff(evals, a, b, False)
    improved = (p['mean'] is not None and p['mean'] > 0 and all(v['mean'] is not None and v['mean'] >= 0 for v in p['per_seed'].values())
                and urg['capture'][0] >= urg['capture'][1] + 1 and p['capture'][0] >= p['capture'][1] and p['unsafe_first'] == 0
                and (eco['mean'] is None or eco['mean'] >= -ECONOMY_TOLERANCE))
    verdict = 'pass' if all(g.values()) else ('improved' if improved else 'no_effect')
    return {'comparison': f'{a} - {b}', **p, 'gates': g, 'urgent': urg, 'economy': eco, 'verdict': verdict}


def readout(root: Path) -> dict:
    evals = {}
    for c in ('EACH', 'SAC'):
        for s in SEEDS:
            evals[f'{c}_s{s}'] = load_eval(root / 'C' / f'{c}_s{s}')
            evals[f'{c}_CERT_s{s}'] = load_eval(root / 'C' / f'{c}_s{s}_cert')
    n = {k: len(v) for k, v in evals.items()}
    main = paired(evals, 'EACH_CERT', 'SAC'); gm = gates(main)
    out = {'evaluations': n, 'complete': all(v == 48 for v in n.values()),
           'arms': {k: arm_summary(list(v.values())) for k, v in evals.items()},
           'interventions': {k: int(sum(r.get('cert_interventions', 0) or 0 for r in v.values())) for k, v in evals.items() if 'CERT' in k},
           'main': {'comparison': 'EACH_CERT - SAC', **main, 'gates': gm, 'verdict': 'pass' if all(gm.values()) else 'no_gain'},
           'u2_each': tier(evals, 'EACH_CERT', 'EACH'), 'u2_sac': tier(evals, 'SAC_CERT', 'SAC'),
           'report_each_vs_sac': paired(evals, 'EACH', 'SAC')}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    res = readout(Path(a.root))
    Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({'complete': res['complete'], 'main': res['main']['verdict'], 'main_gates': res['main']['gates'],
                      'u2_each': res['u2_each']['verdict'], 'u2_sac': res['u2_sac']['verdict'], 'interventions': res['interventions']}, indent=1))


if __name__ == '__main__':
    main()
