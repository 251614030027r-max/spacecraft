"""ECC development round: readout and pre-registered decision order (stage ``ecc``; spec ``docs/final_method/ECC_SPEC.md``).

    python -B -m experiments.ecc_readout --root ECC_DIR --output ECC_DECISION.json

``ECC_DIR``: ``B/TRAIN_{SAC_C,EACH_C,ECC}_s{10,11}`` and ``C/{SAC_C,EACH_C,ECC}_s{10,11}`` (as trained: ECC with its
certification, the others without), ``C/ECC_s{10,11}_nocert`` (the same ECC model, certification off) and
``C/EACH_C_s{10,11}_cert`` (the same EACH_C model with deployment certification), 48 openings each (294320-294367).
Comparisons are seed-matched (same training-opening stream) and opening-paired: 96 pairs.

Decision order (registered before the data):
  1. 'ECC': ECC vs SAC_C passes G1-G5 (``experiments.u2_readout.gates``) and ECC vs EACH_C is not degraded
     (mean >= -8.3, captures >= EACH_C's - 4, unsafe 0) -> the final method is ECC.
  2. 'ECC_deploy': otherwise, EACH_C with deployment certification vs SAC_C passes G1-G5 -> the final method is ECC with
     the certification at deployment only (the critic reads the pass-level opportunity); the full ECC is reported.
  3. 'none': otherwise.
Reported, not gated: ECC with vs without certification (same model), EACH_C vs SAC_C, curves, interventions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.u2_readout import MARGIN, SEEDS, arm_summary, gates, load_eval, paired

CAPTURE_TOLERANCE = 4


def readout(root: Path) -> dict:
    dirs = {'SAC_C': 'SAC_C', 'EACH_C': 'EACH_C', 'ECC': 'ECC', 'ECC_NOCERT': 'ECC_{s}_nocert', 'EACH_C_CERT': 'EACH_C_{s}_cert'}
    evals = {}
    for k, d in dirs.items():
        for s in SEEDS:
            name = d.format(s=f's{s}') if '{s}' in d else f'{d}_s{s}'
            evals[f'{k}_s{s}'] = load_eval(root / 'C' / name)
    n = {k: len(v) for k, v in evals.items()}
    main = paired(evals, 'ECC', 'SAC_C'); gm = gates(main)
    keep = paired(evals, 'ECC', 'EACH_C')
    kept = (keep['mean'] is not None and keep['mean'] >= -MARGIN and keep['capture'][0] >= keep['capture'][1] - CAPTURE_TOLERANCE
            and keep['unsafe_first'] == 0)
    fb = paired(evals, 'EACH_C_CERT', 'SAC_C'); gf = gates(fb)
    final = 'ECC' if (all(gm.values()) and kept) else ('ECC_deploy' if all(gf.values()) else 'none')
    out = {'margin': MARGIN, 'evaluations': n, 'complete': all(v == 48 for v in n.values()), 'final_method': final,
           'arms': {k: arm_summary(list(v.values())) for k, v in evals.items()},
           'interventions': {k: int(sum(r.get('cert_interventions', 0) or 0 for r in v.values())) for k, v in evals.items()},
           'ecc_vs_sac': {**main, 'gates': gm, 'pass': all(gm.values())},
           'ecc_vs_each': {**keep, 'not_degraded': bool(kept)},
           'fallback_each_cert_vs_sac': {**fb, 'gates': gf, 'pass': all(gf.values())},
           'report_certification_same_model': paired(evals, 'ECC', 'ECC_NOCERT'),
           'report_each_vs_sac': paired(evals, 'EACH_C', 'SAC_C')}
    curves = {}
    try:
        from experiments.t2_curves import run_curve
        for c in ('SAC_C', 'EACH_C', 'ECC'):
            for s in SEEDS:
                f = root / 'B' / f'TRAIN_{c}_s{s}' / 'episodes.json'
                if f.exists():
                    cur = run_curve(json.loads(f.read_text(encoding='utf-8')), 50, 2000, grid_from=250)
                    curves[f'{c}_s{s}'] = {k: cur[k] for k in ('curve_auc', 'late_return', 'late_urgent_capture', 'episodes')}
    except Exception as exc:                       # reported, never gated
        curves['error'] = repr(exc)
    out['curves'] = curves
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    res = readout(Path(a.root))
    Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({'complete': res['complete'], 'final_method': res['final_method'], 'ecc_vs_sac': res['ecc_vs_sac']['gates'],
                      'ecc_not_degraded_vs_each': res['ecc_vs_each']['not_degraded'],
                      'fallback_pass': res['fallback_each_cert_vs_sac']['pass']}, indent=1))


if __name__ == '__main__':
    main()
