"""U2 stage B readout and pre-registered decision (collab run order ``RUN_U2``; spec ``docs/final_method/U2_PROBE_SPEC.md``).

    python -B -m experiments.u2_readout decide --root U2_DIR --pretest-root PRETEST_DIR --output U2_DECISION.json
    python -B -m experiments.u2_readout reuse --new DIR --old DIR --output REUSE.json   # a reused model, new code = old record

``U2_DIR``: ``B/TRAIN_{PROBE,EACH_PROBE}_s{10,11}`` and ``C/{PROBE,EACH_PROBE,EACH,SAC}_s{10,11}`` (48 openings each,
294148-294195); ``PRETEST_DIR``: ``B/TRAIN_{EACH,SAC}_s{10,11}`` (the reused runs: their training episodes for the
curves). Comparisons are seed-matched (same training-opening stream) and opening-paired: 96 pairs per comparison.

Main criterion, PROBE against SAC (all of):
  G1  improvement: mean paired true-return difference >= 8.3, or captures >= SAC's + 4 (of 96);
  G2  not a few flips: the mean difference without the two largest positive pairs of each seed (4 of 96) > 0;
  G3  both seeds: each seed's mean paired difference > 0;
  G4  no fewer captures than SAC;
  G5  safety: PROBE unsafe = 0.
  -> 'pass' (systematically generated executor feedback has learning value of its own), else 'no_gain'.
Secondary, EACH_PROBE against EACH (no pass / fail of U2):
  'complementary' if the mean paired difference >= 8.3 with G2-G5 analogues; 'kept' if >= -8.3, captures >= EACH's - 4
  and unsafe 0; else 'degraded' (EACH stays the method; U2 stays separate evidence).
Reported, not gated: PROBE against EACH (the gap to U1), curve AUC and late training return, urgent captures, medians,
entries lost, the both-captured time / delta-v.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MARGIN = 8.3
SEEDS = (10, 11)
CAPTURE_MARGIN = 4


def load_eval(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob('*.json')):
        if f.name.startswith('ERROR_'):
            continue
        r = json.loads(f.read_text(encoding='utf-8'))
        if 'opening' in r and 'event' in r:
            out[int(r['opening'])] = r
    return out


def arm_summary(rs: list) -> dict:
    urg = [r for r in rs if r['mission']['urgent']]
    return {'n': len(rs), 'capture': sum(r['event'] == 'success' for r in rs), 'unsafe': sum(r['event'] == 'unsafe' for r in rs),
            'timeout': sum(r['event'] == 'timeout' for r in rs), 'urgent_n': len(urg), 'urgent_capture': sum(r['event'] == 'success' for r in urg),
            'mean_return': float(np.mean([r['return'] for r in rs])) if rs else None,
            'median_return': float(np.median([r['return'] for r in rs])) if rs else None,
            'entries_lost': int(sum(r.get('entries_lost', 0) or 0 for r in rs))}


def paired(evals: dict, a: str, b: str) -> dict:
    """a - b over seed-matched, opening-paired evaluations."""
    diffs, per_seed, cap_a, cap_b, unsafe_a, both = [], {}, 0, 0, 0, []
    for s in SEEDS:
        A, B = evals[f'{a}_s{s}'], evals[f'{b}_s{s}']
        ops = sorted(set(A) & set(B))
        d = np.array([A[o]['return'] - B[o]['return'] for o in ops])
        per_seed[s] = {'n': len(ops), 'mean': float(d.mean()) if len(d) else None, 'median': float(np.median(d)) if len(d) else None,
                       'without_top2': float(np.sort(d)[:-2].mean()) if len(d) > 2 else None}
        diffs += d.tolist()
        cap_a += sum(A[o]['event'] == 'success' for o in ops); cap_b += sum(B[o]['event'] == 'success' for o in ops)
        unsafe_a += sum(A[o]['event'] == 'unsafe' for o in ops)
        both += [(A[o], B[o]) for o in ops if A[o]['event'] == 'success' and B[o]['event'] == 'success']
    d = np.array(diffs)
    trimmed = []
    for s in SEEDS:
        A, B = evals[f'{a}_s{s}'], evals[f'{b}_s{s}']
        ds = np.sort([A[o]['return'] - B[o]['return'] for o in sorted(set(A) & set(B))])
        trimmed += ds[:-2].tolist() if len(ds) > 2 else []
    return {'pairs': int(len(d)), 'mean': float(d.mean()) if len(d) else None, 'median': float(np.median(d)) if len(d) else None,
            'without_top2_per_seed': float(np.mean(trimmed)) if trimmed else None, 'per_seed': per_seed,
            'capture': [int(cap_a), int(cap_b)], 'unsafe_first': int(unsafe_a),
            'flips': {'only_first_captures': int(sum(1 for s in SEEDS for o in set(evals[f'{a}_s{s}']) & set(evals[f'{b}_s{s}'])
                                                     if evals[f'{a}_s{s}'][o]['event'] == 'success' and evals[f'{b}_s{s}'][o]['event'] != 'success')),
                      'only_second_captures': int(sum(1 for s in SEEDS for o in set(evals[f'{a}_s{s}']) & set(evals[f'{b}_s{s}'])
                                                      if evals[f'{a}_s{s}'][o]['event'] != 'success' and evals[f'{b}_s{s}'][o]['event'] == 'success'))},
            'both_captured': {'n': len(both), 'time_s': [float(np.mean([x['time_s'] for x, _ in both])) if both else None,
                                                         float(np.mean([y['time_s'] for _, y in both])) if both else None],
                              'delta_v': [float(np.mean([x['delta_v'] for x, _ in both])) if both else None,
                                          float(np.mean([y['delta_v'] for _, y in both])) if both else None]}}


def gates(p: dict) -> dict:
    g = {'G1': (p['mean'] is not None and p['mean'] >= MARGIN) or (p['capture'][0] >= p['capture'][1] + CAPTURE_MARGIN),
         'G2': p['without_top2_per_seed'] is not None and p['without_top2_per_seed'] > 0,
         'G3': all(v['mean'] is not None and v['mean'] > 0 for v in p['per_seed'].values()),
         'G4': p['capture'][0] >= p['capture'][1], 'G5': p['unsafe_first'] == 0}
    return {k: bool(v) for k, v in g.items()}


def decide(evals: dict) -> dict:
    main = paired(evals, 'PROBE', 'SAC'); gm = gates(main)
    sec = paired(evals, 'EACH_PROBE', 'EACH'); gs = gates(sec)
    if all(gs.values()):
        secondary = 'complementary'
    elif sec['mean'] is not None and sec['mean'] >= -MARGIN and sec['capture'][0] >= sec['capture'][1] - CAPTURE_MARGIN and sec['unsafe_first'] == 0:
        secondary = 'kept'
    else:
        secondary = 'degraded'
    return {'main': {'comparison': 'PROBE - SAC', **main, 'gates': gm, 'verdict': 'pass' if all(gm.values()) else 'no_gain'},
            'secondary': {'comparison': 'EACH_PROBE - EACH', **sec, 'gates': gs, 'verdict': secondary},
            'report_probe_vs_each': paired(evals, 'PROBE', 'EACH')}


def readout(root: Path, pretest_root: Path | None) -> dict:
    evals = {f'{c}_s{s}': load_eval(root / 'C' / f'{c}_s{s}') for c in ('PROBE', 'EACH_PROBE', 'EACH', 'SAC') for s in SEEDS}
    n = {k: len(v) for k, v in evals.items()}
    out = {'margin': MARGIN, 'capture_margin': CAPTURE_MARGIN, 'evaluations': n, 'complete': all(v == 48 for v in n.values()),
           'arms': {k: arm_summary(list(v.values())) for k, v in evals.items()}}
    out.update(decide(evals))
    curves = {}
    try:
        from experiments.t2_curves import run_curve
        for c in ('PROBE', 'EACH_PROBE', 'EACH', 'SAC'):
            base = root if c in ('PROBE', 'EACH_PROBE') else pretest_root
            if base is None:
                continue
            for s in SEEDS:
                f = base / 'B' / f'TRAIN_{c}_s{s}' / 'episodes.json'
                if f.exists():
                    cur = run_curve(json.loads(f.read_text(encoding='utf-8')), 50, 2000, grid_from=250)
                    curves[f'{c}_s{s}'] = {k: cur[k] for k in ('curve_auc', 'late_return', 'late_urgent_capture', 'episodes')}
    except Exception as exc:                       # curves are reported, never gated
        curves['error'] = repr(exc)
    out['curves'] = curves
    probe_cost = {}
    for c in ('PROBE', 'EACH_PROBE'):
        for s in SEEDS:
            f = root / 'B' / f'TRAIN_{c}_s{s}' / 'episodes.json'
            if f.exists():
                st = [e.get('exec_stats') or {} for e in json.loads(f.read_text(encoding='utf-8'))]
                probe_cost[f'{c}_s{s}'] = {'probes': int(sum(x.get('probes', 0) for x in st)), 'probe_wall_s': float(sum(x.get('probe_wall_s', 0.0) for x in st))}
    out['probe_cost'] = probe_cost
    return out


def reuse(new: Path, old: Path) -> dict:
    """A pretest model re-evaluated with the current code against its pretest record on the same openings: the
    evaluation path of the reused configurations is unchanged, so outcome, return, time and delta-v must be equal."""
    A, B = load_eval(new), load_eval(old)
    keys = ('event', 'return', 'time_s', 'delta_v')
    rows = {o: {k: [A[o].get(k), B[o].get(k)] for k in keys} for o in sorted(A) if o in B}
    same = bool(rows) and len(rows) == len(A) and all(v[0] == v[1] for r in rows.values() for v in r.values())
    return {'openings': sorted(rows), 'identical': same, 'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('decide'); d.add_argument('--root', required=True); d.add_argument('--pretest-root'); d.add_argument('--output', required=True)
    r = sub.add_parser('reuse'); r.add_argument('--new', required=True); r.add_argument('--old', required=True); r.add_argument('--output', required=True)
    a = p.parse_args()
    if a.cmd == 'reuse':
        res = reuse(Path(a.new), Path(a.old))
        Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8')
        print(json.dumps({'identical': res['identical'], 'openings': res['openings']}))
        raise SystemExit(0 if res['identical'] else 1)
    res = readout(Path(a.root), Path(a.pretest_root) if a.pretest_root else None)
    Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({'complete': res['complete'], 'main': res['main']['verdict'], 'gates': res['main']['gates'],
                      'secondary': res['secondary']['verdict']}, indent=1))


if __name__ == '__main__':
    main()
