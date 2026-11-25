"""Read-out of the round-3 diagnostic D1 (``docs/final_method/ROUND3.md`` section 1; evaluation only, not for the paper).

    python -B -m experiments.final diag --root ROOT [--root ROOT2 ...] --output evidence/R3_DIAG

Outputs: ``DIAG_TABLE.csv`` (per arm and seed: outcomes, timeouts before / after an entry, entries lost, re-timings,
corrections, direct entries), ``DIAG_PAIRED.csv`` (the registered comparisons: capture with the round-2 bootstrap,
time and delta-v on both-captured tasks), ``READOUT.json``, ``SHA256SUMS``. Missing results block the read-out.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.final import REGISTRY, Roots, block, sha256, stage_of
from experiments.formal_r2_readout import load, paired, write_csv
from experiments.r2_assets_readout import _reached_entry

STAGE = 'diag3'
#: (a, b): a switch against the same models without it, and the method comparisons of the decision
COMPARISONS = [('ECC+r3_launch_first', 'ECC'), ('ECC+r3_no_correction', 'ECC'), ('ECC+r3_launch_lead32', 'ECC'),
               ('ECC', 'ECC_NO_CORE'), ('ECC+r3_launch_first', 'ECC_NO_CORE'), ('ECC+r3_no_correction', 'ECC_NO_CORE'),
               ('ECC', 'SAC_MPC'), ('ECC+r3_launch_first', 'SAC_MPC'), ('ECC_NO_CORE', 'SAC_MPC'),
               *[(f'ECC+r3_hs_{v}', 'ECC+r3_hs') for v in ('slips', 'force07', 'replan05')]]


def _label(arm: str, seed: str) -> str:
    m, _, sw = arm.partition('+')
    return f'{m}_s{seed}' + (f'_{sw}' if sw else '')


def table(R, rows) -> list:
    out = []
    for arm in sorted({k[0] for k in rows}):
        for s in sorted({k[1] for k in rows if k[0] == arm}):
            n = dict(capture=0, timeout_no_entry=0, timeout_after_entry=0, timeout_in_cone=0, unsafe=0, other=0, entries_lost=0,
                     retimings=0, corrections=0, long_entries=0, episodes=0)
            label = _label(arm, s)
            for o in block(stage_of(STAGE)['eval_openings']):
                raw = json.loads(R.find('C', label, f'{label}_{o}.json').read_text(encoding='utf-8'))
                ev = raw['event']
                key = ('capture' if ev == 'success' else 'unsafe' if ev == 'unsafe' else
                       ('timeout_after_entry' if _reached_entry(raw) else 'timeout_no_entry') if ev == 'timeout' else 'other')
                n[key] += 1; n['episodes'] += 1
                # at the deadline already inside the approach cone (terminal record): the descent did not finish in time
                n['timeout_in_cone'] += int(ev == 'timeout' and bool((raw.get('terminal') or {}).get('in_cone')))
                n['entries_lost'] += int(raw.get('entries_lost', 0) or 0)
                n['retimings'] += len(raw.get('window_recoveries') or [])
                n['corrections'] += int(raw.get('corrections', 0)); n['long_entries'] += int(raw.get('long_entries', 0))
            out.append({'arm': arm, 'seed': s, **n})
    return out


def comparisons(rows) -> list:
    out = []
    for a, b in COMPARISONS:
        for metric in ('capture', 'time_s', 'delta_v'):
            r = paired(rows, a, b, metric)
            if r is not None:
                out.append({'a': a, 'b': b, 'metric': metric, 'n': r['n'], 'estimate': r['estimate'], 'ci95': json.dumps(r['ci95']),
                            'per_seed': json.dumps(r['per_seed']), 'flips': json.dumps(r.get('flips')), 'median': r.get('median')})
    return out


def write_all(roots, out: Path) -> dict:
    R = Roots(roots)
    rows, missing, _, _ = load(R.roots, STAGE)
    out.mkdir(parents=True, exist_ok=True)
    readout = {'stage': STAGE, 'registry_sha256': sha256(REGISTRY), 'missing_or_error': missing}
    if missing:
        (out / 'READOUT.json').write_text(json.dumps(readout, indent=1, ensure_ascii=False), encoding='utf-8')
        return {'verdict': 'incomplete', 'missing': len(missing), 'first': missing[:10]}
    tab, comp = table(R, rows), comparisons(rows)
    write_csv(out / 'DIAG_TABLE.csv', tab); write_csv(out / 'DIAG_PAIRED.csv', comp)
    readout.update(table=tab, paired=comp)
    (out / 'READOUT.json').write_text(json.dumps(readout, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    files = sorted(f for f in out.rglob('*') if f.is_file() and f.name != 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(f'{sha256(f)}  {f.relative_to(out).as_posix()}\n' for f in files), encoding='utf-8')
    return {'verdict': 'complete', 'files': len(files) + 1,
            'capture': {f"{t['arm']}/{t['seed']}": t['capture'] for t in tab}}
