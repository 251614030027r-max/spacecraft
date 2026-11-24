"""Read-out of the supplementary assets (``docs/final_method/FORMAL.md`` section 10; evaluation only).

    python -B -m experiments.final assets --root ROOT [--root ROOT2 ...] --output evidence/R2_ASSETS

The roots hold the formal results (``C/<cfg>_s<seed>``, ``S/pure``) and the generalization results
(``C/<cfg>_s<seed>_gen_*``); the fine case replays are taken from ``<root>/traj_fine/*.json`` when present.
Missing generalization or formal files block the read-out (``READOUT.json`` lists them, nothing else is written).

Outputs (``GEN_FAILURES.csv`` and ``GEN_REFERENCE.csv`` are the post-hoc items of FORMAL.md 10.1): ``GEN_TABLE.csv`` (outcomes per condition, method and seed), ``GEN_PAIRED.csv`` (ECC vs SAC-MPC per condition:
capture, return, and on both-captured tasks time / delta-v), ``GEN_EPISODES.csv`` (per opening and seed),
``TERMINAL.csv`` (terminal errors of every formal episode), ``TERMINAL_SUMMARY.csv`` (median / 95th percentile on each
method's own captures and on the ECC / SAC-MPC both-captured tasks), ``READOUT.json``, ``traj_fine/``, ``SHA256SUMS``.
"""
from __future__ import annotations

import json
import shutil
import statistics as st
from pathlib import Path

import numpy as np

from experiments.final import REGISTRY, Roots, block, registry, sha256, stage_of
from experiments.formal_r2_readout import load, outcome, paired, write_csv

NOMINAL = 'nominal'
METHODS = ('ECC', 'SAC_MPC')
TERMINAL_ARMS = (('ECC', 'C', 'ECC_s{s}'), ('SAC_MPC', 'C', 'SAC_MPC_s{s}'), ('TD3_MPC', 'C', 'TD3_MPC_s{s}'), ('pure', 'S', 'pure'))
ERRORS = ('position_error_m', 'attitude_error_deg', 'speed_m_s', 'angular_velocity_error_rad_s')


def conditions() -> list[str]:
    return [NOMINAL, *stage_of('assets')['eval_switches']]


def arm(method: str, cond: str) -> str:
    return method if cond == NOMINAL else f'{method}+{cond}'


def gen_tables(rows) -> tuple[list, list, list]:
    table, pairs = [], []
    for c in conditions():
        for m in METHODS:
            if not any(k[0] == arm(m, c) for k in rows):
                continue
            o = outcome(rows, arm(m, c))
            for s, x in sorted(o['per_seed'].items()):
                table.append({'condition': c, 'method': m, 'seed': s, **{k: x[k] for k in ('n', 'capture', 'timeout', 'unsafe', 'resource_exhausted', 'other')},
                              'mean_return': x['mean_return']})
            p = o['pooled']
            table.append({'condition': c, 'method': m, 'seed': 'all', **{k: p[k] for k in ('n', 'capture', 'timeout', 'unsafe', 'resource_exhausted')},
                          'other': p['n'] - p['capture'] - p['timeout'] - p['unsafe'] - p['resource_exhausted'], 'mean_return': p['mean_return'],
                          'capture_rate': p['capture_rate'], 'success_time_median_s': p['success_time_median_s'],
                          'success_dv_median_m_s': p['success_dv_median_m_s']})
        sub = {(k[0].split('+')[0], k[1], k[2]): r for k, r in rows.items() if k[0] in (arm('ECC', c), arm('SAC_MPC', c))}
        for metric in ('capture', 'return', 'time_s', 'delta_v'):
            r = paired(sub, 'ECC', 'SAC_MPC', metric)
            if r is not None:
                pairs.append({'condition': c, 'metric': metric, 'n': r['n'], 'estimate': r['estimate'], 'ci95': json.dumps(r['ci95']),
                              'per_seed': json.dumps(r['per_seed']), 'flips': json.dumps(r.get('flips')), 'median': r.get('median')})
    episodes = []
    for (a, s, o), r in sorted(rows.items()):
        m, _, c = a.partition('+')
        if m in METHODS:
            episodes.append({'condition': c or NOMINAL, 'method': m, 'seed': s, 'opening': o, 'event': r['event'], 'return': r['return'],
                             'time_s': r['time_s'], 'delta_v': r['delta_v'], 'deadline_s': r['deadline_s'], 'spin_deg_s': r['spin_deg_s'],
                             'corrections': r['corrections'], 'long_entries': r['long_entries']})
    return table, pairs, episodes


def terminal_rows(R, ops) -> tuple[list, list]:
    seeds = [str(s) for s in stage_of('formal')['seeds']]
    out, missing = [], []
    for name, kind, pat in TERMINAL_ARMS:
        for s in (seeds if '{s}' in pat else ['fixed']):
            label = pat.format(s=s)
            for o in ops:
                f = R.find(kind, label, f'{label}_{o}.json')
                if not f.is_file():
                    missing.append(f'{kind}/{label}/{label}_{o}.json'); continue
                r = json.loads(f.read_text(encoding='utf-8'))
                t = r.get('terminal')
                if r.get('event') == 'error' or t is None:
                    missing.append(f'{kind}/{label}/{label}_{o}.json (no terminal record)'); continue
                out.append({'method': name, 'seed': s, 'opening': o, 'event': r['event'],
                            'position_error_m': t['position_error_m'], 'attitude_error_deg': float(np.degrees(t['attitude_error_rad'])),
                            'speed_m_s': t['speed_m_s'], 'angular_velocity_error_rad_s': t['angular_velocity_error_rad_s'],
                            'in_cone': t['in_cone'], **{f'min_{k}': v for k, v in (r.get('min_margins') or {}).items()}})
    return out, missing


def _q(x, q):
    return float(np.quantile(np.asarray(x, float), q)) if x else None


def terminal_summary(term) -> list:
    ok = [t for t in term if t['event'] == 'success']
    both = {(t['seed'], t['opening']) for t in ok if t['method'] == 'ECC'} & {(t['seed'], t['opening']) for t in ok if t['method'] == 'SAC_MPC'}
    out = []
    for subset, pick in (('own captures', lambda t: True), ('ECC and SAC-MPC both captured', lambda t: (t['seed'], t['opening']) in both)):
        for name, *_ in TERMINAL_ARMS:
            x = [t for t in ok if t['method'] == name and pick(t)]
            if subset != 'own captures' and name not in METHODS:
                continue
            out.append({'subset': subset, 'method': name, 'n': len(x),
                        **{f'{e}_{lab}': _q([t[e] for t in x], q) for e in ERRORS for lab, q in (('median', 0.5), ('p95', 0.95))},
                        **{f'{e}_max': (max(t[e] for t in x) if x else None) for e in ERRORS}})
    return out


def _reached_entry(raw: dict) -> bool:
    """Did the episode ever enter the entry transfer or the terminal descent (decision records of the audit)?"""
    for d in raw.get('decisions') or []:
        if d.get('true_mode_event_t') is not None or d.get('macro_end') in ('entry', 'descent') \
                or (d.get('coord') or {}).get('kind') == 'launch' or d.get('mode') in ('bridge', 'descent'):
            return True
    return False


def failures(R) -> list:
    """FORMAL.md 10.1: outcomes split by phase per condition, method and seed (from the raw records)."""
    out = []
    for c in conditions():
        for m in METHODS:
            for s in [str(x) for x in stage_of('assets')['seeds']]:
                label = f'{m}_s{s}' + ('' if c == NOMINAL else f'_{c}')
                n = {'capture': 0, 'timeout_no_entry': 0, 'timeout_after_entry': 0, 'unsafe': 0, 'other': 0, 'entries_lost': 0}
                for o in block(stage_of('assets')['eval_openings']):
                    raw = json.loads(R.find('C', label, f'{label}_{o}.json').read_text(encoding='utf-8'))
                    ev = raw['event']
                    key = ('capture' if ev == 'success' else 'unsafe' if ev == 'unsafe' else
                           ('timeout_after_entry' if _reached_entry(raw) else 'timeout_no_entry') if ev == 'timeout' else 'other')
                    n[key] += 1; n['entries_lost'] += int(raw.get('entries_lost', 0) or 0)
                out.append({'condition': c, 'method': m, 'seed': s, **n})
    return out


def time_budget_reference(rows) -> list:
    """FORMAL.md 10.1: nominal captures of the same model and opening that finish no later than the short-deadline
    condition's deadline of that opening (a reference, not a bound)."""
    sw = 'gen_deadline_tight'
    out = []
    for m in METHODS:
        for s in sorted({k[1] for k in rows if k[0] == m}):
            ref = tight = 0
            for (a, ss, o), r in rows.items():
                if a != m or ss != s or (f'{m}+{sw}', s, o) not in rows:
                    continue
                q = rows[(f'{m}+{sw}', s, o)]
                ref += int(r['event'] == 'success' and r['time_s'] <= q['deadline_s'] + 1e-9)
                tight += int(q['event'] == 'success')
            out.append({'method': m, 'seed': s, 'nominal_captures_within_short_deadline': ref, 'short_deadline_captures': tight})
    return out


def write_all(roots, out: Path) -> dict:
    R = Roots(roots)
    rows, missing, ops, _ = load(R.roots, 'assets')
    term, tmiss = terminal_rows(R, block(stage_of('formal')['eval_openings']))
    missing += tmiss
    out.mkdir(parents=True, exist_ok=True)
    fine = sorted({f.name: f for root in R.roots for f in (Path(root) / 'traj_fine').glob('*.json')}.values(), key=lambda f: f.name)
    readout = {'stage': 'assets', 'registry_sha256': sha256(REGISTRY),
               'conditions': {c: (registry()['eval_switches'][c]['display'] if c != NOMINAL else 'formal test distribution')
                              for c in conditions()},
               'missing_or_error': missing, 'fine_replays': [f.name for f in fine]}
    if missing:
        (out / 'READOUT.json').write_text(json.dumps(readout, indent=1, ensure_ascii=False), encoding='utf-8')
        return {'verdict': 'incomplete', 'missing': len(missing), 'first': missing[:10]}
    table, pairs, episodes = gen_tables(rows)
    write_csv(out / 'GEN_TABLE.csv', table); write_csv(out / 'GEN_PAIRED.csv', pairs); write_csv(out / 'GEN_EPISODES.csv', episodes)
    write_csv(out / 'TERMINAL.csv', term)
    fails, refs = failures(R), time_budget_reference(rows)
    write_csv(out / 'GEN_FAILURES.csv', fails); write_csv(out / 'GEN_REFERENCE.csv', refs)
    summ = terminal_summary(term); write_csv(out / 'TERMINAL_SUMMARY.csv', summ)
    (out / 'traj_fine').mkdir(exist_ok=True)
    for f in fine:
        shutil.copy2(f, out / 'traj_fine' / f.name)
    readout.update(gen_paired=pairs, terminal_summary=summ, gen_failures=fails, gen_reference=refs,
                   gen_capture={f"{r['condition']}/{r['method']}": r['capture_rate'] for r in table if r['seed'] == 'all'})
    (out / 'READOUT.json').write_text(json.dumps(readout, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    files = sorted(f for f in out.rglob('*') if f.is_file() and f.name != 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(f'{sha256(f)}  {f.relative_to(out).as_posix()}\n' for f in files), encoding='utf-8')
    return {'verdict': 'complete', 'files': len(files) + 1, 'fine_replays': len(fine),
            'capture': readout['gen_capture']}
