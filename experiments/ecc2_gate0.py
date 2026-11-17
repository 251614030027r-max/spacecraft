"""ECC v2 Gate 0: offline feasibility review of consequence-gated option value learning (``docs/ecc2/GATE0.md``).

Read-only with respect to the method: no training, no change to the environment, the executor or the learner.

    python -B -m experiments.ecc2_gate0 logs   --root FORMAL_ROOT --output A.json
    python -B -m experiments.ecc2_gate0 branch --policy RUN_DIR|FIXED --label L --openings A-B --output DIR
                                               [--workers W] [--success-branch-max OPENING] [--limit N]
    python -B -m experiments.ecc2_gate0 report --output DIR [--result B.json] [--table B_OUTCOMES.csv]

``logs`` (Part A) reads the audited formal evaluations with the certifier on (urgent condition) and reports the failure
anatomy, the consequence coverage of the 3x3 option grid, and the calibration of the executed option's predicted event.

``branch`` (Part B) flies base episodes of a policy with its certifier (deterministic actor, urgent mission, audit on)
and, at every opportunity decision (some grid option predicts a mode event) of every failed base episode and of the
successful base episodes with opening <= --success-branch-max, re-flies the episode from reset nine times with the
grid option b forced at that decision (the certifier still runs, so the prober's solver history is unchanged; only
the flown intent is replaced) and the policy with its certifier afterwards. The first branched decision of every
episode also gets a reference branch that forces the base's executed intent and must reproduce the base outcome.

``report`` computes the Part B metrics (determinism, headroom, consequence sufficiency, consequence-based selection
against the base choice, grid coverage) against the criteria of ``docs/ecc2/GATE0.md``.
"""
import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TAU_TOL_S, DV_TOL = 4.0, 0.05            # "similar consequence": same kind, |dt| <= 4 s, |d dv| <= 0.05 m/s
TAU_BINS, BOOT = (16.0, 32.0), 4000


# ----------------------------------------------------------------------------------------------- Part A (logs)
def _cands(d):
    return (d.get('cert') or {}).get('cands') or []


def _executed(d):
    c = d['cert']
    return c.get('a_exec', c['a']), bool(c.get('intervened'))


def logs(root: Path) -> dict:
    groups = sorted([p for p in (root / 'C').glob('*_urgent') if p.name.startswith(('ECC_CERT_LOOP_', 'SAC_CERT_LOOP_'))]
                    + list((root / 'C').glob('*_urgent_cert')))
    out = {}
    for g in groups:
        eps = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(g.glob('*.json'))]
        eps = [e for e in eps if e.get('event') != 'error']
        anatomy = {'F0_no_opportunity': 0, 'F1_rule_acted': 0, 'F2_own_event': 0, 'F3_unused_binding': 0, 'F3_unused_nonbinding': 0}
        cov_n = cov_hit = 0
        cal = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
        kind_outcome = {}
        for e in eps:
            ds = [d for d in e['decisions'] if 'cert' in d]
            if e['event'] != 'success':
                opp = [d for d in ds if any(c['event'] for c in _cands(d))]
                if not opp:
                    anatomy['F0_no_opportunity'] += 1
                elif any(d['cert'].get('intervened') for d in ds):
                    anatomy['F1_rule_acted'] += 1
                elif any(d['cert'].get('own_event') for d in ds):
                    anatomy['F2_own_event'] += 1
                elif any(c.get('binding') for d in opp for c in _cands(d) if c['event']):
                    anatomy['F3_unused_binding'] += 1
                else:
                    anatomy['F3_unused_nonbinding'] += 1
            for d in ds:
                c = d['cert']
                if not c.get('intervened') and c.get('own_event'):
                    cov_n += 1
                    cov_hit += any(x['event'] and x['kind'] == c.get('own_kind') and abs(x['t'] - c['own_t']) <= TAU_TOL_S
                                   for x in _cands(d))
                # executed option: predicted event within this stage (<= 16 s) vs realized mode switch in the stage
                if c.get('intervened'):
                    pred, kind = c.get('alt_t', 1e9) <= 16.0, c.get('alt_kind')
                else:
                    pred, kind = bool(c.get('own_event')) and c['own_t'] <= 16.0, c.get('own_kind')
                real = d.get('macro_end') in ('entry', 'descent')
                cal['tp' if pred and real else 'fn' if real else 'fp' if pred else 'tn'] += 1
                if real:
                    k = kind_outcome.setdefault(d.get('macro_end'), {'n': 0, 'captured': 0})
                    k['n'] += 1; k['captured'] += e['event'] == 'success'
        tpr = cal['tp'] / max(cal['tp'] + cal['fn'], 1); tnr = cal['tn'] / max(cal['tn'] + cal['fp'], 1)
        out[g.name] = {'episodes': len(eps), 'failures': sum(e['event'] != 'success' for e in eps), 'failure_anatomy': anatomy,
                       'grid_coverage_own_event': {'n': cov_n, 'covered': cov_hit, 'rate': cov_hit / max(cov_n, 1)},
                       'stage_event_calibration': {**cal, 'tpr': tpr, 'tnr': tnr, 'balanced_accuracy': (tpr + tnr) / 2},
                       'capture_after_realized_switch': kind_outcome}
    return out


# ----------------------------------------------------------------------------------------------- Part B (branches)
def _actor(policy: str):
    from experiments.t2 import exec_config_of, load_checkpoint, policy_action
    p = Path(policy)
    if p.is_dir():
        ckpt = p / 'final.zip'
        cfg = exec_config_of(ckpt)
        if not cfg['certify']:
            raise SystemExit(f'{policy}: Gate 0 branches a policy flown with its certifier')
        model = load_checkpoint(ckpt)
        return cfg, (lambda e, o: model.predict(o, deterministic=True)[0])
    return {'window': True, 'geometry': True, 'certify': True}, (lambda e, o: policy_action(policy, e, o))


def _env(opening, cfg):
    from env.t2_stage import T2StageEnv, mission_from_name
    env = T2StageEnv(openings=[opening], shaping=False, window=cfg['window'], geometry=cfg['geometry'],
                     mission=mission_from_name('urgent'), certify=True)
    env.audit_logging = True
    return env


def base_job(job):
    import torch; torch.set_num_threads(1)
    from experiments.t2 import dump, run_episode
    file = Path(job['out']) / 'base' / f"{job['label']}_{job['opening']}.json"
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    cfg, act = _actor(job['policy'])
    env = _env(job['opening'], cfg)
    res = run_episode(env, job['opening'], act)
    res.update(label=job['label'], policy=job['policy'])
    dump(file, res)
    return res


def branch_job(job):
    import torch; torch.set_num_threads(1)
    from experiments.t2 import dump
    tag = 'ref' if job['b'] is None else f"b{job['b']}"
    file = Path(job['out']) / 'branch' / f"{job['label']}_{job['opening']}_k{job['k']}_{tag}.json"
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    cfg, act = _actor(job['policy'])
    env = _env(job['opening'], cfg)
    forced = np.asarray(job['forced'], np.float64)
    original = env._certify

    def certify_then_force(a):
        _, rec = original(a)                  # the certifier runs as in the base (same prober history)
        return forced.copy(), rec

    obs, _ = env.reset(options={'opening': job['opening']}); done, ret, idx, check, macro = False, 0.0, 0, None, None
    while not done:
        a = act(env, obs)
        if idx == job['k']:
            check = {'t_start': float(env.env.time_s), 'mode': env.generator.mode}
            env._certify = certify_then_force
        obs, r, term, trunc, i = env.step(a); done = term or trunc; ret += i['true_reward']
        if idx == job['k']:
            env._certify = original
            macro = (i.get('decision_audit') or {}).get('macro_end')
        idx += 1
    res = {'label': job['label'], 'opening': job['opening'], 'k': job['k'], 'b': job['b'], 'forced': list(forced),
           'event': i['event'], 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
           'check': check, 'macro_end_at_k': macro}
    dump(file, res)
    return res


def _guard(fn):
    def run(job):
        try:
            return fn(job)
        except Exception as error:  # noqa: BLE001 -- recorded and reported, never dropped
            import traceback
            return {'error': repr(error), 'traceback': traceback.format_exc(), 'job': {k: v for k, v in job.items() if k != 'forced'}}
    return run


def guarded_base(job):
    return _guard(base_job)(job)


def guarded_branch(job):
    return _guard(branch_job)(job)


def opportunity_decisions(base):
    return [k for k, d in enumerate(base['decisions']) if 'cert' in d and any(c['event'] for c in _cands(d))]


def branch_jobs(base, policy, out, success_max):
    from env.t2_probe import PROBE_GRID
    if base['event'] == 'success' and base['opening'] > success_max:
        return []
    jobs = []
    for n, k in enumerate(opportunity_decisions(base)):
        common = dict(policy=policy, label=base['label'], opening=base['opening'], out=str(out), k=k)
        if n == 0:
            jobs.append({**common, 'b': None, 'forced': _executed(base['decisions'][k])[0]})
        jobs += [{**common, 'b': j, 'forced': list(b)} for j, b in enumerate(PROBE_GRID)]
    return jobs


def run_pool(fn, jobs, workers):
    errors = []
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(fn, j) for j in jobs]):
            r = f.result()
            if 'error' in r:
                errors.append(r)
            print(json.dumps({k: r.get(k) for k in ('label', 'opening', 'k', 'b', 'event', 'return', 'error')}, default=str), flush=True)
    return errors


def branch(a):
    from experiments.final import block
    out = Path(a.output); (out / 'base').mkdir(parents=True, exist_ok=True); (out / 'branch').mkdir(exist_ok=True)
    ops = block(a.openings)
    errors = run_pool(guarded_base, [dict(policy=a.policy, label=a.label, opening=o, out=str(out)) for o in ops], a.workers)
    jobs = []
    for o in ops:
        f = out / 'base' / f'{a.label}_{o}.json'
        if f.exists():
            jobs += branch_jobs(json.loads(f.read_text(encoding='utf-8')), a.policy, out, a.success_branch_max)
    if a.limit:
        jobs = jobs[:a.limit]
    print(json.dumps({'branch_jobs': len(jobs)}), flush=True)
    errors += run_pool(guarded_branch, jobs, a.workers)
    if errors:
        (out / 'ERRORS.json').write_text(json.dumps(errors, indent=1, default=str), encoding='utf-8')
        raise SystemExit(f'{len(errors)} job(s) failed: see {out / "ERRORS.json"}')


# ----------------------------------------------------------------------------------------------- Part B (report)
def _cap(r):
    return int(r['event'] == 'success')


def _boot_lb(diffs_by_opening, n=BOOT, q=0.025, seed=0):
    keys = list(diffs_by_opening); rnd = random.Random(seed); est = []
    if not keys:
        return None
    for _ in range(n):
        s = [diffs_by_opening[rnd.choice(keys)] for _ in keys]
        flat = [x for v in s for x in v]
        est.append(sum(flat) / max(len(flat), 1))
    est.sort()
    return est[int(q * n)]


def _cell(c):
    tb = 0 if c['t'] <= TAU_BINS[0] else 1 if c['t'] <= TAU_BINS[1] else 2
    return (c['kind'], tb)


def report(out: Path) -> dict:
    bases = {(b['label'], b['opening']): b for b in (json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'base').glob('*.json')))}
    br = {}
    for p in sorted((out / 'branch').glob('*.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        br.setdefault((r['label'], r['opening'], r['k']), {})['ref' if r['b'] is None else r['b']] = r
    # determinism: every reference branch reproduces its base
    det = []
    for (lab, o, k), rows in br.items():
        if 'ref' in rows:
            b, r = bases[(lab, o)], rows['ref']
            det.append(r['event'] == b['event'] and abs(r['return'] - b['return']) <= 1e-6)
    # per decision: candidates (consequences from the base log) and branch outcomes
    decisions = []
    for (lab, o, k), rows in br.items():
        base = bases[(lab, o)]; cands = _cands(base['decisions'][k])
        opts = [{**cands[j], 'j': j, 'cap': _cap(rows[j]), 'ret': rows[j]['return']} for j in range(len(cands)) if j in rows]
        if len(opts) != len(cands):
            continue
        decisions.append({'label': lab, 'opening': o, 'k': k, 'base_cap': _cap(base), 'base_ret': base['return'],
                          'intervened': bool(base['decisions'][k]['cert'].get('intervened')),
                          'executed_on_grid': bool(base['decisions'][k]['cert'].get('intervened')) or
                          any(np.allclose(base['decisions'][k]['cert']['a'], c['b']) for c in cands),
                          'opts': opts})
    failed = {(b['label'], b['opening']) for b in bases.values() if b['event'] != 'success'}
    branched_failed = {(d['label'], d['opening']) for d in decisions if not d['base_cap']}
    rescued = {(d['label'], d['opening']) for d in decisions if not d['base_cap'] and any(x['cap'] for x in d['opts'])}
    # B2: consequence sufficiency among event options of the same decision
    same = diff = same_dis = diff_dis = 0
    for d in decisions:
        ev = [x for x in d['opts'] if x['event']]
        for i in range(len(ev)):
            for j in range(i + 1, len(ev)):
                x, y = ev[i], ev[j]
                similar = x['kind'] == y['kind'] and abs(x['t'] - y['t']) <= TAU_TOL_S and abs(x['dv'] - y['dv']) <= DV_TOL
                dis = x['cap'] != y['cap']
                if similar:
                    same += 1; same_dis += dis
                else:
                    diff += 1; diff_dis += dis
    # B3: consequence-based selection (cell table fitted on the other openings) against the base choice
    openings = sorted({d['opening'] for d in decisions})
    folds = {o: n % 3 for n, o in enumerate(openings)}
    gains = {}
    for f in range(3):
        table = {}
        for d in decisions:
            if folds[d['opening']] == f:
                continue
            for x in d['opts']:
                if x['event']:
                    t = table.setdefault(_cell(x), [0, 0]); t[0] += x['cap']; t[1] += 1
        for d in decisions:
            if folds[d['opening']] != f:
                continue
            ev = [x for x in d['opts'] if x['event']]
            score = lambda x: (table.get(_cell(x), [0, 0])[0] + 0.5) / (table.get(_cell(x), [0, 0])[1] + 1.0)
            pick = max(ev, key=lambda x: (score(x), -x['t'], -x['dv']))
            gains.setdefault(d['opening'], []).append(pick['cap'] - d['base_cap'])
    flat = [g for v in gains.values() for g in v]
    # B4: grid coverage where the base flew an off-grid intent and captured
    off = [d for d in decisions if not d['executed_on_grid'] and d['base_cap']]
    covered = [any(x['cap'] for x in d['opts']) for d in off]
    covered_ret = [max(x['ret'] for x in d['opts']) >= d['base_ret'] - 10.0 for d in off]
    res = {
        'bases': len(bases), 'failed_bases': len(failed), 'branched_decisions': len(decisions),
        'determinism': {'reference_branches': len(det), 'reproduced': sum(det), 'pass': bool(det) and all(det)},
        'B1_headroom': {'failed_with_opportunity': len(branched_failed), 'rescued': len(rescued),
                        'rescue_rate_of_failed': len(rescued) / max(len(failed), 1),
                        'rescue_rate_of_branched_failed': len(rescued) / max(len(branched_failed), 1),
                        'pass': len(rescued) / max(len(failed), 1) >= 0.25},
        'B2_consequence_sufficiency': {'similar_pairs': same, 'similar_discordant': same_dis / max(same, 1),
                                       'other_pairs': diff, 'other_discordant': diff_dis / max(diff, 1),
                                       'pass': same > 0 and same_dis / same <= 0.15 and same_dis / same <= 0.5 * diff_dis / max(diff, 1)},
        'B3_selection': {'decisions': len(flat), 'mean_capture_gain': sum(flat) / max(len(flat), 1),
                         'gain_lower_95': _boot_lb(gains), 'pass': bool(flat) and (_boot_lb(gains) or 0) > 0},
        'B4_grid_coverage': {'off_grid_captured_decisions': len(off), 'capture_covered': sum(covered) / max(len(off), 1),
                             'return_covered': sum(covered_ret) / max(len(off), 1),
                             'pass': bool(off) and sum(covered) / len(off) >= 0.90},
    }
    return res


def table(out: Path, path: Path) -> int:
    """One row per base episode and per branch: outcome and, for a branch, the forced option's predicted consequence."""
    import csv
    bases = {(b['label'], b['opening']): b for b in (json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'base').glob('*.json')))}
    rows = [{'label': b['label'], 'opening': b['opening'], 'k': '', 'option': 'base', 'event': b['event'], 'return': b['return'],
             'pred_event': '', 'kind': '', 'tau_s': '', 'dv': ''} for b in bases.values()]
    for p in sorted((out / 'branch').glob('*.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        c = {} if r['b'] is None else _cands(bases[(r['label'], r['opening'])]['decisions'][r['k']])[r['b']]
        rows.append({'label': r['label'], 'opening': r['opening'], 'k': r['k'], 'option': 'ref' if r['b'] is None else r['b'],
                     'event': r['event'], 'return': r['return'], 'pred_event': c.get('event', ''), 'kind': c.get('kind', ''),
                     'tau_s': c.get('t', ''), 'dv': c.get('dv', '')})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sp = ap.add_subparsers(dest='cmd', required=True)
    lg = sp.add_parser('logs'); lg.add_argument('--root', required=True); lg.add_argument('--output', required=True)
    b = sp.add_parser('branch'); b.add_argument('--policy', required=True); b.add_argument('--label', required=True)
    b.add_argument('--openings', required=True); b.add_argument('--output', required=True)
    b.add_argument('--workers', type=int, default=4); b.add_argument('--success-branch-max', type=int, default=294009)
    b.add_argument('--limit', type=int)
    rp = sp.add_parser('report'); rp.add_argument('--output', required=True); rp.add_argument('--result'); rp.add_argument('--table')
    a = ap.parse_args()
    if a.cmd == 'logs':
        res = logs(Path(a.root)); Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))
    elif a.cmd == 'branch':
        branch(a)
    else:
        res = report(Path(a.output))
        if a.result:
            Path(a.result).write_text(json.dumps(res, indent=1), encoding='utf-8')
        if a.table:
            table(Path(a.output), Path(a.table))
        print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
