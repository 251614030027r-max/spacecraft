"""Learner 5.0 pre-training check (sandbox): executor-evaluable (terminal) counterfactual labels against truth branches.

    python -B evidence/T2_14_PRECHECK/脚本/term_probe.py run --calibration CAL.json --openings 2910000-2910039 --workers 3 --output DIR
    python -B evidence/T2_14_PRECHECK/脚本/term_probe.py summary --input DIR --output J

Urgent audit openings (U(140, 200) s), behaviour as experiments.t2_audit. The environment computes the labels with a
second stage (entry_stages 2) whose intent is the decision's executed intent (a deterministic stand-in for the policy's
mean). Each candidate is flown on its own branch: its intent for one decision and, when that decision neither ends the
episode nor opens the entry, the same stage-2 intent for a second; an opened entry is flown to the terminal event.
Compared, per candidate: the label's terminal claim within one / two stages against the truth (true and false terminals),
and on agreeing terminals the label value (stage-1: with and without the learner 5.0 correction by the executed
intent's outcome; stage-2: r1 + r2, undiscounted like the truth's sum) against the true return of the branch.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))


def _truth2(snapshot, a, a2):
    from experiments.t2_audit import branch
    b = branch(snapshot)
    total, ent, ev = 0.0, None, None
    for k, act in enumerate((a, a2)):
        _, _, term, trunc, info = b.step(np.asarray(act, dtype=np.float64)); total += info['true_reward']
        if b._stage_entry is not None and ent is None:
            ent = k + 1
        while not (term or trunc) and b.generator.mode in ('bridge', 'descent'):
            _, _, term, trunc, info = b.step(np.zeros(2)); total += info['true_reward']
        if term or trunc:
            ev = info['event']; return {'stage': k + 1, 'event': ev, 'entered': ent, 'total': total}
        if k == 0:
            s1_total = total
    return {'stage': None, 'event': None, 'entered': ent, 'total': total}


def job(args):
    opening, cal_path, out = args
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'term_{opening}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    from experiments.t2_audit import branch, behaviour, load_calibration
    from env.t2_stage import T2Mission, T2StageEnv, CANDIDATE_MIN_SEP_M_S
    env = T2StageEnv(openings=[opening], candidates=8, shaping=True, entry_stages=2, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)),
                     cand_min_sep=CANDIDATE_MIN_SEP_M_S)
    env.diagnostics = True
    env.reset(options={'opening': opening})
    if not env.mission['urgent']:
        res = {'opening': opening, 'skipped': 'economy'}; file.write_text(json.dumps(res), encoding='utf-8'); return res
    load_calibration(env, cal_path)
    name, act = behaviour(opening)
    rows, done, t0, info = [], False, time.time(), {}
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            _, _, term, trunc, info = env.step(np.zeros(2)); done = term or trunc; continue
        snap = branch(env)
        a = np.asarray(act(env), dtype=np.float64)
        env.label_policy = (lambda o, _a=a: (np.asarray(_a, np.float32), 0.0))
        t_dec = env.env.time_s
        _, _, term, trunc, info = env.step(a); done = term or trunc
        m = info['model']; d = m['diag']
        if d['reason'] not in ('stage', 'entry'):
            continue
        f_real_terminal = bool(done)
        rec = {'t': t_dec, 'left_s': env.mission['deadline_s'] - t_dec, 'a': a.tolist(),
               'fact': {'real_true': info['true_reward'], 'real_terminal': f_real_terminal, 'real_event': info['event'],
                        'same_done': bool(d['same_fact']['done']), 'same_entry': d['same_fact'].get('entry') is not None, 'same_true': d['same_fact']['true']},
               'cands': []}
        for c in m['cands']:
            an = c['anchored']; s2 = c.get('stage2')
            lab = {'s1_done': bool(an['done']), 's1_entry': an.get('entry') is not None, 's1_true': an['true'],
                   's2_done': bool(s2['done']) if s2 else None, 's2_entry': (s2.get('entry') is not None) if s2 else None,
                   's2_true': (an['true'] + s2['true']) if (s2 and s2['done']) else None}
            tr = _truth2(snap, c['a'], a)
            rec['cands'].append({'a': np.asarray(c['a']).tolist(), 'label': lab, 'truth': tr})
        rows.append(rec)
        print(json.dumps({'opening': opening, 't': round(t_dec, 1), 'n': len(rec['cands']), 'wall': round(time.time() - t0)}), flush=True)
    res = {'opening': opening, 'policy': name, 'event': info.get('event'), 'decisions': rows, 'wall_s': time.time() - t0}
    file.write_text(json.dumps(res, default=float), encoding='utf-8'); return res


def guarded(args):
    try:
        return job(args)
    except Exception as error:  # noqa: BLE001
        import traceback
        res = {'opening': args[0], 'error': repr(error), 'traceback': traceback.format_exc()}
        (Path(args[2]) / f'ERROR_{args[0]}.json').write_text(json.dumps(res), encoding='utf-8'); return res


def summary(inp):
    decs = [d for f in sorted(Path(inp).glob('term_*.json')) for d in json.loads(f.read_text(encoding='utf-8')).get('decisions', [])]
    st = lambda x: {'n': len(x), 'rms': float(np.sqrt(np.mean(np.square(x)))) if x else None, 'mean': float(np.mean(x)) if x else None,
                    'abs_median': float(np.median(np.abs(x))) if x else None, 'abs_p90': float(np.percentile(np.abs(x), 90)) if x else None}
    out = {'decisions': len(decs), 'candidates': sum(len(d['cands']) for d in decs), 'errors': len(list(Path(inp).glob('ERROR_*.json')))}
    c1 = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}; c2 = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    e1, e1c, e2 = [], [], []
    for d in decs:
        f = d['fact']
        for c in d['cands']:
            L, T = c['label'], c['truth']
            p1, r1 = L['s1_done'], T['stage'] == 1
            c1['tp' if p1 and r1 else 'fp' if p1 else 'fn' if r1 else 'tn'] += 1
            if p1 and r1:
                e = L['s1_true'] - T['total']; e1.append(e)
                comp = f['same_done'] and f['real_terminal'] and (L['s1_entry'] == f['same_entry'])
                e1c.append(e + ((f['real_true'] - f['same_true']) if comp else 0.0))
            if not p1 and L['s2_done'] is not None and not r1:
                p2, r2 = bool(L['s2_done']), T['stage'] == 2
                c2['tp' if p2 and r2 else 'fp' if p2 else 'fn' if r2 else 'tn'] += 1
                if p2 and r2:
                    e2.append(L['s2_true'] - T['total'])
    out['stage1_terminal'] = c1; out['stage2_terminal'] = c2
    out['stage1_err'] = st(e1); out['stage1_err_corrected'] = st(e1c); out['stage2_err'] = st(e2)
    out['stage2_fp_truth'] = [(c['label']['s2_true'], c['truth']['total'], c['truth']['entered'], c['truth']['stage'])
                              for d in decs for c in d['cands'] if not c['label']['s1_done'] and c['label']['s2_done'] and c['truth']['stage'] != 2][:20]
    return out


def main():
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('run'); r.add_argument('--calibration', required=True); r.add_argument('--openings', required=True)
    r.add_argument('--workers', type=int, default=3); r.add_argument('--output', required=True)
    q = s.add_parser('summary'); q.add_argument('--input', required=True); q.add_argument('--output', required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        lo, _, hi = a.openings.partition('-'); ops = list(range(int(lo), int(hi or lo) + 1))
        Path(a.output).mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers) as pool:
            for res in pool.map(guarded, [(o, a.calibration, a.output) for o in ops]):
                print(json.dumps({'done': res.get('opening'), 'skipped': res.get('skipped'), 'error': res.get('error')}), flush=True)
    else:
        res = summary(a.input); Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
