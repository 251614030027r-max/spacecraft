"""T2.12 pre-training check of the relative (candidate minus executed) labels against truth branches (sandbox).

    python -B evidence/T2_12_PRECHECK/脚本/rel_probe.py run --calibration CAL.json --openings 2910000-2910039 --workers 4 --output DIR
    python -B evidence/T2_12_PRECHECK/脚本/rel_probe.py summary --input DIR --output J

Urgent missions U(140, 200) s only (economy openings are skipped). Behaviour per opening as experiments.t2_audit
(deadline-aware rule or a constant closing intent). At every stage decision with labels the environment is branched
before the executed stage; every candidate is flown on its own branch for one decision, and when that decision opens
the entry the branch flies on to the terminal event. The executed intent's truth is the real step itself.

For each candidate j and the executed intent f, three predictions are recorded in the label's own units (unshaped
return of the stage, or of the stage plus the entry outcome when the label is an entry):
  raw   calibration only (T2.2 model; the factual part of T2.11's precision weight)
  anch  candidate treatment (M1 + M2: realized attitude, executed residual path, entry outcome offset) -- T2.11's label
  same  for f: the executed intent through the candidates' treatment (the anchor of the T2.12 relative label)
Compared: absolute error of each candidate label, error of the candidate-minus-executed difference, stage-end
position errors (absolute and relative), and the label class (entry / deadline-budget end / stage end) against the truth.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _cls(label) -> str:
    if label['done']:
        return 'entry' if label.get('entry') is not None else 'cutoff'
    return 'stage'


def _truth(snapshot, a) -> dict:
    """One decision with intent ``a`` on its own branch; an opened entry is flown to the terminal event."""
    from experiments.t2_audit import branch
    from env.t1_p_stage import relative_inertial
    b = branch(snapshot)
    _, _, term, trunc, info = b.step(np.asarray(a, dtype=np.float64))
    total, event = info['true_reward'], info['event']
    q, _ = relative_inertial(b.env)
    entered = b._stage_entry is not None
    out = {'q1': q.tolist(), 'stage_true': info['true_reward'], 'stage_s': info['stage_s'], 'held': info['held'], 'entered': entered}
    while not (term or trunc) and b.generator.mode in ('bridge', 'descent'):
        _, _, term, trunc, info = b.step(np.zeros(2)); total += info['true_reward']; event = info['event']
    out.update(total_true=total, event=event if (term or trunc) else None,
               cls='entry' if entered else ('cutoff' if (term or trunc) else 'stage'))
    return out


def job(args):
    opening, cal_path, out = args
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'rel_{opening}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    from experiments.t2_audit import branch, behaviour, load_calibration
    from env.t2_stage import T2Mission, T2StageEnv
    from env.t1_p_stage import relative_inertial
    from train.t2_mve_sac import K_CANDIDATES
    env = T2StageEnv(openings=[opening], candidates=K_CANDIDATES, shaping=True, entry_stages=1, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.diagnostics = True
    env.reset(options={'opening': opening})
    if not env.mission['urgent']:
        res = {'opening': opening, 'skipped': 'economy'}
        file.write_text(json.dumps(res), encoding='utf-8'); return res
    load_calibration(env, cal_path)
    name, act = behaviour(opening)
    rows, done, t0 = [], False, time.time()
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            _, _, term, trunc, info = env.step(np.zeros(2)); done = term or trunc; continue
        snap = branch(env)
        a = act(env)
        t_dec = env.env.time_s
        _, _, term, trunc, info = env.step(a); done = term or trunc
        m = info['model']; d = m['diag']
        if d['reason'] not in ('stage', 'entry'):
            continue
        q1, _ = relative_inertial(env.env)
        # the executed intent's truth: the real step, and when it opened the entry, the rest of the episode
        f_truth = {'q1': q1.tolist(), 'stage_true': info['true_reward'], 'entered': bool(d['entered']), 'stage_s': info['stage_s']}
        f_total = info['true_reward']
        if not done and env.generator.mode in ('bridge', 'descent'):
            sub = branch(env); term2 = trunc2 = False; ev = None
            while not (term2 or trunc2) and sub.generator.mode in ('bridge', 'descent'):
                _, _, term2, trunc2, i2 = sub.step(np.zeros(2)); f_total += i2['true_reward']; ev = i2['event']
            f_truth['event'] = ev
        else:
            f_truth['event'] = info['event']
        f_truth['total_true'] = f_total
        f_truth['cls'] = 'entry' if d['entered'] else ('cutoff' if done and info['event'] in ('timeout', 'resource_exhausted') else 'stage')
        pick = lambda p: {'true': p['true'], 'cls': _cls(p), 'q': (np.asarray(p['q']).tolist() if 'q' in p else None)}
        rec = {'t': t_dec, 'left_s': env.mission['deadline_s'] - t_dec, 'a': np.asarray(a).tolist(), 'reason': d['reason'],
               'fact': {'truth': f_truth, 'raw': pick(d['raw_fact']), 'same': pick(d['same_fact'])}, 'cands': []}
        for c in m['cands']:
            tr = _truth(snap, np.asarray(c['a'], dtype=np.float64))
            rec['cands'].append({'a': np.asarray(c['a']).tolist(), 'truth': tr, 'raw': pick(c['raw']), 'anch': pick(c['anchored'])})
        rows.append(rec)
        print(json.dumps({'opening': opening, 't': round(t_dec, 1), 'n': len(rec['cands']), 'wall': round(time.time() - t0)}), flush=True)
    res = {'opening': opening, 'policy': name, 'event': info['event'], 'deadline_s': env.mission['deadline_s'], 'decisions': rows, 'wall_s': time.time() - t0}
    file.write_text(json.dumps(res, default=float), encoding='utf-8')
    return res


def guarded(args):
    try:
        return job(args)
    except Exception as error:  # noqa: BLE001 -- recorded
        import traceback
        res = {'opening': args[0], 'error': repr(error), 'traceback': traceback.format_exc()}
        (Path(args[2]) / f'ERROR_{args[0]}.json').write_text(json.dumps(res), encoding='utf-8'); return res


def _value(label_true_stage, label_cls, truth):
    """Comparable value of a label and its truth: an entry (or deadline / budget end) label is a full return to the
    terminal event; a stage-end label covers the stage only."""
    return label_true_stage


def summary(inp: Path) -> dict:
    files = sorted(inp.glob('rel_*.json'))
    decs = [d for f in files for d in json.loads(f.read_text(encoding='utf-8')).get('decisions', [])]
    out = {'openings': sum(1 for f in files if 'decisions' in json.loads(f.read_text(encoding='utf-8'))), 'decisions': len(decs),
           'candidates': sum(len(d['cands']) for d in decs), 'errors': len(list(inp.glob('ERROR_*.json')))}
    # 1. class agreement (entry / cutoff / stage) of candidate labels against truth, and of the executed intent's same-treatment label
    def agree(pairs):
        return {'n': len(pairs), 'agree': float(np.mean([a == b for a, b in pairs])) if pairs else None,
                'confusion': {f'{a}->{b}': sum(1 for x, y in pairs if (x, y) == (a, b)) for a in ('stage', 'entry', 'cutoff') for b in ('stage', 'entry', 'cutoff')}}
    out['class_raw'] = agree([(c['raw']['cls'], c['truth']['cls']) for d in decs for c in d['cands']])
    out['class_anch'] = agree([(c['anch']['cls'], c['truth']['cls']) for d in decs for c in d['cands']])
    out['class_fact_same'] = agree([(d['fact']['same']['cls'], d['fact']['truth']['cls']) for d in decs])
    out['class_fact_raw'] = agree([(d['fact']['raw']['cls'], d['fact']['truth']['cls']) for d in decs])
    # 2. value errors where label and truth are of the same class (stage: stage cost; entry / cutoff: return to the end)
    def tv(tr):
        return tr['total_true'] if tr['cls'] != 'stage' else tr['stage_true']
    abs_e, rel_e, abs_raw, rel_raw, pos_abs, pos_rel, pos_abs_raw, pos_rel_raw = [], [], [], [], [], [], [], []
    by = {}
    for d in decs:
        f = d['fact']
        f_ok = f['same']['cls'] == f['truth']['cls']
        ef_same = f['same']['true'] - tv(f['truth']) if f_ok else None
        ef_raw = f['raw']['true'] - tv(f['truth']) if f['raw']['cls'] == f['truth']['cls'] else None
        for c in d['cands']:
            tr = c['truth']
            for kind, lab, ef, A, R in (('anch', c['anch'], ef_same, abs_e, rel_e), ('raw', c['raw'], ef_raw, abs_raw, rel_raw)):
                if lab['cls'] != tr['cls']:
                    continue
                e_abs = lab['true'] - tv(tr)
                A.append(e_abs)
                if ef is not None:
                    e_rel = e_abs - ef          # (y_j^ - y_f^) - (y_j - y_f)
                    R.append(e_rel)
                    key = f"{kind}:{tr['cls']}|{f['truth']['cls']}"
                    by.setdefault(key, {'abs': [], 'rel': []}); by[key]['abs'].append(e_abs); by[key]['rel'].append(e_rel)
            if c['anch']['cls'] == 'stage' and tr['cls'] == 'stage' and f['truth']['cls'] == 'stage':
                qj, qt, qf_t = np.asarray(c['anch']['q']), np.asarray(tr['q1']), np.asarray(f['truth']['q1'])
                pos_abs.append(float(np.linalg.norm(qj - qt)))
                if f['same']['q'] is not None:
                    pos_rel.append(float(np.linalg.norm((qj - np.asarray(f['same']['q'])) - (qt - qf_t))))
                if c['raw']['q'] is not None and f['raw']['q'] is not None and c['raw']['cls'] == 'stage' and f['raw']['cls'] == 'stage':
                    pos_abs_raw.append(float(np.linalg.norm(np.asarray(c['raw']['q']) - qt)))
                    pos_rel_raw.append(float(np.linalg.norm((np.asarray(c['raw']['q']) - np.asarray(f['raw']['q'])) - (qt - qf_t))))
    st = lambda x: {'n': len(x), 'abs_median': float(np.median(np.abs(x))) if x else None, 'abs_p90': float(np.percentile(np.abs(x), 90)) if x else None,
                    'mean': float(np.mean(x)) if x else None, 'rms': float(np.sqrt(np.mean(np.square(x)))) if x else None}
    out['value_err'] = {'anch_abs': st(abs_e), 'anch_rel': st(rel_e), 'raw_abs': st(abs_raw), 'raw_rel': st(rel_raw)}
    out['value_err_by_class'] = {k: {'abs': st(v['abs']), 'rel': st(v['rel'])} for k, v in sorted(by.items())}
    out['stage_pos_err_m'] = {'anch_abs': st(pos_abs), 'anch_rel': st(pos_rel), 'raw_abs': st(pos_abs_raw), 'raw_rel': st(pos_rel_raw)}
    # 2b. the T2.12 rule (learner 4.0): reference = the model's same-treatment prediction of the executed intent when the
    # candidate's label is of the same outcome class, else the real executed value (then the error is the absolute one)
    rule, absl, by_rule = [], [], {}
    for d in decs:
        f = d['fact']
        for c in d['cands']:
            tr = c['truth']
            if c['anch']['cls'] != tr['cls'] or f['same']['cls'] != f['truth']['cls']:
                continue
            e_abs = c['anch']['true'] - tv(tr)
            same = c['anch']['cls'] == f['same']['cls']
            e = e_abs - (f['same']['true'] - tv(f['truth'])) if same else e_abs
            rule.append(e); absl.append(e_abs)
            key = f"{tr['cls']}|{f['truth']['cls']}"
            by_rule.setdefault(key, {'abs': [], 'rule': []}); by_rule[key]['abs'].append(e_abs); by_rule[key]['rule'].append(e)
    out['t212_rule_err'] = {'abs': st(absl), 'rule': st(rule)}
    out['t212_rule_err_by_class'] = {k: {'abs': st(v['abs']), 'rule': st(v['rule'])} for k, v in sorted(by_rule.items())}
    # 3. pairwise order of candidates within a decision on terminal truths (entry / end) differing by >= 10
    hits = {'anch': [0, 0, 0], 'raw': [0, 0, 0]}
    for d in decs:
        cs = [c for c in d['cands'] if c['truth']['cls'] != 'stage']
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                dt = tv(cs[i]['truth']) - tv(cs[j]['truth'])
                if abs(dt) < 10:
                    continue
                for kind in ('anch', 'raw'):
                    a, b = cs[i][kind], cs[j][kind]
                    if a['cls'] == 'stage' or b['cls'] == 'stage':
                        hits[kind][2] += 1; continue
                    dm = a['true'] - b['true']
                    hits[kind][0 if dm * dt > 0 and abs(dm) >= 1 else 1] += 1
    out['terminal_pair_order'] = {k: {'right': v[0], 'wrong_or_tie': v[1], 'label_not_terminal': v[2]} for k, v in hits.items()}
    # 4. the executed intent: the same-treatment consistency error (the T2.12 trust term) and the raw error
    fs = [d['fact'] for d in decs]
    out['fact_same_err'] = st([f['same']['true'] - tv(f['truth']) for f in fs if f['same']['cls'] == f['truth']['cls']])
    out['fact_raw_err'] = st([f['raw']['true'] - tv(f['truth']) for f in fs if f['raw']['cls'] == f['truth']['cls']])
    out['fact_class_mismatch_same'] = sum(1 for f in fs if f['same']['cls'] != f['truth']['cls'])
    return out


def openings(text):
    a, _, b = text.partition('-'); return list(range(int(a), int(b or a) + 1))


def main():
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('run'); r.add_argument('--calibration', required=True); r.add_argument('--openings', required=True)
    r.add_argument('--workers', type=int, default=4); r.add_argument('--output', required=True)
    q = s.add_parser('summary'); q.add_argument('--input', required=True); q.add_argument('--output', required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers) as pool:
            for res in pool.map(guarded, [(o, a.calibration, str(out)) for o in openings(a.openings)]):
                print(json.dumps({'done': res.get('opening'), 'event': res.get('event'), 'skipped': res.get('skipped'), 'error': res.get('error')}), flush=True)
    else:
        res = summary(Path(a.input)); Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
