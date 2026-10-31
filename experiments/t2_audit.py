"""T2.3 mechanism audits without training (env/t2_stage.py).

    python -B -m experiments.t2_audit warmup --output CAL.json [--episodes 40] [--workers W]
    python -B -m experiments.t2_audit labels --calibration CAL.json --openings A-B --workers W --output DIR [--policy-checkpoint ZIP]
    python -B -m experiments.t2_audit summary --input DIR --output LABEL_AUDIT.json

``labels`` checks the EACC model labels against the truth. At every stage decision of an episode the
environment is branched *before* the executed stage (fresh environment, copied state: bitwise the same
continuation); after the executed stage each candidate intent the learner would receive is flown on its
own branch (one stage, and a second one holding the intent when its label spans two), and the truth is
compared with the candidate's label in both forms: raw (calibration only, T2.2's model) and anchored
(realized target attitude and executed residual, T2.3). Behaviour policy: per opening either the
deadline-aware rule or a constant intent drawn from the opening (both inside the action space).

``warmup`` runs the behaviour policy on its own openings (2,900,000+, in parallel, merged in opening order)
until the stage calibration and the entry calibration are ready, and stores both, so every audit worker
starts from the same model.
"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np

GEN_FIELDS = ('mode', 'plan', 'plan_start_s', 'hold_position', 'request', 'last_status', 'last_outcome', 'event', 'bridge_tau_s')
T2_FIELDS = ('calibration', 'entry_calibration', 'last', 'stats', '_bridge', '_entry_open', '_trace', '_model_t0', 'return_true',
             'far_field_controls', '_stage_entry', 'episode', 'rng', 'mission')


def branch(r, candidates: int = 0):
    """A fresh T2StageEnv in the state of ``r`` (fresh MPC solver cache, copied state)."""
    from env.t1_counterfactual import _ENV_FIELDS, copy_controller_state, copy_generator_state
    from env.t2_stage import T2StageEnv
    s = T2StageEnv(openings=[r.current_opening], seed_base=r.seed_base, candidates=candidates, mission=r.mission_distribution,
                   shaping=r.shaping, geometry=r.geometry, entry_stages=r.entry_stages, window=r.window)
    s.reset(options={'opening': r.current_opening})
    s.env.config = r.env.config
    s.env.target_state = r.env.target_state.copy(); s.env.chaser_state = r.env.chaser_state.copy()
    for name in _ENV_FIELDS:
        setattr(s.env, name, deepcopy(getattr(r.env, name)))
    copy_controller_state(r.env.controller, s.env.controller)
    cache = getattr(s.env.controller._problem, '_solver_cache', None)
    if cache is not None:
        cache.clear()
    for name in GEN_FIELDS:
        setattr(s.generator, name, deepcopy(getattr(r.generator, name)))
    s.generator._descent = deepcopy(r.generator._descent)
    copy_generator_state(r.generator.bridge, s.generator.bridge)
    s.generator.bridge.recoveries = r.generator.bridge.recoveries
    for name in T2_FIELDS:
        setattr(s, name, deepcopy(getattr(r, name)))
    s.current_opening = r.current_opening
    return s


def behaviour(opening: int):
    """Rule or constant intent, fixed per opening."""
    from experiments.t2 import rally_action
    rng = np.random.default_rng([opening, 2303])
    if rng.uniform() < 0.5:
        return 'rally', lambda env: rally_action(env)
    a = np.round(rng.uniform(-1.0, 1.0, size=2), 2)
    a[0] = abs(a[0])                      # closing (a receding constant intent rarely meets a pass before the deadline)
    return f'const_{a[0]:+.2f}_{a[1]:+.2f}', lambda env: a


def load_calibration(env, path):
    from env.t2_stage import Calibration, EntryCalibration
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    env.calibration = Calibration(rows=[(r[0], r[1], np.array(r[2]), np.array(r[3]), r[4]) for r in c['stage']])
    env.entry_calibration = EntryCalibration(rows=[tuple(r) for r in c['entry']])


def _warmup_episode(opening: int) -> dict:
    """One behaviour episode from empty calibrations; its calibration rows (the rows do not depend on earlier
    rows: the behaviour policies and the executor never read the calibrations, so episodes are independent)."""
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv
    _, act = behaviour(opening)
    env = T2StageEnv(seed_base=2_900_000, candidates=0, shaping=True)
    env.reset(options={'opening': opening}); done = False
    while not done:
        _, _, term, trunc, info = env.step(act(env)); done = term or trunc
    return {'opening': opening, 'event': info['event'],
            'stage': [[r[0], r[1], np.asarray(r[2]).tolist(), np.asarray(r[3]).tolist(), r[4]] for r in env.calibration.rows],
            'entry': [list(r) for r in env.entry_calibration.rows]}


def warmup(out: Path, episodes: int, workers: int = 4):
    """Behaviour episodes on openings 2,900,000 + k, merged in opening order (the same rows a serial run would
    collect), until both calibrations are ready."""
    from env.t2_stage import CALIBRATION_WINDOW, Calibration, EntryCalibration
    with ProcessPoolExecutor(workers) as pool:
        eps = list(pool.map(_warmup_episode, [2_900_000 + k for k in range(episodes)]))
    stage, entry, used = [], [], 0
    for e in eps:
        stage = (stage + e['stage'])[-CALIBRATION_WINDOW:]; entry = (entry + e['entry'])[-CALIBRATION_WINDOW:]; used += 1
        print(json.dumps({'opening': e['opening'], 'event': e['event'], 'stage_rows': len(stage), 'entry_rows': len(entry)}), flush=True)
        if Calibration(rows=stage).ready and EntryCalibration(rows=entry).ready:
            break
    assert Calibration(rows=stage).ready and EntryCalibration(rows=entry).ready, 'calibration not ready: run more warm-up episodes'
    out.write_text(json.dumps({'stage': stage, 'entry': entry, 'episodes': used}, indent=1), encoding='utf-8')


def _truth(snapshot, a, a2=None) -> dict:
    """Fly intent ``a`` from the decision state on its own branch for one decision and, when it does not end the
    episode or open the entry, the stage-2 action ``a2`` (the label's own policy sample) for a second decision."""
    from env.t1_p_stage import relative_inertial
    b = branch(snapshot)
    out = {'true': 0.0, 'entered_stage': None, 'event': None}
    for k, act in enumerate([a] if a2 is None else [a, a2]):
        _, _, term, trunc, info = b.step(np.asarray(act, dtype=np.float64))
        out['true'] += info['true_reward']
        q, v = relative_inertial(b.env)
        if k == 0:
            out.update(q1=q.tolist(), reward1=info['true_reward'], obs1=b.observation().tolist(), stage_s=info['stage_s'], held=info['held'])
        else:
            out.update(q2=q.tolist(), reward2=info['true_reward'], stage2_s=info['stage_s'])
        if b._stage_entry is not None and out['entered_stage'] is None:
            out['entered_stage'] = k + 1
        if term or trunc:
            out['event'] = info['event']; break
        if b.generator.mode in ('bridge', 'descent'):
            break
    return out


def _compare(label: dict, truth: dict) -> dict:
    """Stage-1 label against the truth of the candidate's first decision."""
    entry = bool(label['done']) and label.get('entry') is not None     # T2.6: a deadline / budget end is not an entry
    row = {'pred_entry_stage': 1 if entry else 0, 'true_entry_stage': truth['entered_stage'] if truth['entered_stage'] == 1 else None,
           'pred_true': label['true']}
    if entry:
        row['label_err'] = label['true'] - truth['true'] if truth['entered_stage'] == 1 and truth['event'] else None
    elif label['done']:                 # T2.6: the label ends at the deadline / budget; the truth should end there too
        row['pred_cutoff'] = label.get('event'); row['true_event'] = truth['event']
        row['cutoff_err'] = label['true'] - truth['true'] if truth['event'] else None
    else:
        row['reward_err'] = label['true'] - truth['reward1']
        row['pos_err_m'] = float(np.linalg.norm(np.asarray(label['q']) - np.asarray(truth['q1'])))
        row['obs_err'] = float(np.max(np.abs(np.asarray(label['obs']) - np.asarray(truth['obs1']))))
    return row


def _compare2(s1: dict, s2: dict, truth: dict) -> dict:
    """Stage 2 (T2.5): the predicted entry event of the second decision and the two-decision true return."""
    row = {'pred_entry2': bool(s2['done']) and s2.get('entry') is not None, 'true_entry2': truth['entered_stage'] == 2, 'true_entry1': truth['entered_stage'] == 1}
    if truth['entered_stage'] == 1:
        return row                                       # the truth entered already in stage 1: stage 2 not comparable
    if s2['done']:
        row['label2_err'] = (s1['true'] + s2['true']) - truth['true'] if truth['entered_stage'] == 2 and truth['event'] else None
    elif 'reward2' in truth and truth['entered_stage'] is None:
        row['reward2_err'] = s2['true'] - truth['reward2']
        row['pos2_err_m'] = float(np.linalg.norm(np.asarray(s2['q']) - np.asarray(truth['q2'])))
    return row


def label_policy_from(checkpoint, seed: int):
    """mu for the audit: a trained SAC actor (sampling, seeded per opening), as in training."""
    import torch
    from train.t2_mve_sac import T2SAC
    actor = T2SAC.load(checkpoint, device='cpu').actor
    actor.set_training_mode(False)
    torch.manual_seed(seed)
    def mu(o):
        with torch.no_grad():
            a, logp = actor.action_log_prob(torch.as_tensor(np.asarray(o, np.float32)[None]))
        return a[0].numpy(), float(logp.reshape(-1)[0])
    return mu


def audit_job(job):
    opening, cal_path, out, policy_ckpt = job
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'audit_{opening}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv
    from train.t2_mve_sac import K_CANDIDATES
    name, act = behaviour(opening)
    env = T2StageEnv(openings=[opening], candidates=K_CANDIDATES, shaping=True)
    env.diagnostics = True
    env.reset(options={'opening': opening})
    load_calibration(env, cal_path)
    if policy_ckpt:
        env.label_policy = label_policy_from(policy_ckpt, opening)
    rows, done, t0 = [], False, time.time()
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            _, _, term, trunc, info = env.step(np.zeros(2)); done = term or trunc; continue
        snap = branch(env)
        a = act(env)
        _, _, term, trunc, info = env.step(a); done = term or trunc
        m = info['model']
        d = m['diag']
        rec = {'t': info['time_s'], 'a': np.asarray(a).tolist(), 'reason': d['reason'], 'event': info['event'], 'candidates': []}
        if d['reason'] in ('stage', 'entry'):
            rec['fact_raw_done'] = d['raw_fact']['done']; rec['entered'] = d['entered']
            for c in m['cands']:
                a2 = c['s2']['a'] if (not c['done'] and c.get('s2') is not None) else None
                truth = _truth(snap, np.asarray(c['a'], dtype=np.float64), a2)
                row = {'a': np.asarray(c['a']).tolist(), 'truth': {k: truth.get(k) for k in ('true', 'entered_stage', 'event', 'stage_s', 'held')},
                       'raw': _compare(c['raw'], truth), 'anchored': _compare(c['anchored'], truth)}
                if a2 is not None:
                    row['stage2'] = _compare2(c['anchored'], c['stage2'], truth); row['a2'] = np.asarray(a2).tolist()
                rec['candidates'].append(row)
        rows.append(rec)
        print(json.dumps({'opening': opening, 't': rec['t'], 'reason': rec['reason'], 'n': len(rec['candidates'])}), flush=True)
    res = {'opening': opening, 'policy': name, 'label_policy': policy_ckpt, 'event': info['event'], 'decisions': rows, 'wall_s': time.time() - t0}
    file.write_text(json.dumps(res, default=float), encoding='utf-8')
    return res


def guarded_audit(job):
    try:
        return audit_job(job)
    except Exception as error:  # noqa: BLE001 -- recorded
        import traceback
        res = {'opening': job[0], 'event': 'error', 'error': repr(error), 'traceback': traceback.format_exc()}
        (Path(job[2]) / f'ERROR_{job[0]}.json').write_text(json.dumps(res), encoding='utf-8'); return res


def summary(inp: Path) -> dict:
    """Entry-event agreement and label errors, raw against anchored, over every audited candidate."""
    out = {}
    files = sorted(inp.glob('audit_*.json'))
    cands = [c for f in files for d in json.loads(f.read_text(encoding='utf-8'))['decisions'] for c in d['candidates']]
    for kind in ('raw', 'anchored'):
        rows = [c[kind] for c in cands]
        agree = [r['pred_entry_stage'] == (r['true_entry_stage'] or 0) for r in rows]
        tp = sum(1 for r in rows if r['pred_entry_stage'] and r['true_entry_stage'] == r['pred_entry_stage'])
        fp = sum(1 for r in rows if r['pred_entry_stage'] and r['true_entry_stage'] != r['pred_entry_stage'])
        fn = sum(1 for r in rows if not r['pred_entry_stage'] and r['true_entry_stage'] == 1)
        lab = [r['label_err'] for r in rows if r.get('label_err') is not None]
        rew = [r['reward_err'] for r in rows if 'reward_err' in r]
        pos = [r['pos_err_m'] for r in rows if 'pos_err_m' in r]
        out[kind] = {'candidates': len(rows), 'event_agreement': float(np.mean(agree)) if agree else None, 'entry_tp': tp, 'entry_fp': fp, 'entry_fn': fn,
                     'entry_label_err_mean': float(np.mean(lab)) if lab else None, 'entry_label_abs_err_median': float(np.median(np.abs(lab))) if lab else None,
                     'stage_reward_abs_err_median': float(np.median(np.abs(rew))) if rew else None,
                     'stage_pos_err_median_m': float(np.median(pos)) if pos else None, 'stage_pos_err_p90_m': float(np.percentile(pos, 90)) if pos else None}
    # T2.5 cross-stage information: candidates whose second decision (policy action) opens the entry, in decisions
    # where the executed intent did not enter in its own stage ("decisive"), against the truth branch
    decs = [d for f in files for d in json.loads(f.read_text(encoding='utf-8'))['decisions'] if d['candidates']]
    s2 = [(d, c) for d in decs for c in d['candidates'] if 'stage2' in c]
    st2 = [c['stage2'] for _, c in s2 if not c['stage2']['true_entry1']]
    dec = [x for x in st2]
    pred = [x for x in dec if x['pred_entry2']]; true_ = [x for x in dec if x['true_entry2']]
    not_entered = [(d, c) for d, c in s2 if not d.get('entered')]
    l2 = [x['label2_err'] for x in st2 if x.get('label2_err') is not None]
    r2 = [x['reward2_err'] for x in st2 if 'reward2_err' in x]
    p2 = [x['pos2_err_m'] for x in st2 if 'pos2_err_m' in x]
    out['stage2'] = {'candidates_with_stage2': len(s2), 'non_terminal_candidates': sum(1 for d in decs for c in d['candidates'] if not c['anchored']['pred_entry_stage']),
                     'pred_entry2': len(pred), 'true_entry2': len(true_), 'entry2_tp': sum(1 for x in dec if x['pred_entry2'] and x['true_entry2']),
                     'entry2_fp': sum(1 for x in dec if x['pred_entry2'] and not x['true_entry2']), 'entry2_fn': sum(1 for x in dec if x['true_entry2'] and not x['pred_entry2']),
                     'decisive_pred_where_executed_not_entered': sum(1 for d, c in not_entered if c['stage2']['pred_entry2'] and not c['stage2']['true_entry1']),
                     'decisive_true_where_executed_not_entered': sum(1 for d, c in not_entered if c['stage2']['true_entry2']),
                     'decisions_executed_not_entered': len({id(d) for d, _ in not_entered}),
                     'entry2_label_err_mean': float(np.mean(l2)) if l2 else None, 'entry2_label_abs_err_median': float(np.median(np.abs(l2))) if l2 else None,
                     'stage2_reward_abs_err_median': float(np.median(np.abs(r2))) if r2 else None,
                     'stage2_pos_err_median_m': float(np.median(p2)) if p2 else None, 'stage2_pos_err_p90_m': float(np.percentile(p2, 90)) if p2 else None}
    out['openings'] = len(files); out['errors'] = len(list(inp.glob('ERROR_*.json')))
    return out


def openings(text):
    a, _, b = text.partition('-'); return list(range(int(a), int(b or a) + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['warmup', 'labels', 'summary'])
    p.add_argument('--output', required=True); p.add_argument('--calibration'); p.add_argument('--openings'); p.add_argument('--input')
    p.add_argument('--workers', type=int, default=4); p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--policy-checkpoint', default=None, help='mu of the stage-2 labels: a trained SAC checkpoint (T2.5)')
    a = p.parse_args()
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(k, '1')
    if a.mode == 'warmup':
        warmup(Path(a.output), a.episodes, a.workers)
    elif a.mode == 'labels':
        out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.workers) as pool:
            for f in as_completed([pool.submit(guarded_audit, (o, a.calibration, str(out), a.policy_checkpoint)) for o in openings(a.openings)]):
                r = f.result(); print(json.dumps({k: r.get(k) for k in ('opening', 'policy', 'event', 'wall_s', 'error')}), flush=True)
    else:
        res = summary(Path(a.input)); Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
