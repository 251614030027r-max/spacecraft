"""T1-B V8 deployment protocol and offline readouts (ECPG; train with experiments.t1b_v2 --interface v8).

    python -B -m experiments.t1b_v8 deploy  --checkpoint ZIP --interface {v7,v8} --verify {off,deviation,risk,random,all}
                                            [--q-theta 0] [--rho R] --label L --openings A-B --workers W --output OUT
    python -B -m experiments.t1b_v8 offline --checkpoint ZIP --interface {v7,v8} --episodes DIR[,DIR] --truth GLOB[,GLOB]
                                            --model DIR[,DIR] [--q-theta 0] --output OUT.json

``verify_at_deploy`` (one switch, the same protocol for every arm): at a learning decision the
policy proposes a candidate; when the switch triggers, every verified candidate is rolled out from a
snapshot of the real state with the real-controller full-horizon model (V7 branch, QP every
control, earliest at model re-decisions) and the candidate with the best predicted return among
the predicted successes is executed (none predicted to succeed: the proposal). Triggers:

* ``off``       never (the learned policy alone);
* ``deviation`` the proposal is not the G* earliest candidate (verify deviations only);
* ``risk``      deviation, or the real-TD critic's min(Q1, Q2) of the proposal is below ``q_theta``
                (default 0: Q ~ p*55 + (1-p)*(-130) < 0 <=> predicted capture probability < ~0.7);
* ``random``    with probability ``rho`` (hash of opening and decision index; matched-rate control);
* ``all``       always (model-greedy; compute reference, not real-time).

Every triggered decision records the branch count and wall time; the episode is truth RK45.
Verification is an evaluation-time protocol: it never feeds training.
"""
import argparse, glob, hashlib, json, multiprocessing as mp, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch as th

from env.t1_b_feasible_v1 import bin_center, choice_index
from env.t1_b_feasible_v7 import FeasibleArrivalEnvV7, _init_worker, _run_branch, snapshot
from env.t1_n_review import N_DUE_INDEX
from experiments.t1b_v2 import LEARNERS, default, dump, mission, openings, sha
from train.t1_b_v2_sac import FeasibleSupportSACV2

MODES = ('off', 'deviation', 'risk', 'random', 'all')
Q_THETA = 0.0


def load(checkpoint, interface):
    cls = LEARNERS['v8'] if interface == 'v8' else FeasibleSupportSACV2
    return cls.load(checkpoint, device='cpu', custom_objects={'learning_rate': 1e-4, 'lr_schedule': lambda _: 1e-4})


def uniform(opening, decision):
    return int(hashlib.sha256(f'{opening}:{decision}'.encode()).hexdigest()[:12], 16) / 16 ** 12


def trigger(mode, proposal, q_proposal, q_theta=Q_THETA, u=1., rho=0.):
    if mode == 'off': return False, None
    if mode == 'all': return True, 'all'
    if mode == 'random': return (u < rho), ('random' if u < rho else None)
    if proposal != 0: return True, 'deviation'
    if mode == 'risk' and q_proposal < q_theta: return True, 'risk'
    return False, None


def choose(proposal, predicted):
    """predicted: [(return, event)] per candidate index -> executed index."""
    ok = [j for j, (_, ev) in enumerate(predicted) if ev == 'success']
    return max(ok, key=lambda j: predicted[j][0]) if ok else proposal


def candidate_q(model, z, n):
    with th.no_grad():
        q1, q2 = model.candidate_q(th.as_tensor(z[None], dtype=th.float32), model.critic)
    return th.minimum(q1, q2).numpy()[0, :n], q1.numpy()[0, :n], q2.numpy()[0, :n]


# ------------------------------------------------------------------ closed-loop deployment (truth RK45)
def deploy_job(job):
    opening, checkpoint, interface, label, out, mode, q_theta, rho, branch_workers = job
    th.set_num_threads(1); file = Path(out) / f'{label}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    env = FeasibleArrivalEnvV7(openings=[opening], mission=mission()); obs, reset = env.reset()
    model = load(checkpoint, interface); pool = None
    initial, done, k, disc, decisions, t_start = obs.tolist(), False, 0, 0., [], time.time()
    try:
        while not done:
            z = obs; n = int((z[49:52] > .5).sum()); a = model.predict(obs, deterministic=True)[0]
            if z[N_DUE_INDEX] > .5 and n >= 2:
                choices = [float(c) for c in env.support()]; prop = choice_index(float(np.asarray(a).reshape(-1)[0]), n)
                qmin, q1, q2 = candidate_q(model, z, n); u = uniform(opening, len(decisions))
                fire, why = trigger(mode, prop, float(qmin[prop]), q_theta, u, rho)
                row = {'time_s': env.env.time_s, 'candidates_s': choices, 'proposal': prop, 'q_min': qmin.tolist(), 'q1': q1.tolist(), 'q2': q2.tolist(),
                       'u': u, 'triggered': fire, 'reason': why, 'executed': prop, 'branches': 0, 'wall_s': 0., 'observation': z.tolist()}
                if fire:
                    if pool is None:
                        pool = ProcessPoolExecutor(branch_workers, mp_context=mp.get_context('spawn'), initializer=_init_worker, initargs=(env._spec,))
                    snap = snapshot(env); t0 = time.time()
                    res = list(pool.map(_run_branch, [(snap, bin_center(j, n)) for j in range(n)]))
                    pred = [(float(r['return']), r.get('event')) for r in res]; j = choose(prop, pred)
                    row.update(executed=j, branches=n, wall_s=time.time() - t0,
                               model=[{'j': jj, 'return': g, 'event': ev, 'steps': int(r['steps']), 'delta_v_end': r.get('delta_v'), 'qp_zero_fallbacks': r.get('qp_zero_fallbacks')}
                                      for jj, ((g, ev), r) in enumerate(zip(pred, res))])
                    a = np.array([bin_center(j, n)], np.float32)
                decisions.append(row)
            obs, r, t, x, i = env.step(a); done = t or x; disc += .999 ** k * r; k += 1
    finally:
        if pool is not None: pool.shutdown()
    res = {'opening': opening, 'label': label, 'interface': interface, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
           'verify': mode, 'q_theta': q_theta, 'rho': rho, 'mission': reset['mission'], 'event': i['event'], 'discounted_return': disc,
           'time_s': env.env.time_s, 'delta_v': env.env._delta_v, 'decisions': decisions, 'learning_decisions': len(decisions),
           'triggered_decisions': sum(d['triggered'] for d in decisions), 'model_branches': sum(d['branches'] for d in decisions),
           'verification_wall_s': sum(d['wall_s'] for d in decisions), 'episode_wall_s': time.time() - t_start, 'initial_observation': initial}
    dump(file, res); return res


def deploy(out, checkpoint, interface, label, ops, workers, mode, q_theta, rho, branch_workers):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(deploy_job, (o, checkpoint, interface, label, str(out), mode, q_theta, rho, branch_workers)) for o in ops]
        for f in as_completed(jobs):
            r = f.result()
            print(json.dumps({k: r[k] for k in ('opening', 'label', 'verify', 'event', 'discounted_return', 'learning_decisions', 'triggered_decisions', 'model_branches')}, default=default), flush=True)


# ------------------------------------------------------------------ offline readout (no simulation)
def truth_tables(patterns):
    t = {}
    for pattern in patterns:
        for f in sorted(glob.glob(pattern, recursive=True)):
            d = json.loads(Path(f).read_text(encoding='utf-8-sig'))
            if d.get('skipped') or 'event' not in d: continue
            j = d.get('first_index', d.get('j0'))
            t.setdefault(int(d['opening']), {})[int(j)] = (float(d['discounted_return']), d['event'], float(d['delta_v']), float(d['time_s']))
    return t


def model_tables(dirs):
    m = {}
    for d in dirs:
        for f in sorted(Path(d).glob('model_*.json')):
            x = json.loads(f.read_text(encoding='utf-8-sig'))
            m[int(x['opening'])] = {int(r['j']): (float(r['pred_return']), r['pred_event']) for r in x['rows']}
    return m


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool); pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg): return None
    return float(np.mean([(p > q) + .5 * (p == q) for p in pos for q in neg]))


def offline(out, checkpoint, interface, episodes, truth_globs, model_dirs, q_theta):
    model = load(checkpoint, interface); T = truth_tables(truth_globs); MP = model_tables(model_dirs); rows = []
    for d in episodes:
        for f in sorted(Path(d).glob('*.json')):
            x = json.loads(f.read_text(encoding='utf-8-sig'))
            if 'initial_observation' not in x: continue
            op = int(x['opening']); z = np.asarray(x['initial_observation'], np.float32); n = int((z[49:52] > .5).sum())
            if op not in T or n < 2 or any(j not in T[op] for j in range(n)) or z[N_DUE_INDEX] < .5: continue
            a = model.predict(z, deterministic=True)[0]; prop = choice_index(float(np.asarray(a).reshape(-1)[0]), n)
            qmin, _, _ = candidate_q(model, z, n)
            rows.append({'opening': op, 'n': n, 'proposal': prop, 'q_min': qmin.tolist(), 'truth': [T[op][j] for j in range(n)],
                         'model': [MP[op][j] for j in range(n)] if op in MP and all(j in MP[op] for j in range(n)) else None,
                         'logged_first_index': (x.get('first_decision') or {}).get('index'),
                         'risk_trigger': bool(prop != 0 or qmin[prop] < q_theta), 'earliest_fails': T[op][0][1] != 'success'})
    fail0 = [r['truth'][0][1] != 'success' for r in rows]
    summary = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'interface': interface, 'q_theta': q_theta, 'states': len(rows),
               'auc_minus_q_earliest_vs_earliest_failure': auc([-r['q_min'][0] for r in rows], fail0), 'earliest_failures': int(sum(fail0))}
    ret = lambda r, j: r['truth'][j][0]; cap = lambda r, j: r['truth'][j][1] == 'success'
    with_model = [r for r in rows if r['model'] is not None]
    def greedy(r):
        return choose(0, r['model'])
    arms = {'earliest': lambda r: (0, False), 'policy_off': lambda r: (r['proposal'], False),
            'verify_all': lambda r: (greedy(r), True),
            'verify_deviation': lambda r: (greedy(r), True) if r['proposal'] != 0 else (0, False),
            'verify_risk': lambda r: (greedy(r), True) if (r['proposal'] != 0 or r['q_min'][r['proposal']] < q_theta) else (r['proposal'], False),
            'oracle': lambda r: (max(range(r['n']), key=lambda j: ret(r, j)), False)}
    table = {}
    for name, f in arms.items():
        picks = [f(r) for r in with_model]
        table[name] = {'n': len(with_model), 'return': float(np.mean([ret(r, j) for r, (j, _) in zip(with_model, picks)])),
                       'capture': float(np.mean([cap(r, j) for r, (j, _) in zip(with_model, picks)])),
                       'triggered_fraction': float(np.mean([v for _, v in picks])),
                       'catastrophic_vs_earliest': int(sum(cap(r, 0) and not cap(r, j) for r, (j, _) in zip(with_model, picks)))}
    rate = table['verify_risk']['triggered_fraction']
    table['random_matched_rate_expectation'] = {'n': len(with_model), 'rate': rate,
                                                'return': table['earliest']['return'] + rate * (table['verify_all']['return'] - table['earliest']['return'])}
    summary['first_decision_recombination_truth_continuation_V1_teacher'] = table
    dump(out, {'summary': summary, 'rows': rows}); print(json.dumps(summary, indent=1, default=default), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['deploy', 'offline'])
    p.add_argument('--output', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--interface', choices=['v7', 'v8'], default='v8')
    p.add_argument('--verify', choices=MODES, default='off'); p.add_argument('--q-theta', type=float, default=Q_THETA); p.add_argument('--rho', type=float, default=0.)
    p.add_argument('--label'); p.add_argument('--openings'); p.add_argument('--workers', type=int, default=2); p.add_argument('--branch-workers', type=int, default=3)
    p.add_argument('--episodes'); p.add_argument('--truth'); p.add_argument('--model')
    a = p.parse_args(); out = Path(a.output)
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(k, '1')
    if a.mode == 'deploy':
        deploy(out, a.checkpoint, a.interface, a.label, openings(a.openings), a.workers, a.verify, a.q_theta, a.rho, a.branch_workers)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        offline(out, a.checkpoint, a.interface, a.episodes.split(','), a.truth.split(','), a.model.split(','), a.q_theta)


if __name__ == '__main__':
    main()
