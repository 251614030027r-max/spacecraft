"""Pre-training truth audit of the T2.7 executor opportunity bounds (sandbox; exploratory, read-only).

Urgent missions U(140, 200) of audit openings 2,910,000+, the rally rule as behaviour, the env's own T2.7 labels
(opportunity=True, the shared warm-up calibration). At the first three labelled decisions of each episode, every
candidate and the executed intent are flown on a truth branch: the intent for one decision, then the in-space
plan-tracking continuation (the executor's plan projected onto the stage action space each decision) to the end.
Compared: the bound L (unshaped: L + Phi(s)) with that truth return; decisive-pair ordering by the bounds."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/user/spacecraft')
CAL, OUT, LO, HI = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
FIRST = 3


def track_continuation(b):
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import frame_now
    from experiments.t2 import rally_action
    q, v = relative_inertial(b.env)
    st = {'q': q, 'v': v, 'frame': frame_now(b.env, q), 't': b.env.time_s, 'rotation': b.env.target_state.rotation, 'omega': b.env.target_state.omega}
    a = b._track_intent(st)
    return a if a is not None else rally_action(b)


def truth(snapshot, a):
    from experiments.t2_audit import branch
    b = branch(snapshot)
    _, _, term, trunc, info = b.step(np.asarray(a, dtype=np.float64))
    g, done = float(info['true_reward']), term or trunc
    while not done:
        act = np.zeros(2) if b.generator.mode in ('bridge', 'descent') else track_continuation(b)
        _, _, term, trunc, info = b.step(act); g += float(info['true_reward']); done = term or trunc
    return {'G': g, 'event': info.get('event')}


def job(opening):
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv, T2Mission
    from experiments.t2 import rally_action
    from experiments.t2_audit import branch, load_calibration
    env = T2StageEnv(openings=[opening], seed_base=2_910_000, candidates=8, shaping=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)),
                     entry_stages=1, opportunity=True)
    env.reset(options={'opening': opening})
    if not env.mission['urgent']:
        return {'opening': opening, 'urgent': False}
    load_calibration(env, CAL)
    recs, done, t0 = [], False, time.time()
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            _, _, term, trunc, info = env.step(np.zeros(2)); done = term or trunc; continue
        a = rally_action(env, hold_m=7.5)
        audit = len(recs) < FIRST
        snap, phi0 = (branch(env), env._phi()) if audit else (None, None)
        _, _, term, trunc, info = env.step(a); done = term or trunc
        m = info['model']
        if not audit or m['diag'].get('reason') not in ('stage', 'entry'):
            continue
        items = [{'a': np.asarray(a, float).tolist(), 'executed': True, 'bound': m['fact'].get('bound'), 'done': False}]
        items += [{'a': np.asarray(c['a'], float).tolist(), 'executed': False, 'bound': c.get('bound'), 'done': bool(c['done']),
                   'label': float(c['reward'])} for c in m['cands']]
        for it in items:
            it['L_true'] = None if it['bound'] is None else it['bound'] + phi0
            it['truth'] = truth(snap, it['a'])
        recs.append({'t': info['time_s'] - info['stage_s'], 'left_s': env.mission['deadline_s'] - (info['time_s'] - info['stage_s']),
                     'track_candidate': m['diag'].get('track'), 'items': items})
        print(json.dumps({'opening': opening, 'n': len(items), 'bounds': sum(i['bound'] is not None for i in items), 'wall': round(time.time() - t0)}), flush=True)
    return {'opening': opening, 'urgent': True, 'event': info.get('event'), 'decisions': recs}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    ops = [2_910_000 + k for k in range(LO, HI) if not (OUT / f'{2_910_000 + k}.json').exists()]
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(job, ops):
            (OUT / f"{r['opening']}.json").write_text(json.dumps(r, default=float), encoding='utf-8')
            print('DONE', r['opening'], r.get('urgent'), r.get('event'), len(r.get('decisions', [])), flush=True)
