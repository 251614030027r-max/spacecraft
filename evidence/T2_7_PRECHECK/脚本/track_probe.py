"""Exploratory (sandbox): is the executor planner's capture realizable INSIDE the 2-D stage action space?

Policies on the urgent missions of audit openings 2,910,000+ (U(140, 200) s, standard 3.0 m/s allocation):
  track   at every stage decision, the executor's conditional G* plans its earliest feasible entry on a branch
          (start_direct_gstar; nothing is committed in the real episode); the stage intent (9 x 9 grid) whose stage goal
          lies closest to that plan's state 16 s ahead is flown; no plan -> the rally rule. Inside the action space.
  rally   the deadline-aware rule (hold 9.5 m)
  gstar   the direct G* row (outside the action space; re-planned after a lost bridge)
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/user/spacecraft')
OUT, LO, HI = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
GRID = [np.array([p, z]) for p in np.linspace(-1, 1, 9) for z in np.linspace(-1, 1, 9)]


def track_action(env):
    from env.t1_p_stage import plan_state, relative_inertial
    from env.t2_stage import frame_now, stage_goal
    from experiments.t2 import rally_action
    from experiments.t2_audit import branch
    b = branch(env)
    if not b.start_direct_gstar():
        return rally_action(env), False
    plan, mass = b.generator.bridge.plan, env.env.chaser_parameters.mass
    q16, v16 = plan_state(plan, 16.0, mass)
    q, v = relative_inertial(env.env); frame = frame_now(env.env, q)
    best, cost = None, np.inf
    for a in GRID:
        g = stage_goal(q, v, frame, a, mass, geometry=True)
        c = float(np.linalg.norm(g['q_goal'] - q16) + 10.0 * np.linalg.norm(g['v_goal'] - v16))
        if c < cost:
            best, cost = a, c
    return best, True


def job(arg):
    opening, policy = arg
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv, mission_from
    from experiments.t2 import rally_action
    env = T2StageEnv(openings=[opening], seed_base=2_910_000, shaping=False, mission=mission_from((140.0, 200.0)))
    obs, _ = env.reset(options={'opening': opening})
    if not env.mission['urgent']:
        return {'opening': opening, 'policy': policy, 'urgent': False}
    done, ret, info, planned, n = False, 0.0, {}, 0, 0
    if policy == 'gstar':
        ok = env.start_direct_gstar(); done = not ok; info = {'event': 'no_plan'}
    while not done:
        if policy == 'gstar':
            obs, r, term, trunc, info = env.step(np.zeros(2))
            if not (term or trunc) and env.generator.mode in ('stage', 'hold'):
                env.start_direct_gstar()
        else:
            if env.generator.mode in ('stage', 'hold'):
                a, used = track_action(env) if policy == 'track' else (rally_action(env), False)
                planned += int(used); n += 1
            else:
                a = np.zeros(2)
            obs, r, term, trunc, info = env.step(a)
        ret += info['true_reward']; done = term or trunc
    return {'opening': opening, 'policy': policy, 'urgent': True, 'deadline_s': env.mission['deadline_s'], 'event': info.get('event'),
            'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v, 'decisions': n, 'plan_tracked': planned}


if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(2_910_000 + k, p) for k in range(LO, HI) for p in ('track', 'rally', 'gstar')]
    with ProcessPoolExecutor(4) as pool:
        rows = [r for r in pool.map(job, jobs) if r['urgent']]
    OUT.write_text(json.dumps(rows), encoding='utf-8')
    import collections
    for p in ('track', 'rally', 'gstar'):
        rs = [r for r in rows if r['policy'] == p]
        print(p, len(rs), dict(collections.Counter(r['event'] for r in rs)), 'mean return %.1f' % np.mean([r['return'] for r in rs]),
              'dv %.2f' % np.mean([r['delta_v'] for r in rs]))
    by = collections.defaultdict(dict)
    for r in rows:
        by[r['opening']][r['policy']] = r['event']
    print('opening  track  rally  gstar')
    for o, d in sorted(by.items()):
        print(o, d.get('track'), d.get('rally'), d.get('gstar'))
