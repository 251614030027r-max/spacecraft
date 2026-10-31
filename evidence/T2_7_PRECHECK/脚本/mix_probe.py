"""T2.7 pre-start technical review (research layer 2026-10-31; sandbox, exploratory). Frozen science 2208a51.

Mixed execution: at each stage decision the executor's projected intent (T2StageEnv.track_intent) flies with probability
p, otherwise the base action (uniform random -- the untrained policy and the collector's first 200 decisions -- or the
rally rule). Audit openings 2,910,000+, urgent U(140, 200), 3.0 m/s. Per guided decision the execution consistency:
stage plan feasible / held / rate-limited, the realized position after the stage against the executor plan's state
16 s ahead and against the stage goal."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/user/spacecraft')
OUT, LO, HI = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
CONFIGS = (('random', 0.0), ('random', 0.5), ('random', 0.25), ('rally', 0.5), ('rally', 0.0))


def job(arg):
    opening, base, p = arg
    import torch; torch.set_num_threads(1)
    from env.t1_p_stage import plan_state, relative_inertial
    from env.t2_stage import STAGE_S, T2Mission, T2StageEnv, frame_now, stage_goal
    from experiments.t2 import rally_action
    env = T2StageEnv(openings=[opening], seed_base=2_910_000, shaping=False, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.reset(options={'opening': opening})
    rng = np.random.default_rng([opening, int(p * 100), 7])
    done, ret, info, n, guided, cons = False, 0.0, {}, 0, 0, []
    while not done:
        a, g = np.zeros(2), False
        if env.generator.mode in ('stage', 'hold'):
            n += 1
            a = rng.uniform(-1.0, 1.0, size=2) if base == 'random' else rally_action(env)
            if p > 0 and rng.uniform() < p:
                e = env.env; q, v = relative_inertial(e)
                opp = env._opportunity(q, v, e.time_s, e.target_state.rotation, e.target_state.omega)
                t_ = env.track_intent()
                if t_ is not None:
                    a, g = t_, True; guided += 1
                    q16, _ = plan_state(opp['plan'], STAGE_S, e.chaser_parameters.mass)
                    goal = stage_goal(q, v, frame_now(e, q), a, e.chaser_parameters.mass, geometry=env.geometry)
        _, _, term, trunc, info = env.step(a); ret += info['true_reward']; done = term or trunc
        if g:
            q1, _ = relative_inertial(env.env)
            cons.append({'status': info.get('plan_status'), 'held': bool(info.get('held')), 'rate_limited': bool(info.get('rate_limited')),
                         'mode_after': info.get('mode'), 'stage_s': info.get('stage_s'),
                         'dev_plan_m': float(np.linalg.norm(q1 - q16)) if info.get('stage_s', 0) >= STAGE_S - 1e-6 and info.get('mode') in ('stage', 'hold') else None,
                         'dev_goal_m': float(np.linalg.norm(q1 - goal['q_goal'])) if info.get('stage_s', 0) >= STAGE_S - 1e-6 and info.get('mode') in ('stage', 'hold') else None})
    return {'opening': opening, 'base': base, 'p': p, 'urgent': env.mission['urgent'], 'event': info.get('event'), 'return': ret,
            'delta_v': env.env._delta_v, 'time_s': env.env.time_s, 'decisions': n, 'guided': guided, 'consistency': cons}


if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(2_910_000 + k, b, p) for k in range(LO, HI) for b, p in CONFIGS]
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(job, jobs))
    OUT.write_text(json.dumps(rows, default=float), encoding='utf-8')
    import collections
    for b, p in CONFIGS:
        for u in (True, False):
            rs = [r for r in rows if r['base'] == b and r['p'] == p and r['urgent'] == u]
            print(f'{b:6s} p={p:.2f} {"urgent " if u else "economy"} n {len(rs):2d} events {dict(collections.Counter(r["event"] for r in rs))} '
                  f'return {np.mean([r["return"] for r in rs]):6.1f} dv {np.mean([r["delta_v"] for r in rs]):.2f} '
                  f'guided/decisions {sum(r["guided"] for r in rs)}/{sum(r["decisions"] for r in rs)}')
    c = [x for r in rows for x in r['consistency']]
    if c:
        dp = [x['dev_plan_m'] for x in c if x['dev_plan_m'] is not None]; dg = [x['dev_goal_m'] for x in c if x['dev_goal_m'] is not None]
        print('guided decisions', len(c), 'plan status', dict(collections.Counter(x['status'] for x in c)), 'held', sum(x['held'] for x in c),
              'rate_limited', sum(x['rate_limited'] for x in c), 'mode after', dict(collections.Counter(x['mode_after'] for x in c)))
        print('realized - executor plan @16 s: median %.2f m p90 %.2f m | realized - stage goal: median %.2f m p90 %.2f m (n %d)'
              % (np.median(dp), np.percentile(dp, 90), np.median(dg), np.percentile(dg, 90), len(dp)))
