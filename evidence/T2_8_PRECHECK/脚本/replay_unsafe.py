"""Replay recorded evaluation actions (deterministic eval) and read the truth judge's violation at the unsafe step."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/user/spacecraft')
from env.t2_stage import T2Mission, T2StageEnv
from env.t1_p_stage import relative_inertial

def replay(path):
    r = json.load(open(path))
    env = T2StageEnv(openings=[r['opening']], shaping=False, window=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.reset(options={'opening': r['opening']})
    out, k = None, 0
    acts = [d['a'] for d in r['decisions']]
    done, info = False, {}
    while not done:
        a = np.asarray(acts[k], float) if (env.generator.mode in ('stage', 'hold') and k < len(acts)) else np.zeros(2)
        if env.generator.mode in ('stage', 'hold'):
            k += 1
        _, _, term, trunc, info = env.step(a); done = term or trunc
    m = env.env.metrics()
    q, v = relative_inertial(env.env)
    return {'file': path.split('/')[-1], 'recorded': r['event'], 'replayed': info.get('event'), 't': round(env.env.time_s, 1), 'recorded_t': r['time_s'],
            'violations': {kk: round(vv, 3) for kk, vv in m.violations.items()}, 'range_m': round(m.range_m, 2), 'in_cone': m.in_cone,
            'cone_margin_m': round(m.cone_margin_m, 2), 'union_margin_m': round(m.union_margin_m, 2), 'closing_speed_m_s': round(-float(q @ v) / float(np.linalg.norm(q)), 3)}

if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(replay(p)))
