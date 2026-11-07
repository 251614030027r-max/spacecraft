"""Exploratory (sandbox): how captures happen -- through a G* bridge (the entry the EACC labels model) or by the
executor's direct descent trigger from stage / hold (mouth region, r < 6 m, or in the port cone within 7 m), which the
labels do not model. Fixed policies inside the action space, urgent U(140, 200) missions, frozen science code."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, '/home/user/spacecraft')
OUT = Path(sys.argv[1]); LO, HI = int(sys.argv[2]), int(sys.argv[3]); POLICIES = sys.argv[4].split(',')


def job(arg):
    opening, policy = arg
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv, T2Mission
    from experiments.t2 import policy_action

    class Probe(T2StageEnv):
        def _try_bridge(self, m):
            ok = super()._try_bridge(m)
            if ok and self._first_bridge_t is None:
                self._first_bridge_t = self.env.time_s
            return ok

    env = Probe(openings=[opening], seed_base=2_910_000, shaping=False, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    obs, _ = env.reset(options={'opening': opening}); env._first_bridge_t = None
    done, info, descent_t = False, {}, None
    while not done:
        obs, _, term, trunc, info = env.step(policy_action(policy, env, obs)); done = term or trunc
        if descent_t is None and env.generator is not None and env.generator.mode == 'descent':
            descent_t = env.env.time_s
    route = None
    if info.get('event') == 'success':
        route = 'bridge' if env._first_bridge_t is not None else 'direct'
    return {'opening': opening, 'policy': policy, 'urgent': env.mission['urgent'], 'deadline_s': env.mission['deadline_s'],
            'event': info.get('event'), 'time_s': env.env.time_s, 'route': route, 'first_bridge_t': env._first_bridge_t}


if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(2_910_000 + k, p) for p in POLICIES for k in range(LO, HI)]
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(job, jobs))
    OUT.write_text(json.dumps(rows), encoding='utf-8')
    import collections
    for p in POLICIES:
        for u in (True, False):
            rs = [r for r in rows if r['policy'] == p and r['urgent'] == u]
            print(p, 'urgent' if u else 'economy', len(rs), dict(collections.Counter((r['event'], r['route']) for r in rs)))
