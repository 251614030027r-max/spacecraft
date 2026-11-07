"""Exploratory probe (sandbox, not evidence): which EACC candidates receive a model label and which are dropped,
by range and closing action, on urgent U(140, 200) missions. Openings 2,910,000+ (audit block), frozen science code.
Behaviour: rally rule holding at HOLD m (inside the action space). Stage-2 labels off (no policy copy)."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/user/spacecraft')
CAL = Path(sys.argv[1]); OUT = Path(sys.argv[2]); N = int(sys.argv[3]); HOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 7.5


def job(opening):
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv, T2Mission
    from experiments.t2 import rally_action
    from experiments.t2_audit import load_calibration
    from env.t1_p_stage import relative_inertial

    class Probe(T2StageEnv):
        def candidate_actions(self, action):
            out = super().candidate_actions(action); self._acts = [np.asarray(x, float) for x in out]; return out

        def _plan_candidates(self, q, v, frame, action):
            out = super()._plan_candidates(q, v, frame, action); self._planned = out; return out

        def predict_intent(self, plan, action, *args, **kw):
            p = super().predict_intent(plan, action, *args, **kw)
            if getattr(self, '_rec_on', False):
                self._rec.append((np.asarray(action, float), plan, p))
            return p

        def _labels(self, *args, **kw):
            self._rec_on, self._rec = True, []
            res = super()._labels(*args, **kw)
            self._rec_on = False
            q, _ = relative_inertial(self.env)
            rows = []
            planned = {tuple(np.round(c['a'], 6)) for c in self._planned}
            for a in self._acts:
                if tuple(np.round(a.astype(np.float32), 6)) not in planned:
                    rows.append({'a': a.tolist(), 'class': 'plan_infeasible'})
            for a, plan, p in self._rec[1:]:
                if p['done']:
                    c = 'entry' if p.get('entry') is not None else 'cutoff'
                elif plan.get('terminal_position_error_m', 1.0) > 0.3:
                    c = 'drop_plan_error'
                elif p['r_end'] < 6.5:
                    c = 'drop_r_end_lt_6.5'
                elif p['r_end'] > 25.0:
                    c = 'drop_r_end_gt_25'
                else:
                    c = 'stage'
                rows.append({'a': a.tolist(), 'class': c, 'r_end': p.get('r_end')})
            self._log.append({'t': float(args[2]['t']), 'r': float(np.linalg.norm(args[2]['q'])),
                              'left_s': float(self.mission['deadline_s'] - args[2]['t']), 'a_exec': np.asarray(args[1], float).tolist(),
                              'cands': rows})
            return res

    env = Probe(openings=[opening], seed_base=2_910_000, candidates=8, shaping=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.reset(options={'opening': opening})
    load_calibration(env, CAL)
    env._log = []; done = False; info = {}
    while not done:
        _, _, term, trunc, info = env.step(rally_action(env, hold_m=HOLD)); done = term or trunc
    return {'opening': opening, 'mission': env.mission, 'event': info.get('event'), 'time_s': env.env.time_s, 'decisions': env._log}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(job, [2_910_000 + k for k in range(int(sys.argv[5]) if len(sys.argv) > 5 else 0, N)]):
            (OUT / f"{r['opening']}.json").write_text(json.dumps(r), encoding='utf-8')
            print(r['opening'], r['mission']['urgent'], round(r['mission']['deadline_s'], 1), r['event'], round(r['time_s'], 1),
                  len(r['decisions']), round(time.time() - t0), flush=True)
