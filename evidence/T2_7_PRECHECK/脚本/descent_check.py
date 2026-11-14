"""Exploratory (sandbox): would the executor's own direct-descent trigger, applied at the model's 2 s nodes, see the
direct captures the labels miss -- and how often would it fire where the truth did not descend?

Replays the horizon probe's factual episodes (same rule, same seeds, same candidates; no truth branches) and, for each
candidate, scans the model's predicted nodes (stage 1 anchored, stages 2-4 calibration only, intent held, as the probe)
with the executor's predicate (t1_gstar _descending: mouth region, r < operation sphere, or in the port cone within the
descent radius) and the truth judge's operation-sphere rule (inside the sphere outside the cone: unsafe). Joined to the
stored truth by (opening, decision time, candidate)."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/user/spacecraft')
sys.path.insert(0, str(Path(__file__).parent))
CAL, HZ, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DEPTH, WINDOW = 4, (40.0, 140.0)


def node_events(env, plan, resid, attitude, offset):
    """First node of the stage where the executor's descent trigger fires on the predicted state."""
    from env.t1_p_stage import plan_state
    from env.t1_task import cone_quantities
    from env.t2_stage import ENTRY_NODES_S
    task, mass = env.env.config.task, env.env.chaser_parameters.mass
    r_desc = env.generator.bridge.descent_radius_m
    for s in ENTRY_NODES_S:
        q, _ = plan_state(plan, s, mass)
        q = q + resid(s)[0]
        rot, _ = attitude(offset + s)
        p = rot.T @ q
        r = float(np.linalg.norm(p))
        _, _, cone_margin = cone_quantities(p, task)
        in_cone = cone_margin >= 0.0
        mouth = in_cone and r < task.operation_sphere_radius_m + task.mouth_band_m
        if r < task.operation_sphere_radius_m and not in_cone:
            return {'s': offset + s, 'event': 'unsafe', 'r': r}
        if mouth or r < task.operation_sphere_radius_m or (in_cone and r < r_desc):
            return {'s': offset + s, 'event': 'descent', 'r': r}
    return None


def job(opening):
    import torch; torch.set_num_threads(1)
    from env.t1_p_stage import plan_state
    from env.t2_stage import T2StageEnv, T2Mission, STAGE_S, spin_frame, stage_goal, BRIDGE_RANGE_M, d_authority
    from experiments.t2 import rally_action
    from experiments.t2_audit import load_calibration

    class Probe(T2StageEnv):
        def _plan_candidates(self, q, v, frame, action):
            out = super()._plan_candidates(q, v, frame, action); self._planned = out; return out

        def _labels(self, plan, a, start, cal, ecal, trace, entered, entry_rec, event):
            res = super()._labels(plan, a, start, cal, ecal, trace, entered, entry_rec, event)
            if not self._want:
                return res
            attitude = self._realized_attitude(trace)
            mass = self.env.chaser_parameters.mass
            s_valid = (entered['t'] - start['t']) if entered is not None else STAGE_S
            nodes = {}
            for x in trace:
                if x['s'] <= s_valid + 1e-6:
                    qp, vp = plan_state(plan, x['s'], mass)
                    nodes[round(x['s'], 1)] = (x['q'] - qp, x['v'] - vp, 0.0)
            keys = sorted(nodes)
            anchored = lambda s: nodes[round(float(s), 1)] if round(float(s), 1) in nodes else nodes[max([y for y in keys if y <= round(float(s), 1)] or [keys[0]])]
            rows = []
            for c in self._planned:
                ev, pl, res_fn, q0, v0 = None, c['plan'], anchored, None, None
                for k in range(DEPTH):
                    ev = node_events(self, pl, res_fn, attitude, k * STAGE_S)
                    if ev is not None:
                        ev['stage'] = k + 1; break
                    q1, v1 = plan_state(pl, STAGE_S, mass); dq, dv_, _ = res_fn(STAGE_S); q1, v1 = q1 + dq, v1 + dv_
                    if float(np.linalg.norm(q1)) > BRIDGE_RANGE_M + 2 * d_authority(mass):
                        break
                    rot1, om1 = attitude((k + 1) * STAGE_S)
                    frame1 = spin_frame(rot1, om1, q1, self.env.config.task.approach_axis)
                    goal = stage_goal(q1, v1, frame1, c['a'], mass, geometry=self.geometry)
                    pl = self.generator.planner.solve(q1, v1, goal['q_goal'], goal['v_goal'])
                    if not pl['feasible']:
                        break
                    dq_b, dv_b = cal.residuals(); b1 = frame1['basis']
                    res_fn = (lambda b1, dq_b, dv_b: (lambda s: (b1.T @ dq_b * (s / STAGE_S), b1.T @ dv_b * (s / STAGE_S), 0.0)))(b1, dq_b, dv_b)
                rows.append({'a': np.asarray(c['a']).tolist(), 'descent': ev})
            self._rows = rows
            return res

    env = Probe(openings=[opening], seed_base=2_910_000, candidates=8, shaping=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.reset(options={'opening': opening})
    if not env.mission['urgent']:
        return {'opening': opening, 'urgent': False}
    load_calibration(env, CAL)
    recs, done = [], False
    while not done:
        left = env.mission['deadline_s'] - env.env.time_s
        a = rally_action(env, hold_m=7.5)
        env._want = env.generator.mode in ('stage', 'hold') and WINDOW[0] <= left <= WINDOW[1]
        env._rows = None; t_dec = env.env.time_s
        _, _, term, trunc, info = env.step(a); done = term or trunc
        if env._want and env._rows is not None:
            recs.append({'t': t_dec, 'cands': env._rows})
    return {'opening': opening, 'urgent': True, 'decisions': recs}


if __name__ == '__main__':
    ops = [int(f.stem) for f in sorted(HZ.glob('*.json')) if json.loads(f.read_text()).get('urgent')]
    with ProcessPoolExecutor(4) as pool:
        out = list(pool.map(job, ops))
    OUT.write_text(json.dumps(out, default=float), encoding='utf-8')
    print('done', len(out))
