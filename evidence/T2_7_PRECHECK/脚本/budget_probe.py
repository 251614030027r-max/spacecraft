"""Exploratory horizon probe (sandbox, read-only; not evidence, not the science line).

Urgent missions U(140, 200) s on audit openings 2,910,000+. At each labelled decision 40-140 s before the deadline,
every EACC candidate is predicted 1-4 stages deep by the executor's own model and flown 1-4 stages on a truth branch.
  depth 1   = the production stage-1 label (predict_intent, anchored), unchanged
  depth 2-4 = predict_second_stage's rule chained (calibration only, no anchoring; cutoff at the deadline / budget,
              the budget before an entry included), the same continuation in model and truth: hold the candidate intent
              (no policy copy in the sandbox; production stage 2 samples the policy instead)
Truth: a fresh branch from the decision snapshot (experiments.t2_audit.branch), the same intent held for up to 4 decisions;
an entry runs to its end (T2StageEnv.step does). Frozen science code, untouched."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/user/spacecraft')
CAL, OUT, LO, HI = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
DEPTH, WINDOW = 4, (40.0, 140.0)


def chain_stage(env, prev, a, start, cal, ecal, attitude, k):
    """Stage k >= 2 of one candidate holding intent ``a`` (predict_second_stage's rule, generic depth)."""
    from env.t1_p_stage import plan_state
    from env.t2_stage import (BRIDGE_RANGE_M, ENTRY_MAX_SOLVES, STAGE_S, d_authority, spin_frame, stage_goal)
    e, mass, w = env.env, env.env.chaser_parameters.mass, env.env.config.reward
    cost = lambda s, dv: -w.time_cost_per_s * s - w.delta_v_cost_per_m_s * dv
    q1, v1, rot1, om1 = prev['q'], prev['v'], prev['rot'], prev['omega']
    if float(np.linalg.norm(q1)) > BRIDGE_RANGE_M + 2 * d_authority(mass):
        return {'event': 'out_of_domain'}
    frame1 = spin_frame(rot1, om1, q1, e.config.task.approach_axis)
    goal = stage_goal(q1, v1, frame1, a, mass, geometry=env.geometry)
    nxt = env.generator.planner.solve(q1, v1, goal['q_goal'], goal['v_goal'])
    if not nxt['feasible'] or nxt.get('terminal_position_error_m', 1.0) > 0.3:
        return {'event': 'infeasible'}
    dq_b, dv_b = cal.residuals(); b1 = frame1['basis']
    raw = lambda s: (b1.T @ dq_b * (s / STAGE_S), b1.T @ dv_b * (s / STAGE_S), 0.0)
    t1 = start['t'] + (k - 1) * STAGE_S
    spent = prev['dv_total']
    hit = env._scan(nxt, raw, attitude, (k - 1) * STAGE_S, [ENTRY_MAX_SOLVES], True)
    if hit is not None:
        dv_hit = env._dv_model(nxt, cal, hit['s']) + hit['brake_dv']
        cut = env._stage_cutoff(t1, start['dv'] + spent, dv_hit, hit['s'])
        if cut is not None:
            return {'event': 'cutoff_' + cut[2], 'value': prev['value'] + cost(cut[0], cut[1]) - w.timeout_penalty}
        pr = ecal.predict(hit['tau'], hit['dv_plan'], w, env.mission['deadline_s'] - (t1 + hit['s']))
        return {'event': 'entry', 'entry_t': t1 + hit['s'], 'p_success': pr['p_success'],
                'value': prev['value'] + cost(hit['s'], dv_hit) + pr['reward']}
    q2, v2 = plan_state(nxt, STAGE_S, mass)
    q2, v2 = q2 + b1.T @ dq_b, v2 + b1.T @ dv_b
    dv2 = env._dv_model(nxt, cal, STAGE_S)
    cut = env._stage_cutoff(t1, start['dv'] + spent, dv2)
    if cut is not None:
        return {'event': 'cutoff_' + cut[2], 'value': prev['value'] + cost(cut[0], cut[1]) - w.timeout_penalty}
    if not 6.5 <= float(np.linalg.norm(q2)) <= 25.0:
        return {'event': 'out_of_domain'}
    rot2, om2 = attitude(k * STAGE_S)
    return {'event': None, 'q': q2, 'v': v2, 'rot': rot2, 'omega': om2, 'dv_total': spent + dv2,
            'value': prev['value'] + cost(STAGE_S, dv2)}


def truth(snapshot, a):
    from env.t1_p_stage import relative_inertial
    from experiments.t2_audit import branch
    b = branch(snapshot)
    out = {'stages': [], 'entered_stage': None, 'event': None, 'true': 0.0}
    for k in range(DEPTH):
        _, _, term, trunc, info = b.step(np.asarray(a, dtype=np.float64))
        out['true'] += info['true_reward']
        q, _ = relative_inertial(b.env)
        out['stages'].append({'q': q.tolist(), 'stage_s': info['stage_s'], 'held': info['held'], 't': b.env.time_s})
        if b._stage_entry is not None and out['entered_stage'] is None:
            out['entered_stage'] = k + 1
        if term or trunc:
            out['event'] = info['event']; break
    return out


def job(opening):
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv, T2Mission
    from experiments.t2 import rally_action
    from experiments.t2_audit import branch, load_calibration

    class Probe(T2StageEnv):
        def _plan_candidates(self, q, v, frame, action):
            out = super()._plan_candidates(q, v, frame, action); self._planned = out; return out

        def _labels(self, plan, a, start, cal, ecal, trace, entered, entry_rec, event):
            res = super()._labels(plan, a, start, cal, ecal, trace, entered, entry_rec, event)
            if self._want is None:
                return res
            attitude = self._realized_attitude(trace)
            # the same anchored residual the production labels use (rebuilt as in _labels)
            from env.t1_p_stage import plan_state
            mass = self.env.chaser_parameters.mass
            s_valid = (entered['t'] - start['t']) if entered is not None else 16.0
            nodes = {}
            for x in trace:
                if x['s'] <= s_valid + 1e-6:
                    qp, vp = plan_state(plan, x['s'], mass)
                    nodes[round(x['s'], 1)] = (x['q'] - qp, x['v'] - vp, (x['dv'] - start['dv']) - self._dv_model(plan, cal, x['s']))
            keys = sorted(nodes)
            anchored = lambda s: nodes[round(float(s), 1)] if round(float(s), 1) in nodes else nodes[max([y for y in keys if y <= round(float(s), 1)] or [keys[0]])]
            self._model_t0 = start['t']
            rows = []
            for c in self._planned:              # the production label's own candidates (no extra RNG draw)
                p = self.predict_intent(c['plan'], c['a'], start, cal, ecal, anchored, attitude)
                d = [None] * DEPTH
                if p['done']:
                    d[0] = {'event': 'entry' if p.get('entry') is not None else 'cutoff_' + p.get('event', '?'), 'value': p['true'],
                            'entry_t': start['t'] + p['entry']['s'] if p.get('entry') else None}
                elif c['plan'].get('terminal_position_error_m', 1.0) > 0.3 or not 6.5 <= p['r_end'] <= 25.0:
                    d[0] = {'event': 'dropped'}
                else:
                    d[0] = {'event': None, 'q': p['q'], 'v': p['v'], 'rot': p['rot'], 'omega': p['omega'], 'dv_total': p['dv'], 'value': p['true']}
                    for k in range(2, DEPTH + 1):
                        d[k - 1] = chain_stage(self, d[k - 2], c['a'], start, cal, ecal, attitude, k)
                        if d[k - 1]['event'] is not None:
                            break
                rows.append({'a': np.asarray(c['a']).tolist(),
                             'model': [None if x is None else {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in x.items()
                                                               if kk not in ('rot', 'omega', 'v')} for x in d]})
            self._deep = rows
            return res

    env = Probe(openings=[opening], seed_base=2_910_000, candidates=8, shaping=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0), budget_range_m_s=(1.2, 3.0)))
    env.reset(options={'opening': opening})
    load_calibration(env, CAL)
    env._want = None
    recs, done, t0, info = [], False, time.time(), {}
    while not done:
        left = env.mission['deadline_s'] - env.env.time_s
        a = np.array([1.0, 0.5])            # c_10_5: a fuel-hungry fixed intent, so the allocation binds
        critical = env.generator.mode in ('stage', 'hold') and left >= 40.0 and env.mission['budget_m_s'] - env.env._delta_v <= 1.0
        snap = branch(env) if critical else None
        env._want, env._deep = (np.asarray(a, float) if critical else None), None
        t_dec = env.env.time_s
        _, _, term, trunc, info = env.step(a); done = term or trunc
        if critical and env._deep is not None:
            rows = env._deep
            for row in rows:
                row['truth'] = truth(snap, row['a'])
            recs.append({'t': t_dec, 'left_s': left, 'a': np.asarray(a).tolist(), 'cands': rows})
            print(json.dumps({'opening': opening, 't': t_dec, 'left': round(left), 'n': len(rows), 'wall': round(time.time() - t0)}), flush=True)
    return {'opening': opening, 'urgent': True, 'budget': env.mission['budget_m_s'], 'mission': env.mission, 'event': info.get('event'), 'time_s': env.env.time_s,
            'decisions': recs, 'wall_s': time.time() - t0}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [o for o in range(2_910_000 + LO, 2_910_000 + HI) if not (OUT / f'{o}.json').exists()]
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(job, todo):
            (OUT / f"{r['opening']}.json").write_text(json.dumps(r, default=float), encoding='utf-8')
            print('DONE', r['opening'], r.get('urgent'), r.get('event'), len(r.get('decisions', [])), round(r.get('wall_s', 0)), flush=True)
