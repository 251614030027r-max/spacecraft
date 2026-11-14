"""Exploratory (sandbox, read-only): does the executor's own capture-opportunity evaluation carry the information the
stage labels lack on urgent missions?

Same episodes, decisions and candidates as horizon_probe (rule behaviour, audit openings 2,910,000+, urgent U(140, 200),
decisions 40-140 s before the deadline, the production label's own candidates). For every candidate:
  1. truth: the intent for one stage from the decision snapshot (fresh branch);
  2. opportunity: from that stage-end state the executor's conditional G* (start_direct_gstar: earliest feasible entry
     on its 20-220 s grid within deadline - 45 s, constant-rate attitude model, no truth) -- feasible or not, its tau and
     planned delta-v, and the entry value predicted by the episode's entry-outcome replay (EntryCalibration.predict);
     L = stage-1 true reward + that predicted entry value  (the value of committing to the executor's plan)
  3. plan truth: the same plan flown to the end on that branch (G* re-planned after a lost bridge, as the gstar row);
Joined to horizon_probe's truth (the intent held for up to 4 decisions) by (opening, decision time, candidate)."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/user/spacecraft')
CAL, HZ, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
WINDOW = (40.0, 140.0)


def opportunity(snapshot, a, ecal):
    from experiments.t2_audit import branch
    b = branch(snapshot)
    _, _, term, trunc, info = b.step(np.asarray(a, dtype=np.float64))
    r1 = float(info['true_reward'])
    out = {'r1': r1, 'stage_event': info['event'], 'entered_in_stage': b._stage_entry is not None}
    if term or trunc:
        out.update(feasible=None, L=r1, plan_event=info['event'], plan_true=r1)
        return out
    if b.generator.mode in ('bridge', 'descent'):        # the executor's own entry already started inside the stage
        out['feasible'] = 'entered'
    else:
        t = b.env.time_s
        ok = b.start_direct_gstar()
        out['feasible'] = bool(ok)
        if ok:
            plan = b.generator.bridge.plan
            tau, dvp = float(plan['tau_s']), float(plan['delta_v_m_s'])
            pr = ecal.predict(tau, dvp, b.env.config.reward, b.mission['deadline_s'] - t)
            out.update(tau=tau, dv_plan=dvp, p_success=pr['p_success'], entry_pred=pr['reward'], L=r1 + pr['reward'])
        else:
            out['L'] = None
    ret, done, info = r1, False, {}
    if out['feasible'] is False:
        out.update(plan_event=None, plan_true=None)
        return out
    while not done:                                       # fly the executor's plan to the end
        _, _, term, trunc, info = b.step(np.zeros(2)); ret += float(info['true_reward']); done = term or trunc
        if not done and b.generator.mode in ('stage', 'hold'):
            if not b.start_direct_gstar():
                pass                                      # no plan left: the neutral stage flies on, as the gstar row
    out.update(plan_event=info.get('event'), plan_true=ret)
    return out


def job(opening):
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv, T2Mission
    from experiments.t2 import rally_action
    from experiments.t2_audit import branch, load_calibration

    class Probe(T2StageEnv):
        def _plan_candidates(self, q, v, frame, action):
            out = super()._plan_candidates(q, v, frame, action); self._planned = out; return out

        def _labels(self, plan, a, start, cal, ecal, *args, **kw):
            self._ecal_at_decision = ecal
            return super()._labels(plan, a, start, cal, ecal, *args, **kw)

    env = Probe(openings=[opening], seed_base=2_910_000, candidates=8, shaping=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.reset(options={'opening': opening})
    if not env.mission['urgent']:
        return {'opening': opening, 'urgent': False}
    load_calibration(env, CAL)
    recs, done, t0 = [], False, time.time()
    while not done:
        left = env.mission['deadline_s'] - env.env.time_s
        a = rally_action(env, hold_m=7.5)
        critical = env.generator.mode in ('stage', 'hold') and WINDOW[0] <= left <= WINDOW[1]
        snap = branch(env) if critical else None
        env._planned, env._ecal_at_decision = None, None
        t_dec = env.env.time_s
        _, _, term, trunc, info = env.step(a); done = term or trunc
        m = info['model']
        if critical and m['diag']['reason'] in ('stage', 'entry') and env._planned:
            ecal = env._ecal_at_decision
            rows = [{'a': np.asarray(c['a']).tolist(), **opportunity(snap, c['a'], ecal)} for c in env._planned]
            recs.append({'t': t_dec, 'left_s': left, 'cands': rows})
            print(json.dumps({'opening': opening, 't': t_dec, 'left': round(left), 'n': len(rows), 'wall': round(time.time() - t0)}), flush=True)
    return {'opening': opening, 'urgent': True, 'event': info.get('event'), 'decisions': recs, 'wall_s': time.time() - t0}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    ops = [int(f.stem) for f in sorted(HZ.glob('*.json')) if json.loads(f.read_text()).get('urgent')]
    ops = [o for o in ops if not (OUT / f'{o}.json').exists()]
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(job, ops):
            (OUT / f"{r['opening']}.json").write_text(json.dumps(r, default=float), encoding='utf-8')
            print('DONE', r['opening'], r.get('event'), len(r.get('decisions', [])), round(r.get('wall_s', 0)), flush=True)
