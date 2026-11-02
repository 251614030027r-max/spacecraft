"""T2.8 P4 addendum (read-only, exploratory; written 2026-11-02 before any P4 result was seen): the executor-calibrated
opportunity signal 'opp2' on the P4 key decisions.

    python -B evidence/T2_8_PRECHECK/脚本/opp2_probe.py CAL.json OUT_DIR LO HI WORKERS

Why a second signal: P4's 'opp' plans entries only up to deadline - 45 s (the conditional G*'s planning margin), so
it reports 'no opportunity' for every intent in the last ~100 s while the executor's actual entry (the bridge) has no
such margin and often still captures. 'opp2' asks the executor the question its own entry would answer:

  from the intent's stage goal (end state of the planner's own goal, attitude 16 s ahead from the constant-rate model,
  estimated inertia) the earliest feasible conditional G* entry with arrival tau <= the time left (no margin); its value
  is the entry outcome model's prediction from historical execution replay (EntryCalibration.predict: the recorded
  entries' realized deviations replayed on this plan and remaining time -> capture or timeout, time and delta-v), less
  the stage's time cost; without a feasible entry: -timeout penalty - time cost of the time left.

Re-flies the P4 episodes exactly as info_probe.py (rule rally, same environment, calibration and key-decision rule)
and stores the observation so p4b.py can check it equals P4's on the same machine. Frozen environment code.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (run from anywhere)
CAL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
LO, HI, WORKERS = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
WINDOW = (40.0, 140.0)
GRID = [(p, z) for p in (-0.5, 0.25, 1.0) for z in (-1.0, 0.0, 1.0)]
ECONOMY_DECISIONS = 4


def opp2_signal(env, a):
    from env.t1_gstar import free_body_rotations
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import OPPORTUNITY_TAUS_S, STAGE_S, frame_now, stage_goal
    e = env.env; w = e.config.reward
    q, v = relative_inertial(e)
    goal = stage_goal(q, v, frame_now(e, q), np.asarray(a, float), e.chaser_parameters.mass, geometry=env.geometry, task=e.config.task)
    remaining = float(env.mission['deadline_s']) - (e.time_s + STAGE_S)
    taus = [t for t in OPPORTUNITY_TAUS_S if t <= remaining]
    if not taus:
        return {'feasible': False, 'value': float(-w.timeout_penalty - w.time_cost_per_s * max(remaining + STAGE_S, 0.0))}
    inertia = np.asarray(env.info_env.target_parameters.inertia)
    _, rots, rates = free_body_rotations(e.target_state.rotation, e.target_state.omega, inertia, STAGE_S)
    history = free_body_rotations(rots[-1], rates[-1], inertia, max(taus) + 1.0)
    g = env.generator.bridge
    for tau in taus:
        res = g.planner.solve(np.asarray(goal['q_goal'], float), np.asarray(goal['v_goal'], float), tau, history, None,
                              entry_speed_m_s=g.entry_speed_m_s)
        if res['feasible']:
            pr = env.entry_calibration.frozen().predict(float(res['tau_s']), float(res['delta_v_m_s']), w, remaining)
            return {'feasible': True, 'tau': float(res['tau_s']), 'dv_plan': float(res['delta_v_m_s']), 'p_success': pr['p_success'],
                    'value': float(pr['reward'] - w.time_cost_per_s * STAGE_S)}
    return {'feasible': False, 'value': float(-w.timeout_penalty - w.time_cost_per_s * (remaining + STAGE_S))}


def job(opening):
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2Mission, T2StageEnv
    from experiments.t2 import rally_action
    from experiments.t2_audit import load_calibration
    env = T2StageEnv(openings=[opening], seed_base=2_910_000, candidates=8, shaping=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.reset(options={'opening': opening})
    urgent = bool(env.mission['urgent'])
    load_calibration(env, CAL)
    recs, done, t0, n_dec = [], False, time.time(), 0
    while not done:
        left = env.mission['deadline_s'] - env.env.time_s
        a = rally_action(env, hold_m=7.5)
        stage = env.generator.mode in ('stage', 'hold')
        if stage and ((WINDOW[0] <= left <= WINDOW[1] or n_dec < 2) if urgent else n_dec < ECONOMY_DECISIONS):
            n_dec += 1
            recs.append({'t': env.env.time_s, 'left_s': left, 'urgent': urgent, 'obs': env.observation().tolist(),
                         'cands': [{'a': list(gp), 'opp2': opp2_signal(env, gp)} for gp in GRID]})
        _, _, term, trunc, _ = env.step(a); done = term or trunc
    return {'opening': opening, 'urgent': urgent, 'decisions': recs, 'wall_s': time.time() - t0}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [o for o in range(2_910_000 + LO, 2_910_000 + HI) if not (OUT / f'{o}.json').exists()]
    with ProcessPoolExecutor(WORKERS) as pool:
        for r in pool.map(job, todo):
            (OUT / f"{r['opening']}.json").write_text(json.dumps(r, default=float), encoding='utf-8')
            print('DONE', r['opening'], r['urgent'], len(r['decisions']), round(r['wall_s']), flush=True)
