"""T2.8 pre-check P4 (read-only, exploratory): which information would correct the critic on the key states?

    python -B evidence/T2_8_PRECHECK/脚本/info_probe.py CAL.json STATES_DIR OUT_DIR LO HI WORKERS

Re-flies the key-state probe's episodes (rule rally, hold 7.5 m, mission profile with urgent deadlines U(140, 200) s,
audit openings 2,910,000 + [LO, HI), the same environment configuration and calibration). Key decisions: urgent
missions as the key-state probe (first two decisions and 40-140 s before the deadline; the observation is checked
against the stored one, ``obs_match``); economy missions (added in P4: the night run's fuel loss came from late entries
on economy missions) the first ECONOMY_DECISIONS stage decisions. The observation is stored for critic scoring
(experiments.t2_keyscore). For every 3 x 3 grid intent it adds:

  truth_track  truth branch: the intent for one stage, then the executor's own capture plan projected onto the stage
               action space (T2StageEnv.track_intent) at every later decision, the intent itself when the executor has
               no plan; to the end of the episode. A near-best continuation (the projection captures 19/19 urgent
               audit missions from the start), so it measures what the intent costs when the rest is flown well,
               unlike the probe's 'hold the intent for 4 stages'.
  opp          executor-only signal, no truth: the stage goal's end state (the planner's own goal, known at the
               decision) and the target attitude 16 s ahead (constant-rate model, estimated inertia) -> the executor's
               earliest feasible entry (T2StageEnv._opportunity); value = -time and delta-v cost to the entry +
               capture reward, or -timeout penalty without one. A ranking signal to be audited, not a bound.

Frozen environment code; the script only reads it. One JSON per opening in OUT_DIR (skips openings already written).
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (run from anywhere)

CAL, STATES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LO, HI, WORKERS = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
WINDOW = (40.0, 140.0)
GRID = [(p, z) for p in (-0.5, 0.25, 1.0) for z in (-1.0, 0.0, 1.0)]
MAX_STEPS = 40
ECONOMY_DECISIONS = 4


def truth_track(snapshot, a):
    from experiments.t2_audit import branch
    b = branch(snapshot)
    _, _, term, trunc, info = b.step(np.asarray(a, dtype=np.float64))
    g, n, tracked, done = info['true_reward'], 1, 0, term or trunc
    while not done and n < MAX_STEPS:
        t = b.track_intent()
        tracked += int(t is not None)
        _, _, term, trunc, info = b.step(np.asarray(a if t is None else t, dtype=np.float64))
        g += info['true_reward']; n += 1; done = term or trunc
    return {'true': float(g), 'event': info.get('event'), 'terminal': bool(done), 'time_s': float(b.env.time_s),
            'delta_v': float(b.env._delta_v), 'decisions': n, 'tracked': tracked}


def opp_signal(env, a):
    from env.t1_gstar import free_body_rotations
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import STAGE_S, frame_now, stage_goal
    e = env.env; w = e.config.reward
    q, v = relative_inertial(e)
    goal = stage_goal(q, v, frame_now(e, q), np.asarray(a, float), e.chaser_parameters.mass, geometry=env.geometry, task=e.config.task)
    _, rots, rates = free_body_rotations(e.target_state.rotation, e.target_state.omega,
                                         np.asarray(env.info_env.target_parameters.inertia), STAGE_S)
    opp = env._opportunity(goal['q_goal'], goal['v_goal'], e.time_s + STAGE_S, rots[-1], rates[-1])
    if opp is None:
        return {'feasible': False, 'value': float(-w.timeout_penalty)}
    s = STAGE_S + opp['tau']
    return {'feasible': True, 'tau': opp['tau'], 'dv_plan': opp['dv_plan'],
            'value': float(-w.time_cost_per_s * s - w.delta_v_cost_per_m_s * opp['dv_plan'] + w.capture_reward)}


def job(opening):
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2Mission, T2StageEnv
    from experiments.t2 import rally_action
    from experiments.t2_audit import branch, load_calibration
    stored = {}
    f = STATES / f'{opening}.json'
    if f.exists():
        for d in json.loads(f.read_text(encoding='utf-8')).get('decisions', []):
            stored[round(d['t'], 3)] = d['obs']
    env = T2StageEnv(openings=[opening], seed_base=2_910_000, candidates=8, shaping=True, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    env.reset(options={'opening': opening})
    urgent = bool(env.mission['urgent'])
    load_calibration(env, CAL)
    recs, done, t0, n_dec, info = [], False, time.time(), 0, {}
    while not done:
        left = env.mission['deadline_s'] - env.env.time_s
        a = rally_action(env, hold_m=7.5)
        stage = env.generator.mode in ('stage', 'hold')
        critical = stage and ((WINDOW[0] <= left <= WINDOW[1] or n_dec < 2) if urgent else n_dec < ECONOMY_DECISIONS)
        if critical:
            n_dec += 1
            obs = env.observation().tolist(); ref = stored.get(round(env.env.time_s, 3))
            snap = branch(env)
            rows = [{'a': list(g), 'truth_track': truth_track(snap, g), 'opp': opp_signal(env, g)} for g in GRID]
            recs.append({'t': env.env.time_s, 'left_s': left, 'urgent': urgent, 'obs': obs,
                         'obs_match': (ref is not None and np.array_equal(np.asarray(ref), np.asarray(obs))) if urgent else None,
                         'cands': rows})
            print(json.dumps({'opening': opening, 't': env.env.time_s, 'left': round(left), 'match': recs[-1]['obs_match'],
                              'wall': round(time.time() - t0)}), flush=True)
        _, _, term, trunc, info = env.step(a); done = term or trunc
    return {'opening': opening, 'urgent': urgent, 'mission': env.mission, 'event': info.get('event'), 'decisions': recs, 'wall_s': time.time() - t0}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [o for o in range(2_910_000 + LO, 2_910_000 + HI) if not (OUT / f'{o}.json').exists()]
    with ProcessPoolExecutor(WORKERS) as pool:
        for r in pool.map(job, todo):
            (OUT / f"{r['opening']}.json").write_text(json.dumps(r, default=float), encoding='utf-8')
            print('DONE', r['opening'], r.get('urgent'), r.get('event'), len(r.get('decisions', [])), round(r.get('wall_s', 0)), flush=True)
