"""T2 staged-approach driver (env/t2_stage.py, train/t2_mve_sac.py).

    python -B -m experiments.t2 screen --openings A-B --policies coast,c_06_00,...,rally,gstar --workers W --output DIR
    python -B -m experiments.t2 train  --arm {e_sac,e_mve} --seed S --steps N --workers W --output DIR [--smoke]
    python -B -m experiments.t2 eval   --checkpoint ZIP --label L --openings A-B --workers W --output DIR

``--no-window`` (all modes): the T2.2 rigid entry commitment instead of the pass-window commitment (ablation).

Fixed policies (screen): ``coast`` (a = 0), ``c_PP_AA`` constant action (a_p = PP/10, a_a = AA/10),
``rally`` (deadline-aware rule inside the action space: close to 9.5 m, swing towards the oncoming port only when waiting would miss the deadline),
``gstar`` (reference outside the action space: conditional G* from the far field, earliest feasible),
``pure`` (the shared MPC alone: fixed terminal set-point from the start, no stage layer).
"""
import argparse, hashlib, json, os, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

STAGE_EVAL_KEYS = ('opening', 'policy', 'event', 'return', 'time_s', 'delta_v', 'stages', 'holds', 'rejected', 'mission')


def openings(text):
    a, _, b = text.partition('-'); return list(range(int(a), int(b or a) + 1))


def dump(path, x):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, indent=1, default=lambda z: z.tolist() if isinstance(z, np.ndarray) else (z.item() if isinstance(z, np.generic) else str(z))), encoding='utf-8')
    tmp.replace(path)


def rally_action(env, hold_m=9.5):
    """Fixed deadline-aware rule inside the action space: close at 0.04 (r - hold) m/s (-0.05..0.20);
    swing towards the oncoming port at full speed when waiting for its pass would not leave the
    bridge and descent (~90 s) before the deadline, otherwise wait."""
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import action_for_velocity, frame_now
    q, _ = relative_inertial(env.env)
    f = frame_now(env.env, q)
    v_c = float(np.clip(0.04 * (f["r"] - hold_m), -0.05, 0.20))
    late = f["time_to_pass_s"] + max(f["r"] - hold_m, 0.0) / 0.1 + 90.0 > env.mission["deadline_s"] - env.env.time_s
    return action_for_velocity(f, v_c, 0.15 if late else 0.0)


def policy_action(name, env, obs):
    if name == 'coast': return np.zeros(2)
    if name == 'rally': return rally_action(env)
    if name.startswith('c_'):
        _, p, a = name.split('_'); return np.array([int(p) / 10.0, int(a) / 10.0])
    if name.startswith('b_'):          # basic layering: rotation-unaware closing (env built with geometry=False)
        return np.array([int(name.split('_')[1]) / 10.0, 0.0])
    raise ValueError(name)


def run_episode(env, opening, act, record=None):
    obs, info = env.reset(options={'opening': opening}); done = False; ret = 0.; t0 = time.time(); decisions = []
    while not done:
        a = act(env, obs)
        obs, r, term, trunc, i = env.step(a); done = term or trunc; ret += i['true_reward']
        decisions.append({'a': np.asarray(a).tolist(), 't': i['time_s'], 'mode': i['mode'], 'held': i['held'], 'range_m': i['range_m'],
                          'planned_dv': i['planned_dv'], 'executed_dv': i['executed_dv'], 'model': (i.get('model') or {}).get('diag', {}).get('reason'), 'event': i['event']})
    s = env.stats
    return {'opening': opening, 'event': i['event'], 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
            'stages': s['stages'], 'holds': s['holds'], 'rejected': s['rejected'], 'bridge_attempts': s['bridge_attempts'],
            'macro': s['macro'], 'mission': env.mission, 'wall_s': time.time() - t0, 'decisions': decisions}


def guarded(fn):
    """A failing episode is recorded (event 'error', excluded by the readout) and the pool goes on."""
    def run(job):
        try:
            return fn(job)
        except Exception as error:  # noqa: BLE001 -- recorded, not swallowed
            import traceback
            opening, name, out = job[0], (job[1] if fn is screen_job else job[2]), (job[2] if fn is screen_job else job[3])
            res = {'opening': opening, 'policy_or_label': name, 'event': 'error', 'error': repr(error), 'traceback': traceback.format_exc()}
            dump(Path(out) / f'ERROR_{name}_{opening}.json', res); return res
    return run


def screen_job(job):
    opening, policy, out, window = job
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'{policy}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[opening], shaping=False, geometry=not policy.startswith('b_'), window=window)
    if policy == 'pure':
        # paper baseline: the shared MPC alone (fixed terminal set-point, no stage layer, no G*), same
        # mission, deadline, weights, budget and truth judge
        obs, _ = env.reset(options={'opening': opening}); t0 = time.time(); ret = 0.; event = None
        while event is None:
            r, term, trunc, i = env.step_pure(); ret += r; event = i.get('event')
            if event is None and env.env._delta_v >= env.mission['budget_m_s']:
                event = 'resource_exhausted'; ret -= env.env.config.reward.timeout_penalty
        res = {'opening': opening, 'event': event, 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
               'stages': 0, 'holds': 0, 'rejected': 0, 'mission': env.mission, 'wall_s': time.time() - t0}
    elif policy == 'gstar':
        obs, _ = env.reset(options={'opening': opening}); t0 = time.time()
        ok = env.start_direct_gstar(); done = not ok; ret = 0.; i = {'event': 'no_plan'}
        while not done:
            obs, r, term, trunc, i = env.step(np.zeros(2)); done = term or trunc; ret += i['true_reward']
            if not done and env.generator.mode in ('stage', 'hold'):
                env.start_direct_gstar()      # bridge lost: G* re-plans from the far field (else the step flies a neutral stage)
        res = {'opening': opening, 'event': i['event'], 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
               'stages': env.stats['stages'], 'holds': env.stats['holds'], 'rejected': env.stats['rejected'], 'mission': env.mission, 'wall_s': time.time() - t0}
    else:
        res = run_episode(env, opening, lambda e, o: policy_action(policy, e, o))
    res['policy'] = policy; res['window'] = window; res['entries_lost'] = env.stats.get('entries_lost', 0) if env.stats else 0
    res['window_recoveries'] = env.generator.bridge.recovery_log if env.generator is not None else []
    dump(file, res); return res


def screen(out, ops, policies, workers, window=True):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(guarded_screen, (o, p, str(out), window)) for p in policies for o in ops]):
            r = f.result(); print(json.dumps({k: r.get(k) for k in STAGE_EVAL_KEYS}, default=str), flush=True)


def eval_job(job):
    opening, checkpoint, label, out, window = job
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'{label}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv
    from train.t2_mve_sac import T2SAC
    model = T2SAC.load(checkpoint, device='cpu')
    env = T2StageEnv(openings=[opening], shaping=False, window=window)
    res = run_episode(env, opening, lambda e, o: model.predict(o, deterministic=True)[0])
    res.update(policy=label, checkpoint=str(checkpoint), window=window, entries_lost=env.stats.get('entries_lost', 0),
               window_recoveries=env.generator.bridge.recovery_log); dump(file, res); return res


def evaluate(out, checkpoint, label, ops, workers, window=True):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(guarded_eval, (o, checkpoint, label, str(out), window)) for o in ops]):
            r = f.result(); print(json.dumps({k: r.get(k) for k in STAGE_EVAL_KEYS}, default=str), flush=True)


def guarded_screen(job):
    return guarded(screen_job)(job)


def guarded_eval(job):
    return guarded(eval_job)(job)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['screen', 'train', 'eval'])
    p.add_argument('--output', required=True); p.add_argument('--openings'); p.add_argument('--policies')
    p.add_argument('--workers', type=int, default=4); p.add_argument('--arm', choices=['e_sac', 'e_mve'], default='e_sac')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--checkpoint'); p.add_argument('--label'); p.add_argument('--checkpoints', default='500,1000,1500,2000,3000,4000,5000')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--no-window', action='store_true')
    p.add_argument('--entry-stages', type=int, default=2); p.add_argument('--updates-per-transition', type=float, default=2.0)
    p.add_argument('--model-beta', type=float, default=0.5)
    a = p.parse_args(); out = Path(a.output)
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(k, '1')
    window = not a.no_window
    if a.mode == 'screen': screen(out, openings(a.openings), a.policies.split(','), a.workers, window)
    elif a.mode == 'eval': evaluate(out, a.checkpoint, a.label, openings(a.openings), a.workers, window)
    else:
        from train.t2_mve_sac import train
        train(out, a.arm, a.seed, a.steps, a.workers, [int(x) for x in a.checkpoints.split(',')], a.smoke,
              updates_per_transition=a.updates_per_transition, entry_stages=a.entry_stages, window=window, model_beta=a.model_beta)


if __name__ == '__main__':
    main()
