"""Episode runners of the formal experiments (library; the command-line entry is ``experiments/final.py``).

* ``evaluate``: a trained checkpoint, deterministic actor (mean action), one result file per opening; the executor
  configuration (pass-window commitment, rotation guidance, certification) is read from the run's manifest, with the
  registered evaluation switches (rigid entry commitment, certification forced on / off).
* ``screen``: a fixed policy -- ``pure`` (the frozen MPC alone, no stage layer), ``gstar`` (direct conditional G* from
  the far field, outside the action space), ``rally`` (deadline-aware rule inside the action space), ``c_PP_AA``
  (constant intent a = (PP/10, AA/10)), ``b_PP`` (rotation-unaware closing at a_p = PP/10, no rotation guidance).

Every result file holds the outcome, true return, time, delta-v, executor statistics and the per-decision log; with
``audit`` (formal runs) also the initial state, the read-only decision diagnostics and the terminal / margin record.
An episode that raises is recorded as ``ERROR_<label>_<opening>.json`` (event 'error') and the pool goes on.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

STAGE_EVAL_KEYS = ('opening', 'policy', 'event', 'return', 'time_s', 'delta_v', 'stages', 'holds', 'rejected', 'mission')


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
    initial = {}
    if env.audit_logging:
        from env.t1_p_stage import relative_inertial
        q, v = relative_inertial(env.env)
        target = env.env.target_state
        initial = {'q': q.tolist(), 'v': v.tolist(), 'rotation': target.rotation.tolist(),
                   'omega': target.omega.tolist(), 'axis': list(env.env.config.task.approach_axis)}
    while not done:
        a = act(env, obs)
        obs, r, term, trunc, i = env.step(a); done = term or trunc; ret += i['true_reward']
        decisions.append({'a': np.asarray(a).tolist(), 't': i['time_s'], 'mode': i['mode'], 'held': i['held'], 'range_m': i['range_m'],
                          'planned_dv': i['planned_dv'], 'executed_dv': i['executed_dv'], 'event': i['event'],
                          **({'cert': i['cert']} if i.get('cert') is not None else {}), **i.get('decision_audit', {})})
    s = env.stats
    return {'opening': opening, 'event': i['event'], 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
            'stages': s['stages'], 'holds': s['holds'], 'rejected': s['rejected'], 'bridge_attempts': s['bridge_attempts'],
            'macro': s['macro'], 'mission': env.mission, 'wall_s': time.time() - t0, 'decisions': decisions,
            **({'initial': initial, **env.episode_audit(),
                'cert_wall_s_total': sum(d['cert']['wall_s'] for d in decisions if 'cert' in d),
                'cert_decisions': sum('cert' in d for d in decisions)} if env.audit_logging else {})}


def guarded(fn):
    """A failing episode is recorded (event 'error'; it blocks the readout) and the pool goes on."""
    def run(job):
        try:
            return fn(job)
        except Exception as error:  # noqa: BLE001 -- recorded, not swallowed
            import traceback
            opening, name, out = job['opening'], job['label'], job['out']
            res = {'opening': opening, 'policy_or_label': name, 'event': 'error', 'error': repr(error), 'traceback': traceback.format_exc()}
            dump(Path(out) / f'ERROR_{name}_{opening}.json', res); return res
    return run


def screen_job(job):
    opening, policy, out, window, mission = job['opening'], job['label'], job['out'], job['window'], job['mission']
    audit, condition = job.get('audit', False), job.get('condition')
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'{policy}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv, mission_from_name
    env = T2StageEnv(openings=[opening], shaping=False, geometry=not policy.startswith('b_'), window=window, mission=mission_from_name(mission))
    env.audit_logging = audit
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
                env.start_direct_gstar()      # entry lost: G* re-plans from the far field (else the step flies a neutral stage)
        res = {'opening': opening, 'event': i['event'], 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
               'stages': env.stats['stages'], 'holds': env.stats['holds'], 'rejected': env.stats['rejected'], 'mission': env.mission, 'wall_s': time.time() - t0}
    else:
        res = run_episode(env, opening, lambda e, o: policy_action(policy, e, o))
    if audit:
        res.update(env.episode_audit(), condition=condition)
    res['policy'] = policy; res['window'] = window; res['entries_lost'] = env.stats.get('entries_lost', 0) if env.stats else 0
    res['window_recoveries'] = env.generator.bridge.recovery_log if env.generator is not None else []
    dump(file, res); return res


def screen(out, ops, policies, workers, window=True, mission='natural', audit=False, condition=None):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(opening=o, label=p, out=str(out), window=window, mission=mission, audit=audit, condition=condition)
            for p in policies for o in ops]
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(guarded_screen, j) for j in jobs]):
            r = f.result(); print(json.dumps({k: r.get(k) for k in STAGE_EVAL_KEYS}, default=str), flush=True)


def run_manifest(checkpoint) -> dict:
    """The manifest of the run a checkpoint belongs to (RUN/checkpoints/step_N.zip or RUN/final.zip)."""
    c = Path(checkpoint)
    for m in (c.parent.parent / 'manifest.json', c.parent / 'manifest.json'):
        if m.exists():
            return json.loads(m.read_text(encoding='utf-8'))
    raise FileNotFoundError(f'no manifest.json beside {checkpoint}: the decision interface of the checkpoint is unknown')


def exec_config_of(checkpoint) -> dict:
    """The executor configuration a checkpoint was trained with (pass-window commitment, rotation guidance, certification)."""
    d = run_manifest(checkpoint)
    if d.get('obs_profile') != 'cut_priv':
        raise ValueError(f'{checkpoint}: interface {d.get("obs_profile")!r} is not the formal interface cut_priv')
    return {'window': bool(d.get('window', True)), 'geometry': bool(d.get('geometry', True)), 'certify': bool(d.get('certify', False))}


def load_checkpoint(checkpoint):
    """Load the training algorithm specified by the checkpoint's run manifest."""
    from train.t2_mve_sac import T2SAC, T2TD3
    return (T2TD3 if run_manifest(checkpoint).get('arm') == 'e_td3' else T2SAC).load(checkpoint, device='cpu')


def eval_job(job):
    opening, checkpoint, label, out = job['opening'], job['checkpoint'], job['label'], job['out']
    window, mission, certify = job.get('window'), job['mission'], job.get('certify')
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'{label}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv, mission_from_name
    model = load_checkpoint(checkpoint)
    cfg = exec_config_of(checkpoint)
    window = cfg['window'] if window is None else bool(window)     # None: as trained; False: forced rigid (paired ablation)
    certify = cfg['certify'] if certify is None else bool(certify)    # None: as trained; True / False: forced
    env = T2StageEnv(openings=[opening], shaping=False, window=window, geometry=cfg['geometry'], mission=mission_from_name(mission),
                     certify=certify)
    env.audit_logging = bool(job.get('audit', False))
    res = run_episode(env, opening, lambda e, o: model.predict(o, deterministic=True)[0])
    if env.audit_logging:
        res['condition'] = job.get('condition') or ('urgent' if env.mission['urgent'] else 'economy')
    res['certify'] = certify; res['cert_interventions'] = env.stats.get('cert_interventions', 0)
    res.update(policy=label, checkpoint=str(checkpoint), window=window, geometry=cfg['geometry'], entries_lost=env.stats.get('entries_lost', 0),
               window_recoveries=env.generator.bridge.recovery_log); dump(file, res); return res


def evaluate(out, checkpoint, label, ops, workers, window=None, mission='natural', certify=None, audit=False, condition=None):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(opening=o, checkpoint=str(checkpoint), label=label, out=str(out), window=window, mission=mission, certify=certify,
                 audit=audit, condition=condition) for o in ops]
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(guarded_eval, j) for j in jobs]):
            r = f.result(); print(json.dumps({k: r.get(k) for k in STAGE_EVAL_KEYS}, default=str), flush=True)


def guarded_screen(job):
    return guarded(screen_job)(job)


def guarded_eval(job):
    return guarded(eval_job)(job)
