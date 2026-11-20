"""Episode runners of the experiments (library; the command-line entry is ``experiments/final.py``).

* ``evaluate``: a trained model (final), deterministic actor (mean action), one result file per opening; the executor
  configuration (coordination, correction, direct entry, pass-window commitment, rotation guidance) is read from the
  run's manifest, with registered evaluation switches applied on top (``env`` overrides).
* ``screen``: fixed non-learning policies: ``pure`` (the frozen MPC alone, no stage layer) and the RWG check pair
  ``c_10_0`` / ``b_10`` (the same intent with and without rotation guidance).

Every result file holds the outcome, true return, time, delta-v, executor statistics and the per-decision log; with
``audit`` also the initial state, the read-only decision diagnostics and the terminal / margin record.
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
                          **({'coord': i['coord']} if i.get('coord') is not None else {}), **i.get('decision_audit', {})})
    s = env.stats
    return {'opening': opening, 'event': i['event'], 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
            'stages': s['stages'], 'holds': s['holds'], 'rejected': s['rejected'], 'bridge_attempts': s['bridge_attempts'],
            'macro': s['macro'], 'mission': env.mission, 'wall_s': time.time() - t0, 'decisions': decisions,
            **({'initial': initial, **env.episode_audit(),
                'coord_wall_s_total': sum(d['coord'].get('wall_s', 0.0) for d in decisions if 'coord' in d),
                'coord_decisions': sum('coord' in d for d in decisions)} if env.audit_logging else {}),
            'corrections': s.get('corrections', 0), 'long_entries': s.get('long_entries', 0),
            'value_arbitrations': s.get('value_arbitrations', 0), 'arbitration_changed': s.get('arbitration_changed', 0)}


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


#: fixed non-learning policies (``docs/final_method/ROUND2.md`` section 2): ``pure`` (the frozen MPC alone),
#: ``c_PP_AA`` (constant intent (PP/10, AA/10) with rotation guidance), ``b_PP`` (constant closing PP/10 along the line of
#: sight, rotation guidance off); no coordination
def fixed_intent(policy: str):
    if policy.startswith('c_'):
        _, p, a = policy.split('_'); return np.array([int(p) / 10.0, int(a) / 10.0]), True
    if policy.startswith('b_'):
        return np.array([int(policy.split('_')[1]) / 10.0, 0.0]), False
    raise ValueError(f'unknown fixed policy {policy}')


def screen_job(job):
    opening, policy, out = job['opening'], job['label'], job['out']
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'{policy}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv
    if policy != 'pure':
        action, geometry = fixed_intent(policy)
        env = T2StageEnv(openings=[opening], shaping=False, geometry=geometry)
        env.audit_logging = bool(job.get('audit', False))
        res = run_episode(env, opening, lambda e, o: action)
        res.update(policy=policy, geometry=geometry, entries_lost=env.stats.get('entries_lost', 0),
                   window_recoveries=env.generator.bridge.recovery_log)
        dump(file, res); return res
    env = T2StageEnv(openings=[opening], shaping=False)
    env.audit_logging = bool(job.get('audit', False))
    # the frozen MPC alone (fixed terminal set-point, no stage layer, no entry planner): same task, deadline, weights,
    # budget and truth judge
    obs, _ = env.reset(options={'opening': opening}); t0 = time.time(); ret = 0.; event = None
    while event is None:
        r, term, trunc, i = env.step_pure(); ret += r; event = i.get('event')
        if event is None and env.env._delta_v >= env.mission['budget_m_s']:
            event = 'resource_exhausted'; ret -= env.env.config.reward.timeout_penalty
    res = {'opening': opening, 'event': event, 'return': ret, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v,
           'stages': 0, 'holds': 0, 'rejected': 0, 'mission': env.mission, 'wall_s': time.time() - t0, 'policy': policy}
    if env.audit_logging:
        res.update(env.episode_audit())
    dump(file, res); return res


def screen(out, ops, policies, workers, audit=False):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(opening=o, label=p, out=str(out), audit=audit) for p in policies for o in ops]
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


#: the executor configuration a run records in its manifest (``train.t2_mve_sac.train``)
EXEC_KEYS = ('geometry', 'window', 'coordination', 'correction', 'long_entry')


def exec_config_of(checkpoint) -> dict:
    """The executor configuration a checkpoint was trained with (evaluation does not store experience, so the
    equivalence switch is not part of it)."""
    d = run_manifest(checkpoint)
    if d.get('obs_dim') is None or 'coordination' not in d:
        raise ValueError(f'{checkpoint}: not a round-2 run (first-draft models load at science 72a8d41)')
    return {**{k: bool(d[k]) for k in EXEC_KEYS}, 'arbitration': d.get('arbitration', 'nearest')}


def attach_ranker(env, model) -> None:
    """Value arbitration ranks with the evaluated model's critic (fixed, so the coordination map is deterministic)."""
    if env.arbitration == 'value':
        from train.t2_mve_sac import critic_ranker
        env.ranker = critic_ranker(model)


def load_checkpoint(checkpoint):
    """Load the training algorithm specified by the checkpoint's run manifest."""
    from train.t2_mve_sac import T2SAC, T2TD3
    return (T2TD3 if run_manifest(checkpoint).get('algo') == 'td3' else T2SAC).load(checkpoint, device='cpu')


def eval_job(job):
    opening, checkpoint, label, out = job['opening'], job['checkpoint'], job['label'], job['out']
    import torch; torch.set_num_threads(1)
    file = Path(out) / f'{label}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv
    model = load_checkpoint(checkpoint)
    cfg = {**exec_config_of(checkpoint), **(job.get('env') or {})}     # registered evaluation switches on top
    env = T2StageEnv(openings=[opening], shaping=False, **cfg)
    env.audit_logging = bool(job.get('audit', False))
    attach_ranker(env, model)
    res = run_episode(env, opening, lambda e, o: model.predict(o, deterministic=True)[0])
    res.update(policy=label, checkpoint=str(checkpoint), exec_config=cfg, entries_lost=env.stats.get('entries_lost', 0),
               window_recoveries=env.generator.bridge.recovery_log); dump(file, res); return res


def evaluate(out, checkpoint, label, ops, workers, env=None, audit=False):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(opening=o, checkpoint=str(checkpoint), label=label, out=str(out), env=env, audit=audit) for o in ops]
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(guarded_eval, j) for j in jobs]):
            r = f.result(); print(json.dumps({k: r.get(k) for k in STAGE_EVAL_KEYS}, default=str), flush=True)


def guarded_screen(job):
    return guarded(screen_job)(job)


def guarded_eval(job):
    return guarded(eval_job)(job)
