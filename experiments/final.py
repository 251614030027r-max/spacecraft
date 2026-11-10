"""Final experiments (EACH, ECC; env 0.17.0: the ECC final round, ``docs/final_method/ECC_FINAL_SPEC.md``): the only entry
for the pretest, the development rounds and the formal experiments.

Every configuration, opening block, seed list and budget comes from ``docs/final_method/METHOD_REGISTRY.json``; the
historical arms (counterfactual labels, opportunity maps, actor feedback) cannot be launched from here.

    python -B -m experiments.final train    --stage {pretest,formal} --config NAME --seed S --output DIR [--smoke-steps N]
    python -B -m experiments.final eval     --stage S --run DIR --label L --output DIR [--rigid] [--workers W] [--openings A-B]
    python -B -m experiments.final screen   --stage S --policy NAME --output DIR [--rigid] [--workers W] [--openings A-B]
    python -B -m experiments.final traj     --entry NAME=RUN_DIR|NAME=POLICY --openings A,B --output DIR [--rigid]
    python -B -m experiments.final geoscan  --openings A,B --output DIR [--ranges 8,10,12]
    python -B -m experiments.final resolve  --stage S --root DIR --output REGISTRY_RESOLVED.json
    python -B -m experiments.final check    --stage S --root DIR

``train`` runs ``train.t2_mve_sac.train`` with the registry's arm, observation, window and geometry for NAME, the stage's
budget, checkpoints and urgent-deadline range; a seed outside the stage's list is refused (``--smoke-steps`` runs a short
check into a smoke directory and lifts that rule). ``eval`` evaluates a run's budget checkpoint deterministically on the
stage's evaluation openings under the executor configuration the run was trained with (``--rigid``: forced rigid entry
commitment, the paired window ablation). ``screen`` flies a registered fixed policy. ``traj`` re-flies episodes serially and
records the truth every 2 s (figure B). ``geoscan`` scans the sweep-cone entry geometry and the conditional G* feasibility
(figure A; no training, no evaluation returns). ``resolve`` writes the registry resolved with the science SHA of every run,
model SHA-256 and data locations, refusing mixed SHAs, budgets or configurations; ``check`` reports completeness.
"""
import argparse, hashlib, json, os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'docs' / 'final_method' / 'METHOD_REGISTRY.json'


def registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding='utf-8'))


def block(text: str) -> list[int]:
    if ',' in text:
        return [int(x) for x in text.split(',')]
    a, _, b = text.partition('-')
    return list(range(int(a), int(b or a) + 1))


def stage_of(name: str) -> dict:
    s = registry()['protocol']['stages']
    if name not in s:
        raise SystemExit(f'unknown stage {name}: {sorted(s)}')
    if s[name].get('obsolete'):
        raise SystemExit(f'obsolete stage {name}; see docs/paper/EXPERIMENTS.md')
    return s[name]


def stage_mission(st: dict, natural: bool = False, condition: str | None = None):
    """env 0.17.0: the mission of a stage -- training: ``st['mission']`` (e.g. 'urgent_mix:0.75'); evaluation: the mission of
    ``condition`` in ``st['conditions']`` ('urgent' / 'economy' / 'natural'); ``natural`` or no entry: the natural urgent-deadline
    range of the protocol."""
    if condition is not None and st.get('conditions'):
        return st['conditions'][condition]['mission']
    if st.get('mission') and not natural:
        return st['mission']
    return tuple(registry()['protocol']['urgent_deadline_s'])


def condition_openings(st: dict, condition: str | None) -> str:
    if condition is not None and st.get('conditions'):
        return st['conditions'][condition]['openings']
    return st['eval_openings']


def training_openings(seed: int) -> range:
    """Training openings of a seed (``train.t2_mve_sac.worker``: 1,000,000 (seed + 1) + 100,000 rank + episode)."""
    return range(1_000_000 * (seed + 1), 1_000_000 * (seed + 2))


def _forbidden_eval(stage: str) -> set:
    """Openings an evaluation of ``stage`` may not use: the development block and the other stage's evaluation block."""
    reg = registry(); p = reg['protocol']
    bad = set(block(p['used_before']['dev_eval']))
    for k, v in p['stages'].items():
        if k != stage and not v.get('obsolete'):
            bad |= set(block(v['eval_openings']))
            for c in (v.get('conditions') or {}).values():
                bad |= set(block(c['openings']))
    return bad


# ---------------------------------------------------------------------- train / eval / screen
def cmd_train(a):
    reg = registry(); st = stage_of(a.stage)
    if a.config not in reg['trained']:
        raise SystemExit(f'{a.config} is not a registered trained configuration: {sorted(reg["trained"])}')
    c = reg['trained'][a.config]
    if a.smoke_steps is None:
        if a.seed not in st['seeds']:
            raise SystemExit(f'seed {a.seed} is not a {a.stage} seed {st["seeds"]}')
        if a.config not in st['trained'] and a.config not in (st.get('alternatives') or {}).values():
            raise SystemExit(f'{a.config} is not trained in the {a.stage} stage ({st["trained"]}); use --smoke-steps for a configuration check')
    steps = int(a.smoke_steps) if a.smoke_steps is not None else int(st['steps'])
    marks = [m for m in st['checkpoints'] if m < steps] if a.smoke_steps is None else [max(steps // 2, 1)]
    from train.t2_mve_sac import train
    out = Path(a.output)
    smoke = a.smoke_steps is not None or bool(st.get('smoke', False))       # dryrun stage: smoke learner settings
    train(out, c['arm'], int(a.seed), steps, int(a.workers or reg['protocol']['workers']), marks, smoke,
          updates_per_transition=float(reg['protocol']['updates_per_transition']), window=bool(c['window']),
          urgent_deadline=stage_mission(st), obs_profile=c['obs'], geometry=bool(c['geometry']), certify=bool(c.get('certify', False)),
          govern=c.get('govern'),
          manifest_extra={'final_config': a.config, 'final_stage': a.stage,
                          'final_registry_sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest()})
    verify_manifest(json.loads((out / 'manifest.json').read_text(encoding='utf-8')), a.config)


def verify_manifest(m: dict, config: str) -> None:
    """A run's manifest against its registered configuration (and the main method's switches)."""
    reg = registry(); c = reg['trained'][config]
    want = {'arm': c['arm'], 'obs_profile': c['obs'], 'window': c['window'], 'geometry': c['geometry'], 'actor_view': c['actor_view'],
            'critic_view': c['critic_view']}
    want.update({k: v for k, v in reg['main_switches'].items() if k in m})
    if bool(m.get('certify', False)) != bool(c.get('certify', False)):
        want['certify'] = bool(c.get('certify', False))
    if m.get('govern') != c.get('govern'):
        want['govern'] = c.get('govern')
    bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if m.get('k_candidates', 0) != 0 or (m.get('model_beta') or 0.0) != 0.0:
        bad['labels'] = (m.get('k_candidates'), m.get('model_beta'))
    if bad:
        raise SystemExit(f'manifest of {config} does not match the registry: {bad}')


def budget_checkpoint(run: Path, steps: int) -> Path:
    for p in (run / 'checkpoints' / f'step_{steps}.zip', run / 'final.zip'):
        if p.exists():
            return p
    raise SystemExit(f'no checkpoint at the budget {steps} in {run}')


FORMAL_CONDITIONS = ('urgent', 'economy', 'natural', 'resource', 'validation')


def registered_condition(st, name):
    if name == 'validation':
        return st.get('validation')
    return st.get('conditions', {}).get(name)


def checked_cases(a, st, condition, smoke):
    from math import isfinite

    registered = registered_condition(st, condition)
    if condition in ('resource', 'validation') and (a.stage != 'formal' or registered is None):
        raise SystemExit(f'{condition} is not registered for {a.stage}')
    budget = getattr(a, 'budget', None)
    if budget is not None:
        if not isfinite(budget) or budget <= 0:
            raise SystemExit('--budget must be finite and positive')
        if a.stage == 'formal_prescreen':
            if budget not in st['budgets_m_s']:
                raise SystemExit('unregistered prescreen budget')
        elif not smoke or not a.openings or condition == 'resource':
            raise SystemExit('--budget requires formal_prescreen or a development smoke')
    if a.stage == 'formal_prescreen' and budget is None:
        raise SystemExit('prescreen requires a registered --budget')
    if condition == 'resource':
        budget = registered['budget_m_s']
        if budget is None:
            raise SystemExit('the resource budget has not been frozen')
    if smoke and a.stage == 'formal' and not a.openings:
        raise SystemExit('formal smoke requires explicit development openings')
    default = registered['openings'] if registered else st['eval_openings']
    ops = block(a.openings or default)
    if not ops or len(set(ops)) != len(ops):
        raise SystemExit('openings must be nonempty and unique')
    if smoke and any(o >= 300000 for o in ops):
        raise SystemExit('smoke must stay outside the formal evaluation block')
    if smoke and a.stage == 'formal':
        allowed = set(block('294300-294301')) | set(block(st['validation']['openings']))
        if not set(ops) <= allowed:
            raise SystemExit('openings not registered for the formal preparation smoke')
    else:
        if (a.stage in ('formal', 'formal_prescreen') and not set(ops) <= set(block(default))
                or set(ops) & _forbidden_eval(a.stage)):
            raise SystemExit('openings outside the registered evaluation block or overlapping a reserved block')
    mission = registered['mission'] if registered else (
        st['mission'] if a.stage == 'formal_prescreen' else stage_mission(
            st, natural=bool(a.natural), condition=condition))
    if budget is not None:
        from dataclasses import replace
        from experiments.t2 import mission_profile

        mission = replace(mission_profile(mission), budget_m_s=budget)
    return ops, mission


def verify_existing(output, manifest, openings, label):
    """Check all result files that the evaluator would otherwise silently skip."""
    hashes = {}
    for line in Path(manifest).read_text(encoding='utf-8-sig').splitlines():
        digest, relative = line.split(maxsplit=1)
        key = Path(relative).as_posix()
        if key in hashes or len(digest) != 64:
            raise SystemExit(f'invalid SHA256 row: {line}')
        hashes[key] = digest.lower()
    for opening in openings:
        filename = f'{label}_{opening}.json'
        path = output / filename
        if not path.is_file():
            continue
        keys = (filename, f'{output.name}/{filename}', f'{output.parent.name}/{output.name}/{filename}')
        matches = {hashes[k] for k in keys if k in hashes}
        if matches != {hashlib.sha256(path.read_bytes()).hexdigest()}:
            raise SystemExit(f'existing result SHA256 mismatch or missing: {path}')


def result_directory(output, kind, label):
    path = Path(output)
    return path if path.name == label and path.parent.name == kind else path / kind / label


def cmd_eval(a):
    reg = registry(); st = stage_of(a.stage)
    run = Path(a.run); m = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if m.get('final_config') not in reg['trained']:
        raise SystemExit(f'{run} was not trained through experiments.final')
    cond = getattr(a, 'condition', None) or ('urgent' if st.get('conditions') else None)
    if getattr(a, 'natural', False) and st.get('conditions'):
        cond = 'natural'
    smoke = bool(m.get('smoke'))
    if a.stage == 'formal_prescreen':
        raise SystemExit('formal_prescreen admits fixed policies only')
    ops, mission = checked_cases(a, st, cond, smoke)
    steps = int(m['steps'])
    step = getattr(a, 'checkpoint_step', None)
    if step is not None and cond != 'validation':
        raise SystemExit('--checkpoint-step requires validation')
    output, label = Path(a.output), a.label
    if a.stage == 'formal':
        config, seed = m['final_config'], m['seed']
        if config not in st['trained'] or (not smoke and (steps != st['steps'] or seed not in st['seeds'])):
            raise SystemExit('model is not registered for formal evaluation')
        switches = [name for name, on in (('rigid', a.rigid), ('cert', a.certify), ('nocert', a.no_certify)) if on]
        if len(switches) > 1 or getattr(a, 'govern', None) is not None:
            raise SystemExit('unsupported formal execution switch combination')
        switch = switches[0] if switches else None
        if switch and cond not in st['eval_switches'].get(config, {}).get(switch, ()):
            raise SystemExit('unregistered formal execution switch')
        if cond == 'validation':
            if switch or config not in st['validation']['trained']:
                raise SystemExit('unregistered validation configuration')
            allowed = [max(1, steps // 2), steps] if smoke else [*st['checkpoints'], steps]
            step = steps if step is None else step
            if step not in allowed:
                raise SystemExit('unregistered validation checkpoint')
            label = f'{config}_s{seed}_step{step}'
            output = result_directory(output, 'V', label)
        else:
            label = f'{config}_s{seed}_{cond}' + (f'_{switch}' if switch else '')
            output = result_directory(output, 'C', label)
        if a.label and a.label != label:
            raise SystemExit(f'formal result label must be {label}')
    if not label:
        raise SystemExit('--label is required outside formal evaluation')
    ckpt = budget_checkpoint(run, steps) if step is None else run / 'checkpoints' / f'step_{step}.zip'
    if not ckpt.is_file():
        raise SystemExit(f'missing checkpoint: {ckpt}')
    if getattr(a, 'verify_existing', None):
        verify_existing(output, a.verify_existing, ops, label)
    from experiments.t2 import evaluate
    evaluate(output, str(ckpt), label, ops, int(a.workers), False if a.rigid else None, mission,
             certify=True if getattr(a, 'certify', False) else (False if getattr(a, 'no_certify', False) else None),
             govern=getattr(a, 'govern', None), audit=getattr(a, 'audit', False) or a.stage == 'formal', condition=cond)


def cmd_screen(a):
    reg = registry(); st = stage_of(a.stage)
    if a.policy not in reg['fixed']:
        raise SystemExit(f'{a.policy} is not a registered fixed policy: {sorted(reg["fixed"])}')
    if a.rigid and not reg['fixed'][a.policy]['window_pairable']:
        raise SystemExit(f'{a.policy} has no entry commitment to pair')
    condition = getattr(a, 'condition', None)
    if condition is None and a.stage == 'formal':
        condition = 'urgent'
    if condition == 'validation' or (a.stage == 'formal_prescreen' and condition not in (None, 'economy')):
        raise SystemExit('condition not available for fixed-policy screening')
    if a.stage == 'formal' and a.policy == 'gstar' and condition not in ('urgent', 'economy'):
        raise SystemExit('direct-entry reference is registered for urgent and economy')
    ops, mission = checked_cases(a, st, condition, bool(getattr(a, 'smoke', False)))
    output = Path(a.output)
    label = a.policy + (f'_{a.coordination}' if getattr(a, 'coordination', None) else '')
    if a.stage == 'formal':
        if a.rigid or getattr(a, 'coordination', None):
            raise SystemExit('unregistered fixed-policy switch')
        output = result_directory(output, 'S', f'{a.policy}_{condition}')
    if getattr(a, 'verify_existing', None):
        verify_existing(output, a.verify_existing, ops, label)
    from experiments.t2 import screen
    screen(output, ops, [a.policy], int(a.workers), not a.rigid, mission,
           coord=getattr(a, 'coordination', None),
           condition='prescreen' if a.stage == 'formal_prescreen' else condition,
           audit=a.stage in ('formal', 'formal_prescreen'))


# ---------------------------------------------------------------------- trajectories (figure B)
def _executor_env(opening: int, src: str, rigid: bool, condition=None):
    """Environment of an entry: a run directory (its manifest's interface, window and geometry) or a fixed policy."""
    from env.t2_stage import T2StageEnv
    from experiments.t2 import mission_profile
    reg = registry(); urgent = condition or tuple(reg['protocol']['urgent_deadline_s'])
    if Path(src).is_dir():
        m = json.loads((Path(src) / 'manifest.json').read_text(encoding='utf-8'))
        env = T2StageEnv(openings=[opening], shaping=False, window=bool(m.get('window', True)) and not rigid,
                         geometry=bool(m.get('geometry', True)), mission=mission_profile(urgent), obs_profile=m.get('obs_profile', 'engineered'))
        env.cert_live = False
        env.probe_live = False
        if condition is not None:
            env.certify = bool(m.get('certify', False))
            env.govern = m.get('govern')
        return env, m
    return T2StageEnv(openings=[opening], shaping=False, geometry=not src.startswith('b_'), window=not rigid, mission=mission_profile(urgent)), None


def fly(opening: int, name: str, src: str, rigid: bool = False, condition=None) -> dict:
    import time
    import torch; torch.set_num_threads(1)
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import angle_deg, frame_now
    from experiments.t2 import policy_action
    env, m = _executor_env(opening, src, rigid, condition)
    env.audit_logging = True
    env.env.audit_controls = True
    obs, _ = env.reset(options={'opening': opening})
    axis = np.asarray(env.env.config.task.approach_axis); rows, decisions = [], []

    def record(control=None):
        q, v = relative_inertial(env.env); t = env.env.target_state; f = frame_now(env.env, q)
        rows.append({'t': env.env.time_s, 'q': q.tolist(), 'v': v.tolist(), 'dv': env.env._delta_v, 'rot': t.rotation.tolist(),
                     'omega': t.omega.tolist(), 'range': float(np.linalg.norm(q)), 'port_los_deg': angle_deg(t.rotation @ axis, q),
                     'lat_err_deg': float(f['lat_err_deg']), 'phase_deg': float(np.degrees(f['phase_rad'])),
                     'mode': env.generator.mode if env.generator is not None else 'pure', **(control or {})})
    record()
    step = env.env.step
    def logged(*x, **k):
        out = step(*x, **k); record(out[-1].get('control_audit')); return out
    env.env.step = logged
    t0 = time.perf_counter(); ret, event = 0.0, None
    if src == 'pure':
        while event is None:
            r, term, trunc, i = env.step_pure(); ret += r; event = i.get('event')
            if event is None and env.env._delta_v >= env.mission['budget_m_s']:
                event = 'resource_exhausted'; ret -= env.env.config.reward.timeout_penalty
    else:
        if m is not None:
            from experiments.t2 import load_checkpoint
            model = load_checkpoint(budget_checkpoint(Path(src), int(m['steps'])))
            act = lambda o: model.predict(o, deterministic=True)[0]
        else:
            act = lambda o: policy_action(src, env, o)
        done = False
        while not done:
            a = np.asarray(act(obs), float)
            decisions.append({'t': env.env.time_s, 'a': a.tolist(), 'mode': env.generator.mode})
            obs, r, term, trunc, i = env.step(a); done = term or trunc; ret += i['true_reward']; event = i['event']
            decisions[-1].update(i.get('decision_audit', {}))
            if i.get('cert') is not None:
                decisions[-1]['cert'] = i['cert']
    return {'opening': opening, 'entry': name, 'source': src, 'rigid': rigid, 'event': event, 'return': ret, 'time_s': env.env.time_s,
            'delta_v': env.env._delta_v, 'mission': env.mission, 'entries_lost': env.stats.get('entries_lost', 0) if env.stats else 0,
            'window_recoveries': getattr(env.generator.bridge, 'recovery_log', []) if env.generator is not None else [],
            'approach_axis': axis.tolist(), 'decisions': decisions, 'rows': rows, 'wall_s': time.perf_counter() - t0}


def cmd_traj(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for o in block(a.openings):
        for spec in a.entry:
            name, _, src = spec.partition('=')
            f = out / f'traj_{name}{"_rigid" if a.rigid else ""}_{o}.json'
            if f.exists():
                continue
            res = fly(o, name, src or name, a.rigid, getattr(a, 'condition', None))
            f.write_text(json.dumps(res), encoding='utf-8')
            print(json.dumps({k: res[k] for k in ('opening', 'entry', 'event', 'return', 'time_s', 'delta_v', 'entries_lost')}), flush=True)


# ---------------------------------------------------------------------- sweep-cone geometry scan (figure A)
def _los_for(s_hat: np.ndarray, port: np.ndarray, lam_deg: float, phase_deg: float) -> np.ndarray:
    """Unit line of sight at polar angle ``lam`` from the spin axis whose meridian the port still has to sweep through
    ``phase`` (the convention of ``env.t2_stage.spin_frame``)."""
    p = port - (port @ s_hat) * s_hat; p /= np.linalg.norm(p)
    ph = np.radians(phase_deg); w = np.cross(s_hat, p)
    rp = np.cos(ph) * p + np.sin(ph) * w          # rotate p about s_hat by +phase
    lam = np.radians(lam_deg)
    return np.cos(lam) * s_hat + np.sin(lam) * rp


def geoscan_opening(opening: int, ranges=(8.0, 10.0, 12.0), lat_grid=None, phase_grid=None) -> dict:
    """For one opening's target state: on a (latitude error, phase) grid at each range, with zero relative velocity,
    (i) the nominal geometric condition -- the smallest port / line-of-sight angle over the entry arrival times under the
    constant-rate forecast the executor uses and whether it passes the 35 deg pre-screen, and (ii) the earliest arrival
    time of BRIDGE_TAUS_S at which the conditional G* (the executor's own planner, estimated inertia) finds a feasible
    entry, trying only the arrival times that pass the pre-screen (``T2StageEnv._try_bridge``). Plus the nominal one-stage
    displacement of three intents (wait, intercept, co-rotate) on the same plane."""
    from env.t1_gstar import free_body_rotations
    from env.t2_stage import (BRIDGE_ANGLE_DEG, BRIDGE_TAUS_S, STAGE_S, T2StageEnv, angle_deg, so3_exp, spin_frame,
                              target_velocity)
    from experiments.t2 import mission_profile
    reg = registry()
    env = T2StageEnv(openings=[opening], shaping=False, mission=mission_profile(tuple(reg['protocol']['urgent_deadline_s'])))
    env.reset(options={'opening': opening})
    t = env.env.target_state; R, om = t.rotation.copy(), t.omega.copy()
    axis = np.asarray(env.env.config.task.approach_axis)
    s_hat = R @ om / np.linalg.norm(om); port = R @ axis
    beta = angle_deg(port, s_hat); rate = float(np.linalg.norm(om))
    inertia = np.asarray(env.info_env.target_parameters.inertia)
    planner = env.generator.bridge.planner; v_entry = env.generator.bridge.entry_speed_m_s
    histories = {tau: free_body_rotations(R, om, inertia, tau + 1.0) for tau in BRIDGE_TAUS_S}
    lat_grid = np.arange(-60.0, 60.1, 5.0) if lat_grid is None else np.asarray(lat_grid, float)
    phase_grid = np.arange(0.0, 360.0, 5.0) if phase_grid is None else np.asarray(phase_grid, float)
    port_at = lambda s: R @ so3_exp(s * om) @ axis
    out = {'opening': opening, 'beta_deg': beta, 'rate_deg_s': float(np.degrees(rate)), 'gate_deg': BRIDGE_ANGLE_DEG,
           'taus_s': list(BRIDGE_TAUS_S), 'lat_err_deg': lat_grid.tolist(), 'phase_deg': phase_grid.tolist(), 'ranges': {}}
    for r in ranges:
        min_angle = np.full((len(lat_grid), len(phase_grid)), np.nan); gate = np.zeros_like(min_angle, dtype=bool)
        earliest = np.full_like(min_angle, np.nan); check = []
        for i, dl in enumerate(lat_grid):
            lam = beta + dl
            if not 1.0 < lam < 179.0:
                continue
            for j, ph in enumerate(phase_grid):
                q = r * _los_for(s_hat, port, lam, ph)
                f = spin_frame(R, om, q, axis)
                check.append((abs(f['lat_err_deg'] - dl), abs(((np.degrees(f['phase_rad']) - ph + 180) % 360) - 180)))
                angs = [angle_deg(port_at(s), q) for s in np.arange(15.0, 60.01, 0.5)]
                min_angle[i, j] = min(angs)
                passing = [tau for tau in BRIDGE_TAUS_S if angle_deg(port_at(tau), q) <= BRIDGE_ANGLE_DEG]
                gate[i, j] = bool(passing)
                if r <= 12.0:
                    for tau in passing:
                        res = planner.solve(q, np.zeros(3), tau, histories[tau], None, entry_speed_m_s=v_entry)
                        if res['feasible']:
                            earliest[i, j] = tau
                            break
        err = np.max(np.array(check), axis=0).tolist() if check else [None, None]
        out['ranges'][f'{r:g}'] = {'min_angle_deg': np.where(np.isnan(min_angle), None, np.round(min_angle, 2)).tolist(),
                                   'gate_pass': gate.tolist(), 'earliest_feasible_tau_s': np.where(np.isnan(earliest), None, earliest).tolist(),
                                   'construction_error_deg': err}
    # nominal one-stage displacement of three intents from a representative point (no dynamics: q + STAGE_S v_target,
    # target advanced STAGE_S at constant rate)
    arrows = []
    for r in ranges:
        for ph0 in (60.0, 180.0, 300.0):
            q0 = r * _los_for(s_hat, port, beta, ph0); f0 = spin_frame(R, om, q0, axis)
            R1 = R @ so3_exp(STAGE_S * om)
            for name, act in (('wait', (0.0, 0.0)), ('intercept', (0.0, 1.0)), ('co-rotate', (0.0, -1.0))):
                q1 = q0 + STAGE_S * target_velocity(f0, np.array(act), True)
                f1 = spin_frame(R1, om, q1, axis)
                arrows.append({'range': r, 'from': [0.0, ph0], 'to': [float(f1['lat_err_deg']), float(np.degrees(f1['phase_rad']))],
                               'intent': name, 'action': list(act)})
    out['intent_arrows'] = arrows
    return out


def cmd_geoscan(a):
    from concurrent.futures import ProcessPoolExecutor
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    ranges = tuple(float(x) for x in a.ranges.split(','))
    todo = [o for o in block(a.openings) if not (out / f'geoscan_{o}.json').exists()]
    with ProcessPoolExecutor(int(a.workers)) as pool:
        for o, res in zip(todo, pool.map(geoscan_opening, todo, [ranges] * len(todo))):
            (out / f'geoscan_{o}.json').write_text(json.dumps(res), encoding='utf-8')
            print(json.dumps({'opening': o, 'beta_deg': round(res['beta_deg'], 1),
                              'gate_share': {k: round(float(np.mean(v['gate_pass'])), 3) for k, v in res['ranges'].items()}}), flush=True)


# ---------------------------------------------------------------------- resolve / check
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(stage: str, root: Path) -> dict:
    """The registry resolved for the runs under ``root/B`` (``TRAIN_<CONFIG>_s<seed>``): science SHA, model SHA-256,
    data locations. Refuses mixed science SHAs, budgets, urgent ranges or configurations that do not match the registry."""
    reg = registry(); st = stage_of(stage)
    runs, shas, budgets, urgent = {}, set(), set(), set()
    for d in sorted((root / 'B').glob('TRAIN_*')):
        if not (d / 'manifest.json').exists():
            continue
        m = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
        cfg = m.get('final_config')
        if cfg is None or m.get('final_stage') != stage:
            raise SystemExit(f'{d} is not a {stage} run of experiments.final')
        verify_manifest(m, cfg)
        shas.add(m['git_head']); budgets.add(int(m['steps'])); urgent.add(tuple(m['urgent_deadline_s']))
        ck = budget_checkpoint(d, int(m['steps']))
        runs[d.name] = {'config': cfg, 'display': reg['trained'][cfg]['display'], 'seed': int(m['seed']), 'git_head': m['git_head'],
                        'steps': int(m['steps']), 'model': str(ck.relative_to(root)), 'model_sha256': sha256(ck),
                        'finished': (d / 'finished.json').exists(), 'episodes': str((d / 'episodes.json').relative_to(root))}
    if len(shas) > 1:
        raise SystemExit(f'runs from different science SHAs: {sorted(shas)}')
    if len(budgets) > 1 or (budgets and budgets != {int(st['steps'])}):
        raise SystemExit(f'budgets {sorted(budgets)} differ from the stage budget {st["steps"]}')
    if len(urgent) > 1:
        raise SystemExit(f'urgent ranges differ: {sorted(urgent)}')
    return {'stage': stage, 'registry_sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), 'science_sha': next(iter(shas), None),
            'steps': next(iter(budgets), None), 'eval_openings': st['eval_openings'], 'runs': runs,
            'configs': {k: reg['trained'][k] for k in sorted({r['config'] for r in runs.values()})}}


def cmd_resolve(a):
    res = resolve(a.stage, Path(a.root))
    Path(a.output).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'science_sha': res['science_sha'], 'runs': len(res['runs'])}))


def check_formal(root, st):
    report = {'stage': 'formal', 'missing_runs': [], 'unfinished_runs': [],
              'evaluations': {}, 'screens': {}, 'validation': {}, 'errors': []}
    conditions = dict(st['conditions'])
    if conditions.get('resource', {}).get('budget_m_s') is None:
        conditions.pop('resource', None)

    def check_group(folder, key, opens, result_name=None):
        directory = root / folder / key
        paths = sorted(directory.glob('*.json'))
        expected = {f'{result_name or key}_{opening}.json' for opening in block(opens)}
        actual = {p.name for p in paths}
        report['errors'].extend(str(directory / p) for p in sorted(actual - expected))
        section = {'C': 'evaluations', 'S': 'screens', 'V': 'validation'}[folder]
        report[section][key] = {'files': len(actual & expected), 'expected': len(expected)}

    for config in st['trained']:
        for seed in st['seeds']:
            run = root / 'B' / f'TRAIN_{config}_s{seed}'
            if not (run / 'manifest.json').is_file():
                report['missing_runs'].append(run.name)
            elif not (run / 'finished.json').is_file() or json.loads(
                    (run / 'finished.json').read_text(encoding='utf-8'))['steps'] != st['steps']:
                report['unfinished_runs'].append(run.name)
            if config in st['validation']['trained']:
                for step in [*st['checkpoints'], st['steps']]:
                    check_group('V', f'{config}_s{seed}_step{step}', st['validation']['openings'])
    for label in st['evaluations']:
        cond = next((c for c in st['conditions'] if f'_{c}' in label), None)
        if cond in conditions:
            check_group('C', label, conditions[cond]['openings'])
    for policy in st['fixed']:
        for cond, item in conditions.items():
            if policy == 'gstar' and cond not in ('urgent', 'economy'):
                continue
            check_group('S', f'{policy}_{cond}', item['openings'], policy)
    report['complete'] = not (report['missing_runs'] or report['unfinished_runs'] or report['errors']) and all(
        v['files'] == v['expected'] for key in ('evaluations', 'screens', 'validation') for v in report[key].values())
    return report


def check(stage: str, root: Path) -> dict:
    """Completeness of a stage directory: every planned run finished, every evaluation / screen complete, no duplicates."""
    reg = registry(); st = stage_of(stage); n = len(block(st['eval_openings']))
    if stage == 'formal':
        return check_formal(root, st)
    rep = {'stage': stage, 'missing_runs': [], 'unfinished_runs': [], 'evaluations': {}, 'screens': {}, 'errors': []}
    for cfg0 in st['trained']:
        alt = (st.get('alternatives') or {}).get(cfg0)        # registered fallback (gov_b: ECC_CERT_LOOP for ECC_GOV)
        cfg = alt if (alt and not (root / 'B' / f'TRAIN_{cfg0}_s{st["seeds"][0]}').exists()
                      and (root / 'B' / f'TRAIN_{alt}_s{st["seeds"][0]}').exists()) else cfg0
        for s in st['seeds']:
            d = root / 'B' / f'TRAIN_{cfg}_s{s}'
            if not (d / 'manifest.json').exists():
                rep['missing_runs'].append(d.name)
            elif not (d / 'finished.json').exists():
                rep['unfinished_runs'].append(d.name)
            for tag in (('', '_rigid') if cfg in st.get('rigid_pair_trained', []) else ('',)) if 'evaluations' not in st else ():
                e = root / 'C' / f'{cfg}_s{s}{tag}'
                files = [f for f in e.glob('*.json')] if e.exists() else []
                rep['evaluations'][f'{cfg}_s{s}{tag}'] = {'files': len(files), 'expected': n}
                rep['errors'] += [str(f) for f in files if f.name.startswith('ERROR_')]
    for cfg, tag in ([(c, '') for c in st.get('reused', {})] + [(c, '_cert') for c in st.get('certified', [])]
                     + [(c, '_nocert') for c in st.get('uncertified', [])]):
        for s in st['seeds']:                   # stage u2 / u2c: earlier models re-evaluated here (u2c: also with certification)
            e = root / 'C' / f'{cfg}_s{s}{tag}'
            files = [f for f in e.glob('*.json')] if e.exists() else []
            rep['evaluations'][f'{cfg}_s{s}{tag}'] = {'files': len(files), 'expected': n}
            rep['errors'] += [str(f) for f in files if f.name.startswith('ERROR_')]
    for name in st.get('evaluations', []):     # env 0.17.0 stages: every evaluation directory is listed in the registry
        for k, alt in (st.get('alternatives') or {}).items():      # registered fallback configuration (gov_b: ECC_CERT_LOOP)
            if name.startswith(k + '_') and not (root / 'C' / name).exists() and (root / 'C' / name.replace(k, alt, 1)).exists():
                name = name.replace(k, alt, 1)
        e = root / 'C' / name
        files = [f for f in e.glob('*.json')] if e.exists() else []
        cond = next((c for c in (st.get('conditions') or {}) if name.endswith('_' + c)), 'urgent')
        expected = len(block(condition_openings(st, cond))) if st.get('conditions') else n
        rep['evaluations'][name] = {'files': len(files), 'expected': expected}
        rep['errors'] += [str(f) for f in files if f.name.startswith('ERROR_')]
    for pol in st['fixed']:
        for tag in ('', '_rigid') if pol in st['window_pairs'] else ('',):
            e = root / 'S' / f'{pol}{tag}'
            files = [f for f in e.glob('*.json')] if e.exists() else []
            rep['screens'][f'{pol}{tag}'] = {'files': len(files), 'expected': n}
            rep['errors'] += [str(f) for f in files if f.name.startswith('ERROR_')]
    rep['complete'] = (not rep['missing_runs'] and not rep['unfinished_runs'] and not rep['errors']
                       and all(v['files'] == v['expected'] for v in {**rep['evaluations'], **rep['screens']}.values()))
    return rep


def cmd_check(a):
    rep = check(a.stage, Path(a.root))
    print(json.dumps(rep, indent=1))
    if a.output:
        Path(a.output).write_text(json.dumps(rep, indent=1), encoding='utf-8')


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='cmd', required=True)
    t = s.add_parser('train'); t.add_argument('--stage', required=True); t.add_argument('--config', required=True); t.add_argument('--seed', type=int, required=True)
    t.add_argument('--output', required=True); t.add_argument('--workers', type=int); t.add_argument('--smoke-steps', type=int)
    e = s.add_parser('eval'); e.add_argument('--stage', required=True); e.add_argument('--run', required=True); e.add_argument('--label')
    e.add_argument('--output', required=True); e.add_argument('--rigid', action='store_true'); e.add_argument('--workers', type=int, default=4); e.add_argument('--openings')
    e.add_argument('--certify', action='store_true', help='env 0.15.0: U2 decision-time certification (the executor checks each intent)')
    e.add_argument('--no-certify', action='store_true', help='env 0.16.0: certification off for a run trained with it (paired ablation)')
    e.add_argument('--govern', choices=('act', 'monitor', 'off'), help='env 0.17.0: the execution-opportunity governor (default: as trained)')
    e.add_argument('--audit', action='store_true', help='include read-only decision and terminal diagnostics')
    e.add_argument('--natural', action='store_true', help='env 0.17.0: the natural mission distribution instead of the stage mission')
    e.add_argument('--budget', type=float)
    e.add_argument('--checkpoint-step', type=int)
    e.add_argument('--verify-existing', type=Path)
    e.add_argument('--condition', choices=FORMAL_CONDITIONS, help='env 0.17.0: evaluation condition of the stage (default urgent)')
    c = s.add_parser('screen'); c.add_argument('--stage', required=True); c.add_argument('--policy', required=True); c.add_argument('--output', required=True)
    c.add_argument('--rigid', action='store_true'); c.add_argument('--workers', type=int, default=4); c.add_argument('--openings')
    c.add_argument('--natural', action='store_true')
    c.add_argument('--condition', choices=FORMAL_CONDITIONS)
    c.add_argument('--budget', type=float)
    c.add_argument('--verify-existing', type=Path)
    c.add_argument('--smoke', action='store_true')
    c.add_argument('--coordination', choices=('cert', 'gov'), help='env 0.17.0 (section 6B): the old certification or the governor on a fixed stage policy')
    j = s.add_parser('traj'); j.add_argument('--entry', action='append', required=True); j.add_argument('--openings', required=True)
    j.add_argument('--output', required=True); j.add_argument('--rigid', action='store_true')
    j.add_argument('--condition', choices=('urgent', 'economy', 'natural'))
    g = s.add_parser('geoscan'); g.add_argument('--openings', required=True); g.add_argument('--output', required=True)
    g.add_argument('--ranges', default='8,10,12'); g.add_argument('--workers', type=int, default=4)
    r = s.add_parser('resolve'); r.add_argument('--stage', required=True); r.add_argument('--root', required=True); r.add_argument('--output', required=True)
    k = s.add_parser('check'); k.add_argument('--stage', required=True); k.add_argument('--root', required=True); k.add_argument('--output')
    a = p.parse_args()
    {'train': cmd_train, 'eval': cmd_eval, 'screen': cmd_screen, 'traj': cmd_traj, 'geoscan': cmd_geoscan,
     'resolve': cmd_resolve, 'check': cmd_check}[a.cmd](a)


if __name__ == '__main__':
    main()
