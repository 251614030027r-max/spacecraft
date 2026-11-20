"""The only entry of the round-2 experiments (``docs/final_method/ROUND2.md``). Every configuration, stage, seed list,
budget, opening block and evaluation switch comes from ``docs/final_method/METHOD_REGISTRY.json``.

    python -B -m experiments.final train    --stage STAGE --config NAME --seed S --output DIR [--workers W] [--smoke-steps N]
    python -B -m experiments.final eval     --stage STAGE --run DIR --output ROOT [--switch NAME] [--openings A-B]
                                            [--workers W] [--verify-existing SHA256SUMS]
    python -B -m experiments.final screen   --stage STAGE --policy pure --output ROOT [--openings A-B] [--workers W]
    python -B -m experiments.final traj     --entry NAME=RUN_DIR|NAME=pure --openings A,B --output DIR [--switch NAME]
    python -B -m experiments.final resolve  --stage STAGE --root ROOT --output REGISTRY_RESOLVED.json
    python -B -m experiments.final check    --stage STAGE --root ROOT [--output CHECK.json]

``train`` runs ``train.t2_mve_sac.train`` with the registered algorithm and executor configuration of NAME and the
stage's budget; a seed outside the stage's list is refused (``--smoke-steps`` runs a short check with a seed outside
every stage and lifts that rule). ``eval`` evaluates a run's final model deterministically on the stage's openings
under the executor configuration the run was trained with, or with a registered evaluation switch; results go to
ROOT/C/<config>_s<seed>[_<switch>]. ``screen`` flies the fixed baseline into ROOT/S/<policy>. ``traj`` re-flies episodes
serially and records the truth every 2 s (case figures). ``resolve`` writes the registry resolved with the science SHA of
every run and the model SHA-256; ``check`` reports completeness.
"""
import argparse, hashlib, json, os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'docs' / 'final_method' / 'METHOD_REGISTRY.json'
#: development openings a smoke run may use
SMOKE_OPENINGS = '294300-294301'


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
    return s[name]


def env_kwargs(config: dict) -> dict:
    """``T2StageEnv`` keyword arguments of a registered trained configuration."""
    return {k: config[k] for k in ('geometry', 'window', 'coordination', 'correction', 'long_entry', 'potential', 'arbitration',
                                   'equivalence')}


def _all_stage_seeds() -> set:
    p = registry()['protocol']
    return {s for st in p['stages'].values() for s in st['seeds']} | set(p['reserved']['never_reuse']['seeds'])


def _forbidden_eval(stage: str) -> set:
    """Openings an evaluation of ``stage`` may not use: the never-reused first-draft block and other stages' blocks."""
    p = registry()['protocol']
    bad = set(block(p['reserved']['never_reuse']['openings']))
    for k, v in p['stages'].items():
        if k != stage:
            bad |= set(block(v['eval_openings']))
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
        if a.config not in st['trained']:
            raise SystemExit(f'{a.config} is not trained in the {a.stage} stage ({st["trained"]})')
    elif a.seed in _all_stage_seeds():
        raise SystemExit('a smoke run may not use a registered or reserved seed')
    steps = int(a.smoke_steps) if a.smoke_steps is not None else int(st['steps'])
    from train.t2_mve_sac import train
    out = Path(a.output)
    train(out, c['algo'], int(a.seed), steps, int(a.workers or reg['protocol']['workers']), env_kwargs(c), a.smoke_steps is not None,
          updates_per_transition=float(reg['protocol']['updates_per_transition']),
          manifest_extra={'final_config': a.config, 'final_stage': a.stage,
                          'final_registry_sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest()},
          audit_logging=False)
    verify_manifest(json.loads((out / 'manifest.json').read_text(encoding='utf-8')), a.config)


def verify_manifest(m: dict, config: str) -> None:
    """A run's manifest against its registered configuration."""
    c = registry()['trained'][config]
    want = {'algo': c['algo'], **env_kwargs(c)}
    bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if bad:
        raise SystemExit(f'manifest of {config} does not match the registry: {bad}')


def final_model(run: Path) -> Path:
    p = run / 'final.zip'
    if not p.exists():
        raise SystemExit(f'no final model in {run}')
    return p


def checked_openings(a, st, smoke: bool) -> list[int]:
    ops = block(a.openings or st['eval_openings'])
    if not ops or len(set(ops)) != len(ops):
        raise SystemExit('openings must be nonempty and unique')
    if smoke:
        if not a.openings or not set(ops) <= set(block(SMOKE_OPENINGS)):
            raise SystemExit(f'a smoke model is evaluated on explicit smoke openings {SMOKE_OPENINGS} only')
    elif not set(ops) <= set(block(st['eval_openings'])) or set(ops) & _forbidden_eval(a.stage):
        raise SystemExit('openings outside the registered evaluation block or overlapping a reserved block')
    return ops


def switch_env(stage: dict, config: str, switch: str | None) -> dict | None:
    if switch is None:
        return None
    sw = registry().get('eval_switches', {}).get(switch)
    if sw is None or switch not in stage.get('eval_switches', []) or config not in sw['applies_to']:
        raise SystemExit(f'switch {switch} is not registered for {config} in this stage')
    return dict(sw['env'])


def result_directory(output, kind, label):
    path = Path(output)
    return path if path.name == label and path.parent.name == kind else path / kind / label


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
        if matches != {sha256(path)}:
            raise SystemExit(f'existing result SHA256 mismatch or missing: {path}')


def cmd_eval(a):
    st = stage_of(a.stage)
    run = Path(a.run); m = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    config, seed, steps, smoke = m.get('final_config'), m['seed'], int(m['steps']), bool(m.get('smoke'))
    if config is None:
        raise SystemExit(f'{run} was not trained through experiments.final')
    verify_manifest(m, config)
    if not smoke and (m.get('final_stage') != a.stage or steps != st['steps'] or seed not in st['seeds'] or config not in st['trained']):
        raise SystemExit('model is not registered for evaluation in this stage')
    ops = checked_openings(a, st, smoke)
    env = switch_env(st, config, a.switch)
    label = f'{config}_s{seed}' + (f'_{a.switch}' if a.switch else '')
    output = result_directory(a.output, 'C', label)
    if a.verify_existing:
        verify_existing(output, a.verify_existing, ops, label)
    from experiments.t2 import evaluate
    evaluate(output, str(final_model(run)), label, ops, int(a.workers), env=env, audit=True)


def cmd_screen(a):
    st = stage_of(a.stage)
    if a.policy not in st.get('fixed', []):
        raise SystemExit(f'{a.policy} is not a fixed policy of the {a.stage} stage ({st.get("fixed")})')
    ops = checked_openings(a, st, smoke=False)
    output = result_directory(a.output, 'S', a.policy)
    if a.verify_existing:
        verify_existing(output, a.verify_existing, ops, a.policy)
    from experiments.t2 import screen
    screen(output, ops, [a.policy], int(a.workers), audit=True)


# ---------------------------------------------------------------------- trajectories (case figures)
def fly(opening: int, name: str, src: str, env_override: dict | None = None) -> dict:
    """Re-fly one episode of a run directory (its manifest's executor configuration) or of ``pure``, recording the truth
    every 2 s and every decision."""
    import time
    import torch; torch.set_num_threads(1)
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import T2StageEnv, angle_deg, frame_now
    from experiments.t2 import attach_ranker, exec_config_of, load_checkpoint
    model = None
    if src == 'pure':
        env = T2StageEnv(openings=[opening], shaping=False)
    else:
        ck = final_model(Path(src))
        env = T2StageEnv(openings=[opening], shaping=False, **{**exec_config_of(ck), **(env_override or {})})
        model = load_checkpoint(ck)
        attach_ranker(env, model)
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
    if model is None:
        while event is None:
            r, term, trunc, i = env.step_pure(); ret += r; event = i.get('event')
            if event is None and env.env._delta_v >= env.mission['budget_m_s']:
                event = 'resource_exhausted'; ret -= env.env.config.reward.timeout_penalty
    else:
        done = False
        while not done:
            a = np.asarray(model.predict(obs, deterministic=True)[0], float)
            decisions.append({'t': env.env.time_s, 'a': a.tolist(), 'mode': env.generator.mode})
            obs, r, term, trunc, i = env.step(a); done = term or trunc; ret += i['true_reward']; event = i['event']
            decisions[-1].update(i.get('decision_audit', {}))
            if i.get('coord') is not None:
                decisions[-1]['coord'] = i['coord']
    return {'opening': opening, 'entry': name, 'source': src, 'env_override': env_override, 'event': event, 'return': ret,
            'time_s': env.env.time_s, 'delta_v': env.env._delta_v, 'mission': env.mission,
            'entries_lost': env.stats.get('entries_lost', 0) if env.stats else 0,
            'window_recoveries': getattr(env.generator.bridge, 'recovery_log', []) if env.generator is not None else [],
            'approach_axis': axis.tolist(), 'decisions': decisions, 'rows': rows, 'wall_s': time.perf_counter() - t0}


def cmd_traj(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    override = dict(registry()['eval_switches'][a.switch]['env']) if a.switch else None
    for o in block(a.openings):
        for spec in a.entry:
            name, _, src = spec.partition('=')
            f = out / f'traj_{name}{"_" + a.switch if a.switch else ""}_{o}.json'
            if f.exists():
                continue
            res = fly(o, name, src or name, override)
            f.write_text(json.dumps(res), encoding='utf-8')
            print(json.dumps({k: res[k] for k in ('opening', 'entry', 'event', 'return', 'time_s', 'delta_v', 'entries_lost')}), flush=True)


# ---------------------------------------------------------------------- resolve / check
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(stage: str, root: Path) -> dict:
    """The registry resolved for the runs under ``root/B`` (``TRAIN_<CONFIG>_s<seed>``): science SHA, model SHA-256,
    data locations. Refuses mixed science SHAs, budgets or configurations that do not match the registry."""
    reg = registry(); st = stage_of(stage)
    runs, shas, budgets = {}, set(), set()
    for d in sorted((root / 'B').glob('TRAIN_*')):
        if not (d / 'manifest.json').exists():
            continue
        m = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
        cfg = m.get('final_config')
        if cfg is None or m.get('final_stage') != stage:
            raise SystemExit(f'{d} is not a {stage} run of experiments.final')
        verify_manifest(m, cfg)
        shas.add(m['git_head']); budgets.add(int(m['steps']))
        ck = final_model(d)
        runs[d.name] = {'config': cfg, 'display': reg['trained'][cfg]['display'], 'seed': int(m['seed']), 'git_head': m['git_head'],
                        'steps': int(m['steps']), 'model': str(ck.relative_to(root)), 'model_sha256': sha256(ck),
                        'finished': (d / 'finished.json').exists(), 'episodes': str((d / 'episodes.json').relative_to(root)),
                        'tensorboard': str((d / 'tb').relative_to(root))}
    if len(shas) > 1:
        raise SystemExit(f'runs from different science SHAs: {sorted(shas)}')
    if len(budgets) > 1 or (budgets and budgets != {int(st['steps'])}):
        raise SystemExit(f'budgets {sorted(budgets)} differ from the stage budget {st["steps"]}')
    return {'stage': stage, 'registry_sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), 'science_sha': next(iter(shas), None),
            'steps': next(iter(budgets), None), 'eval_openings': st['eval_openings'], 'runs': runs,
            'configs': {k: reg['trained'][k] for k in sorted({r['config'] for r in runs.values()})}}


def cmd_resolve(a):
    res = resolve(a.stage, Path(a.root))
    Path(a.output).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'science_sha': res['science_sha'], 'runs': len(res['runs'])}))


def check_stage(root: Path, stage: str) -> dict:
    st = stage_of(stage)
    report = {'stage': stage, 'missing_runs': [], 'unfinished_runs': [], 'missing_tensorboard': [], 'evaluations': {},
              'screens': {}, 'errors': []}
    ops = block(st['eval_openings'])

    def check_group(folder, label):
        directory = root / folder / label
        actual = {p.name for p in directory.glob('*.json')}
        expected = {f'{label}_{o}.json' for o in ops}
        report['errors'].extend(str(directory / p) for p in sorted(actual - expected))
        report['evaluations' if folder == 'C' else 'screens'][label] = {'files': len(actual & expected), 'expected': len(expected)}

    report['conditional_not_run'] = []
    for config in st['trained']:
        if config in st.get('conditional', {}) and not any((root / 'B' / f'TRAIN_{config}_s{s}').exists() for s in st['seeds']):
            report['conditional_not_run'].append(config)    # a conditional session that has not been triggered
            continue
        for seed in st['seeds']:
            run = root / 'B' / f'TRAIN_{config}_s{seed}'
            if not (run / 'manifest.json').is_file():
                report['missing_runs'].append(run.name)
            elif not (run / 'finished.json').is_file() or json.loads(
                    (run / 'finished.json').read_text(encoding='utf-8'))['steps'] != st['steps']:
                report['unfinished_runs'].append(run.name)
            elif not any((run / 'tb').glob('events.out.tfevents.*')):
                report['missing_tensorboard'].append(run.name)
            check_group('C', f'{config}_s{seed}')
            for sw in st.get('eval_switches', []):
                if config in registry()['eval_switches'][sw]['applies_to']:
                    check_group('C', f'{config}_s{seed}_{sw}')
    for policy in st.get('fixed', []):
        check_group('S', policy)
    report['complete'] = not (report['missing_runs'] or report['unfinished_runs'] or report['missing_tensorboard'] or report['errors']) and all(
        v['files'] == v['expected'] for key in ('evaluations', 'screens') for v in report[key].values())
    return report


def cmd_check(a):
    rep = check_stage(Path(a.root), a.stage)
    print(json.dumps(rep, indent=1))
    if a.output:
        Path(a.output).write_text(json.dumps(rep, indent=1), encoding='utf-8')


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='cmd', required=True)
    t = s.add_parser('train'); t.add_argument('--stage', required=True); t.add_argument('--config', required=True); t.add_argument('--seed', type=int, required=True)
    t.add_argument('--output', required=True); t.add_argument('--workers', type=int); t.add_argument('--smoke-steps', type=int)
    e = s.add_parser('eval'); e.add_argument('--stage', required=True); e.add_argument('--run', required=True)
    e.add_argument('--output', required=True); e.add_argument('--workers', type=int, default=4); e.add_argument('--openings')
    e.add_argument('--switch', help='registered evaluation switch of the stage')
    e.add_argument('--verify-existing', type=Path)
    c = s.add_parser('screen'); c.add_argument('--stage', required=True); c.add_argument('--policy', required=True); c.add_argument('--output', required=True)
    c.add_argument('--workers', type=int, default=4); c.add_argument('--openings'); c.add_argument('--verify-existing', type=Path)
    j = s.add_parser('traj'); j.add_argument('--entry', action='append', required=True); j.add_argument('--openings', required=True)
    j.add_argument('--output', required=True); j.add_argument('--switch')
    r = s.add_parser('resolve'); r.add_argument('--stage', required=True); r.add_argument('--root', required=True); r.add_argument('--output', required=True)
    k = s.add_parser('check'); k.add_argument('--stage', required=True); k.add_argument('--root', required=True); k.add_argument('--output')
    a = p.parse_args()
    {'train': cmd_train, 'eval': cmd_eval, 'screen': cmd_screen, 'traj': cmd_traj, 'resolve': cmd_resolve, 'check': cmd_check}[a.cmd](a)


if __name__ == '__main__':
    main()
