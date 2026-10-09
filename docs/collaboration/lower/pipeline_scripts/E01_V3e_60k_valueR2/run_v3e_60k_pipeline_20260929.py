"""Run the upper-authorized final V3e pipeline with bounded concurrency."""
from datetime import datetime
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
ROOT = Path(r'D:\py\DRL2_v3e')
OLD = Path(r'D:\py\DRL2')
OUT = ROOT / 'eval' / 'v3e'
STATE = BASE / 'V3E_60K_EXECUTION_20260929.json'
LOCK = BASE / 'V3E_60K_EXECUTION_20260929.lock'
LOGS = OUT / 'formal_60k_logs'
SEEDS = (262420, 262421, 262422)
COMMIT = '5a2abe6d9255a523df701ccdae6408414a65ee70'
MAX_WORKERS = 6
MIN_AVAILABLE_MB = 768

def now():
    return datetime.now().astimezone().isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def git(root, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={root.as_posix()}', '-C', str(root), *args], text=True).strip()

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                ('avail_extended', ctypes.c_ulonglong)]

def available_mb():
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('GlobalMemoryStatusEx failed')
    return value.avail_phys / (1024 * 1024)

def nonfinite(obj, trail='$'):
    if isinstance(obj, float) and not math.isfinite(obj):
        return trail
    if isinstance(obj, dict):
        for key, value in obj.items():
            bad = nonfinite(value, f'{trail}.{key}')
            if bad:
                return bad
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            bad = nonfinite(value, f'{trail}[{index}]')
            if bad:
                return bad
    return None

def module(name, *args):
    return [sys.executable, '-u', '-B', '-m', name, *map(str, args)]

def evaluation(seed, output, arbiter=False):
    args = ['--episodes', '48', '--seed', '262000', '--horizon', '35', '--parametrization',
            'task_state_v3', '--phase-time-observation', '--execution-feedback', '--adaptive-task',
            '--model', f'logs/v3e_{seed}/final_model.zip']
    if arbiter:
        args += ['--arbiter', 'one_way', '--values', f'eval/v3e/{seed}/values']
    return module('experiments.evaluate_hybrid_policy', *args, '--output', output)

def main():
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, json.dumps({'pid': os.getpid(), 'started_at': now()}).encode('utf-8'))
    os.close(lock_fd)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8')
    state = {'started_at': now(), 'supervisor_pid': os.getpid(), 'status': 'preflight',
             'authorization': 'upper ruling 20260929; safety is final outcome, not an interim stop',
             'evaluation_commit': COMMIT, 'max_workers': MAX_WORKERS,
             'min_available_memory_mb': MIN_AVAILABLE_MB, 'jobs': {}, 'errors': [],
             'models': {str(s): {'m6_gate': None} for s in SEEDS},
             'reused_baselines': [], 'training_not_modified': True}
    active = {}
    def save():
        state['updated_at'] = now()
        tmp = STATE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, STATE)
    def event(kind, **fields):
        print(json.dumps({'event': kind, 'at': now(), **fields}, ensure_ascii=False), flush=True)
    def add(name, stage, seed, command, outputs, deps=(), priority=20):
        for path in outputs:
            if path.exists() or Path(str(path) + '.partial').exists():
                raise FileExistsError(f'Existing output must be reconciled, not overwritten: {path}')
        state['jobs'][name] = {'stage': stage, 'seed': seed, 'command': command,
                              'outputs': [str(p) for p in outputs], 'depends_on': list(deps),
                              'priority': priority, 'status': 'pending',
                              'stdout': str(LOGS / f'{name}.stdout.log'),
                              'stderr': str(LOGS / f'{name}.stderr.log')}
    def terminate_active():
        for name, item in list(active.items()):
            if item['process'].poll() is None:
                subprocess.run(['taskkill', '/PID', str(item['process'].pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                try:
                    item['process'].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    item['process'].kill()
                    item['process'].wait(timeout=15)
                state['jobs'][name]['status'] = 'terminated_after_program_fault'
                state['jobs'][name]['exit_code'] = item['process'].returncode
            item['stdout'].close()
            item['stderr'].close()
        active.clear()
    def validate(name, job):
        checks = {}
        for output in map(Path, job['outputs']):
            if not output.exists():
                raise FileNotFoundError(output)
            if output.suffix == '.json':
                payload = read(output)
                bad = nonfinite(payload)
                if bad:
                    raise ValueError(f'{output}: invalid numeric value at {bad}')
                checks[str(output)] = {'sha256': sha(output), 'finite_json': True}
            elif output.suffix == '.npz':
                arrays = {}
                with np.load(output, allow_pickle=False) as data:
                    for key in data.files:
                        value = data[key]
                        if value.dtype.kind in 'biufc' and not np.isfinite(value).all():
                            raise ValueError(f'{output}: nonfinite array {key}')
                        arrays[key] = list(value.shape)
                if not arrays.get('observation') or arrays['observation'][0] == 0:
                    raise ValueError(f'{output}: no observations')
                checks[str(output)] = {'sha256': sha(output), 'finite_arrays': True, 'shapes': arrays}
            else:
                checks[str(output)] = {'sha256': sha(output)}
        if job['stage'] in ('learned', 'm5'):
            payload = read(Path(job['outputs'][0]))
            if payload['episode_seeds'] != list(range(262000, 262048)) or len(payload['records']) != 48:
                raise ValueError(f'{name}: wrong formal evaluation seed block')
            if payload.get('stochastic_policy') is not False:
                raise ValueError(f'{name}: policy must be deterministic')
            fallback = sum(r['zero_fallback_steps_total'] for r in payload['records'])
            checks['evaluation'] = {'completed': sum(r['completed'] for r in payload['records']),
                'episodes_with_truth_violation': sum(r['constraint_violated'] for r in payload['records']),
                'qp_zero_fallback_steps': fallback}
            if job['stage'] == 'm5' and fallback > 50:
                raise ValueError(f'{name}: anomalous M5 QP zero fallback steps {fallback} > 50')
            if job['stage'] == 'm5' and (payload.get('arbiter') != 'one_way' or payload.get('arbiter_z') != 1.0):
                raise ValueError(f'{name}: wrong arbiter configuration')
        if job['stage'] == 'm3':
            payload = read(Path(job['outputs'][0]))
            if payload.get('target') != 'task' or set(payload['V_B']['masked_blocks']) != {'task_state', 'applied_direction'}:
                raise ValueError(f'{name}: wrong target or V_B observation mask')
        if job['stage'] == 'm6':
            payload = read(Path(job['outputs'][0]))
            gate = payload['summary']['gate']
            if gate not in ('PASS', 'INCONCLUSIVE', 'STOP'):
                raise ValueError(f'{name}: unexpected M6 gate {gate}')
            state['models'][str(job['seed'])]['m6_gate'] = gate
            state['models'][str(job['seed'])]['m6_summary'] = payload['summary']
            if gate == 'STOP':
                state['jobs'][f"m5_{job['seed']}"]['status'] = 'skipped_m6_STOP'
                event('m6_STOP_model_only', seed=job['seed'], summary=payload['summary'])
            else:
                event('m6_gate', seed=job['seed'], gate=gate)
        job['checks'] = checks
    try:
        save()
        if git(ROOT, 'rev-parse', 'HEAD') != COMMIT or git(ROOT, 'status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError('Unexpected V3e commit or tracked edits')
        if git(OLD, 'rev-parse', 'HEAD') != 'b05e389483cafab15c94c4852cecf08579f29111':
            raise RuntimeError('V3d checkout changed')
        audit = read(BASE / 'V3E_FINAL_TRAINING_AUDIT_20260929.json')
        sys.path.insert(0, str(ROOT))
        from experiments.check_v3b_training_health import check
        for record in audit['runs']:
            seed = record['seed']
            run = ROOT / 'logs' / f'v3e_{seed}'
            manifest = read(run / 'manifest.json')
            expected = next(a['sha256'] for a in record['artifacts'] if a['path'].endswith('final_model.zip'))
            if sha(run / 'final_model.zip') != expected:
                raise RuntimeError(f'{seed}: final model changed after integrity audit')
            health = check(run)
            if manifest.get('status') != 'completed' or manifest.get('actual_decision_steps') != 60000 or health['status'] != 'OK':
                raise RuntimeError(f'{seed}: final model not completed or unhealthy')
            state['models'][str(seed)].update({'health': health, 'model_sha256': expected,
                'training_code_commit': manifest['code_commit'], 'manifest_sha256': sha(run / 'manifest.json')})
        for filename in ('pure_mpc.json', 'nominal.json'):
            path = OUT / filename
            prior = BASE / 'V3E_50K_DELIVERY_20260928' / 'formal' / filename
            if sha(path) != sha(prior):
                raise RuntimeError(f'Reused baseline changed: {path}')
            payload = read(path)
            if nonfinite(payload) or payload['episode_seeds'] != list(range(262000, 262048)):
                raise ValueError(f'Invalid baseline: {path}')
            done = sum(r['completed'] for r in payload['records'])
            if filename == 'pure_mpc.json' and done != 37:
                raise ValueError('Pure MPC no longer 37/48')
            state['reused_baselines'].append({'path': str(path), 'sha256': sha(path), 'completed': done})
        for seed in SEEDS:
            relative = f'eval/v3e/{seed}'
            add(f'learned_{seed}', 'learned', seed, evaluation(seed, f'{relative}/learned_only.json'),
                [OUT / str(seed) / 'learned_only.json'], priority=0)
        blocks = [('a', '270000-270011'), ('b', '270012-270023'), ('c', '270024-270035'), ('d', '270036-270047')]
        for index, (letter, seed_range) in enumerate(blocks):
            for seed in SEEDS:
                relative = f'eval/v3e/{seed}'
                add(f'm2_{seed}_{letter}', 'm2', seed,
                    module('experiments.v3_collect_value_data', '--run-dir', f'logs/v3e_{seed}', '--seeds', seed_range,
                           '--output', f'{relative}/m2_{letter}'),
                    [OUT / str(seed) / f'm2_{letter}.npz', OUT / str(seed) / f'm2_{letter}.json'], priority=20 + index)
        for seed in SEEDS:
            relative = f'eval/v3e/{seed}'
            m2_names = tuple(f'm2_{seed}_{letter}' for letter, _ in blocks)
            add(f'm3_{seed}', 'm3', seed,
                module('experiments.v3_fit_values', '--data', *[f'{relative}/m2_{letter}.npz' for letter, _ in blocks],
                       '--output-dir', f'{relative}/values'),
                [OUT / str(seed) / 'values/m3_report.json', OUT / str(seed) / 'values/values_L.pt', OUT / str(seed) / 'values/values_B.pt'],
                deps=m2_names, priority=1)
            add(f'm6_{seed}', 'm6', seed,
                module('experiments.v3_calibrate_values', '--run-dir', f'logs/v3e_{seed}', '--values', f'{relative}/values',
                       '--output', f'{relative}/m6.json'), [OUT / str(seed) / 'm6.json'], deps=(f'm3_{seed}',), priority=2)
            add(f'm5_{seed}', 'm5', seed, evaluation(seed, f'{relative}/arbitrated.json', True),
                [OUT / str(seed) / 'arbitrated.json'], deps=(f'm6_{seed}',), priority=3)
        state['status'] = 'running'
        save()
        event('preflight_pass', commit=COMMIT, max_workers=MAX_WORKERS, available_mb=round(available_mb()))
        last_partial_check = 0
        while True:
            for name, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                item['stdout'].close()
                item['stderr'].close()
                del active[name]
                job = state['jobs'][name]
                job.update({'exit_code': code, 'completed_at': now()})
                if code != 0:
                    job['status'] = 'failed'
                    raise RuntimeError(f'{name}: command exit={code}; see {job["stderr"]}')
                try:
                    validate(name, job)
                except BaseException:
                    job['status'] = 'validation_failed'
                    raise
                job['status'] = 'completed'
                save()
                event('completed', job=name, exit_code=code)
            if time.monotonic() - last_partial_check >= 300:
                last_partial_check = time.monotonic()
                for name in active:
                    job = state['jobs'][name]
                    if job['stage'] not in ('learned', 'm5'):
                        continue
                    partial = Path(job['outputs'][0] + '.partial')
                    if not partial.exists():
                        continue
                    try:
                        payload = read(partial)
                    except (OSError, json.JSONDecodeError):
                        continue
                    bad = nonfinite(payload)
                    if bad:
                        raise ValueError(f'{name}: invalid partial numeric value at {bad}')
                    if job['stage'] == 'm5':
                        fallback = sum(r['zero_fallback_steps_total'] for r in payload['records'])
                        if fallback > 50:
                            raise ValueError(f'{name}: anomalous partial M5 QP zero fallback {fallback} > 50')
            pending = [(name, job) for name, job in state['jobs'].items() if job['status'] == 'pending']
            if not pending and not active:
                break
            ready = [(name, job) for name, job in pending if all(state['jobs'][dep]['status'] == 'completed' for dep in job['depends_on'])]
            ready.sort(key=lambda item: item[1]['priority'])
            for name, job in ready:
                if len(active) >= MAX_WORKERS or (active and available_mb() < MIN_AVAILABLE_MB):
                    break
                stdout = open(job['stdout'], 'w', encoding='utf-8')
                stderr = open(job['stderr'], 'w', encoding='utf-8')
                proc = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                job.update({'status': 'running', 'pid': proc.pid, 'started_at': now()})
                active[name] = {'process': proc, 'stdout': stdout, 'stderr': stderr}
                save()
                event('started', job=name, pid=proc.pid, available_mb=round(available_mb()))
                time.sleep(3)
            if not active and not ready:
                raise RuntimeError('Dependency deadlock; inspect model calibration gate')
            time.sleep(60)
        eligible = [s for s in SEEDS if state['jobs'][f'm5_{s}']['status'] == 'completed']
        if eligible:
            args = ['--pure-mpc', 'eval/v3e/pure_mpc.json']
            for seed in eligible:
                args += ['--model', f'{seed}=eval/v3e/{seed}/learned_only.json,eval/v3e/{seed}/arbitrated.json']
            add('readout', 'readout', None, module('experiments.v3_readout', *args, '--output', 'eval/v3e/readout.json'),
                [OUT / 'readout.json'])
        args = ['--pure-mpc', 'eval/v3e/pure_mpc.json', '--nominal', 'eval/v3e/nominal.json', '--v3e']
        args += [f'eval/v3e/{s}/learned_only.json' for s in SEEDS]
        args += [f'eval/v3e/{s}/arbitrated.json' for s in eligible]
        add('table_60k', 'readout', None, module('experiments.v3e_early_readout', *args, '--output', 'eval/v3e/table_60k.json'),
            [OUT / 'table_60k.json'])
        for name in ('readout', 'table_60k'):
            if name not in state['jobs']:
                state['readout_not_run_reason'] = 'All models M6 STOP; no permitted M5 rows'
                continue
            job = state['jobs'][name]
            job.update({'status': 'running', 'started_at': now()})
            save()
            with open(job['stdout'], 'w', encoding='utf-8') as stdout, open(job['stderr'], 'w', encoding='utf-8') as stderr:
                proc = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                job['pid'] = proc.pid
                save()
                code = proc.wait()
            job.update({'exit_code': code, 'completed_at': now()})
            if code != 0:
                job['status'] = 'failed'
                raise RuntimeError(f'{name}: command exit={code}')
            validate(name, job)
            job['status'] = 'completed'
            save()
        state.update({'status': 'evaluations_completed', 'evaluations_completed_at': now()})
        save()
        event('evaluations_completed', eligible_m5_models=eligible)
    except BaseException as error:
        state['errors'].append({'at': now(), 'error': str(error), 'type': type(error).__name__})
        state['status'] = 'program_fault_stop'
        terminate_active()
        save()
        event('program_fault_stop', error=str(error))
    state['package_status'] = 'running'
    save()
    package_script = BASE / 'package_v3e_60k_pipeline_20260929.py'
    with open(BASE / 'V3E_60K_PACKAGING_20260929.stdout.log', 'w', encoding='utf-8') as stdout, open(BASE / 'V3E_60K_PACKAGING_20260929.stderr.log', 'w', encoding='utf-8') as stderr:
        code = subprocess.call([sys.executable, '-u', '-B', str(package_script)], cwd=ROOT, env=env,
                               stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    state['package_status'] = 'completed' if code == 0 else 'failed'
    state['package_exit_code'] = code
    state['finished_at'] = now()
    save()
    event('finished', status=state['status'], package_status=state['package_status'])

if __name__ == '__main__':
    main()
