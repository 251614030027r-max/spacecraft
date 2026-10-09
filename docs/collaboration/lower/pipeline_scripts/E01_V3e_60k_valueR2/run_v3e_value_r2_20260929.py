"""Upper-authorized value-data round two; preserves the running round-one M5."""
from datetime import datetime
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
ROOT = Path(r'D:\py\DRL2_v3e')
PROCESS = BASE / '过程文件/V3e价值第二轮'
RECORDS = PROCESS / '记录'
STATE = RECORDS / 'V3E_VALUE_R2_EXECUTION_20260929.json'
LOCK = STATE.with_suffix('.lock')
OUT = ROOT / 'eval/v3e'
LOGS = OUT / 'value_r2_logs'
SEEDS = (262420, 262421, 262422)
HOLDOUT = '270040-270047,271160-271191'
sys.path.insert(0, str(BASE / '过程文件/V3e60k/脚本'))
from run_v3e_60k_pipeline_20260929 import available_mb, nonfinite, git

def now():
    return datetime.now().astimezone().isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def first_state():
    path = BASE / 'V3E_60K_EXECUTION_20260929.json'
    if not path.exists():
        path = BASE / '过程文件/V3e60k/运行归档' / path.name
    return read(path)

def alive(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)

def module(name, *args):
    return [sys.executable, '-u', '-B', '-m', name, *map(str, args)]

def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, json.dumps({'pid': os.getpid(), 'started_at': now()}).encode())
    os.close(lock_fd)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    state = {'status': 'preflight', 'started_at': now(), 'supervisor_pid': os.getpid(),
             'authorization': 'User execute 20260929 ruling; frozen policy, additional learned_full data only',
             'jobs': {}, 'models': {str(s): {'m6_gate': None} for s in SEEDS}, 'errors': [],
             'max_total_compute_workers_including_r1': 6, 'min_available_memory_mb': 768,
             'supplement_seeds': '271000-271191', 'holdout': HOLDOUT, 'calibration_seeds': '262112-262123',
             'policy_evaluation_seeds': '262000-262047', 'first_round_role': 'supplementary', 'this_round_role': 'main'}
    active = {}
    def save():
        state['updated_at'] = now()
        tmp = STATE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, STATE)
    def event(kind, **fields):
        print(json.dumps({'event': kind, 'at': now(), **fields}, ensure_ascii=False), flush=True)
    def add(name, stage, seed, command, outputs, deps=(), priority=20):
        for output in outputs:
            if output.exists() or Path(str(output) + '.partial').exists():
                raise FileExistsError(f'Existing result must not be overwritten: {output}')
        state['jobs'][name] = {'stage': stage, 'seed': seed, 'command': command,
            'outputs': [str(p) for p in outputs], 'depends_on': list(deps), 'priority': priority, 'status': 'pending',
            'stdout': str(LOGS / f'{name}.stdout.log'), 'stderr': str(LOGS / f'{name}.stderr.log')}
    def terminate_owned():
        for name, item in active.items():
            if item['process'].poll() is None:
                subprocess.run(['taskkill', '/PID', str(item['process'].pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                item['process'].wait(timeout=30)
                state['jobs'][name]['status'] = 'terminated_after_program_fault'
            item['stdout'].close()
            item['stderr'].close()
        active.clear()
    def validate(job):
        checks = {}
        for output in map(Path, job['outputs']):
            if not output.is_file():
                raise FileNotFoundError(output)
            check = {'sha256': sha(output)}
            if output.suffix == '.json':
                payload = read(output)
                bad = nonfinite(payload)
                if bad:
                    raise ValueError(f'Invalid numeric value: {output} {bad}')
                check['finite_json'] = True
            elif output.suffix == '.npz':
                with np.load(output, allow_pickle=False) as arrays:
                    for key in arrays.files:
                        if arrays[key].dtype.kind in 'biufc' and not np.isfinite(arrays[key]).all():
                            raise ValueError(f'Nonfinite NPZ: {output} {key}')
                    if arrays['observation'].ndim != 2 or arrays['observation'].shape[1] != 42 or not len(arrays['observation']):
                        raise ValueError(f'Invalid observations: {output}')
                    check.update({'finite_arrays': True, 'states': len(arrays['observation'])})
            checks[str(output)] = check
        if job['stage'] == 'm2x':
            meta = read(Path(job['outputs'][1]))
            expected = job['expected_episode_seeds']
            if meta.get('learned_only') is not True or meta['probes'] != 0 or [e['seed'] for e in meta['episodes']] != expected:
                raise ValueError('Wrong supplementary seed block or collection mode')
            if any(e['kind'] != 'learned_full' or e['handback_decision'] is not None for e in meta['episodes']):
                raise ValueError('Supplement includes baseline/probe trajectories')
            checks['data_design'] = {'independent_learned_episodes': 48, 'seeds': expected}
        if job['stage'] == 'm3':
            report = read(Path(job['outputs'][0]))
            if report['target'] != 'task' or report['holdout_seeds'] != HOLDOUT or report['episodes'] != 384:
                raise ValueError('Wrong round-two value target, holdout or trajectory count')
            if len(report['V_L']['heads']) != 5 or len(report['V_B']['heads']) != 5 or set(report['V_B']['masked_blocks']) != {'task_state', 'applied_direction'}:
                raise ValueError('Value ensemble or mask changed')
            old = read(OUT / str(job['seed']) / 'values/m3_report.json')
            for split in ('train', 'holdout'):
                if report['V_B'][split]['states'] != old['V_B'][split]['states']:
                    raise ValueError('V_B data selection changed unexpectedly')
            checks['design'] = {'independent_V_L_train_seeds': 200, 'independent_V_L_holdout_seeds': 40,
                                'V_B_state_counts_unchanged': True}
        if job['stage'] == 'm6':
            payload = read(Path(job['outputs'][0]))
            if payload['seeds'] != '262112-262123' or payload['z'] != 1.0 or payload['agreement_gate'] != .8 or payload['min_decisive_checkpoints'] != 5:
                raise ValueError('Round-two calibration block or gate changed')
            gate = payload['summary']['gate']
            if gate not in ('PASS', 'INCONCLUSIVE', 'STOP'):
                raise ValueError('Unknown M6 gate')
            state['models'][str(job['seed'])].update({'m6_gate': gate, 'm6_summary': payload['summary']})
            if gate == 'STOP':
                state['jobs'][f'm5r2_{job["seed"]}']['status'] = 'skipped_m6_STOP'
            event('m6_r2_gate', seed=job['seed'], gate=gate, summary=payload['summary'])
        if job['stage'] == 'm5':
            payload = read(Path(job['outputs'][0]))
            if payload['episode_seeds'] != list(range(262000, 262048)) or len(payload['records']) != 48 or payload.get('stochastic_policy') is not False:
                raise ValueError('Wrong fixed evaluation block or stochastic policy')
            if payload.get('arbiter') != 'one_way' or payload.get('arbiter_z') != 1.0:
                raise ValueError('Arbiter changed')
            fallback = sum(r['zero_fallback_steps_total'] for r in payload['records'])
            if fallback > 50:
                raise ValueError(f'Round-two M5 QP zero fallback {fallback} > 50')
            checks['evaluation'] = {'completed': sum(r['completed'] for r in payload['records']),
                'truth_violation_episodes': sum(r['constraint_violated'] for r in payload['records']),
                'qp_zero_fallback_steps': fallback}
        job['checks'] = checks
    try:
        prior = first_state()
        commit = git(ROOT, 'rev-parse', 'HEAD')
        if not commit.startswith('c84ed74') or git(ROOT, 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError('Unexpected commit or dirty tracked tree')
        state['evaluation_commit'] = commit
        state['training_commit'] = 'a03632a304bf8af50c4be42cc9d3e450fb23d5ad'
        doc = ROOT / 'docs/V3E_60K_CALIBRATION_RULING_20260929.md'
        supplied = PROCESS / '接续' / doc.name
        if doc.read_text(encoding='utf-8').replace('\r\n','\n') != supplied.read_text(encoding='utf-8-sig').replace('\r\n','\n'):
            raise ValueError('Repository ruling differs from the user-supplied document')
        state['ruling_sha256'] = sha(doc)
        state['reused_original_inputs'] = []
        for seed in SEEDS:
            model = ROOT / f'logs/v3e_{seed}/final_model.zip'
            expected = prior['models'][str(seed)]['model_sha256']
            if sha(model) != expected:
                raise ValueError('Policy ZIP changed')
            state['models'][str(seed)]['model_sha256'] = expected
            for chunk in 'abcd':
                old_job = prior['jobs'][f'm2_{seed}_{chunk}']
                if old_job['status'] != 'completed':
                    raise ValueError('Original M2 incomplete')
                for p in map(Path, old_job['outputs']):
                    digest = old_job['checks'][str(p)]['sha256']
                    if sha(p) != digest:
                        raise ValueError(f'Original data changed: {p}')
                    state['reused_original_inputs'].append({'path': str(p), 'sha256': digest})
        for name in ('pure_mpc.json', 'nominal.json'):
            p = OUT / name
            original = next(x for x in prior['reused_baselines'] if Path(x['path']).name == name)
            if sha(p) != original['sha256']:
                raise ValueError('Reused baseline changed')
        for ci, chunk in enumerate('abcd'):
            for seed in SEEDS:
                first = 271000 + ci * 48
                relative = f'eval/v3e/{seed}/m2x_{chunk}'
                name = f'm2x_{seed}_{chunk}'
                add(name, 'm2x', seed, module('experiments.v3_collect_value_data', '--run-dir', f'logs/v3e_{seed}',
                    '--seeds', f'{first}-{first+47}', '--learned-only', '--output', relative),
                    [OUT / str(seed) / f'm2x_{chunk}.npz', OUT / str(seed) / f'm2x_{chunk}.json'], priority=20+ci)
                state['jobs'][name]['expected_episode_seeds'] = list(range(first, first+48))
        for seed in SEEDS:
            relative = f'eval/v3e/{seed}'
            data = [f'{relative}/{stem}_{c}.npz' for stem in ('m2','m2x') for c in 'abcd']
            add(f'm3r2_{seed}', 'm3', seed, module('experiments.v3_fit_values', '--data', *data, '--holdout', HOLDOUT,
                '--output-dir', f'{relative}/values_r2'), [OUT / str(seed) / f'values_r2/{n}' for n in ('m3_report.json','values_L.pt','values_B.pt')],
                deps=[f'm2x_{seed}_{c}' for c in 'abcd'], priority=1)
            add(f'm6r2_{seed}', 'm6', seed, module('experiments.v3_calibrate_values', '--run-dir', f'logs/v3e_{seed}',
                '--values', f'{relative}/values_r2', '--seeds', '262112-262123', '--output', f'{relative}/m6_r2.json'),
                [OUT / str(seed) / 'm6_r2.json'], deps=[f'm3r2_{seed}'], priority=2)
            add(f'm5r2_{seed}', 'm5', seed, module('experiments.evaluate_hybrid_policy', '--episodes','48','--seed','262000',
                '--horizon','35','--parametrization','task_state_v3','--phase-time-observation','--execution-feedback','--adaptive-task',
                '--model',f'logs/v3e_{seed}/final_model.zip','--arbiter','one_way','--values',f'{relative}/values_r2',
                '--output',f'{relative}/arbitrated_r2.json'), [OUT / str(seed) / 'arbitrated_r2.json'], deps=[f'm6r2_{seed}'], priority=3)
        state['status'] = 'running'
        save()
        event('preflight_pass', commit=commit, V_L_train=200, V_L_holdout=40)
        last_partial_check = 0.0
        while any(j['status'] in ('pending','running') for j in state['jobs'].values()):
            prior = first_state()
            first_alive = alive(prior['supervisor_pid'])
            if prior['status'] == 'program_fault_stop' or (prior['status'] == 'running' and not first_alive):
                raise RuntimeError('First-round program fault or missing active supervisor; preserve both rounds')
            for name, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                job = state['jobs'][name]
                item['stdout'].close(); item['stderr'].close()
                del active[name]
                job.update({'exit_code': code, 'completed_at': now()})
                if code != 0:
                    job['status'] = 'failed'
                    raise RuntimeError(f'{name}: exit {code}')
                validate(job)
                job['status'] = 'completed'
                save(); event('completed', job=name)
            if time.monotonic() - last_partial_check >= 300:
                last_partial_check = time.monotonic()
                for name in active:
                    job = state['jobs'][name]
                    if job['stage'] != 'm5':
                        continue
                    partial = Path(job['outputs'][0] + '.partial')
                    if not partial.exists():
                        continue
                    try:
                        payload = read(partial)
                    except json.JSONDecodeError:
                        continue
                    if nonfinite(payload) or sum(r['zero_fallback_steps_total'] for r in payload['records']) > 50:
                        raise ValueError(f'{name}: invalid partial or anomalous QP fallback')
            reserved = 1 if first_alive else 0
            ready = sorted(((n,j) for n,j in state['jobs'].items() if j['status']=='pending' and
                all(state['jobs'][d]['status']=='completed' for d in j['depends_on'])), key=lambda x:x[1]['priority'])
            for name, job in ready:
                if len(active) + reserved >= 6 or available_mb() < 768:
                    break
                stdout, stderr = open(job['stdout'],'w',encoding='utf-8'), open(job['stderr'],'w',encoding='utf-8')
                proc = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
                active[name] = {'process':proc,'stdout':stdout,'stderr':stderr}
                job.update({'status':'running','pid':proc.pid,'started_at':now(),'round1_slot_reserved':reserved})
                save(); event('started', job=name, pid=proc.pid, first_round_slot=reserved)
                time.sleep(3)
            if not active and any(j['status']=='pending' for j in state['jobs'].values()) and not ready:
                raise RuntimeError('Round-two dependency deadlock')
            time.sleep(60)
        eligible = [s for s in SEEDS if state['jobs'][f'm5r2_{s}']['status']=='completed']
        if eligible:
            args = ['--pure-mpc','eval/v3e/pure_mpc.json']
            for seed in eligible:
                args += ['--model',f'{seed}=eval/v3e/{seed}/learned_only.json,eval/v3e/{seed}/arbitrated_r2.json']
            add('readout_r2','readout',None,module('experiments.v3_readout',*args,'--output','eval/v3e/readout_r2.json'),[OUT/'readout_r2.json'])
            job = state['jobs']['readout_r2']; job.update({'status':'running','started_at':now()}); save()
            with open(job['stdout'],'w',encoding='utf-8') as stdout, open(job['stderr'],'w',encoding='utf-8') as stderr:
                proc = subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                job['pid']=proc.pid; save(); code=proc.wait()
            job.update({'exit_code':code,'completed_at':now()})
            if code:
                job['status']='failed'; raise RuntimeError(f'readout_r2 exit {code}')
            validate(job); job['status']='completed'
        else:
            state['readout_not_run_reason']='All round-two models M6 STOP; no permitted M5 rows'
        state.update({'status':'evaluations_completed','eligible_m5_models':eligible,'evaluations_completed_at':now()}); save()
        while first_state().get('package_status') != 'completed':
            prior = first_state()
            if prior.get('package_status') == 'failed' or not alive(prior['supervisor_pid']):
                raise RuntimeError('First-round supplemental delivery unavailable; preserve completed round-two evaluations')
            time.sleep(30)
    except BaseException as error:
        state['errors'].append({'at':now(),'error':str(error),'type':type(error).__name__})
        state['status']='program_fault_stop'
        terminate_owned(); save(); event('program_fault_stop', error=str(error))
    state['package_status']='running'; save()
    with open(RECORDS/'PACKAGING_R2.stdout.log','w',encoding='utf-8') as stdout, open(RECORDS/'PACKAGING_R2.stderr.log','w',encoding='utf-8') as stderr:
        code = subprocess.call([sys.executable,'-u','-B',str(PROCESS/'脚本/package_v3e_value_r2_20260929.py')],
            cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    state.update({'package_status':'completed' if code==0 else 'failed','package_exit_code':code,'finished_at':now()}); save()
    event('finished', status=state['status'], package_status=state['package_status'])

if __name__ == '__main__':
    main()
