"""Serial authorized TD3 queue; abort on an unverified stage, never resumes training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/td3_cloud_runs/TD3_C_7e55a65d_20261118')
REPO = Path('/workspace/spacecraft-td3')
OPS = Path('/workspace/td3-cloud-ops')
SHA = '7e55a65d373666bdb901357e07f5c53713647a4c'
sys.path.insert(0, str(REPO))
from verify import hashes, verify_train, verify_evaluations


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def record(event, **data):
    row={'utc':now(),'event':event,**data}
    with (ROOT/'SCHEDULE.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def assert_frozen():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SHA
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    expected=json.loads((ROOT/'SOURCE_SHA256.json').read_text())
    for name,h in expected.items():
        p=REPO/name
        assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==h, name
    assert shutil.disk_usage(ROOT).free > 512*1024**2, 'less than 512 MiB disk remaining'


def execute(args, label, seed):
    assert_frozen()
    command=['python','-B','-m','experiments.final',*args]
    env=dict(os.environ, PATH='/workspace/td3-venv/bin:'+os.environ['PATH'],
             PYTHONPATH=str(OPS),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLCONFIGDIR='/workspace/td3_cloud_preflight/matplotlib',
             TD3_CLOUD_OBSERVE='1' if args[0]=='train' else '0')
    with (ROOT/'logs'/f'{label}.log').open('x') as log:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status={'status':'running','phase':label,'seed':seed,'command':command,
                'supervisor_pid':os.getpid(),'child_pid':child.pid,'started_utc':now(),
                'keepalive_limit':'detached process survives command exit; provider lifetime is not guaranteed'}
        record('started',**status)
        while True:
            status['observed_utc']=now()
            progress=ROOT/'B'/f'TRAIN_TD3_C_s{seed}'/'progress.json'
            if progress.exists():
                status['train_progress']=json.loads(progress.read_text())
            write_json(ROOT/'STATUS.json',status)
            try:
                rc=child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
    record('exited',phase=label,seed=seed,exit_code=rc)
    if rc:
        raise RuntimeError(f'{label} exit code {rc}; preserve artifacts, do not resume training')


def main():
    pre=json.loads((ROOT/'PREFLIGHT.json').read_text())
    assert pre['pytest_exit_code']==0 and pre['scientific_sha']==SHA
    assert pre['runtime_limits']['background_survival_test']=='passed_across_command_exits'
    (ROOT/'logs').mkdir(exist_ok=True)
    record('queue_started',supervisor_pid=os.getpid(),seeds=[20,21,22],steps=5000,collection_workers=2)
    for seed in (20,21,22):
        run=ROOT/'B'/f'TRAIN_TD3_C_s{seed}'
        execute(['train','--stage','formal','--config','TD3_C','--seed',str(seed),'--output',str(run)],f'TRAIN_TD3_C_s{seed}',seed)
        record('train_verified',**verify_train(ROOT,seed))
        hashes(ROOT, ROOT/f'SHA256SUMS_AFTER_TRAIN_s{seed}')
        execute(['eval','--stage','formal','--run',str(run),'--condition','urgent','--output',str(ROOT),'--workers','2'],f'EVAL_TD3_C_s{seed}_urgent',seed)
        for step in (1000,2000,3000,4000,5000):
            execute(['eval','--stage','formal','--run',str(run),'--condition','validation','--checkpoint-step',str(step),'--output',str(ROOT),'--workers','2'],f'EVAL_TD3_C_s{seed}_step{step}',seed)
        record('seed_verified',**verify_evaluations(ROOT,seed))
        hashes(ROOT,ROOT/f'SHA256SUMS_AFTER_SEED_s{seed}')
    assert_frozen()
    record('authorized_scope_complete',train_decisions=15000,urgent_episodes=288,validation_episodes=360)
    write_json(ROOT/'STATUS.json',{'status':'complete_authorized_TD3_scope','finished_utc':now(),
               'scientific_sha':SHA,'seeds':[20,21,22],'steps_per_seed':5000,
               'full_formal_matrix_complete':False})
    hashes(ROOT,ROOT/'SHA256SUMS')
    archive=shutil.make_archive(str(ROOT)+'_DELIVERY','gztar',ROOT.parent,ROOT.name)
    Path(archive+'.sha256').write_text(hashlib.file_digest(open(archive,'rb'),'sha256').hexdigest()+'  '+Path(archive).name+'\n')


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        record('blocked',error=repr(error),traceback=traceback.format_exc())
        write_json(ROOT/'STATUS.json',{'status':'blocked','utc':now(),'error':repr(error),
                   'supervisor_pid':os.getpid(),'artifacts_preserved':True})
        raise
