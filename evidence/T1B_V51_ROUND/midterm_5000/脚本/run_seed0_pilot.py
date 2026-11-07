"""V5.1 frozen round. Resource admission changes scheduling only, never science."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0='D:/py/DRL2')
import sys,json,time,subprocess,ctypes,zipfile,hashlib,traceback,datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe');FROZEN='b6f5c8a22728b6ca3970d6201525cd4047cba8d8'
ARMS=('e_sac','e_tcpf');MARKS=(2000,5000,10000,20000);OPS=[*range(292140,292148),*range(292120,292128)]
S={};children={};handles=[]
def read(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def dump(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.writing');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def save():S['updated_jst']=now();dump(ROOT/'STATE.json',S)
class MEMORY(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ('total','physical','limit','commit','total_virtual','virtual','extended')]
def memory():
    m=MEMORY();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));g=1024**3
    return {'free_physical_gb':m.physical/g,'free_commit_gb':m.commit/g,'commit_limit_gb':m.limit/g,'used_commit_gb':(m.limit-m.commit)/g}
def ready(p):
    try:
        with zipfile.ZipFile(p) as z:return z.testzip() is None
    except (OSError,zipfile.BadZipFile):return False
def rows(p):
    if not p.exists():return []
    return [json.loads(x) for x in p.read_bytes().splitlines(keepends=True) if x.endswith(b'\n')]
def checked_eval(job):
    sys.path.insert(0,str(ROOT/'脚本'));from readonly_eval import checked_eval as execute
    return execute(job)
def stop(reason):
    for p in children.values():
        if p.poll() is None:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    S.update(training_active=False,stop_reason=reason,phase='stopped_'+reason['type']);save()
def mechanism(arm):
    base=ROOT/f'TRAIN_{arm}_s0';rr=rows(base/'decisions.jsonl');dd=[r for r in rr if r['support']['learning_decision']];post=[r for r in dd if r['step']>500];non=sum(r['support']['candidate_index']!=0 for r in post)
    violations=[{'step':r['step'],'env':r['env']} for r in dd if r['observation_before'][27]>.5];pp=rows(base/'model_prefix_releases.jsonl');valid=[p for p in pp if p.get('gate',{}).get('weight',0)>0];slots={str(j):sum(p.get('cf_slot')==j for p in valid) for j in (0,1)}
    checks={'no_learning_with_plan':not violations,'post500_non_earliest_ge30pct':bool(post) and non/len(post)>=.30}
    if arm=='e_tcpf':checks['both_cf_slots_valid']=all(slots[str(j)]>0 for j in (0,1))
    result={'arm':arm,'seed':0,'pass':all(checks.values()),'checks':checks,'learning_count':len(dd),'post500_count':len(post),'post500_non_earliest':non,'post500_fraction':non/len(post) if post else None,'valid_cf_slots':slots,'valid_pairs':len(valid),'total_pairs':len(pp),'violations':violations,'last_logged_step':max((r['step'] for r in rr),default=0),'jst':now()};dump(ROOT/f'M_{arm}_s0.json',result);return result
def launch_pair(seed):
    assert seed in (0,1) and seed in read(ROOT/'记录/RUN_SCOPE.json')['automatic_seeds']
    for arm in ARMS:
        key=f'{arm}_s{seed}';out=ROOT/f'TRAIN_{key}';assert not out.exists()
        cmd=[PY,'-B','-m','experiments.t1b_v2','train','--interface','v5','--arm',arm,'--seed',str(seed),'--steps','20000','--expert',str(ROOT.parent/'T1B_V3_ROUND_20261019/expert/expert_replay.pkl'),'--output',str(out)]
        stdout=(ROOT/f'记录/{key}.out.log').open('x');stderr=(ROOT/f'记录/{key}.err.log').open('x');handles.extend((stdout,stderr))
        p=subprocess.Popen(cmd,cwd=REPO,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);children[key]=p;S['training'][key]={'pid':p.pid,'command':cmd,'output':str(out)}
    S['started_seeds'].append(seed);S.update(training_active=True,phase='training_and_readout');save()

class AdoptedProcess:
    def __init__(self,pid,key):
        self.pid=pid;self.key=key;self.returncode=None
        api=ctypes.windll.kernel32;api.OpenProcess.restype=ctypes.c_void_p
        self.handle=api.OpenProcess(0x1000|0x100000,False,pid)
        if not self.handle:raise RuntimeError('Cannot adopt original training PID '+str(pid))
    def poll(self):
        code=ctypes.c_ulong();assert ctypes.windll.kernel32.GetExitCodeProcess(ctypes.c_void_p(self.handle),ctypes.byref(code))
        self.returncode=None if code.value==259 else int(code.value);return self.returncode

def main():
    with (ROOT/'记录/PIPELINE_SEED0.lock').open('x') as f:f.write(str(os.getpid()))
    policy=read(ROOT/'记录/CPU_MEMORY_POLICY.json');kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p;h=kernel.GetCurrentProcess();kernel.SetPriorityClass(ctypes.c_void_p(h),0x4000);assert kernel.SetProcessAffinityMask(ctypes.c_void_p(h),ctypes.c_size_t(policy['experiment_affinity_mask']))
    assert read(ROOT/'记录/PREFLIGHT_EXIT.json')['passed'];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==FROZEN;assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    S.update(read(ROOT/'STATE.json'));S.update(pid=os.getpid(),scheduler='run_seed0_pilot.py',scheduler_only_adoption=True)
    for key,record in S['training'].items():children[key]=AdoptedProcess(record['pid'],key)
    initial=read(ROOT/'记录/RESOURCE_ADMISSION.json')['initial'];last_admission=time.monotonic();measured_cost=read(ROOT/'记录/RESOURCE_ADMISSION.json')['estimated_additional_pair']
    scheduled=set();audited=set();gated={f'{a}_s0' for a in ARMS if read(ROOT/f'M_{a}_s0.json').get('pass')}
    shared=False;pending=[];inflight={};pool=None;failure=None
    S['completed_evaluations']=sum(1 for p in ROOT.glob('eval_e_*/*.json') if 'opening' in read(p))
    S['completed_shared_0k']=sum(1 for p in (ROOT/'eval_shared_0k').glob('*.json') if 'opening' in read(p))
    S['resource_adjustment']='User requested modest acceleration; existing scientific PIDs preserved; eval target3 with2.8GB commit/1.2GB physical reserve. Original resource_policy above is admission baseline, RUNTIME_RESOURCE_POLICY is current.';save()
    # Drain only the already submitted read-only evaluations before rebuilding their pool.
    drain=read(ROOT/'记录/SCOPE_CHANGE_DRAIN.json')
    for _ in range(300):
        if all(Path(p).exists() for p in drain.get('result_files',[])):break
        time.sleep(2)
    for pid in drain.get('readonly_worker_pids',[]):
        assert pid not in (26136,14684,1044,20096)
        subprocess.run(['taskkill','/PID',str(pid),'/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    # Preserve any closed partial writes from cancelled read-only attempts; never replace completed science.
    import shutil
    archive=ROOT/'记录/scope_change_partial_eval'
    for folder in ROOT.glob('eval_*'):
        for p in folder.glob('*.json'):
            try:valid='opening' in json.loads(p.read_text(encoding='utf-8-sig'))
            except (json.JSONDecodeError,UnicodeDecodeError):valid=False
            if not valid:
                dest=archive/folder.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();h=hashlib.sha256(p.read_bytes()).hexdigest();shutil.move(str(p),dest);assert hashlib.sha256(dest.read_bytes()).hexdigest()==h
        for p in folder.glob('*.trace.jsonl'):
            if not p.with_name(p.name.replace('.trace.jsonl','.json')).exists():
                dest=archive/folder.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();h=hashlib.sha256(p.read_bytes()).hexdigest();shutil.move(str(p),dest);assert hashlib.sha256(dest.read_bytes()).hexdigest()==h
    S['completed_evaluations']=sum(1 for p in ROOT.glob('eval_e_*/*.json') if 'opening' in read(p))
    S['completed_shared_0k']=sum(1 for p in (ROOT/'eval_shared_0k').glob('*.json') if 'opening' in read(p));save()
    for _ in range(20):
        if memory()['free_commit_gb']>=6.5:break
        time.sleep(1)
    try:
        while True:
            scope=read(ROOT/'记录/RUN_SCOPE.json');allowed=tuple(scope['automatic_seeds']);assert 0 in allowed and set(allowed).issubset({0,1})
            S.update(run_scope=scope,requested_runs=2*len(allowed),planned_evaluations=128*len(allowed),conditional_reserved_seeds=[1] if 1 not in allowed else [],cancelled_seeds=[2],phase='effectiveness_pilot_training_and_readout')
            mem=memory();running=[k for k,p in children.items() if p.poll() is None]
            # The first pair must have finished all environment/model allocations before another is admitted.
            primed=all((ROOT/f'TRAIN_{a}_s0/manifest.json').exists() for a in ARMS)
            if primed and measured_cost is None:
                measured_cost={'commit_gb':max(4.,mem['used_commit_gb']-initial['used_commit_gb']+1.),'physical_gb':max(1.25,initial['free_physical_gb']-mem['free_physical_gb']+.5)}
                dump(ROOT/'记录/RESOURCE_ADMISSION.json',{'initial':initial,'after_first_pair':mem,'estimated_additional_pair':measured_cost,'reserved_physical_core':policy['reserved_physical_core_mask'],'requested_pairs':3,'jst':now()})
            if any(seed not in S['started_seeds'] for seed in allowed) and primed and not inflight and time.monotonic()-last_admission>=30:
                need_commit=measured_cost['commit_gb']+4.;need_physical=measured_cost['physical_gb']+2.
                if not running or (mem['free_commit_gb']>=need_commit and mem['free_physical_gb']>=need_physical):
                    if pool is not None:pool.shutdown(wait=True);pool=None
                    launch_pair(next(seed for seed in allowed if seed not in S['started_seeds']));last_admission=time.monotonic();continue
            for key,p in children.items():
                if p.poll() is not None and p.returncode!=0:
                    failure={'type':'technical_failure','run':key,'exit_code':p.returncode,'stderr_tail':(ROOT/f'记录/{key}.err.log').read_text(errors='replace')[-5000:],'no_automatic_continuation':True};break
                arm,ss=key.rsplit('_s',1);seed=int(ss)
                for mark in MARKS:
                    checkpoint=ROOT/f'TRAIN_{key}/checkpoints/step_{mark}.zip';token=(key,mark)
                    if token in scheduled or not ready(checkpoint):continue
                    if seed==0 and mark==2000 and key not in gated:
                        m=mechanism(arm);gated.add(key)
                        if not m['pass']:failure={'type':'M_failed','run':key,'evidence':m,'retune_or_restart_permitted':False};break
                    # Load a tiny unused Replay in a short-lived read-only subprocess, never in training.
                    if token not in audited:
                        cmd=[PY,'-B',str(ROOT/'脚本/audit_checkpoint.py'),arm,str(seed),str(mark)]
                        with (ROOT/'记录/audit.out.log').open('a') as out,(ROOT/'记录/audit.err.log').open('a') as err:
                            rc=subprocess.run(cmd,cwd=REPO,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW).returncode
                        if rc:failure={'type':'technical_failure','stage':'offline_audit','run':key,'mark':mark,'exit_code':rc};break
                        audited.add(token)
                    scheduled.add(token);out=ROOT/f'eval_{key}_{mark}';out.mkdir(exist_ok=True);pending.extend((op,str(checkpoint),f'{key}_{mark}',str(out),'v5') for op in OPS if not (out/f'{key}_{mark}_{op}.json').exists())
                if failure:break
            if failure:stop(failure);break
            zero=ROOT/'TRAIN_e_sac_s0/checkpoints/step_0.zip'
            if not shared and ready(zero):
                shared=True;out=ROOT/'eval_shared_0k';out.mkdir(exist_ok=True);pending[0:0]=[(op,str(zero),'shared_0k',str(out),'v5') for op in OPS if not (out/f'shared_0k_{op}.json').exists()]
            # Preserve a whole physical core and RAM for desktop; no idle huge pool.
            runtime=read(ROOT/'记录/RUNTIME_RESOURCE_POLICY.json')
            target=int(runtime.get('eval_during_training',3)) if running else int(runtime.get('eval_after_training',4))
            if pool is not None and not inflight and pool._max_workers<target and mem['free_commit_gb']>=4.2 and mem['free_physical_gb']>=1.6:
                pool.shutdown(wait=True);pool=None;mem=memory()
            if pool is not None and not inflight and (mem['free_commit_gb']<2.8 or mem['free_physical_gb']<1.2):
                pool.shutdown(wait=True);pool=None;mem=memory()
            if pool is None:
                capacity=min(int((mem['free_commit_gb']-2.8)/1.25),int((mem['free_physical_gb']-1.2)/.3))
                budget=max(0,min(target,capacity))
            else:budget=0 if mem['free_commit_gb']<2.8 or mem['free_physical_gb']<1.2 else min(target,pool._max_workers)
            if pending and budget and pool is None:pool=ProcessPoolExecutor(max_workers=budget)
            while pending and pool is not None and len(inflight)<budget:
                job=pending.pop(0);inflight[pool.submit(checked_eval,job)]=job
            for f in list(inflight):
                if f.done():
                    job=inflight.pop(f);r=f.result();field='completed_shared_0k' if job[2]=='shared_0k' else 'completed_evaluations';S[field]+=1;print(json.dumps({'label':job[2],'opening':job[0],'event':r['event'],'return':r['discounted_return']}),flush=True)
            S.update(training_active=bool(running),active_runs=running,queued_seeds=[s for s in allowed if s not in S['started_seeds']],last_steps={k:read(ROOT/f'TRAIN_{k}/heartbeat.json').get('steps',0) for k in children},memory=mem,eval_worker_budget=budget,active_evaluations=[list(x) for x in inflight.values()],pending_evaluations=len(pending));save()
            if len(children)==2*len(allowed) and not running and not pending and not inflight and len(scheduled)==8*len(allowed):
                assert all(read(ROOT/f'TRAIN_{k}/finished.json').get('steps')==20000 for k in children);S['phase']='complete_effectiveness_pilot_readout_required';save();break
            time.sleep(30)
    finally:
        if failure and pool:
            for p in list(getattr(pool,'_processes',{}).values()):p.terminate()
        if pool:pool.shutdown(wait=True,cancel_futures=bool(failure))
        for f in handles:f.close()
    S.update(training_active=False,active_evaluations=[]);save();dump(ROOT/'记录/PIPELINE_EXIT.json',{'phase':S['phase'],'failure':failure,'all_science_workers_closed':True,'jst':now()})
if __name__=='__main__':
    try:main()
    except Exception as ex:
        failure={'type':'coordinator_failure','error':repr(ex),'traceback':traceback.format_exc()};stop(failure);dump(ROOT/'记录/PIPELINE_ERROR.json',failure);raise
