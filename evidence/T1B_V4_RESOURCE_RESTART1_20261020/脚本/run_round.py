"""V3 frozen-code scheduling, read-only mechanism checks, conditional seed queue.
No science module or hyperparameter is modified by this scheduler.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0='D:/py/DRL2')
import sys,time,json,hashlib,subprocess,zipfile,traceback,statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from experiments.t1b_v2 import eval_job,dump
from env.t1_b_feasible_v4 import FeasibleArrivalEnvV4
from train.t1_b_v4_sac import FeasibleSupportSACV4 as FeasibleSupportSACV2
ROOT=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe')
FROZEN='8684cca3d3c3a2997f3ed3f2c106c247b94e20e4';ARMS=['e_sac','e_tcpf'];MARKS=[2000,5000,10000,20000]
OPS=[*range(292140,292148),*range(292120,292128)];STATE={};children={};handles=[]
def read(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    if not Path(p).exists():return []
    return [json.loads(x) for x in Path(p).read_bytes().splitlines(keepends=True) if x.endswith(b'\n')]
def git(*a):return subprocess.check_output(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),*a],encoding='utf-8').strip()
def save():
    STATE['updated_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();dump(ROOT/'STATE.json',STATE)
def ready(p):
    try:
        with zipfile.ZipFile(p) as z:return z.testzip() is None
    except (OSError,zipfile.BadZipFile):return False
def terminate(reason):
    for p in children.values():
        if p.poll() is None:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    STATE.update(training_active=False,stop_reason=reason);save()
from readonly_eval import checked_eval
from policy_audit import audit
def mechanism(arm,seed):
    base=ROOT/f'TRAIN_{arm}_s0';rr=rows(base/'decisions.jsonl');learning=[r for r in rr if r['support']['learning_decision']];post=[r for r in learning if r['step']>500];non=sum(r['support']['candidate_index']!=0 for r in post);viol=[{'step':r['step'],'env':r['env'],'has_plan':r['observation_before'][27]} for r in learning if r['observation_before'][27]>.5]
    pairs=rows(base/'model_prefix_releases.jsonl');valid=[r for r in pairs if r.get('gate',{}).get('weight',0)>0];slots={str(k):sum(r.get('cf_slot')==k for r in valid) for k in (0,1)};checks={'no_learning_with_plan':not viol,'post500_non_earliest_ge30pct':bool(post) and non/len(post)>=.30}
    if arm=='e_tcpf':checks['both_cf_slots_valid']=all(slots[str(k)]>0 for k in (0,1))
    result={'arm':arm,'seed':0,'pass':all(checks.values()),'checks':checks,'learning_count':len(learning),'post500_count':len(post),'post500_non_earliest':non,'post500_non_earliest_fraction':non/len(post) if post else None,'violations':viol,'valid_cf_slot_counts':slots,'last_complete_logged_step':max((r['step'] for r in rr),default=0),'gate_scope':'Seed0 only, cumulative complete log prefix when own2000 checkpoint is first observed; entropy and episode-count are diagnostic, not V4 stop criteria.'};dump(ROOT/f'M_{arm}_s0.json',result);return result
def launch(seed):
    for arm in ARMS:
        cmd=[PY,'-B','-m','experiments.t1b_v2','train','--interface','v4','--arm',arm,'--seed',str(seed),'--steps','20000','--expert',str(ROOT.parent/'T1B_V3_ROUND_20261019/expert/expert_replay.pkl'),'--output',str(ROOT/f'TRAIN_{arm}_s{seed}')]
        out=(ROOT/f'记录/{arm}_s{seed}.out.log').open('x');err=(ROOT/f'记录/{arm}_s{seed}.err.log').open('x');handles.extend([out,err]);p=subprocess.Popen(cmd,cwd=REPO,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);children[arm]=p;STATE['training'][f'{arm}_s{seed}']={'pid':p.pid,'command':cmd,'output':str(ROOT/f'TRAIN_{arm}_s{seed}')}
    STATE.update(seed=seed,training_active=True,phase=f'seed{seed}_training_and_readout');save()
def stage(seed,pool):
    launch(seed);pending=[];inflight={};scheduled=set();audited=set();gated=set();failure=None
    try:
        while True:
            running=sum(p.poll() is None for p in children.values())
            for arm,p in children.items():
                if p.poll() is not None and p.returncode!=0:
                    err=(ROOT/f'记录/{arm}_s{seed}.err.log').read_text(errors='replace');failure={'type':'technical_failure','arm':arm,'seed':seed,'exit_code':p.returncode,'stderr_tail':err[-4000:],'restart_rule':'Only verified Traceback/nonfinite/numerical permits fresh_restart1; no checkpoint continuation.'};break
                for mark in MARKS:
                    checkpoint=ROOT/f'TRAIN_{arm}_s{seed}/checkpoints/step_{mark}.zip'
                    if not ready(checkpoint):continue
                    key=(arm,mark)
                    if key not in audited:audit(arm,seed,mark);audited.add(key)
                    if mark==2000 and seed==0 and arm not in gated:
                        m=mechanism(arm,seed);gated.add(arm)
                        if not m['pass']:failure={'type':'M_failed','arm':arm,'seed':seed,'evidence':m,'retune_or_restart_permitted':False};break
                    if key not in scheduled:
                        scheduled.add(key);out=ROOT/f'eval_{arm}_s{seed}_{mark}';out.mkdir(exist_ok=True);pending.extend((op,str(checkpoint),f'{arm}_s{seed}_{mark}',str(out),'v4') for op in OPS)
                if failure:break
            if failure:terminate(failure);STATE['phase']='stopped_'+failure['type'];save();break
            budget=12-4*running
            while pending and len(inflight)<budget:
                job=pending.pop(0);inflight[pool.submit(checked_eval,job)]=job
            done,_=wait(inflight,timeout=30,return_when=FIRST_COMPLETED) if inflight else (set(),set())
            if not inflight:time.sleep(30)
            for f in done:
                job=inflight.pop(f)
                try:
                    r=f.result();STATE['completed_evaluations']+=1;print(json.dumps({'label':job[2],'opening':job[0],'event':r['event'],'return':r['discounted_return']}),flush=True)
                except Exception as ex:failure={'type':'technical_failure','stage':'eval','job':str(job),'error':repr(ex)};break
            if failure:terminate(failure);STATE['phase']='stopped_technical_failure';save();break
            STATE.update(training_active=running>0,active_evaluations=[list(j) for j in inflight.values()],eval_worker_budget=budget,last_steps={a:read(ROOT/f'TRAIN_{a}_s{seed}/heartbeat.json').get('steps',0) for a in ARMS});save()
            if not running and not pending and not inflight and len(scheduled)==8:
                assert all(read(ROOT/f'TRAIN_{a}_s{seed}/finished.json').get('steps')==20000 for a in ARMS);break
    finally:
        if failure:
            for p in list(getattr(pool,'_processes',{}).values()):
                if p.is_alive():p.terminate()
    return failure
def main():
    torch.set_num_threads(1)
    with (ROOT/'记录/PIPELINE.lock').open('x') as f:f.write(str(os.getpid()))
    assert git('rev-parse','HEAD')==FROZEN and not git('status','--porcelain')
    expert=ROOT.parent/'T1B_V3_ROUND_20261019/expert/expert_replay.pkl'
    assert sha(expert)=='9c91cb76f10999ca36aecf4172bb092e4e52de9fcfbf62402a76306e701e1d0f'
    assert read(ROOT/'记录/PREFLIGHT_EXIT.json')['passed']
    STATE.update(topic='t1b_v4_resource_restart1',phase='resource_safe_pair_queue',pid=os.getpid(),scientific_commit=FROZEN,run_order='T1_B_V4_ROUND_20261019 plus2026-10-20 user autonomous deadline authorization',training={},training_active=False,completed_evaluations=0,active_evaluations=[],cpu_policy={'each_arm_envs':4,'active_training_runs':2,'eval_slots':'4/8/12 as2/1/0 arms remain; pool destroyed before next seed','blas_threads':1},authorization='Fresh technical_restart1, same frozen learner/epsilon/task; seed0,1,2 pairs sequential after real host commit-limit failure. No V3 L gate.',restart_provenance=str(ROOT.parent/'T1B_V4_ROUND_20261019/记录/RESOURCE_STOP_RECEIPT.json'));save()
    failure=None;completed=[]
    for seed in (0,1,2):
        pool=ProcessPoolExecutor(max_workers=12)
        try:failure=stage(seed,pool)
        finally:pool.shutdown(wait=True,cancel_futures=bool(failure))
        if failure:break
        completed.append(seed);STATE['completed_seeds']=completed;save()
    if not failure:STATE['phase']='complete_three_seeds_readout_required'
    STATE.update(training_active=False,active_evaluations=[]);save()
    for f in handles:f.close()
    dump(ROOT/'记录/PIPELINE_EXIT.json',{'pid':os.getpid(),'phase':STATE['phase'],'failure':failure,'completed_seeds':completed,'all_science_workers_closed':True,'completed_evaluations':STATE['completed_evaluations']})
if __name__=='__main__':
    try:main()
    except Exception as ex:
        failure={'type':'coordinator_failure','error':repr(ex),'traceback':traceback.format_exc()}
        terminate(failure);STATE['phase']='stopped_coordinator_failure';save();dump(ROOT/'记录/PIPELINE_ERROR.json',failure);raise
