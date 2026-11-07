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
from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3
from train.t1_b_v2_sac import FeasibleSupportSACV2
ROOT=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe')
FROZEN='e33a0dd7b0bd0d73b7d3ad16561f1dd494d242fa';ARMS=['e_sac','e_tcpf'];MARKS=[2000,5000,10000,20000]
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
def checked_eval(job):
    op,checkpoint,label,out,interface=job;file=Path(out)/f'{label}_{op}.json'
    if file.exists():return read(file)
    original=FeasibleArrivalEnvV3.step;trace=[]
    def logged(env,a):
        before=env._observation().copy();result=original(env,a);obs,r,t,x,i=result
        trace.append({'observation_before':before.tolist(),'observation_after':obs.tolist(),'raw_action':np.asarray(a).tolist(),'reward':float(r),'event':i['event'],'support':i['support_transition'],'review':i['review'],'execution':i['execution']});return result
    FeasibleArrivalEnvV3.step=logged
    try:res=eval_job(job)
    finally:FeasibleArrivalEnvV3.step=original
    with (Path(out)/f'{label}_{op}.trace.jsonl').open('w',encoding='utf-8') as f:
        for row in trace:f.write(json.dumps(row)+'\n')
    if res['event']=='numerical_failure':raise RuntimeError('True numerical_failure during frozen eval')
    return res
def audit(arm,seed,mark):
    out=ROOT/f'audit_{arm}_s{seed}_{mark}.json'
    if out.exists():return read(out)
    z=[]
    for op in OPS:
        if op>=292140:p=next((REPO/'evidence/T1B_V1/episodes/interface_0k').glob(f'*_{op}.json'))
        else:p=REPO/f'evidence/T1B_V1/episodes/E-SAC/10000/expanded_292120_292127/e_sac_rl_10000_expanded_{op}.json'
        z.append(read(p)['initial_observation'])
    z=torch.tensor(np.array(z,np.float32));models={}
    for step in (0,mark):
        path=ROOT/f'TRAIN_{arm}_s{seed}/checkpoints/step_{step}.zip';m=FeasibleSupportSACV2.load(path,device='cpu',custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4})
        mean,ls,_=m.actor.get_action_dist_params(z);p=m.support_probabilities(z);q1,q2=m.candidate_q(z,m.critic);q=torch.minimum(q1,q2).detach();alpha=m.alpha_of(z)
        loss=(p[:8]*(alpha[:8]*p[:8].clamp_min(1e-8).log()-q[:8])).sum(1).mean();g=torch.autograd.grad(loss,list(m.actor.parameters()),allow_unused=True)
        norm=float(torch.sqrt(sum((x*x).sum() for x in g if x is not None)));entropy=-(p*p.clamp_min(1e-8).log()).sum(1);ss=[]
        for k,op in enumerate(OPS):
            n=int((z[k,49:52]>.5).sum());raw=float(mean[k,0].tanh().detach());j=min(n-1,max(0,int((raw+1)*.5*n)))
            ss.append({'opening':op,'p':p[k,:n].detach().tolist(),'entropy':float(entropy[k].detach()),'Gaussian_mean':float(mean[k,0].detach()),'Gaussian_std':float(ls[k,0].exp().detach()),'Q1':q1[k,:n].detach().tolist(),'Q2':q2[k,:n].detach().tolist(),'Q_min':q[k,:n].tolist(),'Q_argmax':int(q[k,:n].argmax()),'selected_index':j,'tau_candidates_s':(z[k,46:46+n]*200).tolist()})
        models[str(step)]={'sha256':sha(path),'actor_grad_norm':norm,'per_state':ss,'alpha':float(alpha.mean())}
    n=sum(s['entropy']>.05 for s in models[str(mark)]['per_state'][:8]);result={'arm':arm,'seed':seed,'mark':mark,'models':models,'original8_entropy_gt_005':n,'entropy_gate_pass':n>=4,'observation_provenance':'Same frozen V1 initial observations; before any plan, V3 support/observation unchanged. Expanded8 diagnostic only.'};dump(out,result);return result
def mechanism(arm,seed):
    base=ROOT/f'TRAIN_{arm}_s{seed}';data=rows(base/'decisions.jsonl');counts={};completed=[];invalid=[];decisions=0
    for r in data:
        e=r['env'];counts.setdefault(e,0)
        if r['support']['learning_decision']:
            counts[e]+=1;decisions+=1
            if r['observation_before'][27]>.5:invalid.append({'step':r['step'],'env':e,'has_plan':r['observation_before'][27]})
        if r.get('event'):
            completed.append({'end_step':r['step'],'env':e,'decisions':counts[e],'event':r['event']});counts[e]=0
    pairs=rows(base/'model_prefix_releases.jsonl');valid=[r for r in pairs if r.get('gate',{}).get('weight',0)>0];directions={'earlier':0,'later':0,'same':0}
    for r in valid:
        pay=r['payload'];d=pay['alternative']['tau']-pay['factual']['tau'];directions['earlier' if d<0 else 'later' if d>0 else 'same']+=1
    mean=statistics.mean(r['decisions'] for r in completed) if completed else None;a=audit(arm,seed,2000)
    tests={'no_learning_with_live_plan':len(invalid)==0,'mean_decisions_1_to_4':mean is not None and 1<=mean<=4,'entropy_at_least4_of8':a['entropy_gate_pass']}
    if arm=='e_tcpf':tests['valid_pairs_both_directions']=bool(valid) and directions['earlier']>0 and directions['later']>0
    result={'arm':arm,'seed':seed,'pass':all(tests.values()),'tests':tests,'completed_episodes':completed,'mean_learning_decisions_per_completed_episode':mean,'learning_decisions':decisions,'violations':invalid,'valid_pairs':len(valid),'directions':directions,'entropy_states_gt005':a['original8_entropy_gt_005'],'window':'Cumulative complete log records when that arm first reaches2000 checkpoint; not retrospective finalrun.'};dump(ROOT/f'M_{arm}_s{seed}.json',result);return result
def launch(seed):
    for arm in ARMS:
        cmd=[PY,'-B','-m','experiments.t1b_v2','train','--interface','v3','--arm',arm,'--seed',str(seed),'--steps','20000','--expert',str(ROOT/'expert/expert_replay.pkl'),'--output',str(ROOT/f'TRAIN_{arm}_s{seed}')]
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
                    if mark==2000 and arm not in gated:
                        m=mechanism(arm,seed);gated.add(arm)
                        if not m['pass']:failure={'type':'M_failed','arm':arm,'seed':seed,'evidence':m,'retune_or_restart_permitted':False};break
                    if key not in scheduled:
                        scheduled.add(key);out=ROOT/f'eval_{arm}_s{seed}_{mark}';out.mkdir(exist_ok=True);pending.extend((op,str(checkpoint),f'{arm}_s{seed}_{mark}',str(out),'v3') for op in OPS)
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
def judge_l():
    results={}
    for arm in ARMS:
        data=[read(ROOT/f'eval_{arm}_s0_20000/{arm}_s0_20000_{op}.json') for op in range(292140,292148)]
        assert all('discounted_return' in r for r in data)
        mean=statistics.mean(r['discounted_return'] for r in data);non=sum(r['first_decision'] is not None and r['first_decision']['index']!=0 for r in data)
        results[arm]={'mean_return':mean,'captures':sum(r['event']=='success' for r in data),'first_non_earliest':non,'pass':mean>36.54 and non>0}
    result={'pass':any(r['pass'] for r in results.values()),'arms':results,'threshold':36.54,'automatic_seed1_seed2_authorized_only_if_pass':True};dump(ROOT/'L_JUDGMENT.json',result);return result
def main():
    torch.set_num_threads(1)
    with (ROOT/'记录/PIPELINE.lock').open('x') as f:f.write(str(os.getpid()))
    STATE.update(topic='t1b_v3_round',phase='waiting_preflight',pid=os.getpid(),scientific_commit=FROZEN,run_order='docs/collaboration/upper/run_orders/T1_B_V3_ROUND_20261019.md',training={},training_active=False,completed_evaluations=0,active_evaluations=[],cpu_policy={'logical':12,'each_arm_envs':4,'evaluation_slots':'4 with two arms;8 with one;12 after both','blas_threads':1},authorization='seed0x2x20k; M fail stop; L pass automatic seed1,2 eachpaired20k; frozen scientific modules unchanged');save()
    while not (ROOT/'记录/PREFLIGHT_EXIT.json').exists():time.sleep(30)
    assert read(ROOT/'记录/PREFLIGHT_EXIT.json')['passed'],'Preflight failed; inspect original logs'
    assert git('rev-parse','HEAD')==FROZEN and not git('status','--porcelain')
    receipt=read(ROOT/'expert/EXPERT_RECEIPT.json');assert receipt['interface']=='v3' and receipt['transitions']==1000 and receipt['sha256']==sha(ROOT/'expert/expert_replay.pkl')
    failure=None;completed_seeds=[];pool=ProcessPoolExecutor(max_workers=12)
    try:
        failure=stage(0,pool)
        if not failure:
            completed_seeds.append(0);l=judge_l();STATE['L']=l;save()
            if l['pass']:
                for seed in (1,2):
                    failure=stage(seed,pool)
                    if failure:break
                    completed_seeds.append(seed)
            else:STATE['phase']='complete_L_failed_no_extra_seeds';save()
        if not failure and len(completed_seeds)==3:STATE['phase']='complete_three_seeds_readout_required';save()
    finally:
        pool.shutdown(wait=True,cancel_futures=bool(failure));STATE.update(training_active=False,active_evaluations=[],completed_seeds=completed_seeds);save()
        for f in handles:f.close()
    dump(ROOT/'记录/PIPELINE_EXIT.json',{'pid':os.getpid(),'phase':STATE['phase'],'failure':failure,'completed_seeds':completed_seeds,'all_science_workers_closed':True,'completed_evaluations':STATE['completed_evaluations']})
if __name__=='__main__':
    try:main()
    except Exception as ex:
        failure={'type':'coordinator_failure','error':repr(ex),'traceback':traceback.format_exc()}
        terminate(failure);STATE['phase']='stopped_coordinator_failure';save();dump(ROOT/'记录/PIPELINE_ERROR.json',failure);raise
