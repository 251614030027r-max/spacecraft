"""Native scheduling/gates only. Frozen science a1698ac is never edited here."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,hashlib,subprocess,zipfile,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from experiments.t1b_v2 import eval_job,scan_job,dump,FeasibleArrivalEnvV2
from train.t1_b_v2_sac import FeasibleSupportSACV2
ROOT=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe')
EXPERT=ROOT.parent/'T1B学习修复V1_20261019/结果/expert/expert_replay.pkl'
FROZEN='a1698acefd0f3f4cf7298438bd842ef63ae0aee0'
ARMS=['e_sac','e_tcpf'];STATE={};handles=[];children={}
def read(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),*args],encoding='utf-8').strip()
def save():
    STATE['last_native_check_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();dump(ROOT/'STATE.json',STATE)
def trace_rows(path):
    if not path.exists():return []
    data=path.read_bytes();return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b'\n')]
def zip_ready(path):
    if not path.exists():return False
    try:
        with zipfile.ZipFile(path) as z:return z.testzip() is None
    except (OSError,zipfile.BadZipFile):return False
def terminate_training(reason):
    for arm,p in children.items():
        if p.poll() is None:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    STATE.update(training_active=False,stop_reason=reason);save()
def checked_eval(job):
    # Pure recording wrapper around the frozen evaluator. Original deterministic
    # actions/rewards/termination are returned unchanged, providing B2 states.
    op,checkpoint,label,out=job;file=Path(out)/f'{label}_{op}.json'
    if file.exists():return read(file)
    original=FeasibleArrivalEnvV2.step;rows=[]
    def logged(env,action):
        before=env._observation().copy();result=original(env,action);obs,r,t,x,i=result
        rows.append({'observation_before':before.tolist(),'observation_after':obs.tolist(),'raw_action':np.asarray(action).tolist(),'reward':float(r),'event':i['event'],'support':i['support_transition'],'review':i['review'],'execution':i['execution']})
        return result
    FeasibleArrivalEnvV2.step=logged
    try:res=eval_job(job)
    finally:FeasibleArrivalEnvV2.step=original
    trace=Path(out)/f'{label}_{op}.trace.jsonl'
    with trace.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    if res['event']=='numerical_failure':raise RuntimeError('True numerical_failure during deterministic evaluation')
    return res
def compute_audit(arm,mark):
    path=ROOT/f'TRAIN_{arm}_s0/checkpoints';zero=path/'step_0.zip';checkpoint=path/f'step_{mark}.zip'
    if not zip_ready(zero) or not zip_ready(checkpoint):return None
    observations=[];ops=[]
    for op in range(292140,292148):
        f=next((REPO/'evidence/T1B_V1/episodes/interface_0k').glob(f'*_{op}.json'));observations.append(read(f)['initial_observation']);ops.append(op)
    for op in range(292120,292128):
        f=REPO/f'evidence/T1B_V1/episodes/E-SAC/10000/expanded_292120_292127/e_sac_rl_10000_expanded_{op}.json';observations.append(read(f)['initial_observation']);ops.append(op)
    z=torch.tensor(np.array(observations,np.float32));rows={}
    for model_path in [zero,checkpoint]:
        m=FeasibleSupportSACV2.load(model_path,device='cpu',custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4})
        mean,logstd,_=m.actor.get_action_dist_params(z);p=m.support_probabilities(z);q1,q2=m.candidate_q(z,m.critic);q=torch.minimum(q1,q2).detach();alpha=m.alpha_of(z)
        loss=-(p[:8]*q[:8]).sum(1).mean()+(alpha[:8]*p[:8]*p[:8].clamp_min(1e-8).log()).sum(1).mean()
        gradients=torch.autograd.grad(loss,list(m.actor.parameters()),allow_unused=True)
        norm=float(torch.sqrt(sum((g*g).sum() for g in gradients if g is not None)));entropy=-(p*p.clamp_min(1e-8).log()).sum(1)
        states=[]
        for k,op in enumerate(ops):
            n=int((z[k,49:52]>.5).sum());raw=float(torch.tanh(mean[k,0]).detach());index=min(n-1,max(0,int((raw+1)*.5*n)))
            states.append({'opening':op,'p':p[k,:n].detach().tolist(),'entropy':float(entropy[k].detach()),'Gaussian_mean':float(mean[k,0].detach()),'Gaussian_std':float(logstd[k,0].exp().detach()),'Q1':q1[k,:n].detach().tolist(),'Q2':q2[k,:n].detach().tolist(),'Q_min':q[k,:n].tolist(),'Q_argmax':int(q[k,:n].argmax()),'selected_index':index,'tau_candidates_s':(z[k,46:46+n]*200).tolist()})
        rows[str(model_path)]={'alpha':float(alpha.mean()),'actor_grad_norm':norm,'entropy':float(entropy[:8].mean().detach()),'per_state':states,'sha256':sha(model_path)}
    initial,current=rows[str(zero)],rows[str(checkpoint)];entropy_pass=sum(s['entropy']>.05 for s in current['per_state'][:8]);ratio=current['actor_grad_norm']/initial['actor_grad_norm'] if initial['actor_grad_norm']>0 else 0
    result={'arm':arm,'mark':mark,'models':rows,'A1':{'pass':entropy_pass>=4 and ratio>=.01,'original8_entropy_gt_005':entropy_pass,'actor_gradient_fraction_of_0k':ratio,'baseline_observations':'Original8 gate; expanded8 reported separately; same actual V1 reset observations, unchanged frozen task/support.'}}
    dump(ROOT/f'audit_{arm}_{mark}.json',result);return result
def coverage(arm):
    rows=trace_rows(ROOT/f'TRAIN_{arm}_s0/decisions.jsonl');last=max((r['step'] for r in rows),default=0);bins={}
    for r in rows:
        step=r['step'];s=r['support']
        if step<=500 or not s['learning_decision']:continue
        end=500+((step-501)//1000+1)*1000
        b=bins.setdefault(end,{'begin':end-999,'end':end,'n':0,'non_earliest':0,'candidate_counts':{}});b['n']+=1;b['non_earliest']+=int(s['candidate_index']!=0);key=str(s['candidate_index']);b['candidate_counts'][key]=b['candidate_counts'].get(key,0)+1
    results=[]
    for end,b in sorted(bins.items()):
        b.update(complete=end<=last,fraction=b['non_earliest']/b['n'],pass_=b['non_earliest']/b['n']>=.05);results.append(b)
    failed=[b for b in results if b['complete'] and not b['pass_']]
    return {'arm':arm,'last_logged_global_step':last,'rule':'Non-overlapping1000 global-transition windows after warmup:501–1500,1501–2500,...; only complete windows gated;9501–10000 tail reported, not falsely treated as1000. All genuine learning decisions, not canonical holds.','bins':results,'A2_pass_complete_windows':not failed,'failed_bins':failed}
def launch_training():
    for arm in ARMS:
        cmd=[PY,'-B','-m','experiments.t1b_v2','train','--arm',arm,'--seed','0','--steps','10000','--expert',str(EXPERT),'--output',str(ROOT/f'TRAIN_{arm}_s0')]
        out=(ROOT/f'记录/{arm}_s0.out.log').open('x',encoding='utf-8');err=(ROOT/f'记录/{arm}_s0.err.log').open('x',encoding='utf-8');handles.extend([out,err])
        p=subprocess.Popen(cmd,cwd=REPO,env=os.environ.copy(),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);children[arm]=p
        STATE['training'][arm]={'pid':p.pid,'command':cmd,'output':str(ROOT/f'TRAIN_{arm}_s0')}
    STATE['training_active']=True;save()
def main():
    torch.set_num_threads(1)
    assert git('rev-parse','HEAD')==FROZEN and not git('status','--porcelain')
    assert read(ROOT/'记录/PYTEST_EXIT.json')['exit_code']==0 and read(ROOT/'contract/CONTRACT.json')['passed']
    assert sha(EXPERT)=='bf3c24f9f8db7eb156a5ab42835f6faeefee2332efbd59ba7c35ed733f5ffe80'
    with (ROOT/'记录/PIPELINE.lock').open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    STATE.update(topic='t1b_v2_pilot',phase='training_and_extended_truth_scan',pid=os.getpid(),scientific_commit=FROZEN,run_order='docs/collaboration/upper/run_orders/T1_B_V2_PILOT_20261019.md',training={},active=[],completed_evaluations=0,completed_scan=0,training_active=False,scope='seed0 two arms10k only; A1/A2 failures stop; no multi-seed',gate_protocol={'A1_marks':[2000,5000],'A2_windows':'501–1500,1501–2500,...; complete1000-transition windows only','A2_min_non_earliest':.05},cpu_policy={'logical':12,'training_envs_per_arm':4,'remaining_slots_for_true_jobs':'4 /8 /12 for2/1/0 active training arms','numerical_library_threads_per_process':1},next_action='Monitor preregistered gates; frozen eval/scan; deliver actual evidence, no automatic parameter changes or multi-seed.')
    dump(ROOT/'记录/GATE_PREREGISTRATION.json',STATE['gate_protocol']);save();launch_training()
    pending=[('scan',(op,j,str(ROOT/'scan_292120'))) for op in range(292120,292128) for j in range(3)]
    (ROOT/'scan_292120').mkdir(exist_ok=True)
    scheduled=set();audited=set();inflight={};pool=ProcessPoolExecutor(max_workers=12);failure=None
    try:
        while True:
            running=sum(p.poll() is None for p in children.values());STATE['training_active']=running>0
            for arm,p in children.items():
                if p.poll() is not None and p.returncode!=0:
                    failure={'type':'technical_failure','arm':arm,'exit_code':p.returncode,'next':'Inspect raw stderr; only technical Traceback/nonfinite/numerical failure permits fresh zero restart with_restart1. No automatic algorithm repair.'};break
                cov=coverage(arm);dump(ROOT/f'coverage_{arm}.json',cov)
                if not cov['A2_pass_complete_windows']:failure={'type':'A2_failed','arm':arm,'evidence':cov['failed_bins'],'retune_or_restart_permitted':False};break
                for mark in [2000,5000,10000]:
                    checkpoint=ROOT/f'TRAIN_{arm}_s0/checkpoints/step_{mark}.zip'
                    if not zip_ready(checkpoint):continue
                    key=(arm,mark)
                    if key not in audited:
                        audit=compute_audit(arm,mark)
                        if audit is None:continue
                        audited.add(key)
                        if mark in [2000,5000] and not audit['A1']['pass']:
                            failure={'type':'A1_failed','arm':arm,'mark':mark,'evidence':audit['A1'],'retune_or_restart_permitted':False};break
                    if key not in scheduled:
                        scheduled.add(key);out=ROOT/f'eval_{arm}_{mark}';out.mkdir(exist_ok=True)
                        pending.extend(('eval',(op,str(checkpoint),f'{arm}_{mark}',str(out))) for op in [*range(292140,292148),*range(292120,292128)])
                if failure:break
            if failure:
                terminate_training(failure);STATE['phase']='stopped_'+failure['type'];save();break
            budget=max(0,12-4*running)
            while pending and len(inflight)<budget:
                kind,job=pending.pop(0);inflight[pool.submit(scan_job if kind=='scan' else checked_eval,job)]=(kind,job)
            ready,_=wait(inflight,timeout=30,return_when=FIRST_COMPLETED) if inflight else (set(),set())
            if not inflight:time.sleep(30)
            for future in ready:
                kind,job=inflight.pop(future)
                try:
                    result=future.result()
                    if result.get('event')=='numerical_failure':raise RuntimeError('True numerical failure')
                    STATE['completed_scan' if kind=='scan' else 'completed_evaluations']+=1
                    print(json.dumps({'kind':kind,'opening':job[0],'event':result.get('event'),'return':result.get('discounted_return')},ensure_ascii=False),flush=True)
                except Exception as ex:failure={'type':'technical_failure','stage':kind,'job':str(job),'error':repr(ex)};break
            if failure:
                terminate_training(failure);STATE['phase']='stopped_technical_failure';save();break
            STATE.update(active=[{'kind':k,'job':list(j)} for k,j in inflight.values()],eval_worker_budget=budget,last_steps={a:read(ROOT/f'TRAIN_{a}_s0/heartbeat.json').get('steps',0) for a in ARMS});save()
            if not running and not pending and not inflight and len(scheduled)==6:
                STATE['phase']='pilot_complete_readout_required';STATE['next_action']='Produce A1–A3/B1–B2 readout; publish authorized evidence and collab handoff. No multi-seed without newGO.';save();break
    finally:
        if failure:
            for p in list(getattr(pool,'_processes',{}).values()):
                if p.is_alive():p.terminate()
        pool.shutdown(wait=True,cancel_futures=bool(failure))
        for f in handles:f.close()
        STATE['active']=[];STATE['training_active']=False;save()
    dump(ROOT/'记录/PIPELINE_EXIT.json',{'pid':os.getpid(),'phase':STATE['phase'],'failure':failure,'completed_scan':STATE['completed_scan'],'completed_evaluations':STATE['completed_evaluations'],'all_science_workers_closed':True})
if __name__=='__main__':
    try:main()
    except Exception as ex:
        failure={'type':'coordinator_failure','error':repr(ex),'traceback':traceback.format_exc()}
        if STATE.get('pid')==os.getpid():terminate_training(failure);STATE['phase']='stopped_coordinator_failure';save()
        dump(ROOT/'记录/PIPELINE_ERROR.json',failure);raise
