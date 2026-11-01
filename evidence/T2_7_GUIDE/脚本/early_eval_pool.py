"""Resource-only early evaluation of registered cases, using the unchanged native eval_job."""
import os,sys,time,json,traceback,multiprocessing as mp
from pathlib import Path
import psutil
import runtime as a
sys.path.insert(0,str(a.REPO))
R=a.R
def worker(conn):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 import torch;torch.set_num_threads(1)
 from experiments.t2 import guarded_eval
 while True:
  job=conn.recv()
  if job is None:break
  try:
   result=guarded_eval(job);conn.send({'ok':not bool(result.get('error')),'result':{k:result.get(k) for k in ('opening','event','return')}})
  except BaseException:conn.send({'ok':False,'error':traceback.format_exc()})
 conn.close()
def main():
 me=psutil.Process();me.nice(psutil.NORMAL_PRIORITY_CLASS);me.cpu_affinity(list(range(12)))
 assert not any('codex' in p.name().lower() or 'chatgpt' in p.name().lower() for p in me.parents())
 owner=a.read(R/'记录/OS_ANCESTRY.json');root=psutil.Process(owner['pid']);assert root.create_time()==owner['created']
 state=a.read(R/'RUN_STATE.json');assert state['phase'] in ('training','training_and_early_evaluation'),'queue already left training; must reconcile before overlap'
 initial=state['processes'];train_ids={k:v['pid'] for k,v in initial.items() if k.startswith('train_')};assert train_ids
 with (R/'记录/EARLY_EVAL_POOL.lock').open('x') as f:f.write(str(me.pid))
 a.frozen();root.suspend();pool={};busy={};completed=set();claimed=set();ctx=mp.get_context('spawn');failures=[]
 meta={'jst':a.now(),'pid':me.pid,'created':me.create_time(),'controller_only_suspended':root.pid,'existing_training_PIDs':train_ids,'scientific_implementation':'unchanged experiments.t2.guarded_eval/eval_job','scope':'16 checkpoints x48 registered openings only','priority':'BelowNormal eval; Normal unchanged training','cpu_affinity':list(range(12)),'no_training_restart_or_hot_edit':True}
 a.dump(R/'输入/EARLY_EVALUATION_RESOURCE_ORDER.json',meta)
 cur=a.read(a.CURRENT);cur.update(updated_at=a.now(),evaluation_active=True,execution_state='training_and_early_evaluation',early_evaluation_state=str(R/'记录/EARLY_EVAL_POOL_STATE.json'));a.dump(a.CURRENT,cur)
 try:
  while True:
   active_training=[];done_training=[]
   for arm in ('e_mve','e_sac'):
    for s in (0,1):
     tag=f'{arm}_s{s}';d=R/f'B/TRAIN_{tag}'
     if (d/'finished.json').exists():
      assert a.read(d/'finished.json')['steps']==4000;done_training.append(tag)
     else:
      match=[pid for key,pid in train_ids.items() if tag in key];assert match and psutil.pid_exists(match[0]),'training exited unexpectedly; return to original supervisor'
      active_training.append(tag)
   for n,p in list(pool.items()):
    proc,conn=p
    if n in busy and conn.poll():
     result=conn.recv();job=busy.pop(n);key=job[0]
     if not result['ok']:failures.append({'case':key,'result':result});raise RuntimeError('native evaluation error; original records preserved')
     completed.add(key)
    if not proc.is_alive():raise RuntimeError('early evaluation worker exited unexpectedly')
   candidates=[]
   for arm in ('e_mve','e_sac'):
    for s in (0,1):
     for k in a.MARKS:
      ck=R/f'B/TRAIN_{arm}_s{s}/checkpoints/step_{k}.zip'
      if not ck.exists() or time.time()-ck.stat().st_mtime<10:continue
      out=R/f'E_guided/{arm}_s{s}/step_{k}';out.mkdir(parents=True,exist_ok=True);label=f'g_{arm}_s{s}_step_{k}'
      for opening in range(294000,294048):
       key=f'{arm}_s{s}_{k}_{opening}';file=out/f'{label}_{opening}.json'
       if file.exists():
        assert a.read(file)['opening']==opening;completed.add(key);continue
       if key not in claimed:candidates.append((key,(opening,str(ck),label,str(out),True,(140.,200.))))
   target=max(1,11-3*len(active_training))
   # Add workers only within free memory, keeping 1.5GiB for OS and live training.
   add_cap=max(0,int((psutil.virtual_memory().available-1.5*2**30)/(.5*2**30)))
   while len(pool)<target and add_cap>0 and candidates:
    n=max(pool,default=-1)+1;parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(child,),daemon=True);p.start();pool[n]=(p,parent);add_cap-=1
   for n,(proc,conn) in pool.items():
    if n not in busy and candidates:
     job=candidates.pop(0);claimed.add(job[0]);busy[n]=job;conn.send(job[1])
   st={'jst':a.now(),'resource_order':meta,'active_training':active_training,'completed_training':done_training,'completed_evaluations':len(completed),'target_eval_workers':target,'actual_eval_workers':len(pool),'busy':{str(n):j[0] for n,j in busy.items()},'worker_pids':[p.pid for p,c in pool.values()],'available_GiB':psutil.virtual_memory().available/2**30,'failures':failures}
   a.dump(R/'记录/EARLY_EVAL_POOL_STATE.json',st)
   state.update(phase='training_and_early_evaluation' if active_training else 'early_evaluation',updated_jst=a.now(),current_job='registered_early_evaluation_pool',processes={**{k:v for k,v in initial.items() if psutil.pid_exists(v['pid'])},'early_eval_pool':{'pid':me.pid,'running':True}},rolling={f'{arm}_s{s}':a.read(R/f'B/TRAIN_{arm}_s{s}/live/progress.json') for arm in ('e_mve','e_sac') for s in (0,1)},training_done=done_training,early_evaluation_completed=len(completed));a.dump(R/'RUN_STATE.json',state)
   if len(completed)==768 and not busy and not active_training:break
   time.sleep(5)
  a.dump(R/'记录/EARLY_EVAL_POOL_EXIT.json',{'returncode':0,'jst':a.now(),'completed':768,'native_queue_resumption':'native per-checkpoint calls skip complete case files; no science recomputed'})
 except BaseException:
  a.dump(R/'记录/EARLY_EVAL_POOL_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc(),'failures':failures});raise
 finally:
  for p,c in pool.values():
   if p.is_alive():
    try:c.send(None)
    except Exception:pass
  for p,c in pool.values():p.join(timeout=5)
  for p,c in pool.values():
   if p.is_alive():p.terminate()
  root.resume()
  import subprocess
  subprocess.run(['schtasks','/Change','/TN','Spacecraft-T27-ParallelEval-20261101','/Disable'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
if __name__=='__main__':main()
