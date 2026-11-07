"""OS-owned administrative scope reduction; never launches training."""
import os,time,traceback,hashlib,shutil
from pathlib import Path
import psutil
import runtime as a
R=a.R
def main():
 lock=R/'记录/FOURPOINT_CONTROLLER.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 old=a.read(R/'RUN_STATE.json');assert old['phase'] in ('C_evaluation','C_evaluation_scope_drain')
 for arm in ('e_sac','e_mve'):
  for seed in (0,1):assert a.read(R/f'B/TRAIN_{arm}_s{seed}/finished.json')['steps']==4000
 for n,h in a.read(R/'记录/EXTERNAL_ADAPTER_SHA256.json').items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
 me=psutil.Process();parents=[{'pid':p.pid,'name':p.name(),'created':p.create_time()} for p in me.parents()]
 assert not any('codex' in p['name'].lower() or 'chatgpt' in p['name'].lower() for p in parents)
 assert a.read(R/'记录/DESKTOP_CPU_RESERVATION.json')['reserved_logical_cpus']==[]
 me.cpu_affinity(list(range(12)));me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 parent=psutil.Process(old['pid']);assert parent.create_time()==old['created'] and 'run_t26w_reduced.py' in ' '.join(parent.cmdline())
 pool=a.read(R/'记录/EVAL_POOL_STATE.json');identities=[]
 for pid in pool['worker_pids']:
  try:
   p=psutil.Process(pid);assert 'parallel_eval.py' in ' '.join(p.cmdline());identities.append({'pid':pid,'created':p.create_time(),'command':p.cmdline()})
  except psutil.NoSuchProcess:pass
 a.dump(R/'记录/FOURPOINT_HANDOFF.json',{'jst':a.now(),'old_main':{'pid':parent.pid,'created':parent.create_time()},'old_workers':identities,'required_cases_not_interrupted':True,'ancestor_proof':parents})
 parent.kill();parent.wait(timeout=20)
 a.S.update(old,pid=me.pid,created=me.create_time(),ancestor_proof=parents,scope='User fourpoint768:1000/2000/3000/4000; full native gate not applicable',expected_eval_cases=768)
 retired=[]
 while identities:
  for ident in list(identities):
   try:
    p=psutil.Process(ident['pid'])
    if p.create_time()!=ident['created']:identities.remove(ident);continue
    tree=[p]+p.children(recursive=True)
    states=[a.read(R/f'记录/C_worker_{q.pid}.json') for q in tree]
    if any(s.get('case') and int(s['case'].split('_step_')[1].split('_')[0]) not in (1000,2000,3000,4000) for s in states):
     retired.append({'identity':ident,'cancelled_case_states':states,'jst':a.now()})
     for q in reversed(tree):
      try:q.kill()
      except psutil.NoSuchProcess:pass
     psutil.wait_procs(tree,timeout=10);identities.remove(ident)
   except psutil.NoSuchProcess:identities.remove(ident)
  a.dump(R/'记录/FOURPOINT_DRAIN_STATE.json',{'jst':a.now(),'remaining_original_workers':identities,'retired_only_cancelled_points':retired})
  a.progress(phase='C_evaluation_scope_drain',original_workers_remaining=len(identities))
  if identities:time.sleep(3)
 # Partial cancelled-point results are preserved outside registered evaluation, never counted as complete.
 archived=[]
 for arm in ('e_sac','e_mve'):
  for seed in (0,1):
   for k in (0,500,1500,2500,3500):
    src=R/f'C_eval/{arm}_s{seed}/step_{k}'
    if src.exists():
     dest=R/f'记录/fourpoint_cancelled/{arm}_s{seed}/step_{k}';assert not dest.exists() and dest.resolve().is_relative_to(R.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
     hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*') if f.is_file()};shutil.move(str(src),str(dest));assert hashes=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.glob('*') if f.is_file()};archived.append({'path':str(dest),'hashes':hashes})
 a.dump(R/'记录/FOURPOINT_CANCELLED_PRESERVED.json',archived)
 import parallel_eval_fourpoint
 parallel_eval_fourpoint.run(a)
 assert len(list((R/'C_eval').rglob('*.json')))==768 and not list((R/'C_eval').rglob('ERROR_*.json'))
 a.run('C_readout',['-m','experiments.t2_readout','curve','--eval-root',R/'C_eval','--train-root',R/'B','--screen',R/'A_ref','--expected','48','--output',R/'C_READOUT.json']);assert a.read(R/'C_READOUT.json')['all_complete']
 a.run('DECIDE',['-m','experiments.t2_decide','--criteria','t26w','--readout',R/'C_READOUT.json','--train-root',R/'B','--output',R/'DECISION.json'])
 native=a.read(R/'DECISION.json');a.dump(R/'DECISION_NATIVE.json',native)
 native.update(verdict='scope_revised_descriptive_only',native_verdict=native.get('verdict'),scope_revision={'marks':[1000,2000,3000,4000],'full_t26w_window':'MISSING; cancelled by user','no_full_pass_claim':True})
 a.dump(R/'DECISION.json',native)
 a.progress(phase='final_publication',verdict=a.read(R/'DECISION.json')['verdict']);ok=a.publish('final')
 a.progress(phase='complete_waiting_review' if ok else 'complete_publication_pending',finished_jst=a.now(),no_more_training=True);a.dump(R/'记录/PIPELINE_EXIT.json',a.S)
if __name__=='__main__':
 try:main()
 except Exception:a.dump(R/'记录/FOURPOINT_CONTROLLER_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc()});raise
