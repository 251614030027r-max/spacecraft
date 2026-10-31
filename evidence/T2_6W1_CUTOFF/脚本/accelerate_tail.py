"""User-authorized audit tail overlap. Suspend only administrative queue, never science processes."""
from pathlib import Path
import os,time,json,traceback,subprocess
import psutil
import runtime as a
R=a.R
def write(p,x):a.dump(p,x)
def main():
 me=psutil.Process();assert not any('codex' in p.name().lower() or 'chatgpt' in p.name().lower() for p in me.parents())
 me.nice(psutil.NORMAL_PRIORITY_CLASS)
 state=a.read(R/'RUN_STATE.json');owner=a.read(R/'记录/OS_ANCESTRY.json')
 root=psutil.Process(owner['pid']);assert root.create_time()==owner['created']
 if state.get('current_job')!='D_e_sac_s0' or state.get('phase')!='mechanism_audit':
  write(R/'记录/ACCELERATION_EXIT.json',{'jst':a.now(),'status':'not_applied_queue_already_advanced','job':state.get('current_job')});return
 assert len(list((R/'D_rank/runs').glob('e_sac_s1_step_4000_*.json')))==0
 a.frozen(a.AUDIT)
 oldpid=state['processes']['D_e_sac_s0']['pid'];old=psutil.Process(oldpid);oldtime=old.create_time()
 assert old.ppid()==root.pid and 'experiments.t2_rank' in old.cmdline()
 assert psutil.virtual_memory().available>=4*2**30,'insufficient available RAM for 6 extra workers'
 parent_cmd=old.cmdline();args=parent_cmd[parent_cmd.index('-m'):]
 args[args.index('--checkpoint')+1]=str(a.W/'B/TRAIN_e_sac_s1/checkpoints/step_4000.zip')
 args[args.index('--label')+1]='e_sac_s1_step_4000';args[args.index('--workers')+1]='6'
 assert a.sha(args[args.index('--checkpoint')+1])=='f0e3dbe597463b8fb0273477f9569af3cf0c3199e8827fa5b49088c367db0697'
 record={'jst':a.now(),'controller_pid':root.pid,'controller_created':root.create_time(),'existing_science_pid':oldpid,'existing_created':oldtime,'added_workers':6,'preserved_existing_pool_workers':11,'cpu_affinity':list(range(12)),'scientific_configuration_unchanged':True,'controller_only_suspended':True}
 root.suspend();child=None
 try:
  with (R/'记录/D_e_sac_s1_overlap.out.log').open('ab') as fo,(R/'记录/D_e_sac_s1_overlap.err.log').open('ab') as fe:
   child=subprocess.Popen([a.PY,'-B',*args],cwd=a.REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.NORMAL_PRIORITY_CLASS)
  record.update(added_pid=child.pid,command=[a.PY,'-B',*args]);write(R/'输入/AUDIT_PARALLEL_RESOURCE_CHANGE.json',record)
  cur=a.read(a.CURRENT);cur['audit_parallel_resource_change']=str(R/'输入/AUDIT_PARALLEL_RESOURCE_CHANGE.json');write(a.CURRENT,cur)
  while True:
   rc=child.poll()
   try:oldalive=psutil.Process(oldpid).create_time()==oldtime
   except psutil.NoSuchProcess:oldalive=False
   state.update(updated_jst=a.now(),current_job='D_e_sac_s0_and_s1_parallel',parallel_resource_adjustment=record,
    processes={'D_e_sac_s0':{'pid':oldpid,'running':oldalive},'D_e_sac_s1_overlap':{'pid':child.pid,'running':rc is None}})
   write(R/'RUN_STATE.json',state)
   if rc is not None:
    assert rc==0,'overlap audit failed; serial queue will retry only missing openings'
    assert len(list((R/'D_rank/runs').glob('e_sac_s1_step_4000_*.json')))==48
    break
   time.sleep(30)
  write(R/'记录/ACCELERATION_EXIT.json',{'jst':a.now(),'returncode':0,'SAC1_cases':48,'queue_resumes':'native SAC1 command will skip all completed opening files; no truth rerun'})
 except Exception:
  write(R/'记录/ACCELERATION_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc()})
  if child is not None and child.poll() is None:
   # Keep the controller suspended until the owned audit exits, avoiding duplicate opening writers.
   child.wait()
  raise
 finally:
  root.resume()
  subprocess.run(['schtasks','/Change','/TN','Spacecraft-T26W1-AuditTail-20261031','/Disable'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
if __name__=='__main__':main()
