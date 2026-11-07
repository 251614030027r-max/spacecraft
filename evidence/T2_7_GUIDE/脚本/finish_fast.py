"""After all768 registered results exist: avoid redundant evaluation subprocess startup; same native readouts/decision."""
import os,time,json,shutil,traceback,subprocess
from pathlib import Path
import psutil
import runtime as a
R=a.R
def main():
 me=psutil.Process();me.nice(psutil.NORMAL_PRIORITY_CLASS)
 assert not any('codex' in p.name().lower() or 'chatgpt' in p.name().lower() for p in me.parents())
 a.frozen();assert a.read(R/'记录/EARLY_EVAL_POOL_EXIT.json')['returncode']==0
 hashes={}
 for arm in ('e_mve','e_sac'):
  for s in (0,1):
   assert a.read(R/f'B/TRAIN_{arm}_s{s}/finished.json')['steps']==4000
   for k in a.MARKS:
    d=R/f'E_guided/{arm}_s{s}/step_{k}';files=list(d.glob('*.json'));assert len(files)==48 and not list(d.glob('ERROR_*.json'));rows=[a.read(p) for p in files];assert {x['opening'] for x in rows}==set(range(294000,294048))
    for p in files:hashes[str(p.relative_to(R))]=a.sha(p)
 oldstate=a.read(R/'RUN_STATE.json');owner=a.read(R/'记录/OS_ANCESTRY.json');root=psutil.Process(owner['pid']);assert root.create_time()==owner['created']
 assert oldstate['phase']=='evaluation' and oldstate['current_job'].startswith('eval_'),'native queue has advanced; do not replace an active publisher'
 root.suspend();drained=[]
 try:
  for info in oldstate['processes'].values():
   try:
    p=psutil.Process(info['pid']);assert 'experiments.t2' in p.cmdline() and 'eval' in p.cmdline()
    # Every output exists: this invocation only skips existing files. Let it exit cleanly.
    p.wait(timeout=120);drained.append(info['pid'])
   except psutil.NoSuchProcess:pass
  root.terminate();root.wait(timeout=20)
 except BaseException:root.resume();raise
 a.S.update(oldstate);a.S.pop('error',None)
 a.dump(R/'记录/FAST_FINISH_HANDOFF.json',{'jst':a.now(),'new_pid':me.pid,'new_created':me.create_time(),'parents':[{'pid':p.pid,'name':p.name()} for p in me.parents()],'retired_controller':owner,'drained_skip_only_processes':drained,'complete_case_hashes':hashes,'no_scientific_run_repeated_or_interrupted':True})
 a.current('readout_and_publication',evaluation=False);a.state(phase='readout_and_publication',current_job='four_native_readouts')
 pairs={'C_eval':(R/'E_guided','e_mve',a.W/'C_eval','e_sac'),'C_guide_only':(R/'E_guided','e_sac',a.W/'C_eval','e_sac'),'C_labels':(R/'E_guided','e_mve',R/'E_guided','e_sac'),'C_vs_eacc':(R/'E_guided','e_mve',a.W1/'C_eval','e_mve')}
 for name,(mr,ma,sr,sa) in pairs.items():
  for s in (0,1):
   for k in a.MARKS:
    for srcroot,arm,target in ((mr,ma,'e_mve'),(sr,sa,'e_sac')):
     src=srcroot/f'{arm}_s{s}/step_{k}';dest=R/f'{name}/{target}_s{s}/step_{k}'
     if not dest.exists():shutil.copytree(src,dest)
     else:assert {p.name:a.sha(p) for p in src.glob('*.json')}=={p.name:a.sha(p) for p in dest.glob('*.json')}
 for s in (0,1):
  src=R/f'B/TRAIN_e_mve_s{s}';d=R/f'B_mve/TRAIN_e_mve_s{s}';d.mkdir(parents=True,exist_ok=True)
  for n in ('model_stats.json','finished.json','episodes.json','heartbeat.json','manifest.json'):shutil.copy2(src/n,d/n)
 jobs={}
 for name in pairs:
  jobs[name]=a.launch(name+'_readout',['-m','experiments.t2_readout','curve','--eval-root',R/name,'--train-root',R/'B_mve','--screen',a.W/'A_ref','--expected','48','--output',R/(name+'_READOUT.json')])
 for name,p in jobs.items():
  assert p.wait()==0,name;a.finish(name+'_readout');assert a.read(R/(name+'_READOUT.json'))['all_complete']
 p=a.launch('decide',['-m','experiments.t2_decide','--criteria','t27','--checkpoints','1000,2000,3000,4000','--readout',R/'C_eval_READOUT.json','--train-root',R/'B_mve','--output',R/'DECISION.json']);assert p.wait()==0;a.finish('decide')
 import deliver
 a.current('publishing');a.state(phase='publishing',current_job='batch_evidence_publication');deliver.final()
 a.state(phase='complete_delivered',current_job=None,finished_jst=a.now());a.dump(R/'记录/PIPELINE_EXIT.json',{'returncode':0,'jst':a.now(),'verdict':a.read(R/'DECISION.json')['verdict'],'administrative_finish':'fast native readouts and batch Git index; no new evaluations'})
 for name in ('Spacecraft-T27-20261031','Spacecraft-T27-ParallelEval-20261101','Spacecraft-T27-Finish-20261101'):
  subprocess.run(['schtasks','/Change','/TN',name,'/Disable'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
if __name__=='__main__':
 try:main()
 except BaseException:
  a.dump(R/'记录/FAST_FINISH_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc()});a.state(phase='publication_or_readout_error',error=traceback.format_exc());raise
