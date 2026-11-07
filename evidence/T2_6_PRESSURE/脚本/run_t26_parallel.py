"""OS-owned T2.6: complete A -> native selection -> publish A -> frozen four-run B -> native C -> publish/stop."""
from pathlib import Path
import os,time,json,re,subprocess,shutil,traceback,hashlib
import psutil
import runtime as a
R=a.R;SHA=a.SHA
PROFILES={'p170':[170.,210.],'p155':[155.,195.],'p140':[140.,180.]}
POLICIES=('rally','c_10_0','c_10_-5','c_6_5','c_10_5','gstar','pure')
MARKS=(0,500,1000,1500,2000,2500,3000)
PROBES={'sac_s0_4000':R.parent/'T2_3_TEST_20261025/TRAIN_e_sac_s0/checkpoints/step_4000.zip','mve_s0_4000':R.parent/'T2_5_1_VALIDATE_20261026/TRAIN_e_mve_s0/checkpoints/step_4000.zip'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cap():
 while True:
  m=a.memory();n=min(11,int(max(0,(m['physical_free_GiB']-1.2)//.75)),int(max(0,(m['commit_free_GiB']-4)//1.0)))
  if n:return n
  a.progress(resource_hold=True);time.sleep(30)
def group(name,labels,out,args):
 out=Path(out);expected={f'{label}_{o}.json' for label in labels for o in range(294000,294024)}
 for attempt in range(2):
  if expected.issubset({p.name for p in out.glob('*.json')}):return
  a.run(f'{name}_try{attempt}',[*args,'--workers',str(cap()),'--output',out])
  if expected.issubset({p.name for p in out.glob('*.json')}):return
  if attempt==0:
   dest=R/'记录/episode_errors'/name;dest.mkdir(parents=True,exist_ok=True)
   for p in out.glob('ERROR_*.json'):shutil.move(str(p),str(dest/(p.stem+f'_{time.time_ns()}.json')))
 raise RuntimeError('registered cases incomplete '+name)
def early_mechanism():
 if (R/'记录/MECHANISM1000_CAPTURED.json').exists():return
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 result={}
 for seed in (0,1):
  live=a.read(R/f'B/TRAIN_e_mve_s{seed}/live/progress.json')
  if live.get('steps',0)<1050:return
  ea=EventAccumulator(str(R/f'B/TRAIN_e_mve_s{seed}/live/tensorboard'),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars'];rows={}
  for field in ('two_stage_share','decisive_share','stage2_rows','w2_mean','pi_gap_sigma_median','stage2_entry_share','c_mean','increment_abs_mean','grad_cos','loss_share','w_mean'):
   key=('critic/' if field in ('grad_cos','loss_share') else 'model/')+field
   v=[x for x in ea.Scalars(key) if x.step>1000] if key in tags else []
   rows[field]={'mean':sum(x.value for x in v)/len(v) if v else None,'n':len(v),'first_step':v[0].step if v else None,'last_step':v[-1].step if v else None,'MISSING':not bool(v)}
  mt=ea.Scalars('critic/model_td') if 'critic/model_td' in tags else []
  result[f'e_mve_s{seed}']={'after1000':rows,'first_native_model_row_step':mt[0].step if mt else None,'T251_reference_first_model_row_step':(512,537)[seed]}
 alerts=[tag for tag,v in result.items() if any(v['after1000'][f]['MISSING'] for f in ('w2_mean','pi_gap_sigma_median','grad_cos')) or (v['after1000']['w2_mean']['mean'] or 0)<.5 or (v['after1000']['pi_gap_sigma_median']['mean'] is not None and v['after1000']['pi_gap_sigma_median']['mean']>.25) or (v['after1000']['grad_cos']['mean'] or 0)<=0]
 a.dump(R/'MECHANISM1000.json',{'jst':a.now(),'selected':a.S['selected'],'deadline':PROFILES[a.S['selected']],'source':'Actual native probes from TensorBoard,steps>1000;early window not final M','readings':result,'alerts':alerts})
 a.dump(R/'记录/MECHANISM1000_CAPTURED.json',{'jst':a.now(),'alerts':alerts});a.publish('mechanism1000')
def main():
 with (R/'记录/PIPELINE_PARALLEL.lock').open('x') as f:f.write(str(os.getpid()))
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);me.cpu_affinity(list(range(psutil.cpu_count())))
 anc=[{'pid':p.pid,'name':p.name(),'created':p.create_time()} for p in me.parents()];assert not any('codex' in x['name'].lower() or 'chatgpt' in x['name'].lower() for x in anc)
 a.S.update(pid=me.pid,started_jst=a.now(),science_SHA=SHA,order_collab='c1a962c',phase='pytest',ancestor_proof=anc,scope='A648;native selection;only if selected B4runs3000/2workers;C672;t26;no branch')
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==SHA and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 assert a.read(R/'记录/PYTEST_VERIFIED.json').get('status')=='passed' and a.read(R/'记录/PYTEST_VERIFIED.json').get('science_SHA')==SHA
 a.dump(R/'记录/PROBE_SOURCE_SHA256.json',[{'label':tag,'path':str(p),'SHA256':sha(p)} for tag,p in PROBES.items()])
 a.progress(phase='A_parallel')
 import parallel_A
 parallel_A.run(a)
 assert len(list((R/'A').rglob('*.json')))==648
 a.run('A_readout',['-m','experiments.t2_readout','pressure','--dir',R/'A','--tags','p170,p155,p140','--output',R/'A_PRESSURE.json'])
 sel=a.read(R/'A_PRESSURE.json')['selected'];assert sel in (None,'p155','p140');a.progress(phase='A_publication',selected=sel)
 while not a.publish('A'):time.sleep(180)
 if sel is None:
  a.progress(phase='A_no_profile_waiting_review',finished_jst=a.now(),no_training=True);a.dump(R/'记录/PIPELINE_EXIT.json',a.S);return
 while not a.read(R/'记录/LOGGING_VERIFIED.json'):
  assert not a.read(R/'记录/LOGGING_VERIFICATION_ERROR.json'),'Logging gate failed: preserve A, no formal training';a.progress(phase='B_wait_logging');time.sleep(15)
 assert a.read(R/'记录/LOGGING_VERIFIED.json')['status']=='passed'
 while a.memory()['commit_free_GiB']<14 or a.memory()['physical_free_GiB']<6:a.progress(phase='B_resource_hold');time.sleep(30)
 interval=','.join(str(int(x)) for x in PROFILES[sel]);running={};attempts={(arm,s):0 for arm in ('e_sac','e_mve') for s in (0,1)};done=[]
 def start(arm,s):
  out=R/f'B/TRAIN_{arm}_s{s}';assert not out.exists();n=f'TRAIN_{arm}_s{s}_attempt{attempts[arm,s]}'
  a.launch(n,[R/'脚本/train_with_logs.py','train','--arm',arm,'--seed',str(s),'--steps','3000','--workers','2','--urgent-deadline',interval,'--checkpoints','500,1000,1500,2000,2500','--output',out]);running[n]=(arm,s)
 a.progress(phase='B_training',selected=sel,deadline=PROFILES[sel])
 for arm,s in attempts:start(arm,s)
 while running:
  for n,(arm,s) in list(running.items()):
   rc=a.finish(n)
   if rc is None:continue
   del running[n];out=R/f'B/TRAIN_{arm}_s{s}';fin=a.read(out/'finished.json')
   if fin.get('steps')==3000:
    manifest=a.read(out/'manifest.json');assert manifest['learner_version']=='t2-learner-2.5.1' and manifest['urgent_deadline_s']==PROFILES[sel]
    if rc:a.dump(R/f'记录/{arm}_s{s}_LOGGING_ONLY_FAILURE.json',{'jst':a.now(),'rc':rc,'science_finished3000':True,'not_retrained':True})
    done.append(f'{arm}_s{s}');continue
   if attempts[arm,s]:
    for other in list(running):a.stop_owned(other)
    raise RuntimeError(n+' second genuine failure: stop segment, preserve evidence')
   for pid,created in reversed(list(a.created[n].items())):
    try:
     proc=psutil.Process(pid)
     if proc.create_time()==created:proc.kill()
    except psutil.NoSuchProcess:pass
   dest=R/f'记录/restarts/TRAIN_{arm}_s{s}_attempt0';assert out.resolve().is_relative_to(R.resolve()) and dest.resolve().is_relative_to(R.resolve()) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(dest));a.dump(R/f'记录/RESTART_{arm}_s{s}.json',{'jst':a.now(),'from_zero':True,'same_config':True,'optimizer_replay_RNG_not_spliced':True});attempts[arm,s]=1;start(arm,s)
  early_mechanism();a.progress(phase='B_training',training_done=done);time.sleep(15)
 while not a.read(R/'记录/EARLY_EVAL_EXIT.json'):a.progress(phase='C_drain_early',training_done=done);time.sleep(10)
 a.progress(phase='C_evaluation')
 for arm in ('e_sac','e_mve'):
  for s in (0,1):
   for k in MARKS:
    label=f'{arm}_s{s}_step_{k}'
    group(f'C_{arm}_s{s}_{k}',[label],R/f'C_eval/{arm}_s{s}/step_{k}',['-m','experiments.t2','eval','--checkpoint',R/f'B/TRAIN_{arm}_s{s}/checkpoints/step_{k}.zip','--label',label,'--openings','294000-294023','--urgent-deadline',interval])
 assert len(list((R/'C_eval').rglob('*.json')))==672
 a.run('C_readout',['-m','experiments.t2_readout','curve','--eval-root',R/'C_eval','--train-root',R/'B','--screen',R/f'A/{sel}/screen','--output',R/'C_READOUT.json']);assert a.read(R/'C_READOUT.json')['all_complete']
 a.run('DECIDE',['-m','experiments.t2_decide','--criteria','t26','--readout',R/'C_READOUT.json','--train-root',R/'B','--output',R/'DECISION.json']);assert a.read(R/'DECISION.json')['verdict']!='incomplete'
 a.progress(phase='final_publication',verdict=a.read(R/'DECISION.json')['verdict']);ok=a.publish('final')
 a.progress(phase='complete_waiting_review' if ok else 'complete_publication_pending',finished_jst=a.now(),no_more_training=True);a.dump(R/'记录/PIPELINE_EXIT.json',a.S)
if __name__=='__main__':
 try:main()
 except Exception:
  a.dump(R/'记录/PIPELINE_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc(),'phase':a.S.get('phase'),'active_jobs_preserved_for_identity_checked_driver_repair':list(a.active)})
  raise
