"""OS-owned frozen T26W: admission, four simultaneous4000 runs, 48-opening evaluation, native t26w, deliver and stop."""
from pathlib import Path
import os,time,subprocess,traceback,shutil,psutil,hashlib,math
import runtime as a
R=a.R;SHA=a.SHA;MARKS=range(0,4001,500)
def early_mechanism():
 if (R/'记录/MECHANISM1000_CAPTURED.json').exists():return
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 result={}
 for seed in (0,1):
  if a.read(R/f'B/TRAIN_e_mve_s{seed}/live/progress.json').get('steps',0)<1050:return
  ea=EventAccumulator(str(R/f'B/TRAIN_e_mve_s{seed}/live/tensorboard'),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars'];rows={}
  for field in ('two_stage_share','decisive_share','stage2_rows','w2_mean','pi_gap_sigma_median','stage2_entry_share','c_mean','increment_abs_mean','grad_cos','loss_share','w_mean'):
   key=('critic/' if field in ('grad_cos','loss_share') else 'model/')+field
   values=[x for x in ea.Scalars(key) if x.step>1000] if key in tags else []
   rows[field]={'mean':sum(x.value for x in values)/len(values) if values else None,'n':len(values),'first_step':values[0].step if values else None,'last_step':values[-1].step if values else None,'MISSING':not bool(values)}
  values=ea.Scalars('critic/model_td') if 'critic/model_td' in tags else []
  result[f'e_mve_s{seed}']={'after1000':rows,'first_model_step':values[0].step if values else None,'T251_3worker_reference_step':(512,537)[seed],'sandbox_2worker_pooled_reference_step':362,'reference_not_local_observation':True}
 alerts=[]
 for tag,v in result.items():
  rows=v['after1000']
  if any(rows[f]['MISSING'] for f in ('w2_mean','pi_gap_sigma_median','grad_cos')) or (rows['w2_mean']['mean'] or 0)<.5 or (rows['pi_gap_sigma_median']['mean'] is not None and rows['pi_gap_sigma_median']['mean']>.25) or (rows['grad_cos']['mean'] or 0)<=0:alerts.append(tag)
 a.dump(R/'MECHANISM1000.json',{'jst':a.now(),'source_SHA':SHA,'deadline':[140.,200.],'readings':result,'alerts':alerts,'scope':'Actual native probes steps>1000, early window not final M; c reported only; no performance claim'})
 a.dump(R/'记录/MECHANISM1000_CAPTURED.json',{'jst':a.now(),'alerts':alerts});a.publish('mechanism1000')
def verify_live():
 if (R/'记录/FIRST_LIVE_GROWTH_VERIFIED.json').exists():return
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 result={}
 for arm in ('e_sac','e_mve'):
  for seed in (0,1):
   out=R/f'B/TRAIN_{arm}_s{seed}';manifest=a.read(out/'manifest.json')
   if not manifest:return
   assert manifest['git_head']==SHA and manifest['learner_version']=='t2-learner-2.6' and manifest['pooled_calibration']==(arm=='e_mve') and manifest['workers']==2 and manifest['steps']==4000 and manifest['urgent_deadline_s']==[140.,200.]
   path=out/'live/progress.jsonl'
   if not path.exists():return
   import json
   try:rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
   except json.JSONDecodeError:return
   if len(rows)<2 or not any(y['steps']>x['steps'] for x,y in zip(rows,rows[1:])):return
   ev=EventAccumulator(str(out/'live/tensorboard'));ev.Reload()
   if 'progress/steps' not in ev.Tags()['scalars']:return
   samples=ev.Scalars('progress/steps');assert all(math.isfinite(x.value) for x in samples) and all(y.step>=x.step for x,y in zip(samples,samples[1:]))
   result[f'{arm}_s{seed}']={'manifest_verified':True,'pooled_calibration':manifest['pooled_calibration'],'live_samples':len(rows),'first_steps':rows[0]['steps'],'last_steps':rows[-1]['steps'],'TB_progress_count':len(samples),'event_files':[{'path':str(p),'bytes':p.stat().st_size} for p in (out/'live/tensorboard').glob('events.out.tfevents.*')]}
 a.dump(R/'记录/FIRST_LIVE_GROWTH_VERIFIED.json',{'jst':a.now(),'four_runs':result})
def main():
 with (R/'记录/PIPELINE.lock').open('x') as f:f.write(str(os.getpid()))
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);me.cpu_affinity(list(range(psutil.cpu_count())))
 ancestors=[{'pid':p.pid,'name':p.name(),'created':p.create_time()} for p in me.parents()];assert not any('codex' in p['name'].lower() or 'chatgpt' in p['name'].lower() for p in ancestors)
 a.S.update(pid=me.pid,created=me.create_time(),started_jst=a.now(),science_SHA=SHA,order_collab='69d49fa',phase='admission',ancestor_proof=ancestors,deadline=[140.,200.],scope='No screen gate;4runs4000/2workers;48openings9points=1728;A_ref336;native t26w;no branch')
 while not (a.read(R/'记录/PREPARE_VERIFIED.json').get('status')=='passed' and a.read(R/'记录/LOGGING_VERIFIED.json').get('status')=='passed'):
  assert not a.read(R/'记录/PREPARE_ERROR.json') and not a.read(R/'记录/LOGGING_VERIFICATION_ERROR.json'),'Admission failure; no formal training';a.progress(phase='admission');time.sleep(15)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==SHA and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 hashes=a.read(R/'记录/EXTERNAL_ADAPTER_SHA256.json');assert all(hashlib.sha256((R/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
 while a.memory()['commit_free_GiB']<14 or a.memory()['physical_free_GiB']<3.5:a.progress(phase='B_resource_hold');time.sleep(30)
 running={};attempts={(arm,seed):0 for arm in ('e_sac','e_mve') for seed in (0,1)};done=[]
 def start(arm,seed):
  out=R/f'B/TRAIN_{arm}_s{seed}';assert not out.exists();name=f'TRAIN_{arm}_s{seed}_attempt{attempts[arm,seed]}'
  a.launch(name,[R/'脚本/train_with_logs.py','train','--arm',arm,'--seed',str(seed),'--steps','4000','--workers','2','--urgent-deadline','140,200','--checkpoints','500,1000,1500,2000,2500,3000,3500','--output',out]);running[name]=(arm,seed)
 a.progress(phase='B_training')
 for arm,seed in attempts:start(arm,seed)
 a.dump(R/'记录/FOUR_TRAINING_LAUNCH.json',{'jst':a.now(),'jobs':{n:{'pid':a.active[n].pid,'created':a.created[n][a.active[n].pid]} for n in running},'all_from_zero':True,'steps_each':4000,'workers_each':2})
 while running:
  for name,(arm,seed) in list(running.items()):
   rc=a.finish(name)
   if rc is None:continue
   del running[name];out=R/f'B/TRAIN_{arm}_s{seed}';finished=a.read(out/'finished.json')
   if finished.get('steps')==4000:
    manifest=a.read(out/'manifest.json');assert manifest['learner_version']=='t2-learner-2.6' and manifest['pooled_calibration']==(arm=='e_mve') and manifest['urgent_deadline_s']==[140.,200.]
    if rc:a.dump(R/f'记录/{arm}_s{seed}_LOGGING_ONLY_FAILURE.json',{'jst':a.now(),'returncode':rc,'science_finished4000':True,'not_retrained':True})
    done.append(f'{arm}_s{seed}');continue
   if attempts[arm,seed]:
    for other in list(running):a.stop_owned(other)
    raise RuntimeError(name+' second genuine failure: stop segment and preserve evidence')
   for pid,created in reversed(list(a.created[name].items())):
    try:
     p=psutil.Process(pid)
     if p.create_time()==created:p.kill()
    except psutil.NoSuchProcess:pass
   dest=R/f'记录/restarts/TRAIN_{arm}_s{seed}_attempt0';assert out.resolve().is_relative_to(R.resolve()) and dest.resolve().is_relative_to(R.resolve()) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(dest));a.dump(R/f'记录/RESTART_{arm}_s{seed}.json',{'jst':a.now(),'same_config':True,'from_zero':True,'optimizer_Replay_RNG_not_spliced':True});attempts[arm,seed]=1;start(arm,seed)
  verify_live();early_mechanism();a.progress(phase='B_training',training_done=done);time.sleep(15)
 while not (a.read(R/'记录/EARLY_EVAL_EXIT.json').get('drained') and a.read(R/'记录/REFERENCE_EXIT.json').get('all_complete')):
  assert not a.read(R/'记录/REFERENCE_ERROR.json'),'Reference administrative failure: preserve finished trainings';a.progress(phase='C_drain_aux',training_done=done);time.sleep(15)
 import parallel_eval
 parallel_eval.run(a)
 assert len(list((R/'C_eval').rglob('*.json')))==1728 and not list((R/'C_eval').rglob('ERROR_*.json'))
 a.run('C_readout',['-m','experiments.t2_readout','curve','--eval-root',R/'C_eval','--train-root',R/'B','--screen',R/'A_ref','--expected','48','--output',R/'C_READOUT.json']);assert a.read(R/'C_READOUT.json')['all_complete']
 a.run('DECIDE',['-m','experiments.t2_decide','--criteria','t26w','--readout',R/'C_READOUT.json','--train-root',R/'B','--output',R/'DECISION.json']);assert a.read(R/'DECISION.json')['verdict']!='incomplete'
 a.progress(phase='final_publication',verdict=a.read(R/'DECISION.json')['verdict']);ok=a.publish('final')
 a.progress(phase='complete_waiting_review' if ok else 'complete_publication_pending',finished_jst=a.now(),no_more_training=True);a.dump(R/'记录/PIPELINE_EXIT.json',a.S)
if __name__=='__main__':
 try:main()
 except Exception:a.dump(R/'记录/PIPELINE_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc(),'phase':a.S.get('phase'),'active_jobs_preserved_for_identity_checked_admin_repair':list(a.active)});raise
