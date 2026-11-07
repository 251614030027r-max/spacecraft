import os,time,json,re,shutil,traceback
from pathlib import Path
import psutil
import runtime as a
R=a.R;ARMS=('e_mve','e_sac');JOBS=[(arm,s) for arm in ARMS for s in (0,1)]
def manifest(arm,s):
 m=a.read(R/f'B/TRAIN_{arm}_s{s}/manifest.json')
 assert m['git_head']==a.SHA and m['env_version']=='t2-stage-0.5.0' and m['learner_version']=='t2-learner-2.7'
 assert m['guide'] and m['guide_p0']==.5 and m['guide_until']==2000 and m['workers']==2 and m['steps']==4000 and not m['smoke']
 assert m['k_candidates']==(8 if arm=='e_mve' else 0) and m['pooled_calibration']==(arm=='e_mve') and m['urgent_deadline_s']==[140.,200.]
def preflight():
 a.current('preflight');a.state(phase='preflight',started_jst=a.now(),order_collab=a.ORDER,science_SHA=a.SHA);a.frozen()
 old=[p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.info['name'].lower() in ('python.exe','pythonw.exe') and 'experiments.t2' in (p.info['cmdline'] or [])];assert not old,old
 a.run('pytest',['-m','pytest','-q','tests/test_t2_stage.py','-o','cache_dir='+str(R/'记录/pytest_cache'),'--basetemp',R/'记录/pytest_tmp'])
 raw=(R/'记录/pytest.out.log').read_bytes();(R/'pytest.log').write_bytes(raw);m=re.search(rb'(\d+) passed',raw);assert m and int(m[1])==52
 refs=[]
 for src,arm in ((a.W,'e_sac'),(a.W1,'e_mve')):
  for s in (0,1):
   for k in a.MARKS:
    d=src/f'C_eval/{arm}_s{s}/step_{k}';files=list(d.glob('*.json'));assert len(files)==48 and not list(d.glob('ERROR_*.json'))
    assert {a.read(p)['opening'] for p in files}==set(range(294000,294048))
    refs += [{'path':str(p),'SHA256':a.sha(p)} for p in files]
 for arm,s in JOBS:assert not (R/f'B/TRAIN_{arm}_s{s}').exists()
 a.dump(R/'记录/REFERENCE_SHA256.json',refs)
 a.run('logging_gate',[R/'脚本/verify_logging.py']);assert a.read(R/'记录/LOGGING_VERIFIED.json')['status']=='passed'
 # Also validate guidance-on SAC through the same actual entry point; no model feedback expected.
 out=R/'记录/日志验收/SAC_GUIDE_SMOKE'
 a.run('sac_logging_smoke',[R/'脚本/train_with_logs.py','train','--arm','e_sac','--seed','0','--steps','48','--workers','2','--checkpoints','16,32','--urgent-deadline','140,200','--smoke','--guide','--output',out])
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 fin=a.read(out/'finished.json');ep=a.read(out/'episodes.json');logs=[json.loads(x) for x in (out/'live/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
 assert fin['steps']==48 and fin['updates']>0 and fin['replay']['guided']>0 and len(ep)==len(logs)>0
 assert [(x['steps'],x['worker'],x['event'],x['guided']) for x in ep]==[(x['steps'],x['worker'],x['event'],x['guided']) for x in logs]
 ev=EventAccumulator(str(out/'live/tensorboard'));ev.Reload();assert all(t in ev.Tags()['scalars'] for t in ('progress/steps','rollout/return_true','rollout/guided','critic/real_td'))
 for t in ev.Tags()['scalars']:
  vals=ev.Scalars(t);assert all(__import__('math').isfinite(x.value) for x in vals) and all(y.step>=x.step for x,y in zip(vals,vals[1:]))
 assert a.read(out/'live/TORCH_THREADS.json')['intra_op']==1 and a.read(out/'manifest.json')['guide']
 a.frozen();a.dump(R/'记录/OBSERVER_SHA256.json',{n:a.sha(R/'脚本'/n) for n in ('logging_adapter.py','train_with_logs.py')})
 a.dump(R/'记录/PREFLIGHT_PASS.json',{'status':'passed','jst':a.now(),'pytest_native_exit':0,'pytest_passed':52,'source_SHA':a.SHA,'reused_evaluations_verified':len(refs),'EACC_and_SAC_guided_logging_validated':True})
def early():
 if (R/'EARLY1000.json').exists():return
 if not all((R/f'B/TRAIN_{arm}_s{s}/checkpoints/step_1000.zip').exists() for arm,s in JOBS):return
 out={}
 for arm,s in JOBS:
  root=R/f'B/TRAIN_{arm}_s{s}';hb=a.read(root/'heartbeat.json');eps=[x for x in a.read(root/'episodes.json') if x['steps']<=1000];ms=[x for x in a.read(root/'model_stats.json') if x['steps']<=1000 and x.get('model_rows')]
  assert hb['steps']==1000,'early heartbeat no longer1000; use checkpoint receipts rather than a later window'
  v={'stored_steps':1000,'guided':hb['replay'].get('guided',0),'guided_share':hb['replay'].get('guided',0)/1000,'mean_guided_per_episode':sum(x['guided'] for x in eps)/len(eps),'wall_s_per_decision':hb['wall_s']/1000,'first_model_step':ms[0]['steps'] if ms else None,'episodes':len(eps)}
  for urgent in (True,False):
   rows=[x for x in eps if x['mission']['urgent']==urgent];v['urgent' if urgent else 'economy']={'n':len(rows),'capture':sum(x['event']=='success' for x in rows),'capture_rate':sum(x['event']=='success' for x in rows)/len(rows) if rows else None}
  for k in ('two_stage_share','decisive_share','w2_mean','pi_gap_sigma_median','grad_cos','stage2_cutoff_share','entry_bias_now'):
   vals=[x[k] for x in ms if x.get(k) is not None];v[k]={'n':len(vals),'mean':sum(vals)/len(vals) if vals else None}
  out[f'{arm}_s{s}']=v
 a.dump(R/'EARLY1000.json',{'jst':a.now(),'not_a_stop_gate':True,'readings':out})
 try:
  import deliver;deliver.early()
 except Exception:a.dump(R/'记录/EARLY_PUBLICATION_PENDING.json',{'error':traceback.format_exc()})
def trainings():
 a.current('training',training=True);attempt={f'{arm}_s{s}':0 for arm,s in JOBS};done=set();a.state(phase='training',current_job='four_guided_trainings')
 def start(arm,s):
  key=f'{arm}_s{s}';a.launch('train_'+key+'_attempt'+str(attempt[key]),[R/'脚本/train_with_logs.py','train','--arm',arm,'--seed',s,'--steps','4000','--workers','2','--urgent-deadline','140,200','--guide','--checkpoints','1000,2000,3000','--output',R/f'B/TRAIN_{key}'])
 for arm,s in JOBS:start(arm,s)
 while len(done)<4:
  a.frozen()
  for arm,s in JOBS:
   key=f'{arm}_s{s}';root=R/f'B/TRAIN_{key}'
   if key in done:continue
   if (root/'manifest.json').exists():manifest(arm,s)
   if (root/'live/LOGGER_ERROR.json').exists():raise RuntimeError('logging failed '+key)
   if (root/'live/TORCH_THREADS.json').exists():assert a.read(root/'live/TORCH_THREADS.json')['intra_op']==1
   rc=a.finish('train_'+key+'_attempt'+str(attempt[key]))
   if rc is None:continue
   if rc==0:
    assert a.read(root/'finished.json')['steps']==4000;done.add(key)
   else:
    if attempt[key]>=1:raise RuntimeError(key+' failed twice; stop')
    dest=R/f'记录/restarts/TRAIN_{key}_attempt0';assert dest.resolve().is_relative_to(R.resolve()) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(root),str(dest));attempt[key]+=1;start(arm,s)
  # Snapshot each arm's actual1000 window immediately, preventing faster arms overwriting it at2000.
  for arm,s in JOBS:
   root=R/f'B/TRAIN_{arm}_s{s}';dest=R/f'记录/early1000/{arm}_s{s}'
   if not dest.exists() and (root/'checkpoints/step_1000.zip').exists():
    hb=a.read(root/'heartbeat.json')
    if hb.get('steps')==1000:
     dest.mkdir(parents=True);[shutil.copy2(root/n,dest/n) for n in ('heartbeat.json','episodes.json','model_stats.json')]
  # Reporting reads the preserved1000 snapshots, never stops or tunes training.
  if not (R/'EARLY1000.json').exists() and all((R/f'记录/early1000/{arm}_s{s}/heartbeat.json').exists() for arm,s in JOBS):
   import deliver
   try:deliver.early_snapshots()
   except Exception:a.dump(R/'记录/EARLY_PUBLICATION_PENDING.json',{'error':traceback.format_exc()})
  a.state(phase='training',current_job='four_guided_trainings',training_done=sorted(done),attempts=attempt)
  if len(done)<4:time.sleep(60)
def evaluations():
 a.current('evaluation',evaluation=True);a.state(phase='evaluation')
 for arm,s in JOBS:
  for k in a.MARKS:
   out=R/f'E_guided/{arm}_s{s}/step_{k}'
   a.run(f'eval_{arm}_s{s}_{k}',['-m','experiments.t2','eval','--checkpoint',R/f'B/TRAIN_{arm}_s{s}/checkpoints/step_{k}.zip','--label',f'g_{arm}_s{s}_step_{k}','--openings','294000-294047','--urgent-deadline','140,200','--workers','11','--output',out],retries=1)
   assert len(list(out.glob('*.json')))==48 and not list(out.glob('ERROR_*.json'));a.frozen()
 pairs={'C_eval':(R/'E_guided','e_mve',a.W/'C_eval','e_sac'),'C_guide_only':(R/'E_guided','e_sac',a.W/'C_eval','e_sac'),'C_labels':(R/'E_guided','e_mve',R/'E_guided','e_sac'),'C_vs_eacc':(R/'E_guided','e_mve',a.W1/'C_eval','e_mve')}
 for name,(mr,ma,sr,sa) in pairs.items():
  for s in (0,1):
   for k in a.MARKS:
    for root,arm,target in ((mr,ma,'e_mve'),(sr,sa,'e_sac')):
     src=root/f'{arm}_s{s}/step_{k}';dest=R/f'{name}/{target}_s{s}/step_{k}';shutil.copytree(src,dest)
 # Only small mechanism metadata required by readout/decision; no duplicate model ZIP or event tree.
 for s in (0,1):
  src=R/f'B/TRAIN_e_mve_s{s}';dest=R/f'B_mve/TRAIN_e_mve_s{s}';dest.mkdir(parents=True)
  for n in ('model_stats.json','finished.json','episodes.json','heartbeat.json','manifest.json'):shutil.copy2(src/n,dest/n)
 for name in pairs:
  a.run(name+'_readout',['-m','experiments.t2_readout','curve','--eval-root',R/name,'--train-root',R/'B_mve','--screen',a.W/'A_ref','--expected','48','--output',R/(name+'_READOUT.json')]);assert a.read(R/(name+'_READOUT.json'))['all_complete']
 a.run('decide',['-m','experiments.t2_decide','--criteria','t27','--checkpoints','1000,2000,3000,4000','--readout',R/'C_eval_READOUT.json','--train-root',R/'B_mve','--output',R/'DECISION.json']);assert a.read(R/'DECISION.json')['verdict']!='incomplete'
def main():
 me=psutil.Process();me.nice(psutil.NORMAL_PRIORITY_CLASS);me.cpu_affinity(list(range(12)))
 parents=[{'pid':p.pid,'created':p.create_time(),'name':p.name()} for p in me.parents()];assert not any('codex' in p['name'].lower() or 'chatgpt' in p['name'].lower() for p in parents)
 with (R/'记录/PIPELINE.lock').open('x') as f:f.write(str(me.pid))
 a.dump(R/'记录/OS_ANCESTRY.json',{'jst':a.now(),'pid':me.pid,'created':me.create_time(),'parents':parents,'affinity':me.cpu_affinity(),'priority':'Normal'})
 try:
  preflight();trainings();evaluations();a.current('publishing');a.state(phase='publishing',current_job='deliver')
  import deliver;deliver.final();a.state(phase='complete_delivered',current_job=None,finished_jst=a.now());a.dump(R/'记录/PIPELINE_EXIT.json',{'returncode':0,'jst':a.now(),'verdict':a.read(R/'DECISION.json')['verdict']})
 except BaseException:
  err=traceback.format_exc();a.stop_owned();a.current('failed_stopped');a.state(phase='failed_stopped',error=err);a.dump(R/'记录/PIPELINE_EXIT.json',{'returncode':1,'jst':a.now(),'traceback':err});raise
 finally:
  import subprocess
  subprocess.run(['schtasks','/Change','/TN','Spacecraft-T27-20261031','/Disable'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
if __name__=='__main__':main()
