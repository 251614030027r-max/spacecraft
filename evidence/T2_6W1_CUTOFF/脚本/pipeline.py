"""OS-owned one-shot T2.6W1: gates -> two EACC runs -> evaluation -> audit -> delivery."""
import os, sys, time, re, json, shutil, traceback, math
from pathlib import Path
import psutil
import runtime as a
R=a.R
def manifest(seed):
 m=a.read(R/f'B/TRAIN_e_mve_s{seed}/manifest.json')
 assert m['git_head']==a.SHA and m['learner_version']=='t2-learner-2.6.1' and m['env_version']=='t2-stage-0.4.3'
 assert m['pooled_calibration'] is True and m['urgent_deadline_s']==[140.,200.]
 assert m['workers']==2 and m['steps']==4000 and m['seed']==seed and not m['smoke']
def checkpoints():
 expected={'e_sac_s0':'b1da71f7e14513cdaa248d1993d8abbad3da77b22ab23ca8fdd48a488130b8c3','e_sac_s1':'f0e3dbe597463b8fb0273477f9569af3cf0c3199e8827fa5b49088c367db0697'}
 paths={}
 index=a.read(R.parent/'本地资产清理_20261029/SAC_MODEL_KEEP_PATHS.json')
 for label,h in expected.items():
  p=a.W/f'B/TRAIN_{label}/checkpoints/step_4000.zip'
  if not p.exists():
   found=[Path(s) for s in index if label in s and Path(s).exists() and a.sha(s)==h];assert len(found)==1,(label,found);p=found[0]
  assert a.sha(p)==h,(label,'SAC SHA256 mismatch');paths[label]={'path':str(p),'SHA256':h}
 for s in (0,1):
  p=R/f'B/TRAIN_e_mve_s{s}/checkpoints/step_4000.zip'
  if p.exists():paths[f'e_mve_s{s}']={'path':str(p),'SHA256':a.sha(p)}
 a.dump(R/'记录/AUDIT_CHECKPOINT_SHA256.json',paths);return paths
def preflight():
 a.frozen();a.current('preflight');a.state(phase='preflight',started_jst=a.now(),science_SHA=a.SHA,order_collab=a.ORDER)
 old=[]
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.info['name'].lower() in ('python.exe','pythonw.exe') and '-m' in (p.info['cmdline'] or []) and any(x in ('experiments.t2','experiments.t2_rank') for x in p.info['cmdline']):old.append(p.info)
 assert not old,old
 assert not a.git('diff',a.SHA,a.AUDIT,'--','env','train'),'audit changed env/train'
 a.run('pytest',['-m','pytest','-q','tests/test_t2_stage.py','-o','cache_dir='+str(R/'记录/pytest_cache'),'--basetemp',R/'记录/pytest_tmp'])
 raw=(R/'记录/pytest.out.log').read_bytes();(R/'pytest.log').write_bytes(raw)
 m=re.search(rb'(\d+) passed',raw);assert m and int(m[1])==48,raw[-2000:]
 a.dump(R/'记录/PYTEST_VERIFIED.json',{'status':'passed','native_returncode':0,'actual_passed':48,'science_SHA':a.SHA,'jst':a.now()})
 reused=[]
 for s in (0,1):
  assert not (R/f'B/TRAIN_e_mve_s{s}').exists(),'training output already exists'
  for k in a.MARKS:
   src=a.W/f'C_eval/e_sac_s{s}/step_{k}';dest=R/f'C_eval/e_sac_s{s}/step_{k}'
   files=list(src.glob('*.json'));assert len(files)==48 and not list(src.glob('ERROR_*.json')),(s,k,len(files))
   assert {int(a.read(p)['opening']) for p in files}==set(range(294000,294048))
   shutil.copytree(src,dest)
   for p in files:
    assert a.sha(p)==a.sha(dest/p.name);reused.append({'source':str(p),'destination':str(dest/p.name),'SHA256':a.sha(p)})
 a.dump(R/'记录/SAC_EVAL_REUSE_SHA256.json',reused);checkpoints()
 a.run('logging_gate',[R/'脚本/verify_logging.py'])
 assert a.read(R/'记录/LOGGING_VERIFIED.json')['status']=='passed'
 a.frozen();a.dump(R/'记录/ADAPTER_SHA256.json',{str(p.relative_to(R)):a.sha(p) for p in (R/'脚本').glob('*.py')})
 a.dump(R/'记录/PREFLIGHT_PASS.json',{'jst':a.now(),'status':'passed','science_SHA':a.SHA,'pytest':48,'SAC_reused_cases':len(reused),'logging_verified':True,'no_training_output_preexists':True})
def train_args(s):return [R/'脚本/train_with_logs.py','train','--arm','e_mve','--seed',s,'--steps','4000','--workers','2','--urgent-deadline','140,200','--checkpoints','1000,2000,3000','--output',R/f'B/TRAIN_e_mve_s{s}']
def early():
 if (R/'MECHANISM1000.json').exists():return
 paths=[R/f'B/TRAIN_e_mve_s{s}' for s in (0,1)]
 if not all((p/'checkpoints/step_1000.zip').exists() and (p/'model_stats.json').exists() for p in paths):return
 keys=('stage2_cutoff_share','stage1_cutoff_share','w2_mean','pi_gap_sigma_median','grad_cos','entry_bias_now','two_stage_share','decisive_share')
 out={}
 for s,p in enumerate(paths):
  rows=a.read(p/'model_stats.json');used=[x for x in rows if x.get('model_rows') and x['steps']<=1000]
  out[f'e_mve_s{s}']={'first_model_step':used[0]['steps'] if used else None,'window':'actual probes <=1000; no later values included','probe_count':len(used)}
  for k in keys:
   vals=[float(x[k]) for x in used if x.get(k) is not None];out[f'e_mve_s{s}'][k]={'n':len(vals),'mean':sum(vals)/len(vals) if vals else None}
 a.dump(R/'MECHANISM1000.json',{'jst':a.now(),'not_a_stop_gate':True,'readings':out})
 try:
  import publish
  publish.early()
 except Exception:a.dump(R/'记录/EARLY_PUBLICATION_PENDING.json',{'jst':a.now(),'error':traceback.format_exc()})
def trainings():
 a.current('training',training=True);a.state(phase='training')
 attempts={0:0,1:0};done=set()
 for s in (0,1):a.launch(f'train_s{s}_attempt0',train_args(s))
 while len(done)<2:
  a.frozen()
  for s in (0,1):
   if s in done:continue
   out=R/f'B/TRAIN_e_mve_s{s}';name=f'train_s{s}_attempt{attempts[s]}'
   if (out/'manifest.json').exists():manifest(s)
   if (out/'live/LOGGER_ERROR.json').exists():raise RuntimeError('formal logger failed '+str(out))
   rc=a.finish(name)
   if rc is None:continue
   if rc==0:
    fin=a.read(out/'finished.json');assert fin['steps']==4000 and fin['updates']>0;manifest(s);done.add(s)
   else:
    if attempts[s]>=1:raise RuntimeError(f'seed{s} failed twice; stop per run order')
    # Preserve incomplete attempt; no checkpoint resume. Only one authorized from-zero restart.
    dest=R/f'记录/restarts/TRAIN_e_mve_s{s}_attempt0';assert dest.resolve().is_relative_to(R.resolve()) and not dest.exists()
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(dest));attempts[s]+=1
    a.launch(f'train_s{s}_attempt1',train_args(s))
  early();a.state(phase='training',training_done=sorted(done),attempts=attempts)
  if len(done)<2:time.sleep(60)
 early();checkpoints();a.current('evaluation',evaluation=True)
def evaluate():
 a.frozen();a.state(phase='evaluation')
 for s in (0,1):
  for k in a.MARKS:
   a.run(f'eval_s{s}_{k}',['-m','experiments.t2','eval','--checkpoint',R/f'B/TRAIN_e_mve_s{s}/checkpoints/step_{k}.zip','--label',f'e_mve_s{s}_step_{k}','--openings','294000-294047','--urgent-deadline','140,200','--workers','11','--output',R/f'C_eval/e_mve_s{s}/step_{k}'],retries=1)
   a.frozen();d=R/f'C_eval/e_mve_s{s}/step_{k}';assert len(list(d.glob('*.json')))==48 and not list(d.glob('ERROR_*.json'))
 a.run('C_readout',['-m','experiments.t2_readout','curve','--eval-root',R/'C_eval','--train-root',R/'B','--screen',a.W/'A_ref','--expected','48','--output',R/'C_READOUT.json'])
 assert a.read(R/'C_READOUT.json')['all_complete'] and len(list((R/'C_eval').rglob('*.json')))==768
 a.run('decide',['-m','experiments.t2_decide','--criteria','t26w','--checkpoints','1000,2000,3000,4000','--readout',R/'C_READOUT.json','--train-root',R/'B','--output',R/'DECISION.json'])
 assert a.read(R/'DECISION.json')['verdict']!='incomplete'
 a.dump(R/'记录/PERFORMANCE_FROZEN.json',{'jst':a.now(),'science_SHA':a.SHA,'C_READOUT_SHA256':a.sha(R/'C_READOUT.json'),'DECISION_SHA256':a.sha(R/'DECISION.json'),'verdict':a.read(R/'DECISION.json')['verdict'],'all_cases':768})
 a.current('performance_complete_audit_pending');a.state(phase='performance_complete_audit_pending',verdict=a.read(R/'DECISION.json')['verdict'])
def audit():
 assert not a.ACTIVE
 for s in (0,1):assert a.read(R/f'B/TRAIN_e_mve_s{s}/finished.json')['steps']==4000
 assert (R/'DECISION.json').exists();a.frozen()
 assert not a.git('diff',a.SHA,a.AUDIT,'--','env','train')
 a.git('checkout','--detach',a.AUDIT);a.frozen(a.AUDIT)
 a.current('mechanism_audit',audit=True);a.state(phase='mechanism_audit',audit_SHA=a.AUDIT)
 a.run('D_pytest',['-m','pytest','-q','tests/test_t2_stage.py','-o','cache_dir='+str(R/'记录/D_pytest_cache'),'--basetemp',R/'记录/D_pytest_tmp'])
 raw=(R/'记录/D_pytest.out.log').read_bytes();(R/'D_pytest.log').write_bytes(raw);m=re.search(rb'(\d+) passed',raw);assert m and int(m[1])==49
 a.run('D_warmup',['-m','experiments.t2_audit','warmup','--output',R/'D_rank/CAL.json','--episodes','40','--workers','11'])
 paths=checkpoints()
 for label in ('e_mve_s0','e_mve_s1','e_sac_s0','e_sac_s1'):
  a.frozen(a.AUDIT)
  a.run('D_'+label,['-m','experiments.t2_rank','run','--checkpoint',paths[label]['path'],'--label',label+'_step_4000','--openings','294000-294047','--urgent-deadline','140,200','--calibration',R/'D_rank/CAL.json','--window','48,112','--max-decisions','3','--workers','11','--output',R/'D_rank/runs'],retries=1)
  files=list((R/'D_rank/runs').glob(label+'_step_4000_*.json'));assert len(files)==48
 a.run('D_summary',['-m','experiments.t2_rank','summary','--input',R/'D_rank/runs','--output',R/'D_RANK.json'])
 assert len(a.read(R/'D_RANK.json'))==4
 a.dump(R/'D_rank/run.log',{'logs':'Original per-run stdout/stderr are in 记录/D_e_* logs; copied to D_rank below.'})
 with (R/'D_rank/run.log').open('wb') as f:
  for p in sorted((R/'记录').glob('D_e_*.out.log')):f.write(p.read_bytes())
 a.frozen(a.AUDIT)
def main():
 for fd,name in ((1,'pipeline.out.log'),(2,'pipeline.err.log')):
  f=(R/'记录'/name).open('ab',buffering=0);os.dup2(f.fileno(),fd)
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 parents=[{'pid':p.pid,'created':p.create_time(),'name':p.name()} for p in me.parents()]
 assert not any('codex' in x['name'].lower() or 'chatgpt' in x['name'].lower() for x in parents),parents
 with (R/'记录/PIPELINE.lock').open('x') as f:f.write(json.dumps({'pid':me.pid,'created':me.create_time()}))
 a.dump(R/'记录/OS_ANCESTRY.json',{'jst':a.now(),'pid':me.pid,'created':me.create_time(),'parents':parents,'scheduler_owns_process':True})
 try:
  preflight();trainings();evaluate();audit();a.current('publishing');a.state(phase='publishing')
  import publish
  publish.final()
  a.state(phase='complete_delivered',finished_jst=a.now(),training_active=False,evaluation_active=False,audit_active=False)
  a.dump(R/'记录/PIPELINE_EXIT.json',{'returncode':0,'jst':a.now(),'verdict':a.read(R/'DECISION.json')['verdict']})
 except BaseException:
  error=traceback.format_exc();a.stop_owned();a.current('failed_stopped');a.state(phase='failed_stopped',error=error)
  a.dump(R/'记录/PIPELINE_EXIT.json',{'returncode':1,'jst':a.now(),'traceback':error});raise
 finally:
  try:a.run('disable_task',['schtasks','/Change','/TN','Spacecraft-T26W1-20261030','/Disable'],python=False)
  except Exception:pass
if __name__=='__main__':main()
