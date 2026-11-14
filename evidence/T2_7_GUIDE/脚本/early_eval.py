"""Overlap one registered evaluation with training. Resource scheduling only."""
from pathlib import Path
import os,time,json,subprocess,traceback
import psutil
import runtime as a
R=a.R;OUT=R/'E_guided/e_sac_s0/step_4000'
def rate(tag,window=900):
 p=R/f'B/TRAIN_{tag}/live/progress.jsonl'
 rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()];end=rows[-1];start=min(rows,key=lambda x:abs(x['wall_s']-max(0,end['wall_s']-window)));seconds=end['wall_s']-start['wall_s']
 return (end['steps']-start['steps'])/seconds if seconds>window*.8 else None
def stop_owned(child):
 try:tree=[psutil.Process(child.pid)]+psutil.Process(child.pid).children(recursive=True)
 except psutil.NoSuchProcess:return
 for p in reversed(tree):
  try:p.kill()
  except psutil.NoSuchProcess:pass
 psutil.wait_procs(tree,timeout=15)
 # Preserve any interrupted partial JSON; only complete valid result files remain reusable.
 for p in OUT.glob('*.json'):
  try:json.loads(p.read_text(encoding='utf-8'))
  except (json.JSONDecodeError,UnicodeDecodeError):
   d=R/'记录/early_eval_partial'/p.name;d.parent.mkdir(parents=True,exist_ok=True);assert not d.exists();p.rename(d)
def main():
 me=psutil.Process();assert not any('codex' in p.name().lower() or 'chatgpt' in p.name().lower() for p in me.parents());me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 state=a.read(R/'RUN_STATE.json');assert state['phase']=='training','main evaluation already started; no early task needed'
 assert a.read(R/'B/TRAIN_e_sac_s0/finished.json')['steps']==4000 and psutil.virtual_memory().available>3*2**30
 a.frozen();assert not OUT.exists()
 owner=a.read(R/'记录/OS_ANCESTRY.json');root=psutil.Process(owner['pid']);assert root.create_time()==owner['created']
 tags=[f'{arm}_s{s}' for arm in ('e_mve','e_sac') for s in (0,1)];base={tag:rate(tag) for tag in tags if not (R/f'B/TRAIN_{tag}/finished.json').exists()}
 cmd=[a.PY,'-B','-m','experiments.t2','eval','--checkpoint',str(R/'B/TRAIN_e_sac_s0/checkpoints/step_4000.zip'),'--label','g_e_sac_s0_step_4000','--openings','294000-294047','--urgent-deadline','140,200','--workers','1','--output',str(OUT)]
 with (R/'记录/early_eval.out.log').open('ab') as fo,(R/'记录/early_eval.err.log').open('ab') as fe:
  child=subprocess.Popen(cmd,cwd=a.REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
 record={'jst':a.now(),'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'command':cmd,'priority':'BelowNormal','workers':1,'baseline_rates':base,'existing_science_not_interrupted':True,'criterion_unchanged':True,'target':'registered SAC+guide seed0 4000; native queue reuses complete JSON'}
 a.dump(R/'输入/EARLY_EVAL_RESOURCE_ORDER.json',record)
 cur=a.read(a.CURRENT);cur['early_evaluation_state']=str(R/'记录/EARLY_EVAL_STATE.json');cur['evaluation_active']=True;cur['updated_at']=a.now();a.dump(a.CURRENT,cur)
 bad=0;lastcheck=time.monotonic();why=None
 try:
  while child.poll() is None:
   st=a.read(R/'RUN_STATE.json');job=st.get('current_job','')
   # The main queue reaches this point much later. Stop only this owned early evaluation before its output can collide.
   if st['phase']=='failed_stopped' or job in ('eval_e_sac_s0_2000','eval_e_sac_s0_3000','eval_e_sac_s0_4000'):
    root.suspend()
    try:stop_owned(child)
    finally:root.resume()
    why='yield_to_native_queue; complete opening files retained';break
   if psutil.virtual_memory().available<2*2**30:stop_owned(child);why='yield_for_memory';break
   if st['phase']=='training' and time.monotonic()-lastcheck>=300:
    ratios={tag:rate(tag)/v for tag,v in base.items() if v and not (R/f'B/TRAIN_{tag}/finished.json').exists() and rate(tag) is not None}
    bad=bad+1 if any(x<.75 for x in ratios.values()) else 0;lastcheck=time.monotonic()
    if bad>=2:stop_owned(child);why='yield_for_sustained_training_throughput_drop; operational trigger not causal attribution';break
   a.dump(R/'记录/EARLY_EVAL_STATE.json',dict(record,updated_jst=a.now(),completed=len(list(OUT.glob('*.json'))),state='running'))
   time.sleep(30)
  if why is None:
   assert child.returncode==0 and len(list(OUT.glob('*.json')))==48 and not list(OUT.glob('ERROR_*.json'));why='completed48_native_queue_will_skip'
 finally:
  if child.poll() is None:stop_owned(child)
  a.dump(R/'记录/EARLY_EVAL_STATE.json',dict(record,updated_jst=a.now(),state=why or 'error_stopped',completed=len(list(OUT.glob('*.json')))))
  subprocess.run(['schtasks','/Change','/TN','Spacecraft-T27-EarlyEval-20261101','/Disable'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
if __name__=='__main__':
 try:main()
 except Exception:a.dump(R/'记录/EARLY_EVAL_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc()});raise
