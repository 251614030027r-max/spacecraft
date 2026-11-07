"""One low-priority registered evaluation using spare resources. No scientific configuration change."""
import os,time,json,subprocess,traceback,shutil
from pathlib import Path
import psutil
import runtime as a
R=a.R;TAG='e_sac_s0';K=4000;OUT=R/f'记录/early_eval_cache/{TAG}/step_{K}'
def rate(tag,window=900):
 p=R/f'B/TRAIN_{tag}/live/progress.jsonl'
 rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()];end=rows[-1];start=min(rows,key=lambda x:abs(x['wall_s']-(end['wall_s']-window)))
 return (end['steps']-start['steps'])/max(end['wall_s']-start['wall_s'],1)
def main():
 me=psutil.Process();assert not any('codex' in p.name().lower() or 'chatgpt' in p.name().lower() for p in me.parents());me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 assert not (R/'记录/EARLY_SPARE_LAUNCH.json').exists();a.frozen()
 checkpoint=R/f'B/TRAIN_{TAG}/checkpoints/step_{K}.zip';assert checkpoint.exists() and a.read(R/f'B/TRAIN_{TAG}/finished.json')['steps']==4000
 state=a.read(R/'RUN_STATE.json');active=[tag for tag,p in state['rolling'].items() if p and p['steps']<4000];baseline={tag:rate(tag) for tag in active}
 cmd=[a.PY,'-B','-m','experiments.t2','eval','--checkpoint',str(checkpoint),'--label',f'g_{TAG}_step_{K}','--openings','294000-294047','--urgent-deadline','140,200','--workers','1','--output',str(OUT)]
 with (R/'记录/early_spare.out.log').open('ab') as fo,(R/'记录/early_spare.err.log').open('ab') as fe:p=subprocess.Popen(cmd,cwd=a.REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
 ident=psutil.Process(p.pid).create_time();started=time.time();rec={'jst':a.now(),'pid':p.pid,'created':ident,'command':cmd,'priority':'BelowNormal','workers':1,'baseline_last15min_rates':baseline,'cache_not_native_output':True};a.dump(R/'记录/EARLY_SPARE_LAUNCH.json',rec)
 cur=a.read(a.CURRENT);cur['early_eval_resource_receipt']=str(R/'记录/EARLY_SPARE_LAUNCH.json');cur['evaluation_active']=True;cur['evaluation_scope']='one registered4000 checkpoint, low-priority single worker; training has priority';a.dump(a.CURRENT,cur)
 paused=None
 while p.poll() is None:
  try:
   root=psutil.Process(p.pid)
   for q in [root]+root.children(recursive=True):
    if q.name().lower().startswith('python'):q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
  except psutil.NoSuchProcess:pass
  state=a.read(R/'RUN_STATE.json')
  if psutil.virtual_memory().available<2*2**30:paused='available RAM below2GiB'
  if time.time()-started>=1000:
   for tag,b in baseline.items():
    if state.get('rolling',{}).get(tag,{}).get('steps',4000)<4000 and b>0 and rate(tag)<.8*b:paused='training throughput degraded over20% for '+tag
  if state.get('current_job')=='eval_e_sac_s0_4000':paused='native queue reached same target; avoid duplicate work'
  if paused:
   try:
    root=psutil.Process(p.pid);tree=root.children(recursive=True)+[root]
    for q in tree:q.terminate()
   except psutil.NoSuchProcess:pass
   break
  time.sleep(30)
 rc=p.wait();files=list(OUT.glob('*.json'));copied=0
 # Preserve every finished case; native evaluation skips these, partial work is never fabricated.
 dest=R/f'E_guided/{TAG}/step_{K}';dest.mkdir(parents=True,exist_ok=True)
 for f in files:
  if f.name.startswith('ERROR_'):continue
  row=a.read(f)
  if row.get('opening') not in range(294000,294048):continue
  target=dest/f.name
  if not target.exists():shutil.copy2(f,target);assert a.sha(f)==a.sha(target);copied+=1
 result={'jst':a.now(),'returncode':rc,'paused_reason':paused,'completed_cached_cases':len([f for f in files if not f.name.startswith('ERROR_')]),'new_cases_copied_bytewise':copied,'native_queue_skips_completed_cases':True,'no_training_restart_or_hot_observer_change':True};a.dump(R/'记录/EARLY_SPARE_EXIT.json',result)
 cur=a.read(a.CURRENT)
 if cur.get('execution_state')=='training':cur['evaluation_active']=False;cur['early_eval_result']=result;a.dump(a.CURRENT,cur)
 subprocess.run(['schtasks','/Change','/TN','Spacecraft-T27-SpareEval-20261101','/Disable'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
if __name__=='__main__':
 try:main()
 except Exception:a.dump(R/'记录/EARLY_SPARE_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc()});raise
