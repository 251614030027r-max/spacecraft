"""Dynamic deterministic evaluation scheduling; frozen guarded_eval and arguments unchanged."""
from pathlib import Path
import os,sys,json,time,subprocess,psutil,shutil
R=Path(__file__).resolve().parents[1];PY='D:/py/DRL2/.venv/Scripts/python.exe';MARKS=(0,500,1000,1500,2000,2500,3000,4000)
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
def read(p):
 try:return json.loads(p.read_text(encoding='utf-8-sig'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
def jobs():
 for k in MARKS:
  for arm in ('e_sac','e_mve'):
   for seed in (0,1):
    label=f'{arm}_s{seed}_step_{k}';out=R/f'C_eval/{arm}_s{seed}/step_{k}';checkpoint=R/f'B/TRAIN_{arm}_s{seed}/checkpoints/step_{k}.zip'
    for opening in range(294000,294048):yield f'{label}_{opening}',opening,str(checkpoint),label,out
def alive(identity):
 if not identity or 'pid' not in identity:return True
 try:return psutil.Process(identity['pid']).create_time()==identity['created']
 except psutil.NoSuchProcess:return False
def worker():
 sys.path.insert(0,'D:/py/DRL2');os.chdir('D:/py/DRL2');from experiments.t2 import guarded_eval
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);claims=R/'记录/C_case_claims';claims.mkdir(exist_ok=True)
 for key,opening,checkpoint,label,out in jobs():
  target=out/(key+'.json');error=out/('ERROR_'+key+'.json')
  if target.exists() or error.exists():continue
  claim=claims/(key+'.lock')
  try:
   with claim.open('x') as f:f.write(json.dumps({'pid':me.pid,'created':me.create_time()}))
  except FileExistsError:continue
  try:
   if target.exists() or error.exists():continue
   out.mkdir(parents=True,exist_ok=True);(R/f'记录/C_worker_{me.pid}.json').write_text(json.dumps({'pid':me.pid,'created':me.create_time(),'case':key,'time':time.time()}),encoding='utf-8')
   guarded_eval((opening,checkpoint,label,str(out),True,[140.,200.]));assert target.exists() or error.exists()
  finally:claim.unlink(missing_ok=True)
def run(a):
 assert a.read(R/'记录/EARLY_EVAL_EXIT.json').get('drained') and a.read(R/'记录/REFERENCE_EXIT.json').get('all_complete')
 jobs_=list(jobs());pool={};serial=0;retried=set();last_dispatch=0;unexpected_failures=0
 a.dump(R/'记录/EVAL_POOL_READY.json',{'jst':a.now(),'pid':os.getpid(),'method':'native guarded_eval unchanged, per-case exclusive claims','max_workers':11,'registered_cases':1536,'physical_reserve_GiB':1.2,'commit_reserve_GiB':4})
 while True:
  for pid,(child,name) in list(pool.items()):
   if child.poll() is None:continue
   a.dump(R/f'记录/{name}_EXIT.json',{'jst':a.now(),'returncode':child.returncode});del pool[pid]
   if child.returncode:unexpected_failures+=1
  assert unexpected_failures<=1,'Evaluation worker infrastructure failed twice; preserve original stderr'
  for claim in (R/'记录/C_case_claims').glob('*.lock'):
   if not alive(read(claim)):
    dest=R/'记录/episode_errors/stale_claims';dest.mkdir(parents=True,exist_ok=True);shutil.move(str(claim),str(dest/(claim.stem+f'_{time.time_ns()}.json')))
  pending=[j for j in jobs_ if not (j[4]/(j[0]+'.json')).exists() and not (j[4]/('ERROR_'+j[0]+'.json')).exists()]
  if not pool and not pending:
   missing=[j for j in jobs_ if not (j[4]/(j[0]+'.json')).exists()]
   if not missing:break
   for j in missing:
    assert j[0] not in retried,'Second native episode failure '+j[0]
    src=j[4]/('ERROR_'+j[0]+'.json');assert src.exists();retried.add(j[0]);dest=R/'记录/episode_errors/C_eval';dest.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dest/src.name))
   pending=missing
  m=a.memory();cpu=psutil.cpu_percent(interval=3)
  if pending and len(pool)<11 and cpu<92 and m['physical_free_GiB']>=1.75 and m['commit_free_GiB']>=4.8 and time.time()-last_dispatch>=15:
   name=f'C_pool_worker_{serial}';serial+=1
   with (R/f'记录/{name}.out.log').open('ab') as fo,(R/f'记录/{name}.err.log').open('ab') as fe:child=subprocess.Popen([PY,'-B',str(Path(__file__).resolve()),'--worker'],cwd='D:/py/DRL2',stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
   pool[child.pid]=(child,name);last_dispatch=time.time();a.dump(R/f'记录/{name}_LAUNCH.json',{'jst':a.now(),'pid':child.pid,'created':psutil.Process(child.pid).create_time()})
  state={'jst':a.now(),'active_workers':len(pool),'max_workers':11,'complete':sum((j[4]/(j[0]+'.json')).exists() for j in jobs_),'pending':len(pending),'memory':m,'cpu_percent':cpu,'worker_pids':list(pool)}
  a.dump(R/'记录/EVAL_POOL_STATE.json',state);a.progress(phase='C_evaluation',eval_pool=state);time.sleep(10)
 a.dump(R/'记录/EVAL_POOL_EXIT.json',{'jst':a.now(),'cases':1536,'retry_cases':sorted(retried),'all_complete':True})
if __name__=='__main__':assert sys.argv[1:]==['--worker'];worker()
