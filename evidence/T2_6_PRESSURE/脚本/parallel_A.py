"""Administrative case scheduling only; calls frozen native guarded jobs unchanged."""
from pathlib import Path
import os,json,time,subprocess,sys,traceback,hashlib,shutil
import psutil
R=Path(__file__).resolve().parents[1];PY='D:/py/DRL2/.venv/Scripts/python.exe'
PROFILES={'p170':[170.,210.],'p155':[155.,195.],'p140':[140.,180.]}
POLICIES=('rally','c_10_0','c_10_-5','c_6_5','c_10_5','gstar','pure')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
def read(p):
 try:return json.loads(p.read_text(encoding='utf-8-sig'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
def jobs():
 for profile,deadline in PROFILES.items():
  for kind,label,checkpoint in [('screen',p,None) for p in POLICIES]+[('eval','sac_s0_4000',R.parent/'T2_3_TEST_20261025/TRAIN_e_sac_s0/checkpoints/step_4000.zip'),('eval','mve_s0_4000',R.parent/'T2_5_1_VALIDATE_20261026/TRAIN_e_mve_s0/checkpoints/step_4000.zip')]:
   out=R/'A'/profile/('screen' if kind=='screen' else 'probe')
   for opening in range(294000,294024):
    key=f'{profile}_{label}_{opening}';target=out/f'{label}_{opening}.json'
    yield key,profile,kind,label,checkpoint,opening,out,deadline,target
def worker():
 sys.path.insert(0,'D:/py/DRL2');os.chdir('D:/py/DRL2')
 from experiments import t2
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 claims=R/'记录/A_case_claims';claims.mkdir(exist_ok=True)
 for key,profile,kind,label,checkpoint,opening,out,deadline,target in jobs():
  if profile=='p170' and kind=='screen' and not (R/'记录/ORIGINAL_A_SCREEN_EXIT.json').exists():continue
  error=out/f'ERROR_{label}_{opening}.json'
  if target.exists() or error.exists():continue
  claim=claims/(key+'.lock')
  try:
   with claim.open('x') as f:f.write(json.dumps({'pid':me.pid,'created':me.create_time()}))
  except FileExistsError:continue
  try:
   if target.exists() or error.exists():continue
   out.mkdir(parents=True,exist_ok=True)
   status=R/f'记录/A_worker_{me.pid}.json';status.write_text(json.dumps({'pid':me.pid,'created':me.create_time(),'case':key,'phase':'running','time':time.time()}),encoding='utf-8')
   if kind=='screen':t2.guarded_screen((opening,label,str(out),True,deadline))
   else:t2.guarded_eval((opening,str(checkpoint),label,str(out),True,deadline))
   assert target.exists() or error.exists(),'native job produced no result'
  finally:claim.unlink(missing_ok=True)
def alive(identity):
 if not identity or 'pid' not in identity:return True
 try:
  p=psutil.Process(identity['pid']);return p.is_running() and p.create_time()==identity['created']
 except psutil.NoSuchProcess:return False
def run(a):
 a.dump(R/'记录/A_PARALLEL_READY.json',{'pid':os.getpid(),'method':'unchanged frozen guarded_screen/guarded_eval; exclusive per-case claims; original p170 screen retained','max_total_case_workers':11,'physical_reserve_GiB':1.2,'commit_reserve_GiB':4})
 original=a.read(R/'记录/A_PARALLEL_HANDOFF.json')['original_screen_identity'];pool={};serial=0;errors_retried=set();last_dispatch=0
 while True:
  old_alive=any(alive(x) for x in original)
  if not old_alive and not (R/'记录/ORIGINAL_A_SCREEN_EXIT.json').exists():a.dump(R/'记录/ORIGINAL_A_SCREEN_EXIT.json',{'jst':a.now(),'all_original_screen_processes_exited':True})
  for pid,(child,name) in list(pool.items()):
   if child.poll() is None:continue
   a.dump(R/f'记录/{name}_EXIT.json',{'jst':a.now(),'returncode':child.returncode});del pool[pid]
  # A dead worker can leave only its own claim; clear by exact process identity.
  for claim in (R/'记录/A_case_claims').glob('*.lock'):
   if not alive(read(claim)):
    dest=R/'记录/parallel_migration/stale_claims'/claim.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(claim),str(dest.with_name(dest.stem+f'_{time.time_ns()}.json')))
  all_jobs=list(jobs());pending=[j for j in all_jobs if not j[-1].exists() and not (j[6]/f'ERROR_{j[3]}_{j[5]}.json').exists() and not (j[1]=='p170' and j[2]=='screen' and old_alive)]
  if not pool and not old_alive and not pending:
   failures=[j for j in all_jobs if not j[-1].exists()]
   if not failures:break
   for j in failures:
    assert j[0] not in errors_retried,'second native case error '+j[0]
    errors_retried.add(j[0]);src=j[6]/f'ERROR_{j[3]}_{j[5]}.json';assert src.exists()
    dst=R/'记录/episode_errors/parallel_A'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst.with_name(j[0]+'.json')))
   pending=failures
  memory=a.memory();cpu=psutil.cpu_percent(interval=3)
  gate_running=not bool(a.read(R/'记录/LOGGING_GATE_EXIT.json'));reserved=(2 if old_alive else 0)+(2 if gate_running else 0)
  allowed=max(0,11-reserved)
  if pending and len(pool)<allowed and cpu<92 and memory['physical_free_GiB']>=1.75 and memory['commit_free_GiB']>=4.8 and time.time()-last_dispatch>=15:
   name=f'A_parallel_worker_{serial}';serial+=1
   fo=(R/f'记录/{name}.out.log').open('ab');fe=(R/f'记录/{name}.err.log').open('ab')
   cmd=[PY,'-B',str(Path(__file__).resolve()),'--worker'];child=subprocess.Popen(cmd,cwd='D:/py/DRL2',stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);fo.close();fe.close();pool[child.pid]=(child,name);last_dispatch=time.time()
   a.dump(R/f'记录/{name}_LAUNCH.json',{'jst':a.now(),'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'command':cmd})
  state={'jst':a.now(),'extra_case_workers':len(pool),'original_case_workers':2 if old_alive else 0,'logging_smoke_reserved_workers':2 if gate_running else 0,'max_total_case_workers':11,'cpu_percent':cpu,'memory':memory,'complete':sum(j[-1].exists() for j in all_jobs),'pending':len(pending)}
  a.dump(R/'记录/A_PARALLEL_STATE.json',state);a.progress(phase='A_parallel',parallel=state)
  time.sleep(10)
 a.dump(R/'记录/A_PARALLEL_EXIT.json',{'jst':a.now(),'complete':648,'no_duplicate_case_dispatch':True,'original_screen_preserved':True,'retry_cases':sorted(errors_retried)})
if __name__=='__main__':
 assert sys.argv[1:]==['--worker'];worker()
