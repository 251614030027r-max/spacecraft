"""Registered deterministic checkpoints only; spare CPU one case at a time; drain before final pool."""
from pathlib import Path
import os,json,time,subprocess,traceback,datetime,ctypes
import psutil
R=Path(__file__).resolve().parents[1];PY='D:/py/DRL2/.venv/Scripts/python.exe'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
def read(p):
 try:return json.loads(p.read_text(encoding='utf-8-sig'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
def dump(n,d):
 p=R/'记录'/n;t=p.with_suffix('.writing');t.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def done():return all(read(R/f'B/TRAIN_{arm}_s{s}/finished.json').get('steps')==4000 for arm in ('e_sac','e_mve') for s in (0,1))
def main():
 me=psutil.Process();anc=[{'pid':p.pid,'name':p.name(),'created':p.create_time()} for p in me.parents()];assert not any('codex' in p['name'].lower() or 'chatgpt' in p['name'].lower() for p in anc);me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);me.cpu_affinity(read(R/'记录/DESKTOP_CPU_RESERVATION.json').get('experiment_allowed_cpus',list(range(psutil.cpu_count()))))
 dump('EARLY_FAST_READY.json',{'pid':me.pid,'ancestors':anc,'max_workers':1,'CPU_admission_percent':92})
 while True:
  if done():return
  if read(R/'记录/PIPELINE_ERROR.json') or read(R/'记录/PIPELINE_EXIT.json'):return
  state=read(R/'RUN_STATE.json')
  if state.get('phase')!='B_training':time.sleep(15);continue
  cpu=psutil.cpu_percent(interval=3);ram=psutil.virtual_memory().available/2**30
  class M(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','physical','limit','commit','total_virtual','virtual','extended')]
  mm=M();mm.length=ctypes.sizeof(mm);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mm))
  if cpu>=92 or ram<2.8 or mm.commit/2**30<5.5:
   dump('EARLY_EVAL_STATE.json',{'jst':now(),'phase':'resource_hold','cpu':cpu,'RAM_GiB':ram,'max_workers':1,'CPU_admission_percent':92});time.sleep(30);continue
  job=None
  for k in (500,1000,1500,2000,2500,3000,3500,0):
   for arm,s in ((a,b) for a in ('e_sac','e_mve') for b in (0,1)):
    checkpoint=R/f'B/TRAIN_{arm}_s{s}/checkpoints/step_{k}.zip';label=f'{arm}_s{s}_step_{k}';out=R/f'C_eval/{arm}_s{s}/step_{k}'
    if read(R/f'B/TRAIN_{arm}_s{s}/live/progress.json').get('steps',0)<=k:continue
    if not checkpoint.exists():continue
    for o in range(294000,294048):
     if not (out/f'{label}_{o}.json').exists() and not (out/f'ERROR_{label}_{o}.json').exists():job=(arm,s,k,o,checkpoint,label,out);break
    if job:break
   if job:break
  if not job:time.sleep(30);continue
  arm,s,k,o,checkpoint,label,out=job
  cmd=[PY,'-B','-m','experiments.t2','eval','--checkpoint',str(checkpoint),'--label',label,'--openings',str(o),'--workers','1','--urgent-deadline','140,200','--output',str(out)]
  with (R/f'记录/EARLY_{arm}_s{s}_{k}_{o}.out.log').open('ab') as fo,(R/f'记录/EARLY_{arm}_s{s}_{k}_{o}.err.log').open('ab') as fe:
   p=subprocess.Popen(cmd,cwd='D:/py/DRL2',stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
   while p.poll() is None:dump('EARLY_EVAL_STATE.json',{'jst':now(),'phase':'evaluating','pid':p.pid,'created':psutil.Process(p.pid).create_time(),'seed':s,'checkpoint':k,'opening':o,'max_workers':1,'CPU_admission_percent':92});time.sleep(15)
   assert p.returncode==0 and (out/f'{label}_{o}.json').exists(),'Preserve original stderr; final pool may handle registered one retry'
if __name__=='__main__':
 ok=False
 try:main();ok=True
 except Exception:dump('EARLY_EVAL_ERROR.json',{'jst':now(),'traceback':traceback.format_exc()})
 finally:dump('EARLY_EVAL_EXIT.json',{'jst':now(),'ok':ok,'drained':True})
