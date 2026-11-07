"""OS-owned frozen T2.4: logging gate -> two MVE trainings -> evaluation -> native decision -> delivery."""
from pathlib import Path
import os,json,time,datetime as dt,ctypes,subprocess,traceback,shutil
import psutil
R=Path(__file__).resolve().parents[1];OLD=R.parent/'T2_3_TEST_20261025';REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe');SHA='e1166eb5daee069e7f14b6f48d24dcf43c36d66f'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(REPO))
os.chdir(REPO)
for fd,n in ((1,'pipeline.out.log'),(2,'pipeline.err.log')):
 f=(R/'记录'/n).open('ab',buffering=0);os.dup2(f.fileno(),fd)
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
def read(p):
 for _ in range(40):
  try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
  except FileNotFoundError:return {}
  except (PermissionError,json.JSONDecodeError):time.sleep(.1)
 return {}
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.writing');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
 for _ in range(100):
  try:os.replace(t,p);return
  except PermissionError:time.sleep(.1)
 raise RuntimeError('locked '+str(p))
def memory():
 class M(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','physical','limit','commit','total_virtual','virtual','extended')]
 m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'commit_free_GiB':m.commit/2**30,'physical_free_GiB':m.physical/2**30}
S={};active={};created={};starts={}
def progress(**kw):
 S.update(kw,updated_jst=now(),memory=memory(),processes={n:p.pid for n,p in active.items()},rolling={f'{arm}_s{s}':read(R/f'B/TRAIN_{arm}_s{s}/live/progress.json') for arm in ('e_sac','e_mve') for s in (0,1)},preparation=read(R/'STATE.json'),eval_completed=len(list((R/'C_eval').rglob('*.json'))),A_ref_completed=len(list((R/'A_ref').rglob('*.json'))))
 dump(R/'RUN_STATE.json',S)
def launch(n,args):
 fo=(R/f'记录/{n}.out.log').open('ab');fe=(R/f'记录/{n}.err.log').open('ab');p=subprocess.Popen([PY,'-B',*map(str,args)],cwd=REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);fo.close();fe.close();active[n]=p;created[n]={p.pid:psutil.Process(p.pid).create_time()};starts[n]=time.time()
 dump(R/f'记录/{n}_LAUNCH.json',{'jst':now(),'pid':p.pid,'created':created[n][p.pid],'command':[PY,'-B',*map(str,args)]});return p
def capture(n):
 try:
  for p in psutil.Process(active[n].pid).children(recursive=True):created[n][p.pid]=p.create_time()
 except psutil.NoSuchProcess:pass
def stop_owned(n):
 capture(n)
 for pid,t in reversed(list(created[n].items())):
  try:
   p=psutil.Process(pid)
   if p.create_time()==t:p.kill()
  except psutil.NoSuchProcess:pass
def finish(n):
 rc=active[n].poll()
 if rc is None:capture(n);return None
 dump(R/f'记录/{n}_EXIT.json',{'jst':now(),'returncode':rc,'wall_s':time.time()-starts[n]});del active[n];return rc
def run(n,args):
 launch(n,args)
 while True:
  rc=finish(n)
  if rc is not None:assert rc==0,n+' failed; read original stderr';return
  progress(current_job=n);time.sleep(15)
def publish(stage):
 for i in range(3):
  try:run(f'publish_{stage}_{i}',[R/'脚本/publish_t26w.py','--stage',stage]);return True
  except AssertionError:time.sleep(15)
 progress(publication_pending=stage);return False
