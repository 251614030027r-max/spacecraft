"""Concurrent low-priority fixed system references, never a training prerequisite."""
from pathlib import Path
import os,json,time,subprocess,traceback,psutil,ctypes,datetime,shutil
R=Path(__file__).resolve().parents[1];PY='D:/py/DRL2/.venv/Scripts/python.exe'
POLICIES=('rally','c_10_0','c_10_-5','c_6_5','c_10_5','gstar','pure')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def read(p):
 try:return json.loads(p.read_text(encoding='utf-8-sig'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
def dump(n,v):
 p=R/'记录'/n;t=p.with_suffix('.writing');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def mem():
 class M(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','physical','limit','commit','total_virtual','virtual','extended')]
 m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return m.physical/2**30,m.commit/2**30
def main():
 p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);anc=[{'pid':q.pid,'created':q.create_time(),'name':q.name()} for q in p.parents()];assert not any('codex' in x['name'].lower() or 'chatgpt' in x['name'].lower() for x in anc)
 with (R/'记录/REFERENCE.lock').open('x') as f:f.write(str(p.pid))
 dump('REFERENCE_READY.json',{'pid':p.pid,'created':p.create_time(),'ancestors':anc,'fixed_workers_during_training':2})
 while True:
  if read(R/'记录/PIPELINE_ERROR.json'):return
  phase=read(R/'RUN_STATE.json').get('phase');ram,commit=mem()
  if phase in ('B_training','C_drain_aux') and ram>=1.75 and commit>=5.5:break
  dump('REFERENCE_STATE.json',{'jst':now(),'phase':'waiting_training_or_resources','RAM_GiB':ram,'commit_GiB':commit});time.sleep(15)
 out=R/'A_ref';expected={f'{label}_{o}.json' for label in POLICIES for o in range(294000,294048)}
 for attempt in range(2):
  if expected.issubset({f.name for f in out.glob('*.json')}):break
  training=read(R/'RUN_STATE.json').get('phase')=='B_training';workers=2 if training else 11
  cmd=[PY,'-B','-m','experiments.t2','screen','--openings','294000-294047','--policies',','.join(POLICIES),'--urgent-deadline','140,200','--workers',str(workers),'--output',str(out)]
  with (R/f'记录/A_ref_try{attempt}.out.log').open('ab') as fo,(R/f'记录/A_ref_try{attempt}.err.log').open('ab') as fe:
   child=subprocess.Popen(cmd,cwd='D:/py/DRL2',stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);dump('REFERENCE_LAUNCH.json',{'jst':now(),'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'command':cmd})
   while child.poll() is None:dump('REFERENCE_STATE.json',{'jst':now(),'phase':'screening','pid':child.pid,'workers':workers,'completed':len(list(out.glob('*.json')))});time.sleep(15)
   assert child.returncode==0,'A_ref driver failed; original stderr preserved'
  errors=list(out.glob('ERROR_*.json'))
  if not errors and expected.issubset({f.name for f in out.glob('*.json')}):break
  if attempt==0:
   dst=R/'记录/episode_errors/A_ref';dst.mkdir(parents=True,exist_ok=True)
   for f in errors:shutil.move(str(f),str(dst/f.name))
 assert len(list(out.glob('*.json')))==336 and not list(out.glob('ERROR_*.json'))
 dump('REFERENCE_EXIT.json',{'jst':now(),'ok':True,'cases':336,'all_complete':True})
if __name__=='__main__':
 try:main()
 except Exception:dump('REFERENCE_ERROR.json',{'jst':now(),'traceback':traceback.format_exc()});raise
