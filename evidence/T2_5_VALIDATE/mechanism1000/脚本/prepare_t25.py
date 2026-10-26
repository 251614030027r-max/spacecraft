"""Frozen T2.4 preparation only; formal training requires the missing logging gate."""
from pathlib import Path
import os,json,time,datetime as dt,subprocess,hashlib,shutil,traceback,ctypes
import psutil
R=Path(__file__).resolve().parents[1];OLD=R.parent/'T2_3_TEST_20261025';REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe');SHA='84ee9c1045f32bbbacfef6cd220fdad55c5765c8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(REPO))
os.chdir(REPO)
for fd,n in ((1,'prepare.out.log'),(2,'prepare.err.log')):
 f=(R/'记录'/n).open('ab',buffering=0);os.dup2(f.fileno(),fd)
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);t=p.with_name(p.name+f'.{os.getpid()}.writing');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
 for _ in range(100):
  try:os.replace(t,p);return
  except PermissionError:time.sleep(.1)
 raise RuntimeError('locked '+str(p))
active={};started={};S={}
def progress(**kw):
 S.update(kw,updated_jst=now(),active={n:p.pid for n,p in active.items()},audit_completed=len(list((R/'A1_audit').glob('audit_*.json'))),sac_s1_3k_completed=len(list((R/'B_eval/e_sac_s1/step_3000').glob('e_sac_s1_step_3000_*.json'))));dump(R/'STATE.json',S)
def launch(name,args):
 fo=(R/f'记录/{name}.out.log').open('ab');fe=(R/f'记录/{name}.err.log').open('ab')
 p=subprocess.Popen([PY,'-B',*map(str,args)],cwd=REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);fo.close();fe.close();active[name]=p;started[name]=time.time()
 dump(R/f'记录/{name}_LAUNCH.json',{'jst':now(),'pid':p.pid,'created':psutil.Process(p.pid).create_time(),'command':[PY,'-B',*map(str,args)]});return p
def finish(name):
 p=active[name];rc=p.poll()
 if rc is None:return False
 dump(R/f'记录/{name}_EXIT.json',{'jst':now(),'returncode':rc,'wall_s':time.time()-started[name]});del active[name];assert rc==0,name+' failed; inspect original stderr';return True
def main():
 with (R/'记录/PREPARE.lock').open('x') as f:f.write(str(os.getpid()))
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);me.cpu_affinity(list(range(psutil.cpu_count())))
 ancestors=[{'pid':p.pid,'name':p.name(),'created':p.create_time()} for p in me.parents()];assert not any('codex' in p['name'].lower() or 'chatgpt' in p['name'].lower() for p in ancestors)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==SHA
 assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 S.update(pid=os.getpid(),science_SHA=SHA,order_collab='4c52ece',started_jst=now(),phase='pytest',ancestor_proof=ancestors,formal_training_started=False,logging_gate='NOT_IMPLEMENTED_IN_FREEZE',deadline_jst='2026-10-26T09:00:00+09:00')
 dump(R/'记录/LOGGING_GATE.json',{'science_SHA':SHA,'rolling_30s':False,'tensorboard_events':False,'episode_return':False,'checkpoint_logs':True,'verified_source':'train/t2_mve_sac.py disables standard logger; only checkpoint heartbeat and episode/model snapshots','action':'Do not silently modify frozen modules or start formal training before logging gate; await logging-only freeze or explicit external observer authorization'})
 assert '33 passed' in (R/'pytest.log').read_text(encoding='utf-8',errors='replace')
 models=[{'path':str(p),'bytes':p.stat().st_size,'SHA256':sha(p)} for p in sorted(OLD.glob('TRAIN_*/**/*.zip'))];dump(R/'记录/OLD_ASSETS_SHA256.json',models)
 reused=[]
 for a,ks in (('e_sac_s0',(0,1000,2000,3000,4000)),('e_sac_s1',(1000,2000,3000))):
  for k in ks:
   src=(R.parent/'T2_4_TEST_20261026'/'B_eval'/a/f'step_{k}') if a.endswith('s1') and k==3000 else OLD/('B_eval' if a.endswith('s0') else 'B_eval_s1_partial')/a/f'step_{k}';dest=R/'B_eval'/a/f'step_{k}';dest.mkdir(parents=True,exist_ok=False)
   fs=sorted(src.glob(f'{a}_step_{k}_*.json'));assert len(fs)==24
   for f in fs:
    d=json.loads(f.read_text(encoding='utf-8'));assert d['event']!='error' and d['window'] is True
    shutil.copy2(f,dest/f.name);h=sha(f);assert sha(dest/f.name)==h;reused.append({'source':str(f),'destination':str(dest/f.name),'SHA256':h})
 dump(R/'记录/SAC_REUSE.json',reused)
 progress(phase='audit_and_control_eval',reuse_complete=192)
 shutil.copy2(R.parent/'T2_4_TEST_20261026/A1_CAL.json',R/'A1_CAL.json')
 launch('A1_audit',['-m','experiments.t2_audit','labels','--calibration',R/'A1_CAL.json','--openings','2910000-2910005','--workers','2','--policy-checkpoint',R.parent/'T2_4_TEST_20261026/TRAIN_e_mve_s0/checkpoints/step_4000.zip','--output',R/'A1_audit'])
 while active:
  for n in list(active):
   if not finish(n):continue
   if n=='A1_audit':launch('A1_summary',['-m','experiments.t2_audit','summary','--input',R/'A1_audit','--output',R/'A1_LABEL_AUDIT.json'])
  progress();time.sleep(15)
 assert len(list((R/'A1_audit').glob('audit_*.json')))==6
 assert len(list((R/'B_eval').rglob('*.json')))==192
 progress(phase='preparation_complete_training_held_for_logging',finished_jst=now());dump(R/'记录/PREPARE_EXIT.json',S)
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'记录/PREPARE_ERROR.json',{'jst':now(),'traceback':traceback.format_exc(),'formal_training_started':False});raise
