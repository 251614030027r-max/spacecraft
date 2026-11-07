"""OS-owned frozen T2.4: logging gate -> two MVE trainings -> evaluation -> native decision -> delivery."""
from pathlib import Path
import os,json,time,datetime as dt,ctypes,subprocess,traceback,shutil
import psutil
R=Path(__file__).resolve().parents[1];OLD=R.parent/'T2_3_TEST_20261025';REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe');SHA='84ee9c1045f32bbbacfef6cd220fdad55c5765c8'
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
 S.update(kw,updated_jst=now(),memory=memory(),processes={n:p.pid for n,p in active.items()},rolling={f'e_mve_s{s}':read(R/f'TRAIN_e_mve_s{s}/live/progress.json') for s in (0,1)},preparation=read(R/'STATE.json'),eval_completed=len(list((R/'B_eval').rglob('*.json'))))
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
  try:run(f'publish_{stage}_{i}',[R/'脚本/publish_t25.py','--stage',stage]);return True
  except AssertionError:time.sleep(15)
 progress(publication_pending=stage);return False
def evaluate(seed,k):
 label=f'e_mve_s{seed}_step_{k}';out=R/f'B_eval/e_mve_s{seed}/step_{k}';expected={f'{label}_{o}.json' for o in range(294000,294024)}
 for attempt in range(2):
  if {p.name for p in out.glob('*.json')}==expected:return
  while True:
   m=memory();cap=min(10,int(max(0,(m['physical_free_GiB']-1.2)//.5)),int(max(0,(m['commit_free_GiB']-4)//.8)))
   if cap>0:break
   progress(resource_hold=True);time.sleep(30)
  run(f'EVAL_s{seed}_{k}_try{attempt}',['-m','experiments.t2','eval','--checkpoint',R/f'TRAIN_e_mve_s{seed}/checkpoints/step_{k}.zip','--label',label,'--openings','294000-294023','--workers',str(cap),'--output',out])
  if {p.name for p in out.glob('*.json')}==expected:return
  if attempt==0:
   dest=R/f'记录/episode_errors/e_mve_s{seed}_{k}';dest.mkdir(parents=True,exist_ok=True)
   for p in out.glob('ERROR_*.json'):shutil.move(str(p),str(dest/(p.stem+f'_{time.time_ns()}.json')))
 raise RuntimeError('registered evaluation incomplete '+label)
def early_mechanism():
 if (R/'记录/MECHANISM1000_CAPTURED.json').exists():return
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 result={}
 for seed in (0,1):
  live=read(R/f'TRAIN_e_mve_s{seed}/live/progress.json')
  if live.get('steps',0)<1050:return
  ea=EventAccumulator(str(R/f'TRAIN_e_mve_s{seed}/live/tensorboard'),size_guidance={'scalars':0});ea.Reload();rows={}
  for field in ('two_stage_share','decisive_share','c_mean','grad_cos'):
   tag=('critic/' if field=='grad_cos' else 'model/')+field
   vals=[x for x in ea.Scalars(tag) if x.step>1000] if tag in ea.Tags()['scalars'] else []
   rows[field]={'mean':sum(x.value for x in vals)/len(vals) if vals else None,'n':len(vals),'first_step':vals[0].step if vals else None,'last_step':vals[-1].step if vals else None,'missing':not bool(vals)}
  result[f'e_mve_s{seed}']=rows
 alerts=[seed for seed,rows in result.items() if any(v['missing'] for v in rows.values()) or rows['decisive_share']['mean']==0 or (rows['two_stage_share']['mean'] is not None and rows['two_stage_share']['mean']<.5)]
 dump(R/'MECHANISM1000.json',{'jst':now(),'source':'Actual native probes parsed from external TensorBoard; rows strictly after1000; provisional,not final criteria','readings':result,'immediate_alert_seeds':alerts})
 dump(R/'记录/MECHANISM1000_CAPTURED.json',{'jst':now(),'alerts':alerts})
 publish('mechanism1000')

def main():
 with (R/'记录/PIPELINE.lock').open('x') as f:f.write(str(os.getpid()))
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);me.cpu_affinity(list(range(psutil.cpu_count())))
 ancestors=[{'pid':p.pid,'name':p.name(),'created':p.create_time()} for p in me.parents()];assert not any('codex' in p['name'].lower() or 'chatgpt' in p['name'].lower() for p in ancestors)
 S.update(pid=os.getpid(),started_jst=now(),science_SHA=SHA,order_collab='4c52ece',scope='Only two MVE seeds0/1 each4000; reuse T23 SAC; audit6; no branch/no figures',phase='logging_gate',ancestor_proof=ancestors,deadline_jst=None)
 while not read(R/'记录/LOGGING_VERIFIED.json'):
  assert not read(R/'记录/LOGGING_VERIFICATION_ERROR.json'),'Logging gate failed; formal training not started'
  progress();time.sleep(15)
 assert read(R/'记录/LOGGING_VERIFIED.json')['status']=='passed'
 while not (R/'pytest.log').exists():
  assert not read(R/'记录/PREPARE_ERROR.json'),'Preparation/test failure';progress(phase='waiting_pytest');time.sleep(10)
 assert '33 passed' in (R/'pytest.log').read_text(encoding='utf-8',errors='replace')
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==SHA and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 pending=[0,1];running={};done=[];attempts={0:0,1:0};progress(phase='training',logging_gate='passed')
 def start(s):
  out=R/f'TRAIN_e_mve_s{s}';assert not out.exists();n=f'TRAIN_e_mve_s{s}_attempt{attempts[s]}';launch(n,[R/'脚本/train_with_logs.py','train','--arm','e_mve','--seed',str(s),'--steps','4000','--workers','3','--checkpoints','1000,2000,3000','--output',out]);running[n]=s
 for s in pending:start(s)
 while running:
  for n,s in list(running.items()):
   rc=finish(n)
   if rc is None:continue
   del running[n];out=R/f'TRAIN_e_mve_s{s}';fin=read(out/'finished.json')
   if fin.get('steps')==4000:
    if rc:dump(R/f'记录/TRAIN_s{s}_LOGGING_ONLY_FAILURE.json',{'jst':now(),'rc':rc,'science_finished4000':True,'not_retrained':True})
    done.append(s);continue
   assert attempts[s]==0,n+' second genuine failure; stop segment'
   # Original stderr is preserved and copied verbatim before the one registered restart.
   stderr=(R/f'记录/{n}.err.log').read_text(encoding='utf-8',errors='replace');assert 'Traceback' in stderr or rc!=0
   dest=R/f'记录/restarts/TRAIN_e_mve_s{s}_attempt0';assert out.resolve().is_relative_to(R.resolve()) and dest.resolve().is_relative_to(R.resolve()) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
   for pid,t in reversed(list(created[n].items())):
    try:
     p=psutil.Process(pid)
     if p.create_time()==t:p.kill()
    except psutil.NoSuchProcess:pass
   shutil.move(str(out),str(dest));dump(R/f'记录/RESTART_s{s}.json',{'jst':now(),'original_stderr':stderr,'from_zero':True,'same_seed_same_config':True,'optimizer_replay_RNG_not_spliced':True});attempts[s]=1;start(s)
  early_mechanism();progress(training_done=done);time.sleep(15)
 progress(phase='evaluation',training_done=done)
 for s in (0,1):
  for k in (0,1000,2000,3000,4000):evaluate(s,k)
 while not read(R/'记录/PREPARE_EXIT.json'):
  assert not read(R/'记录/PREPARE_ERROR.json'),'Preparation failure; preserve scientific outputs and repair driver'
  progress(waiting_preparation=True);time.sleep(20)
 assert len(list((R/'B_eval').rglob('*.json')))==432
 run('READOUT',['-m','experiments.t2_readout','curve','--eval-root',R/'B_eval','--train-root',R,'--screen',OLD/'A2_window','--output',R/'B_READOUT.json'])
 d=read(R/'B_READOUT.json');assert d['all_complete']
 for s in (0,1):assert set(d['arms'][f'e_mve_s{s}'])=={'0','1000','2000','3000','4000'}
 assert set(d['arms']['e_sac_s0'])=={'0','1000','2000','3000','4000'} and set(d['arms']['e_sac_s1'])=={'1000','2000','3000'}
 run('DECIDE',['-m','experiments.t2_decide','--criteria','v25','--train-root',R,'--readout',R/'B_READOUT.json','--output',R/'DECISION.json']);assert read(R/'DECISION.json')['verdict']!='incomplete'
 progress(phase='final_publication',verdict=read(R/'DECISION.json')['verdict']);ok=publish('final')
 progress(phase='complete_waiting_review' if ok else 'complete_publication_pending',finished_jst=now(),no_more_training=True);dump(R/'记录/PIPELINE_EXIT.json',S)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(R/'记录/PIPELINE_ERROR.json',{'jst':now(),'traceback':traceback.format_exc(),'phase':S.get('phase'),'scientific_scope_unchanged':True})
  for n in list(active):stop_owned(n)
  raise
