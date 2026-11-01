"""Administrative runner only; frozen science is never edited."""
from pathlib import Path
import os, sys, json, time, datetime as dt, subprocess, hashlib, shutil
import psutil
R=Path(__file__).resolve().parents[1]
W=R.parent/'T2_6W_WIDE_140_200'
SPACE=R.parents[1]
REPO=Path('D:/py/DRL2'); PY=str(REPO/'.venv/Scripts/python.exe')
COLLAB=R.parent/'协作/Git工作树'
SHA='2208a5176e88718c2c1db3e4a19c58853da49739'
AUDIT=SHA
W1=R.parent/'T2_6W1_CUTOFF_20261029'
SCIENCE='claude/sac-mpc-coupling-design-ns7g6i'
ORDER='8708f39'; MARKS=(1000,2000,3000,4000)
CURRENT=COLLAB/'docs/collaboration/CURRENT.json'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):os.environ[k]='1'
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(REPO))
os.chdir(REPO)
ACTIVE={}; START={}; S={}
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
def read(p):
 for _ in range(20):
  try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
  except FileNotFoundError:return {}
  except (PermissionError,json.JSONDecodeError):time.sleep(.1)
 raise RuntimeError('Cannot read '+str(p))
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.writing');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 for _ in range(60):
  try:os.replace(tmp,p);return
  except PermissionError:time.sleep(.1)
 raise RuntimeError('locked '+str(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args,repo=REPO,env=None):
 p=subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),*map(str,args)],text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
 if p.returncode:raise RuntimeError('git '+repr(args)+'\n'+p.stderr)
 return p.stdout.strip()
def frozen(expected=SHA):
 assert git('rev-parse','HEAD')==expected,'wrong scientific HEAD'
 assert git('rev-parse','--abbrev-ref','HEAD')=='HEAD','scientific checkout must be detached'
 assert not git('status','--porcelain','--untracked-files=no'),'tracked scientific files changed'
def current(phase,training=False,evaluation=False):
 old=read(CURRENT)
 x=dict(old,topic='t2_7_guide_20261031',updated_at=now(),scientific_commit=SHA,run_order='docs/collaboration/upper/run_orders/T2_7_GUIDE_20261031.md',order_collab_commit=ORDER,
 execution_state=phase,training_active=training,evaluation_active=evaluation,mechanism_audit_active=False,automatic_followup='DISABLED' if phase in ('complete_delivered','failed_stopped') else 'OS_PIPELINE_ACTIVE',native_state=str(R/'RUN_STATE.json'),
 handoff='docs/collaboration/lower/handoffs/t2_7_guide_20261031/RESULT_FINAL.md',local_report=str(SPACE/'上层交付/最新/RESULT_FINAL.md'),
 current_scope={'arms':['e_mve','e_sac'],'seeds':[0,1],'guide':True,'guide_p0':0.5,'guide_until':2000,'steps_each':4000,'workers_each':2,'cpu_affinity':'CPU0-11 shared','priority':'Normal','checkpoints':MARKS,'openings':'294000-294047','criterion':'t27'},
 pending_experiment={'status':'completed' if phase=='complete_delivered' else 'approved_executing','not_started':phase=='preflight'},
 conclusion_boundary='2x2: primary guided EACC vs reused SAC; guide-only, labels incremental and vs unguided EACC are reported separately. No presumed gain.',next_action='Execute authorized T2.7 only; no later experiment authorized.')
 if phase=='preflight':
  x.update(previous_round_snapshot={'path':str(R/'记录/CURRENT_BEFORE.json'),'SHA256':sha(R/'记录/CURRENT_BEFORE.json')},previous_completed_result=old.get('result'),evidence_commit=None,collab_delivery_commit=None)
  x.pop('result',None)
  for k in ('reference_review','reference_review_local','audit_parallel_resource_change','resource_policy_path','torch_resource_verification','stable_throughput_state'):x.pop(k,None)
 dump(CURRENT,x)
def state(**kw):
 S.update(kw,updated_jst=now(),pid=os.getpid(),processes={k:{'pid':p.pid,'running':p.poll() is None} for k,p in ACTIVE.items()},rolling={f'{arm}_s{s}':read(R/f'B/TRAIN_{arm}_s{s}/live/progress.json') for arm in ('e_mve','e_sac') for s in (0,1)})
 dump(R/'RUN_STATE.json',S)
def launch(name,args,python=True):
 cmd=[PY,'-B',*map(str,args)] if python else list(map(str,args))
 fo=(R/f'记录/{name}.out.log').open('ab');fe=(R/f'记录/{name}.err.log').open('ab')
 p=subprocess.Popen(cmd,cwd=REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.NORMAL_PRIORITY_CLASS);fo.close();fe.close()
 ACTIVE[name]=p;START[name]=time.time();dump(R/f'记录/{name}_LAUNCH.json',{'jst':now(),'pid':p.pid,'created':psutil.Process(p.pid).create_time(),'command':cmd})
 return p
def finish(name):
 p=ACTIVE[name];rc=p.poll()
 if rc is None:return None
 dump(R/f'记录/{name}_EXIT.json',{'jst':now(),'returncode':rc,'wall_s':time.time()-START[name]});del ACTIVE[name];return rc
def run(name,args,retries=0,python=True):
 for attempt in range(retries+1):
  n=name if attempt==0 else name+f'_retry{attempt}';launch(n,args,python=python)
  while True:
   rc=finish(n)
   if rc is not None:break
   state(current_job=n);time.sleep(30)
  if rc==0:return
 raise RuntimeError(f'{name} exit {rc}; original stdout/stderr retained')
def stop_owned():
 identities=[]
 for name,p in list(ACTIVE.items()):
  if p.poll() is not None:continue
  try:
   root=psutil.Process(p.pid);tree=[root]+root.children(recursive=True)
   identities += [(q.pid,q.create_time()) for q in tree]
  except psutil.NoSuchProcess:pass
 for pid,created in reversed(identities):
  try:
   q=psutil.Process(pid)
   if q.create_time()==created:q.terminate()
  except psutil.NoSuchProcess:pass
 dump(R/'记录/STOP_OWNED.json',{'jst':now(),'identities':identities,'reason':'pipeline failure; no external processes touched'})
