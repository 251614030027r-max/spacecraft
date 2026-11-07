"""Administrative runner only; frozen science is never edited."""
from pathlib import Path
import os, sys, json, time, datetime as dt, subprocess, hashlib, shutil
import psutil
R=Path(__file__).resolve().parents[1]
W=R.parent/'T2_6W_WIDE_140_200'
SPACE=R.parents[1]
REPO=Path('D:/py/DRL2'); PY=str(REPO/'.venv/Scripts/python.exe')
COLLAB=R.parent/'协作/Git工作树'
SHA='e2d4c14bc4d3bbe385c44e65f3c3dcdbd41372d5'
AUDIT='91b346b664fe59e1f9c00af38f45e65735b27839'
SCIENCE='claude/sac-mpc-coupling-design-ns7g6i'
ORDER='185765b'; MARKS=(1000,2000,3000,4000)
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
def current(phase,training=False,evaluation=False,audit=False):
 old=read(CURRENT)
 x=dict(old,topic='t2_6w1_cutoff_20261029',updated_at=now(),scientific_commit=git('rev-parse','HEAD'),
 scientific_commit_role='actual frozen checkout for current phase',run_order='docs/collaboration/upper/run_orders/T2_6W1_CUTOFF_20261029.md',order_collab_commit=ORDER,
 execution_state=phase,training_active=training,evaluation_active=evaluation,mechanism_audit_active=audit,
 automatic_followup='OS_PIPELINE_ACTIVE' if phase not in ('complete_delivered','failed_stopped') else 'DISABLED',native_state=str(R/'RUN_STATE.json'),
 handoff='docs/collaboration/lower/handoffs/t2_6w1_cutoff_20261029/RESULT_FINAL.md',local_report=str(SPACE/'上层交付/最新/RESULT_FINAL.md'),
 current_scope={'EACC_seeds':[0,1],'steps_each':4000,'workers_each':2,'SAC':'reuse T2.6W; no retraining or evaluation','checkpoints':MARKS,'openings':'294000-294047','urgent_deadline':[140,200],'criterion':'t26w','rank_audit':AUDIT},
 pending_experiment={'status':'approved_executing' if phase!='complete_delivered' else 'completed','not_started':phase=='preflight','science_fix_reference':SHA},
 conclusion_boundary='Execution feedback independently improving SAC long-term decisions remains the research claim; no gain is presumed.',
 next_action='Execute approved sections 0-3 and deliver; no later mechanism edits or new training authorized.')
 if 'previous_round_snapshot' not in x:x['previous_round_snapshot']={'path':str(R/'记录/CURRENT_BEFORE.json'),'SHA256':sha(R/'记录/CURRENT_BEFORE.json')}
 if phase=='preflight':x['previous_completed_result']=old.get('result');x.pop('result',None);x['evidence_commit']=None;x['collab_delivery_commit']=None
 dump(CURRENT,x)
def state(**kw):
 S.update(kw,updated_jst=now(),pid=os.getpid(),processes={k:{'pid':p.pid,'running':p.poll() is None} for k,p in ACTIVE.items()},rolling={f'e_mve_s{s}':read(R/f'B/TRAIN_e_mve_s{s}/live/progress.json') for s in (0,1)})
 dump(R/'RUN_STATE.json',S)
def launch(name,args,python=True):
 cmd=[PY,'-B',*map(str,args)] if python else list(map(str,args))
 fo=(R/f'记录/{name}.out.log').open('ab');fe=(R/f'记录/{name}.err.log').open('ab')
 p=subprocess.Popen(cmd,cwd=REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS);fo.close();fe.close()
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
