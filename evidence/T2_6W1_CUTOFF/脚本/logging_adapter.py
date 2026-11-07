"""User-authorized external read-only logging hooks. No science source edits or extra forwards."""
from pathlib import Path
import os,json,time,threading,queue,inspect,math,collections,datetime as dt
ACTIVE=None
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.writing');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
 for _ in range(100):
  try:os.replace(t,p);return
  except PermissionError:time.sleep(.05)
 raise RuntimeError('log file locked '+str(p))
def logged_worker(conn,rank,seed,candidates,entry_stages,window,urgent_deadline=None):
 import train.t2_mve_sac as tm
 original_worker=tm.worker;assert original_worker is not logged_worker
 old=tm._slim;counter=0
 out=Path(os.environ['SPACECRAFT_T24_LOG_OUT'])/'live/worker_truth';out.mkdir(parents=True,exist_ok=True)
 with (out/f'worker_{rank}.jsonl').open('a',encoding='utf-8',buffering=1) as f:
  def slim(info):
   nonlocal counter
   value=old(info);counter+=1
   f.write(json.dumps({'worker':rank,'worker_step':counter,'true_reward':float(info['true_reward']),'event':info.get('event'),'time_s':info.get('time_s'),'delta_v':info.get('delta_v'),'mission':info.get('mission')},ensure_ascii=False)+'\n')
   return value
  tm._slim=slim
  try:original_worker(conn,rank,seed,candidates,entry_stages,window,urgent_deadline)
  finally:tm._slim=old
class Observer:
 def __init__(self,out,writer):
  self.out=Path(out);self.live=self.out/'live';self.live.mkdir(parents=True,exist_ok=True);self.writer=writer;self.model=None;self.q=queue.SimpleQueue();self.stop=threading.Event();self.error=None
  self.offsets={};self.truth={};self.counts=collections.Counter();self.accum={};self.pending=[];self.probe=0;self.episodes=0;self.success=0;self.recent=collections.deque(maxlen=50);self.start=time.monotonic();self.last=self.start;self.last_steps=0
  self.thread=threading.Thread(target=self.run,name='read-only-progress',daemon=True)
 def bind(self,model):
  self.model=model;old=model.replay_buffer.add
  def add(*args,**kwargs):
   result=old(*args,**kwargs)
   frame=inspect.currentframe().f_back;loc=frame.f_locals
   if 'conn' in loc and 'pipes' in loc:
    rank=loc['pipes'].index(loc['conn']);self.counts[rank]+=1
    reward=float(args[3][0]);done=bool(args[4][0]);info=args[5][0]
    self.q.put({'steps':int(model.num_timesteps)+1,'worker':rank,'worker_step':self.counts[rank],'reward_train':reward,'done':done,'event':info.get('event'),'mission':info.get('mission')})
   return result
  model.replay_buffer.add=add;self.thread.start()
 def scan_truth(self):
  for p in (self.live/'worker_truth').glob('worker_*.jsonl'):
   with p.open('r',encoding='utf-8') as f:
    f.seek(self.offsets.get(str(p),0))
    while True:
     pos=f.tell();line=f.readline()
     if not line or not line.endswith('\n'):f.seek(pos);break
     d=json.loads(line);self.truth[(d['worker'],d['worker_step'])]=d
    self.offsets[str(p)]=f.tell()
 def drain(self):
  while not self.q.empty():self.pending.append(self.q.get())
  self.scan_truth();remaining=[]
  with (self.live/'episodes.jsonl').open('a',encoding='utf-8') as f:
   for d in self.pending:
    key=(d['worker'],d['worker_step']);v=self.truth.pop(key,None)
    if v is None:remaining.append(d);continue
    state=self.accum.setdefault(d['worker'],{'true':0.,'train':0.})
    state['true']+=v['true_reward'];state['train']+=d['reward_train']
    if d['done']:
     self.episodes+=1;ok=int(d['event']=='success');self.success+=ok;self.recent.append(ok)
     row={**d,'episode_index':self.episodes,'return_true':state['true'],'return_train':state['train'],'success':ok,'success_rate_all':self.success/self.episodes,'success_rate_last50':sum(self.recent)/len(self.recent),'window_n':len(self.recent),'time_s':v['time_s'],'delta_v':v['delta_v']};f.write(json.dumps(row,ensure_ascii=False)+'\n')
     for k in ('return_true','return_train','success','success_rate_all','success_rate_last50','window_n'):self.writer.add_scalar('rollout/'+k,row[k],d['steps'])
     self.accum[d['worker']]={'true':0.,'train':0.}
   self.pending=remaining
  stats=self.model.model_stats
  for st in stats[self.probe:]:
   step=int(st['steps'])
   for k,val in st.items():
    if k in ('steps','update') or val is None or isinstance(val,bool) or not isinstance(val,(float,int)):continue
    if not math.isfinite(float(val)):raise RuntimeError('non-finite existing probe '+k)
    prefix='critic' if k in ('real_td','model_td','actor_loss','q_pi_mean','loss_share','grad_cos') else 'model'
    self.writer.add_scalar(prefix+'/'+k,float(val),step)
   self.writer.add_scalar('progress/probe_update',st['update'],step)
  self.probe=len(stats)
 def snapshot(self,final=False):
  t=time.monotonic();step=int(self.model.num_timesteps);updates=int(self.model._n_updates);elapsed=t-self.start;rate=(step-self.last_steps)/max(t-self.last,1e-6)
  row={'jst':dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),'steps':step,'updates':updates,'episodes':self.episodes,'wall_s':elapsed,'recent_decisions_per_s':rate,'overall_decisions_per_s':step/max(elapsed,1e-6),'eta_s_estimate':max(0,int(os.environ.get('SPACECRAFT_T24_TOTAL_STEPS','0'))-step)/rate if rate>0 else None,'snapshot_not_prediction':True,'replay':self.model.replay_buffer.model_counts,'model_probes':self.probe,'pending_truth_records':len(self.pending),'final':final}
  dump(self.live/'progress.json',row)
  with (self.live/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  for k in ('steps','updates','episodes','wall_s','recent_decisions_per_s','overall_decisions_per_s'):self.writer.add_scalar('progress/'+k,row[k],step)
  for k,val in row['replay'].items():
   if isinstance(val,(int,float)):self.writer.add_scalar('model/replay_'+k,val,step)
  self.writer.flush();self.last=t;self.last_steps=step
  print(json.dumps({'rolling_progress':row},ensure_ascii=False),flush=True)
 def run(self):
  try:
   while not self.stop.wait(1):
    self.drain()
    if time.monotonic()-self.last>=30:self.snapshot()
   self.drain();self.snapshot(final=True)
  except Exception as e:self.error=repr(e);dump(self.live/'LOGGER_ERROR.json',{'error':self.error})
 def close(self):
  self.stop.set();self.thread.join(30);self.writer.flush();self.writer.close()
  if self.thread.is_alive():raise RuntimeError('logging thread failed to drain')
  if self.error:raise RuntimeError(self.error)
  assert not self.pending,'unmatched accepted transitions in external logs'
def install(out,steps):
 global ACTIVE
 # Initialize logging before native make_model resets the scientific RNGs.
 from torch.utils.tensorboard import SummaryWriter
 import train.t2_mve_sac as tm
 out=Path(out);(out/'live').mkdir(parents=True,exist_ok=True)
 writer=SummaryWriter(str(out/'live/tensorboard'),max_queue=1000,flush_secs=30)
 ACTIVE=Observer(out,writer);old=tm.make_model
 def make(*args,**kwargs):
  model=old(*args,**kwargs);ACTIVE.bind(model);return model
 os.environ['SPACECRAFT_T24_LOG_OUT']=str(out);os.environ['SPACECRAFT_T24_TOTAL_STEPS']=str(steps)
 tm.make_model=make;tm.worker=logged_worker
 dump(out/'live/ADAPTER_MANIFEST.json',{'mode':'read-only external adapter, explicit user authorization2026-10-26','native_source_unchanged':True,'native_worker_output_slim_unchanged':True,'no_added_forward_backward_or_RNG':True,'interval_s':30,'episode_returns':'true_reward before native _slim plus exact accepted replay training reward; only accepted transitions counted','Q':'existing q_pi_mean is min-Q on policy actions; separate Q1/Q2 unavailable in frozen probes, not fabricated','event_file_validation_required':True})
 return ACTIVE
