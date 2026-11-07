"""Isolated telemetry smoke, never included in scientific curves or counts."""
import sys, os, json, time, subprocess, hashlib, random, traceback
from pathlib import Path
import runtime as a
sys.path.insert(0,str(a.REPO))
BASE=a.R/'记录/日志验收'
def main():
 import numpy as np, torch as th
 import train.t2_mve_sac as tm
 from env.t2_stage import OBS_DIM
 from stable_baselines3.common.logger import configure
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 th.set_num_threads(1);BASE.mkdir(parents=True,exist_ok=False)
 files=[a.REPO/p for p in ('train/t2_mve_sac.py','env/t2_stage.py','experiments/t2.py')];before={str(p):a.sha(p) for p in files}
 original=tm.make_model
 def populate(m):
  m.model_start=0;m.learning_starts=0;m.batch_size=8;m.set_logger(configure(None,[]));m._current_progress_remaining=1.
  for k in range(16):
   fact={'reward':float(k-8),'next_obs':np.zeros(OBS_DIM,np.float32),'done':False} if k%2 else None
   cands=[{'a':np.full(2,.1,np.float32),'reward':-1.,'next_obs':np.zeros(OBS_DIM,np.float32),'done':False,'kind':0,'s2':{'a':np.full(2,.2,np.float32),'logp':-1.,'reward':1.,'next_obs':np.ones(OBS_DIM,np.float32),'done':bool(k%4==0),'kind':0}}]
   m.replay_buffer.add(np.zeros((1,OBS_DIM)),np.zeros((1,OBS_DIM)),np.zeros((1,2)),np.array([-1.]),np.array([False]),[{'model':{'fact':fact,'cands':cands,'diag':{'reason':'stage'}}}])
  m.num_timesteps=16;m.train(gradient_steps=3,batch_size=8)
  return {k:v.detach().clone() for k,v in m.policy.state_dict().items()}
 base=original(tm.SpacesEnv(),'e_mve',0,smoke=True);w0=populate(base)
 import logging_adapter as la
 obs=la.install(BASE/'SYNTHETIC_FEEDBACK',16);m=tm.make_model(tm.SpacesEnv(),'e_mve',0,smoke=True);w1=populate(m)
 assert all(th.equal(w0[k],w1[k]) for k in w0),'logging changed native optimizer update'
 def rng():return repr(random.getstate()),repr(np.random.get_state()),th.get_rng_state().numpy().tobytes()
 r0=rng();obs.close();assert rng()==r0
 ev=EventAccumulator(str(BASE/'SYNTHETIC_FEEDBACK/live/tensorboard'));ev.Reload()
 req=['critic/real_td','critic/q_pi_mean','critic/model_td','model/w_mean','critic/loss_share','critic/grad_cos','model/stage2_rows','model/w2_mean','model/pi_gap_sigma_median']
 assert all(t in ev.Tags()['scalars'] and ev.Scalars(t) for t in req)
 out=BASE/'ASYNC_SMOKE'
 cmd=[sys.executable,'-B',str(a.R/'脚本/train_with_logs.py'),'train','--arm','e_mve','--seed','0','--steps','48','--workers','2','--checkpoints','16,32','--urgent-deadline','140,200','--smoke','--output',str(out)]
 histories=[]
 with (BASE/'smoke.out.log').open('ab') as fo,(BASE/'smoke.err.log').open('ab') as fe:
  p=subprocess.Popen(cmd,cwd=a.REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  a.dump(BASE/'SMOKE_LAUNCH.json',{'pid':p.pid,'command':cmd})
  while p.poll() is None:
   if (out/'live/tensorboard').exists():
    e=EventAccumulator(str(out/'live/tensorboard'));e.Reload();tags=e.Tags()['scalars']
    histories.append({'time':time.time(),'size':sum(f.stat().st_size for f in (out/'live/tensorboard').glob('events.out*')),'counts':{t:len(e.Scalars(t)) for t in tags}})
   time.sleep(30)
  assert p.returncode==0,p.returncode
 fin=a.read(out/'finished.json');eps=a.read(out/'episodes.json');logged=[json.loads(x) for x in (out/'live/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
 assert fin['steps']==48 and fin['updates']>0 and len(eps)==len(logged)>0
 assert [(x['steps'],x['worker'],x['event']) for x in eps]==[(x['steps'],x['worker'],x['event']) for x in logged]
 e=EventAccumulator(str(out/'live/tensorboard'));e.Reload();tags=e.Tags()['scalars']
 req2=['progress/steps','progress/updates','progress/episodes','rollout/return_true','rollout/return_train','rollout/success_rate_last50','critic/real_td','critic/q_pi_mean']
 assert all(t in tags and e.Scalars(t) for t in req2),tags
 for reader in (ev,e):
  for t in reader.Tags()['scalars']:
   vals=reader.Scalars(t);assert all(np.isfinite(v.value) for v in vals) and all(y.step>=x.step for x,y in zip(vals,vals[1:])),t
 snapshots=[json.loads(x) for x in (out/'live/progress.jsonl').read_text(encoding='utf-8').splitlines()]
 assert len(snapshots)>1 and snapshots[-1]['steps']==48 and any(y['steps']>x['steps'] for x,y in zip(snapshots,snapshots[1:]))
 histories.append({'time':time.time(),'size':sum(f.stat().st_size for f in (out/'live/tensorboard').glob('events.out*')),'counts':{t:len(e.Scalars(t)) for t in tags}})
 assert len(histories)>=2 and histories[-1]['size']>histories[0]['size'],'event file did not grow after flushing'
 assert before=={str(p):a.sha(p) for p in files}
 a.dump(a.R/'记录/LOGGING_VERIFIED.json',{'status':'passed','jst':a.now(),'science_SHA':a.SHA,'source_sha256':before,'native_updates_bitwise_equal':True,'logger_close_rng_unchanged':True,'native_smoke_steps':48,'native_updates':fin['updates'],'completed_episodes':len(eps),'episode_identity_matches':True,'event_reader_growth':histories,'feedback_tags':req,'async_tags':tags,'progress_snapshots':len(snapshots),'limits':'Synthetic feedback verifies telemetry only; native async --smoke validates actual entry point through completed episodes and updates. No scientific performance inference.'})
if __name__=='__main__':
 try:main()
 except Exception:a.dump(a.R/'记录/LOGGING_ERROR.json',{'jst':a.now(),'traceback':traceback.format_exc()});raise
