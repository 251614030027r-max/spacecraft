"""Mechanical synthetic feedback and native async smoke; never scientific performance evidence."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,traceback,random
sys.path.insert(0,'D:/py/DRL2')
R=Path(__file__).resolve().parents[1];BASE=R/'记录/日志验收';BASE.mkdir(parents=True,exist_ok=True)
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0='D:/py/DRL2')
os.chdir('D:/py/DRL2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import numpy as np,torch as th
 import train.t2_mve_sac as tm
 from stable_baselines3.common.logger import configure
 from env.t2_stage import OBS_DIM
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 th.set_num_threads(1)
 frozen=[Path('D:/py/DRL2')/p for p in ('train/t2_mve_sac.py','env/t2_stage.py','experiments/t2.py')];before={str(p):sha(p) for p in frozen}
 original=tm.make_model
 def populate(m):
  m.model_start=0;m.learning_starts=0;m.batch_size=8;m.set_logger(configure(None,[]));m._current_progress_remaining=1.
  for k in range(16):
   fact={'reward':float(k-8),'next_obs':np.zeros(OBS_DIM,np.float32),'done':False} if k%2 else None
   cands=[{'a':np.full(2,.1,np.float32),'reward':-1.,'next_obs':np.zeros(OBS_DIM,np.float32),'done':False,'kind':0,'s2':{'a':np.full(2,.2,np.float32),'logp':-1.,'reward':1.,'next_obs':np.ones(OBS_DIM,np.float32),'done':bool(k%4==0),'kind':0}}]
   m.replay_buffer.add(np.zeros((1,OBS_DIM)),np.zeros((1,OBS_DIM)),np.zeros((1,2)),np.array([-1.]),np.array([False]),[{'model':{'fact':fact,'cands':cands,'diag':{'reason':'stage'}}}])
  m.num_timesteps=16;m.train(gradient_steps=3,batch_size=8)
  return {k:v.detach().clone() for k,v in m.policy.state_dict().items()}
 baseline=original(tm.SpacesEnv(),'e_mve',0,smoke=True);w0=populate(baseline)
 import logging_adapter as la
 observer=la.install(BASE/'SYNTHETIC_FEEDBACK',16);observed=tm.make_model(tm.SpacesEnv(),'e_mve',0,smoke=True);w1=populate(observed)
 assert all(th.equal(w0[k],w1[k]) for k in w0),'observer changed native update weights'
 def rng():return (repr(random.getstate()),repr(np.random.get_state()),th.get_rng_state().numpy().tobytes())
 r0=rng();observer.close();assert rng()==r0,'logging changed scientific RNG'
 ev=EventAccumulator(str(BASE/'SYNTHETIC_FEEDBACK/live/tensorboard'));ev.Reload();tags=ev.Tags()['scalars'];required=['critic/real_td','critic/q_pi_mean','critic/model_td','model/w_mean','critic/loss_share','critic/grad_cos','model/stage2_rows','model/w2_mean','model/pi_gap_sigma_median']
 assert all(t in tags and len(ev.Scalars(t)) for t in required),tags
 assert all(np.isfinite(s.value) for t in tags for s in ev.Scalars(t))
 report={'mechanical_data_only':True,'frozen_source_SHA256_before':before,'native_update_state_bitwise_equal':True,'logger_close_RNG_unchanged':True,'actual_computed_feedback_tags':required,'synthetic_events':tags}
 # Each smoke uses native --smoke; isolated and never mixed with formal4000 evidence.
 out=BASE/'ASYNC_SMOKE';cmd=[sys.executable,'-B',str(R/'脚本/train_with_logs.py'),'train','--arm','e_mve','--seed','0','--steps','48','--workers','2','--checkpoints','16,32','--smoke','--output',str(out)]
 with (BASE/'async_smoke.out.log').open('ab') as fo,(BASE/'async_smoke.err.log').open('ab') as fe:
  p=subprocess.Popen(cmd,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  (BASE/'SMOKE_LAUNCH.json').write_text(json.dumps({'pid':p.pid,'time':time.time(),'command':cmd},ensure_ascii=False,indent=2),encoding='utf-8');rc=p.wait();assert rc==0,rc
 fin=json.loads((out/'finished.json').read_text());eps=json.loads((out/'episodes.json').read_text());logged=[json.loads(s) for s in (out/'live/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
 assert fin['steps']==48 and len(eps)==len(logged)>0
 assert [(x['steps'],x['worker'],x['event']) for x in eps]==[(x['steps'],x['worker'],x['event']) for x in logged]
 ev=EventAccumulator(str(out/'live/tensorboard'));ev.Reload();tags=ev.Tags()['scalars'];required=['progress/steps','progress/updates','progress/episodes','rollout/return_true','rollout/return_train','rollout/success_rate_last50','critic/real_td','critic/q_pi_mean']
 assert all(t in tags and ev.Scalars(t) for t in required),tags
 snapshots=[json.loads(s) for s in (out/'live/progress.jsonl').read_text(encoding='utf-8').splitlines()];assert len(snapshots)>=2 and snapshots[-1]['steps']==48 and any(b['steps']>a['steps'] for a,b in zip(snapshots,snapshots[1:]))
 for t in tags:
  values=ev.Scalars(t);assert all(np.isfinite(s.value) for s in values);assert all(b.step>=a.step for a,b in zip(values,values[1:])),t
 assert before=={str(p):sha(p) for p in frozen}
 report.update(async_native_steps=48,actual_episodes=len(eps),episode_steps_workers_events_equal=True,live_progress_samples=len(snapshots),first_step=snapshots[0]['steps'],last_step=snapshots[-1]['steps'],event_files=[str(p) for p in out.rglob('events.out.tfevents.*')],async_tags=tags,scientific_source_unchanged=True,status='passed',limits='Synthetic feedback validates logging actual native loss/gradient only, not task performance. Async smoke uses native --smoke only. Formal parameters untouched.')
 (R/'记录/LOGGING_VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  (R/'记录/LOGGING_VERIFICATION_ERROR.json').write_text(json.dumps({'time':time.time(),'traceback':traceback.format_exc()},ensure_ascii=False,indent=2),encoding='utf-8');raise
