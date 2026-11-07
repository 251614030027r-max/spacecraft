"""Bounded same-task repair experiment: no hidden model selection or oracle."""
import argparse,json,time,hashlib,os,subprocess,pickle,sys
from pathlib import Path
from dataclasses import asdict
from functools import partial
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from env.t1_b_task import MissionDistribution,T1BReviewEnv
from env.t1_gym import T1GymConfig
from env.t1_b_feasible_v1 import FeasibleArrivalEnv,choice_index,bin_center,VERSION
from train.t1_b_feasible_sac import FeasibleSupportSAC,transfer_cold_weights
from train.t1_tcpf import CosineLearningRate

ROOT=Path('C:/Users/35884/Documents/Spacecraft/过程文件/T1B学习修复V1_20261019')
OLD=ROOT.parent/'通宵研究_20261019'
PROFILE=OLD/'记录/profile_mission_v3.json'
COLD=OLD/'结果/TRAIN_r3_seed0/checkpoints/step_0.zip'
def default(x):return x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x))
def dump(path,x):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default),encoding='utf-8');temp.replace(path)
def mission():return MissionDistribution(**json.loads(PROFILE.read_text(encoding='utf-8-sig')))
def make_env(rank,feedback=False,seed=0):return FeasibleArrivalEnv(config=T1GymConfig(seed_base=1_000_000*(seed+1)+rank*100_000),mission=mission(),feedback=feedback)
def model_for(env,arm='e_sac',smoke=False,output=None,seed=0):
    kwargs={'feedback_audit_path':str(Path(output)/'model_prefix_releases.jsonl')} if output else {}
    m=FeasibleSupportSAC('MlpPolicy',env,seed=seed,device='cpu',verbose=0,policy_kwargs={'net_arch':[128,128]},gamma=.999,buffer_size=20000 if smoke else 100000,batch_size=64 if smoke else 256,learning_starts=32 if smoke else 500,learning_rate=CosineLearningRate(1e-4,1e-5),ent_coef=.005,tau=.005,train_freq=(1,'step'),gradient_steps=4,interval_bellman=True,actor_every_transitions=3,mc_lambda=0.,eq_lambda=0.,audit_every=0,delta_lambda=0.,delta_target_share=.05 if arm=='e_tcpf' else None,delta_cap_share=.1,delta_warmup_decisions=32 if smoke else 500,delta_calibration_steps=32 if smoke else 500,delta_probe_every=5,delta_min_probes=5 if smoke else 30,delta_diagnostics_every=20,preserve_warmup_options=False,replay_buffer_kwargs=kwargs)
    transfer_cold_weights(m,COLD);return m

def contract(out):
    from train.t1_b_interval_sac import T1BIntervalSAC
    from env.t1_n_review import N_DUE_INDEX
    torch.set_num_threads(1);rows=[]
    env=FeasibleArrivalEnv(openings=[292140],mission=mission());obs,_=env.reset();model=model_for(env,smoke=True);old=T1BIntervalSAC.load(COLD,device='cpu')
    z=torch.tensor(obs[None]);same=max(abs(old.predict(obs[:46],deterministic=True)[0]-model.predict(obs,deterministic=True)[0]));assert same==0
    p=model.support_probabilities(z);assert abs(float(p.sum())-1)<1e-6 and bool((p>=0).all())
    from copy import deepcopy
    base=T1BReviewEnv(openings=[292140],mission=mission());base.reset()
    encode=lambda c:json.dumps(asdict(c),sort_keys=True,default=default)
    assert encode(env.env.config)==encode(base.env.config)
    # Mathematical gradient check with actual executed candidate coordinates.
    q=torch.tensor([[1.,2.,-1.]]);loss=-(p*q).sum();gs=torch.autograd.grad(loss,list(model.actor.parameters()),allow_unused=True);gn=float(torch.sqrt(sum((g*g).sum() for g in gs if g is not None)));assert gn>0
    # True initial decisions, all candidates and no safety/task changes.
    for j in range(len(env.support())):
        fresh=FeasibleArrivalEnv(openings=[292140],mission=mission());zz,_=fresh.reset();choices=fresh.support();raw=bin_center(j,len(choices));n,r,t,x,i=fresh.step(np.array([raw],np.float32))
        assert i['review']['outcome']=='accepted';assert abs(i['execution']['tau_request_s']-choices[j])<1e-5
        assert abs(i['executed_action'][0]-(choices[j]/100-1))<1e-6
        rows.append({'raw':raw,'tau':choices[j],'accepted':True,'executed_coordinate':float(i['executed_action'][0])})
    # Same real first transition with feedback enabled vs disabled.
    plain=FeasibleArrivalEnv(openings=[292140],mission=mission(),feedback=False);withcf=FeasibleArrivalEnv(openings=[292140],mission=mission(),feedback=True);plain.reset();withcf.reset()
    za,ra,ta,xa,ia=plain.step(np.array([0.],np.float32));zb,rb,tb,xb,ib=withcf.step(np.array([0.],np.float32))
    assert ra==rb and np.array_equal(za,zb) and ta==tb and xa==xb
    assert np.array_equal(plain.env.chaser_state.position,withcf.env.chaser_state.position)
    assert 'counterfactual_pending' in ib
    dump(out/'CONTRACT.json',{'passed':True,'raw_actor_transfer_max_abs':float(same),'exact_probability_sum':float(p.sum()),'actual_probability_gradient_norm':gn,'physics_reward_config_equal':True,'actual_initial_actions':rows,'feedback_on_off_first_real_transition_bitwise_equal':True,'feedback_payload_created':True,'discrete_gradient_semantics':'Exact Gaussian CDF bin probabilities times Q at actual executed tau; no derivative of bin selection.'})
    print('CONTRACT PASS',flush=True)

def eval_job(job):
    opening,checkpoint,label,out=job;torch.set_num_threads(1);out=Path(out);file=out/f'{label}_{opening}.json'
    if file.exists():return json.loads(file.read_text(encoding='utf-8'))
    start=time.perf_counter();env=FeasibleArrivalEnv(openings=[opening],mission=mission(),feedback=False);obs,reset=env.reset()
    model=FeasibleSupportSAC.load(checkpoint,device='cpu') if checkpoint else model_for(env,smoke=True)
    steps=[];done=False;initial=obs.tolist()
    while not done:
        action=model.predict(obs,deterministic=True)[0];obs,r,t,x,i=env.step(action);done=t or x;e=env.env
        steps.append({'time_s':e.time_s,'reward':r,'raw_action':action.tolist(),'executed_action':i['executed_action'].tolist(),'execution':i['execution'],'review':i['review'],'support':i['support_transition'],'budget_remaining':i['budget_remaining_m_s'],'observation':obs.tolist(),'control_trace':i['resource_control_trace'],'q':(e.chaser_state.position-e.target_state.position).tolist(),'v':(e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity).tolist()})
    result={'opening':opening,'stage':label,'checkpoint':str(checkpoint) if checkpoint else 'same cold weights, added columns zero','mission':reset['mission'],'event':i['event'],'discounted_return':sum(.999**k*s['reward'] for k,s in enumerate(steps)),'return':sum(s['reward'] for s in steps),'summary':e.summary(),'initial_observation':initial,'steps':steps,'wall_s':time.perf_counter()-start,'version':VERSION}
    dump(file,result);return {k:result[k] for k in ['opening','stage','event','discounted_return','wall_s']}

def evaluate(out,checkpoint=None,label='interface_0k',openings=range(292140,292148),workers=4):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    out.mkdir(parents=True,exist_ok=True);jobs=[(op,checkpoint,label,str(out)) for op in openings]
    dump(out/'MANIFEST.json',{'jobs':jobs,'fixed_openings':list(openings),'policy_readout':'Deterministic original tanh(mean) scalar mapped to its verified support bin (not outcome oracle).','training':False})
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(eval_job,j) for j in jobs]):print(json.dumps(f.result(),ensure_ascii=False),flush=True)
    dump(out/'finished.json',{'completed':True,'count':len(jobs)})

def expert(out,transitions=1000):
    assert transitions%4==0
    env=SubprocVecEnv([partial(make_env,k,False) for k in range(4)],start_method='spawn');obs=env.reset();records=[];events={};start=time.perf_counter()
    for step in range(transitions//4):
        # Earliest no-plan; keep existing plan. No future reward or opening lookup.
        actions=[]
        for z in obs:
            n=int((z[49:52]>.5).sum());choices=z[46:49]*200
            j=int(np.argmin(abs(choices[:n]-z[28]*200))) if n and z[27]>.5 else 0
            actions.append([bin_center(j,n) if n else 0.])
        nxt,rewards,dones,infos=env.step(np.array(actions,np.float32))
        terminal=np.stack([info.get('terminal_observation',nxt[k]) for k,info in enumerate(infos)])
        records.append({'obs':obs.copy(),'next_obs':terminal.copy(),'action':np.stack([i['executed_action'] for i in infos]),'reward':rewards.copy(),'done':dones.copy(),'infos':[{k:i.get(k) for k in ['event','TimeLimit.truncated','review','execution','mission_deadline_s','mission_budget_m_s','budget_remaining_m_s','support_transition']} for i in infos]})
        for i in infos:
            if i['event']:events[i['event']]=events.get(i['event'],0)+1
        obs=nxt
        if step%25==0:dump(out/'EXPERT_STATUS.json',{'real_transitions':4*(step+1),'events':events,'wall_s':time.perf_counter()-start})
    env.close()
    with (out/'expert_replay.pkl').open('wb') as f:pickle.dump(records,f)
    dump(out/'EXPERT_RECEIPT.json',{'completed':True,'actual_real_transitions':transitions,'policy':'same-task earliest no-plan, keep live plan; includes all failures; no outcome lookup','events':events,'wall_s':time.perf_counter()-start,'sha256':hashlib.sha256((out/'expert_replay.pkl').read_bytes()).hexdigest(),'intended_sampling_fraction':.2,'BC_updates':0})
    print('EXPERT COMPLETE',events,flush=True)

class RecordV1(BaseCallback):
    def __init__(self,out,steps):super().__init__();self.out=out;self.marks={0,500,2000,5000,10000,steps};self.saved=set();self.started=time.perf_counter();self.events={}
    def save(self,mark):
        self.model.save(self.out/f'checkpoints/step_{mark}');self.model.save_replay_buffer(self.out/'replay_latest.pkl')
        import random
        with (self.out/'rng_latest.pkl').open('wb') as f:pickle.dump({'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'python':random.getstate(),'actual_stored_steps':self.model.replay_buffer.n_adds*4},f)
        self.saved.add(mark)
    def _on_training_start(self):self.handle=(self.out/'decisions.jsonl').open('w',encoding='utf-8');self.save(0)
    def _on_step(self):
        n=self.num_timesteps
        for k,i in enumerate(self.locals['infos']):
            r={'step':n,'env':k,'observation_before':self.model._last_obs[k],'observation_after':i.get('terminal_observation',self.locals['new_obs'][k]),'reward':self.locals['rewards'][k],'raw_action':self.locals['actions'][k],'executed_action':i['executed_action'],'review':i['review'],'physical_review':i['physical_review'],'execution':i['execution'],'support':i['support_transition'],'event':i['event'],'mission_deadline_s':i['mission_deadline_s'],'mission_budget_m_s':i['mission_budget_m_s'],'budget_remaining_m_s':i['budget_remaining_m_s']}
            self.handle.write(json.dumps(r,default=default)+'\n')
            if i['event']:self.events[i['event']]=self.events.get(i['event'],0)+1
            if i['event']=='numerical_failure':raise RuntimeError('Real numerical failure; stop, never hide')
        if n%100<4:
            self.handle.flush();dump(self.out/'heartbeat.json',{'steps':n,'critic_updates':self.model._n_updates,'actor_updates':self.model.actor_updates,'events':self.events,'pid':os.getpid(),'wall_s':time.perf_counter()-self.started,'model_prefixes_closed':self.model.replay_buffer.feedback_stats['closed'],'valid_model_prefixes':self.model.replay_buffer.feedback_stats['valid'],'lambda':self.model.delta_calibration['lambda']})
            dump(self.out/'learning_audit.json',self.model.support_stats)
        for mark in sorted(self.marks-self.saved):
            if n>=mark:self.save(mark)
        if not all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters()):raise RuntimeError('Nonfinite parameters')
        return True
    def _on_training_end(self):self.handle.close()

def train(out,arm,steps=5000,smoke=False,seed=0):
    if not smoke and subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Formal paired short training requires clean frozen science source')
    out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    torch.set_num_threads(1);env=SubprocVecEnv([partial(make_env,k,arm=='e_tcpf',seed) for k in range(4)],start_method='spawn')
    model=model_for(env,arm,smoke,out,seed);expert_file=ROOT/'结果/expert/expert_replay.pkl'
    records=pickle.loads(expert_file.read_bytes())
    for r in records:model.replay_buffer.add(r['obs'],r['next_obs'],r['action'],r['reward'],r['done'],r['infos'])
    b=model.replay_buffer;b.demo_slots=b.pos;b._open=[None]*4;b._prefix=[None]*4;b._episode=[[] for _ in range(4)]
    # Replay initialization only: actor weights untouched, guided_init==interface_0k.
    model.save(out/'guided_init')
    dump(out/'manifest.json',{'arm':arm,'seed':seed,'steps':steps,'smoke_not_formal':smoke,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'profile':asdict(mission()),'profile_sha256':hashlib.sha256(PROFILE.read_bytes()).hexdigest(),'cold_model_sha256':hashlib.sha256(COLD.read_bytes()).hexdigest(),'expert_sha256':hashlib.sha256(expert_file.read_bytes()).hexdigest(),'expert_real_transitions':len(records)*4,'expert_sampling_fraction':.2,'BC_updates':0,'guided_init_weights_unchanged':True,'learning_starts':model.learning_starts,'critic_loss':'real SMDP Huber10 + capped paired MPC relative Huber10 (complete arm only)','actor_loss':'exact sum over induced Gaussian-bin probabilities at actual executed tau Q; categorical entropy','network':[128,128],'actual_review_period':3,'MPC':'unchanged h35, real0.1s; nominal model QP stride5 controls, max3 real2s steps'})
    model.learn(total_timesteps=steps,callback=RecordV1(out,steps));model.save(out/'final');model.save_replay_buffer(out/'replay_final.pkl')
    import random
    with (out/'rng_final.pkl').open('wb') as f:pickle.dump({'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'python':random.getstate(),'online_steps':model.num_timesteps,'actual_stored_steps_including_expert':b.n_adds*4,'clock':'after all final gradient updates, matching final.zip and replay_final.pkl'},f)
    dump(out/'learning_audit.json',model.support_stats);dump(out/'feedback_audit.json',b.feedback_stats);dump(out/'delta_calibration.json',model.delta_calibration)
    dump(out/'finished.json',{'steps':model.num_timesteps,'critic_updates':model._n_updates,'actor_updates':model.actor_updates,'events':'see heartbeat','finite':True,'expert_sampling':b.sample_counts,'interval_counts':b.interval_counts,'interval_lengths':b.interval_lengths})
    env.close();print('TRAIN COMPLETE',arm,steps,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','eval','expert','train']);p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--label',default='interface_0k');p.add_argument('--arm',choices=['e_sac','e_tcpf'],default='e_sac');p.add_argument('--steps',type=int,default=5000);p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=0);a=p.parse_args();out=Path(a.output)
    if a.mode!='train':out.mkdir(parents=True,exist_ok=True)
    if a.mode=='contract':contract(out)
    elif a.mode=='expert':expert(out,a.steps)
    elif a.mode=='eval':evaluate(out,a.checkpoint,a.label)
    else:train(out,a.arm,a.steps,a.smoke,a.seed)
if __name__=='__main__':main()
