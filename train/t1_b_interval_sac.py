"""Real-interval SAC with delayed factual validation of paired model intervals."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch as th
import torch.nn.functional as F
from train.t1_tcpf import IntervalReplayBuffer
from train.t1_b_sac import T1BSAC
from env.t1_n_review import N_DUE_INDEX,N_PLAN_REMAINING_INDEX,N_DESCENT_INDEX
from env.t1_b_interval_cf import calibrate_interval

def json_default(x):
    return x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x))

class DelayedModelReplayBuffer(IntervalReplayBuffer):
    due_index=N_DUE_INDEX
    def __init__(self,*args,feedback_audit_path=None,**kwargs):
        super().__init__(*args,**kwargs)
        shape=(self.buffer_size,self.n_envs,1)
        self.cf_discount=np.zeros(shape,np.float32);self.cf_done=np.zeros(shape,np.float32)
        self.cf_fact_rewards=np.zeros(shape,np.float32);self.cf_fact_discount=np.zeros(shape,np.float32)
        self.cf_fact_done=np.zeros(shape,np.float32)
        self.cf_fact_obs=np.zeros_like(self.cf_next_observations)
        self._pending={};self._release_queue=[];self._closing_events=[None]*self.n_envs
        self.feedback_stats={'closed':0,'valid':0,'reasons':Counter(),'weights':[],
                             'return_errors':[],'position_errors':[],'review_steps':[]}
        self.feedback_audit_path=Path(feedback_audit_path) if feedback_audit_path else None

    def _close(self,e,end_obs,terminal,how):
        slot,index,ret,steps=self._open[e]
        pending=self._pending.pop((slot,e),None)
        super()._close(e,end_obs,terminal,how)
        if pending is not None and pending[0]==index and self.interval_ready[slot,e]:
            event=self._closing_events[e] if how!='closed_by_review' else None
            self._release_queue.append((slot,e,index,ret,steps,np.asarray(end_obs).copy(),event,
                                        how=='closed_by_truncation',pending[1]))

    def add(self,obs,next_obs,action,reward,done,infos):
        i=self.pos
        for e in range(self.n_envs):
            self._pending.pop((i,e),None)
            payload=infos[e].get('counterfactual_pending')
            if payload is not None:self._pending[(i,e)]=(self.n_adds,payload)
            self._closing_events[e]=infos[e].get('event')
        super().add(obs,next_obs,action,reward,done,infos)
        for slot,e,index,ret,steps,end,event,truncated,payload in self._release_queue:
            if self.add_index[slot,e]!=index:continue
            gate=calibrate_interval(payload,ret,end,steps,event,truncated)
            weight=gate['weight'];f=payload['factual'];a=payload['alternative']
            self.cf_weights[slot,e]=weight
            if weight>0:
                self.cf_actions[slot,e]=payload['a_tilde'];self.cf_rewards[slot,e]=a['return']
                self.cf_next_observations[slot,e]=a['observation']
                self.cf_discount[slot,e]=a['discount'];self.cf_done[slot,e]=a['done']
                self.cf_fact_rewards[slot,e]=f['return'];self.cf_fact_discount[slot,e]=f['discount']
                self.cf_fact_done[slot,e]=f['done'];self.cf_fact_obs[slot,e]=f['observation']
            s=self.feedback_stats;s['closed']+=1;s['valid']+=int(weight>0)
            s['reasons'].update(gate['reasons']);s['weights'].append(weight)
            s['return_errors'].append(gate['errors']['return']);s['position_errors'].append(gate['errors']['position_m'])
            s['review_steps'].append(steps)
            if self.feedback_audit_path:
                with self.feedback_audit_path.open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps({'slot':slot,'env':e,'add_index':index,'actual_return':ret,
                        'actual_end_observation':end,'actual_steps':steps,'actual_event':event,
                        'gate':gate,'payload':payload},default=json_default)+'\n')
        self._release_queue=[]

    def model_end(self):
        i,e=self._last
        return self.to_torch(self.cf_discount[i,e]),self.to_torch(self.cf_done[i,e])
    def factual_model(self):
        i,e=self._last
        return (self.to_torch(self.cf_fact_rewards[i,e]),self.to_torch(self.cf_fact_discount[i,e]),
                self.to_torch(self.cf_fact_done[i,e]),self.to_torch(self.cf_fact_obs[i,e]))

class T1BIntervalSAC(T1BSAC):
    def __init__(self,*args,preserve_warmup_options=False,**kwargs):
        self.preserve_warmup_options=bool(preserve_warmup_options)
        self.warmup_options_preserved=0
        kwargs.setdefault('replay_buffer_class',DelayedModelReplayBuffer)
        super().__init__(*args,**kwargs)
    def _sample_action(self,learning_starts,action_noise=None,n_envs=1):
        action,stored=super()._sample_action(learning_starts,action_noise,n_envs)
        if self.preserve_warmup_options and self.num_timesteps<learning_starts:
            z=self._last_obs
            keep=(z[:,N_PLAN_REMAINING_INDEX]>0)|(z[:,N_DUE_INDEX]<.5)|(z[:,N_DESCENT_INDEX]>.5)
            action[keep]=0.;stored[keep]=0.
            self.warmup_options_preserved+=int(keep.sum())
        return action,stored
    def _delta_target(self,data,cf,v_next):
        da,donea=self.replay_buffer.model_end()
        rf,df,donef,zf=self.replay_buffer.factual_model()
        with th.no_grad(),th.random.fork_rng(devices=[]):
            state=th.get_rng_state();va=self.target_value(cf.next_observations)
            th.set_rng_state(state);vf=self.target_value(zf)
        return (cf.rewards+da*(1-donea)*va)-(rf+df*(1-donef)*vf)
    def _relative_loss(self,alt_q,current_q,delta,weights):
        return .5*sum((weights*F.huber_loss(qa-qr,delta,reduction='none',delta=self.delta_huber)).mean()
                      for qa,qr in zip(alt_q,current_q))
