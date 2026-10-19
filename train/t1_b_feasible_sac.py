"""Exact finite-support SAC induced by the original scalar Gaussian actor.

Gradients are of categorical probabilities, NOT of a discontinuous tau map.
Same two Q networks receive real interval TD and paired MPC-relative loss.
"""
import json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from train.t1_b_interval_sac import T1BIntervalSAC,DelayedModelReplayBuffer
from train.t1_tcpf import IntervalReplayBuffer
from env.t1_n_review import N_DUE_INDEX
from env.t1_b_interval_cf import calibrate_interval

class ExpertPrefixReplay(DelayedModelReplayBuffer):
    def __init__(self,*args,expert_fraction=.2,**kwargs):
        super().__init__(*args,**kwargs);self.expert_fraction=expert_fraction;self.demo_slots=0;self._prefix=[None]*self.n_envs;self.sample_counts={'expert':0,'online':0,'actor_expert':0,'actor_online':0}
    def _close(self,e,end_obs,terminal,how):
        # Main SMDP closes at the next genuine task decision. The independent
        # model prefix is validated at its own predeclared <=6s actual endpoint.
        IntervalReplayBuffer._close(self,e,end_obs,terminal,how)
    def add(self,obs,next_obs,action,reward,done,infos):
        slot,index=self.pos,self.n_adds;payloads=[i.get('counterfactual_pending') for i in infos]
        clean=[{k:v for k,v in i.items() if k not in ('counterfactual_pending','counterfactual')} for i in infos]
        super().add(obs,next_obs,action,reward,done,clean)
        for e,payload in enumerate(payloads):
            if payload is not None:
                if self._prefix[e] is not None:
                    prior=self._prefix[e];s=self.feedback_stats;s['closed']+=1;s['reasons'].update(['real_early_review_before_predicted_horizon']);s['weights'].append(0.);s['return_errors'].append(abs(prior[2]-prior[4]['factual']['return']));s['position_errors'].append(float(np.linalg.norm(prior[4]['factual']['observation'][:3]-np.asarray(obs[e])[:3])*15));s['review_steps'].append(prior[3])
                    if self.feedback_audit_path:
                        with self.feedback_audit_path.open('a',encoding='utf-8') as f:f.write(json.dumps({'slot':int(prior[0]),'env':e,'add_index':int(prior[1]),'actual_return':prior[2],'actual_steps':prior[3],'gate':{'weight':0.,'reasons':['real_early_review_before_predicted_horizon']},'payload':prior[4]},default=lambda z:z.tolist() if isinstance(z,np.ndarray) else str(z))+'\n')
                self._prefix[e]=[slot,index,0.,0,payload]
            p=self._prefix[e]
            if p is None:continue
            p[2]+=self.gamma**p[3]*float(reward[e]);p[3]+=1
            horizon=p[4]['factual']['steps']
            if p[3]>=horizon or bool(done[e]):
                self._prefix[e]=None;i,ix,ret,d,pay=p
                if self.add_index[i,e]!=ix:continue
                event=infos[e].get('event') if bool(done[e]) else None
                gate=calibrate_interval(pay,ret,np.asarray(next_obs[e]),d,event,infos[e].get('TimeLimit.truncated',False))
                if pay['alternative']['steps']!=d:
                    gate['reasons'].append('unequal_prefix_horizon');gate['weight']=0.
                w=gate['weight'];fact,alt=pay['factual'],pay['alternative'];self.cf_weights[i,e]=w
                if w>0:
                    self.cf_actions[i,e]=pay['a_tilde'];self.cf_rewards[i,e]=alt['return'];self.cf_discount[i,e]=alt['discount'];self.cf_done[i,e]=alt['done'];self.cf_next_observations[i,e]=alt['observation']
                    self.cf_fact_rewards[i,e]=fact['return'];self.cf_fact_discount[i,e]=fact['discount'];self.cf_fact_done[i,e]=fact['done'];self.cf_fact_obs[i,e]=fact['observation']
                s=self.feedback_stats;s['closed']+=1;s['valid']+=int(w>0);s['weights'].append(w);s['reasons'].update(gate['reasons']);s['return_errors'].append(gate['errors']['return']);s['position_errors'].append(gate['errors']['position_m']);s['review_steps'].append(d)
                if self.feedback_audit_path:
                    with self.feedback_audit_path.open('a',encoding='utf-8') as f:f.write(json.dumps({'slot':int(i),'env':e,'add_index':int(ix),'actual_return':ret,'actual_steps':d,'actual_event':event,'actual_end_observation':np.asarray(next_obs[e]).tolist(),'gate':gate,'payload':pay},default=lambda z:z.tolist() if isinstance(z,np.ndarray) else (z.item() if isinstance(z,np.generic) else str(z)))+'\n')
    def sample(self,batch_size,env=None):
        upper=self.buffer_size if self.full else self.pos
        n=int(round(batch_size*self.expert_fraction)) if self.demo_slots and upper>self.demo_slots else 0
        online_low=self.demo_slots if upper>self.demo_slots else 0
        indices=np.r_[np.random.randint(0,self.demo_slots,n) if n else [],np.random.randint(online_low,upper,batch_size-n)].astype(int)
        np.random.shuffle(indices);self.sample_counts['expert']+=n;self.sample_counts['online']+=batch_size-n
        return self._get_samples(indices,env)
    def sample_closed_review(self,batch_size,env=None):
        upper=self.buffer_size if self.full else self.pos;flat=np.flatnonzero((self.due[:upper]&self.interval_ready[:upper]).reshape(-1))
        if not len(flat):return None
        demos=flat[flat<self.demo_slots*self.n_envs];online=flat[flat>=self.demo_slots*self.n_envs]
        n=int(round(batch_size*self.expert_fraction)) if len(demos) and len(online) else (batch_size if len(demos) else 0)
        chosen=np.r_[np.random.choice(demos,n) if n else [],np.random.choice(online,batch_size-n) if batch_size>n else []].astype(int)
        i,e=np.divmod(chosen,self.n_envs);self.sample_counts['actor_expert']+=n;self.sample_counts['actor_online']+=batch_size-n
        return self.to_torch(self._normalize_obs(self.observations[i,e],env))

class FeasibleSupportSAC(T1BIntervalSAC):
    def __init__(self,*args,**kwargs):
        kwargs['replay_buffer_class']=ExpertPrefixReplay
        super().__init__(*args,**kwargs)
        self.support_stats=[]
    def _sample_action(self,learning_starts,action_noise=None,n_envs=1):
        action,stored=super()._sample_action(learning_starts,action_noise,n_envs)
        if self.num_timesteps<learning_starts:
            for k,z in enumerate(self._last_obs):
                if z[N_DUE_INDEX]<.5:action[k]=0.;stored[k]=0.
                elif z[27]>.5:
                    n=int((z[49:52]>.5).sum());j=int(np.argmin(abs(z[46:46+n]*200-z[28]*200)))
                    raw=2*(j+.5)/n-1;action[k]=raw;stored[k]=raw
        return action,stored
    def _excluded_save_params(self):return [*super()._excluded_save_params(),'support_stats']
    def support_probabilities(self,obs):
        mean,logstd,_=self.actor.get_action_dist_params(obs);sigma=logstd.exp();normal=th.distributions.Normal(mean,sigma)
        counts=(obs[:,49:52]>.5).sum(1).clamp(min=1);p=th.zeros((len(obs),3),device=obs.device)
        for n in [1,2,3]:
            edges=[th.zeros_like(mean)]
            for j in range(1,n):edges.append(normal.cdf(th.full_like(mean,float(np.arctanh(2*j/n-1)))))
            edges.append(th.ones_like(mean));pn=th.cat([edges[j+1]-edges[j] for j in range(n)],1)
            pn=F.pad(pn,(0,3-n));p=th.where((counts==n)[:,None],pn,p)
        return p.clamp_min(0.)
    def candidate_q(self,obs,critic):
        a=obs[:,46:49]*2.-1.;z=obs[:,None,:].expand(-1,3,-1).reshape(-1,52)
        qs=critic(z,a.reshape(-1,1));return tuple(q.reshape(-1,3) for q in qs)
    def target_value(self,obs):
        with th.no_grad():
            p=self.support_probabilities(obs);q=th.minimum(*self.candidate_q(obs,self.critic_target));v=(p*(q-self.alpha_of(obs)*p.clamp_min(1e-8).log())).sum(1,keepdim=True)
            cont=th.cat(self.critic_target(obs,th.zeros((len(obs),1),device=obs.device)),1).min(1,keepdim=True).values
            return th.where(obs[:,N_DUE_INDEX:N_DUE_INDEX+1]>.5,v,cont)
    def _td_loss(self,current,target,mask):
        mask=th.ones_like(target) if mask is None else mask
        return .5*sum((mask*F.huber_loss(q,target,reduction='none',delta=10.)).sum()/mask.sum().clamp(min=1) for q in current)
    def train(self,gradient_steps,batch_size=256):
        self.policy.set_training_mode(True);self._update_learning_rate([self.actor.optimizer,self.critic.optimizer])
        owed=max((self.num_timesteps-self.learning_starts)//3-self.actor_updates,0)
        params=list(self.critic.parameters());norm=lambda gs:float(th.sqrt(sum((g*g).sum() for g in gs if g is not None)))
        for step in range(gradient_steps):
            data=self.replay_buffer.sample(batch_size);self._n_updates_in_call=self._n_updates+step
            target,mask=self._td_targets(data,self.target_value(data.next_observations));current=self.critic(data.observations,data.actions);td=self._td_loss(current,target,mask);loss=td
            cf=self.replay_buffer.counterfactual();weights=self._cf_weights(cf);valid=float(weights.sum());applied=share=gtd=gcf=relative=0.;phase=self.delta_phase();lam=self.current_delta_lambda()
            if self.delta_enabled() and valid>0 and phase in ('calibrating','on'):
                with th.no_grad():delta=self._delta_target(data,cf,None)
                aq=self.critic(data.observations,cf.actions);rel=self._relative_loss(aq,current,delta,weights)
                gtd=norm(th.autograd.grad(td,params,retain_graph=True,allow_unused=True));gcf=norm(th.autograd.grad(rel,params,retain_graph=True,allow_unused=True));relative=float(rel.item())
                if phase=='calibrating' and self._n_updates_in_call%5==0 and gtd>1e-8:
                    self.delta_calibration['ratios'].append(gcf/gtd);self.delta_calibration['decisions'].append(int(self.num_timesteps))
                if phase=='on' and gtd>1e-8 and gcf>0:
                    applied=min(lam,.1*gtd/gcf);share=applied*gcf/gtd;loss=td+applied*rel
            self.critic.optimizer.zero_grad();loss.backward();self.critic.optimizer.step()
            aloss=qgrad=0.
            if step>=gradient_steps-owed:
                obs=self.replay_buffer.sample_closed_review(batch_size)
                if obs is not None:
                    p=self.support_probabilities(obs);q=th.minimum(*self.candidate_q(obs,self.critic)).detach()
                    qterm=-(p*q).sum(1).mean();entropy=(self.alpha_of(obs)*p*p.clamp_min(1e-8).log()).sum(1).mean();actor_loss=qterm+entropy
                    qgrad=norm(th.autograd.grad(qterm,list(self.actor.parameters()),retain_graph=True,allow_unused=True))
                    self.actor.optimizer.zero_grad();actor_loss.backward();self.actor.optimizer.step();self.actor_updates+=1;aloss=float(actor_loss.item())
            polyak_update(self.critic.parameters(),self.critic_target.parameters(),self.tau)
            if self._n_updates_in_call%20 in (0,3):
                self.support_stats.append({'steps':int(self.num_timesteps),'update':int(self._n_updates_in_call),'actor_updates':int(self.actor_updates),'real_huber_td':float(td.item()),'relative_loss':relative,'valid_weight_mean':valid/batch_size,'g_td':gtd,'g_mpc_relative':gcf,'applied_lambda':applied,'applied_gradient_share':share,'actor_loss':aloss,'actor_gradient_via_same_Q':qgrad,'phase':phase,'expert_sampling':dict(self.replay_buffer.sample_counts)})
        self._n_updates+=gradient_steps

def transfer_cold_weights(model,old_path):
    old=T1BIntervalSAC.load(old_path,device='cpu');old_state=old.policy.state_dict();state=model.policy.state_dict()
    with th.no_grad():
        for key,t in state.items():
            src=old_state[key]
            if t.shape==src.shape:t.copy_(src)
            elif t.ndim==2 and src.shape[0]==t.shape[0] and t.shape[1]==src.shape[1]+6:
                t.zero_()
                if key.startswith('actor.'):t[:,:46].copy_(src)
                else:t[:,:46].copy_(src[:,:46]);t[:,-1:].copy_(src[:,-1:])
            else:raise ValueError((key,t.shape,src.shape))
        model.policy.load_state_dict(state)
    return old
