"""MPC model feedback at the same decision interval as the real Bellman target.

Only the shadow replans controls in 0.5 s blocks; real MPC remains 0.1 s.
Model rollouts stop at the next review/event, before any unknown future Actor
decision. Their confidence is assigned later, when the factual real interval
has actually closed. No full-mission ranking or truth rollout enters training.
"""
from dataclasses import dataclass,asdict
from time import perf_counter
import numpy as np
from controllers.mpc.prediction import relative_to_vector
from dynamics.relative import relative_state
from env.t1_b_task import T1BReviewEnv
from env.t1_b_cf import MissionPredictor,alternate_action

@dataclass(frozen=True)
class IntervalModelConfig:
    qp_stride_controls: int = 5
    maximum_macro_steps: int = 3
    max_return_error: float = .75
    max_position_error_m: float = .15
    max_velocity_error_m_s: float = .03
    max_observation_error: float = .75
    min_tau_difference_s: float = 4.
    min_physical_difference_m: float = .005
    min_reward_difference: float = .01

class IntervalMissionPredictor(MissionPredictor):
    def __init__(self,real,config=None):
        super().__init__(real);self.interval_config=config or IntervalModelConfig()
    def rollout(self,action,h=None):
        cfg=self.interval_config;h=cfg.maximum_macro_steps if h is None else h
        sh=self.shadow;e=sh.env;g=sh.generator;c=e.config
        controller=e.controller;model=controller.reference_model
        limits=np.array([*[c.max_torque_per_axis_nm]*3,*[c.max_force_per_axis_n]*3])
        start=perf_counter();rewards=[];observations=[];event=None;solves=0;fallbacks=0;first_tau=None;first_outcome=None
        last_plan=None;last_mode=None;last_mouth=None
        for k in range(h):
            if k and sh._due:break
            reviewed=sh._due;sh._continuing=not reviewed
            tau,_,_=sh.apply_intent(action if reviewed else sh.continue_action())
            g.decide(e,tau);sh._last_request=tau
            if k==0:first_tau=g.remaining_s(e);first_outcome=g.last_outcome
            r=0.
            for j in range(c.decision_steps):
                remaining=max(0.,sh.mission['budget_m_s']-e._delta_v)
                if remaining<=1e-10:event='resource_exhausted';r-=c.reward.timeout_penalty;break
                m=e.metrics();waypoint,path=g(e,m)
                need_qp=(j%cfg.qp_stride_controls==0 or id(g.plan)!=last_plan
                         or g.last_status!=last_mode or m.mouth_region!=last_mouth)
                x=relative_to_vector(relative_state(e.target_state,e.chaser_state))
                if need_qp:
                    u,diagnostics=controller.command(x,target_state=e.target_state,time_seconds=e.time_s,
                        external_reference=waypoint,external_reference_path=path,terminal_latched=bool(m.mouth_region))
                    solves+=1;fallbacks+=int(diagnostics.used_zero_fallback)
                else:
                    controller._nominal_controls=np.vstack((controller._nominal_controls[1:],np.zeros((1,6))))
                    u=controller._nominal_controls[0].copy();controller._control_step+=1
                last_plan,last_mode,last_mouth=id(g.plan),g.last_status,m.mouth_region
                u=np.clip(np.asarray(u,dtype=float),-limits,limits)
                dv=float(np.linalg.norm(u[3:]))*c.dt_s/e.chaser_parameters.mass
                if dv>remaining:u[3:]*=remaining/dv;dv=remaining
                xn=controller._predict(x,u);tn=model.propagate_target(e.target_state,e.time_s)
                e.target_state=tn;e.chaser_state=model.reconstruct_chaser(tn,xn)
                e.time_s=round(e.time_s+c.dt_s,10);e._control_steps+=1;e._delta_v+=dv
                r-=c.reward.time_cost_per_s*c.dt_s+c.reward.delta_v_cost_per_m_s*dv
                m=e.metrics();e._track(m)
                e._capture_streak=e._capture_streak+1 if m.instantaneous_capture and not m.hard_violation else 0
                if m.hard_violation or m.range_m>c.task.max_distance_m:
                    event='unsafe';r-=c.reward.unsafe_penalty;break
                if e._capture_streak>=c.task.completion_required_steps(c.dt_s):
                    event='success';r+=c.reward.capture_reward;break
                if e.time_s>=c.max_time_s-1e-9:
                    event='timeout';r-=c.reward.timeout_penalty;break
                if e._delta_v>=sh.mission['budget_m_s']-1e-10:
                    event='resource_exhausted';r-=c.reward.timeout_penalty;break
            if reviewed:sh._last_outcome=g.last_outcome;sh._review_request=tau
            elif g.last_outcome=='replan_failed':sh._last_outcome=g.last_outcome
            e.decision+=1;sh._continuing=False;sh.advance_review(reviewed)
            rewards.append(r);observations.append(sh._observation().copy())
            if event:break
        q=e.chaser_state.position-e.target_state.position
        v=e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity
        return {'rewards':rewards,'return':sum(c.reward.gamma**k*r for k,r in enumerate(rewards)),
            'discount':c.reward.gamma**len(rewards),'steps':len(rewards),'done':bool(event),'event':event,
            'observation':observations[-1],'observations':observations,'q':q,'v':v,
            'tau':first_tau,'outcome':first_outcome,'solves':solves,'qp_zero_fallbacks':fallbacks,'wall_s':perf_counter()-start}

def make_payload(real,predictor,action,cfg):
    alt=alternate_action(real,action)
    if alt is None:return None
    predictor.sync(real);fact=predictor.rollout(action)
    predictor.sync(real);counter=predictor.rollout(alt)
    if fact['steps']==counter['steps']:
        effect=float(np.linalg.norm(counter['q']-fact['q']))
    else:
        effect=float(np.linalg.norm(counter['observations'][0][:3]-fact['observations'][0][:3])*15.)
    return {'a_tilde':np.asarray(alt,dtype=np.float32),'factual':fact,'alternative':counter,
            'physical_difference_m':effect,'first_reward_difference':counter['rewards'][0]-fact['rewards'][0],
            'config':asdict(cfg),'kind':'review_interval_paired_model',
            'overhead_wall_s':fact['wall_s']+counter['wall_s']}

def calibrate_interval(payload,actual_return,actual_end_obs,actual_steps,actual_event,truncated=False):
    f=payload['factual'];a=payload['alternative'];cfg=IntervalModelConfig(**payload['config'])
    errors={'return':abs(f['return']-float(actual_return)),
            'position_m':float(np.linalg.norm(np.asarray(f['observation'])[:3]-actual_end_obs[:3])*15.),
            'velocity_m_s':float(np.linalg.norm(np.asarray(f['observation'])[3:6]-actual_end_obs[3:6])*.2),
            'observation':float(np.linalg.norm(np.asarray(f['observation'])-actual_end_obs))}
    reasons=[]
    if truncated:reasons.append('actual_numerical_truncation')
    if f['steps']!=actual_steps:reasons.append('factual_review_time_mismatch')
    if f['event']!=actual_event:reasons.append('factual_event_mismatch')
    if f['outcome']!='accepted' or a['outcome']!='accepted':reasons.append('intent_not_accepted')
    if abs((f['tau'] or 0)-(a['tau'] or 0))<cfg.min_tau_difference_s:reasons.append('same_effective_tau')
    if a['done']:reasons.append('alternative_terminal_not_validated')
    if f.get('qp_zero_fallbacks',0) or a.get('qp_zero_fallbacks',0):reasons.append('model_qp_zero_fallback')
    for name,limit in [('return',cfg.max_return_error),('position_m',cfg.max_position_error_m),
                       ('velocity_m_s',cfg.max_velocity_error_m_s),('observation',cfg.max_observation_error)]:
        if errors[name]>limit:reasons.append(name)
    effect=payload['physical_difference_m']
    if effect<cfg.min_physical_difference_m and abs(payload['first_reward_difference'])<cfg.min_reward_difference:
        reasons.append('no_physical_effect')
    if effect<2*errors['position_m'] and abs(a['return']-f['return'])<2*errors['return']:
        reasons.append('effect_below_factual_error')
    weight=0. if reasons else max(0.,1-max(errors['return']/cfg.max_return_error,
        errors['position_m']/cfg.max_position_error_m,errors['velocity_m_s']/cfg.max_velocity_error_m_s))
    return {'weight':weight,'reasons':reasons,'errors':errors}

class T1BIntervalFeedbackEnv(T1BReviewEnv):
    def __init__(self,*args,interval_model=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.interval_model=interval_model or IntervalModelConfig()
        self.predictor=IntervalMissionPredictor(self,self.interval_model)
    def step(self,action):
        payload=make_payload(self,self.predictor,action,self.interval_model) if self._due else None
        obs,r,t,x,info=super().step(action)
        if payload is not None:
            info['counterfactual_pending']=payload
            info['counterfactual']={'w':0,'kind':'pending_actual_interval','payload':payload}
        return obs,r,t,x,info
