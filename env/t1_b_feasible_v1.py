"""Same frozen T1-B truth; verified finite arrival support, no reward oracle.

The scalar Gaussian latent selects a bin. Q sees the actual executed tau
coordinate. The learner integrates bin probabilities exactly, never treats
the staircase as a differentiable physical-time mapping.
"""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import gymnasium as gym
from env.t1_b_task import T1BReviewEnv,OBSERVATION_NAMES
from env.t1_n_review import N_DUE_INDEX
from env.t1_gstar import free_body_rotations,_plan_position
from env.t1_b_interval_cf import IntervalMissionPredictor,IntervalModelConfig
from env.t1_b_cf import NominalRK4Model
from env.t1_counterfactual import CounterfactualPredictor

VERSION='t1b-feasible-induced-categorical-sac-v1'
N_BASE=46
OBS_NAMES=OBSERVATION_NAMES+tuple(f'support_tau_{j}_200s' for j in range(3))+tuple(f'support_valid_{j}' for j in range(3))

def choice_index(a,n):return min(n-1,max(0,int(np.floor((float(a)+1)*.5*n)))) if n else None
def bin_center(j,n):return 2*(j+.5)/n-1
def executed_coordinate(tau):return float(tau/100.-1.)

class FeasibleArrivalEnv(T1BReviewEnv):
    def __init__(self,*args,feedback=False,**kwargs):
        self._support_cache=None;self.support_solves=0;self.support_wall=0.;self.feedback_enabled=False;self.predictor=None
        super().__init__(*args,**kwargs)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(52,),np.float32)
        self.feedback_enabled=feedback
        if feedback:self.predictor=FeasiblePredictor(self)

    def reset(self,**kwargs):
        self._support_cache=None
        _,info=super().reset(**kwargs)
        self._support_cache=None
        return self._observation(),info

    def support(self):
        from time import perf_counter
        e,g=self.env,self.generator
        if not self._due or g._descending(e,e.metrics()):return ()
        key=(e.time_s,e.config.max_time_s,g.deadline_margin_s,e._delta_v,e.chaser_state.position.tobytes())
        if self._support_cache and self._support_cache[0]==key:return self._support_cache[1]
        start=perf_counter();remaining=g.remaining_s(e);horizon=min(200.,e.config.max_time_s-g.deadline_margin_s-e.time_s)
        if remaining>0:requested=[remaining-15.,remaining,remaining+15.]
        else:
            first=self.earliest_feasible_summary()
            requested=[] if first is None else [first,first+20.,first+40.]
        q=e.chaser_state.position-e.target_state.position
        v=e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity
        budget=max(0.,self.mission['budget_m_s']-e._delta_v);valid=[];audit=[]
        for tau in sorted(set(float(t) for t in requested if 0<t<=horizon+1e-8)):
            if remaining>0 and abs(tau-remaining)<=g.replan_tau_tolerance_s and g.plan is not None:
                deviation=np.linalg.norm(_plan_position(g.plan,e.time_s-g.plan_start_s,g.planner.mass)-q)
                if deviation<=g.replan_deviation_m:
                    valid.append(tau);audit.append({'tau':tau,'status':'kept_verified','necessary_dv_lower_bound':0.});continue
            history=free_body_rotations(e.target_state.rotation,e.target_state.omega,e.target_parameters.inertia,tau+1.)
            result=g.planner.solve(q,v,tau,history,None,entry_speed_m_s=g.entry_speed_m_s);self.support_solves+=1
            if not result['feasible']:audit.append({'tau':tau,'status':result['status']});continue
            # Conservative impulse necessary condition. Differential gravity/J2
            # over the unchanged <=30m, ~500km task is bounded loosely by 1e-3 m/s².
            # This is NOT a nominal-plan fuel cap or a guarantee of task completion.
            dv_min=max(0.,float(np.linalg.norm(result['q'][-1]-q-v*tau))/tau-.001*tau/2)
            audit.append({'tau':tau,'status':result['status'],'necessary_dv_lower_bound':dv_min,'budget_remaining':budget})
            if dv_min<=budget+1e-8:valid.append(tau)
        self.support_wall+=perf_counter()-start
        self._support_cache=(key,tuple(valid),audit)
        return tuple(valid)

    def _observation(self):
        base=super()._observation();choices=self.support()
        # A genuine policy decision requires >=2 admissible alternatives.
        # Singleton/empty support and descent have a canonical nondecision action.
        base[N_DUE_INDEX]=float(self._due and len(choices)>=2)
        return np.r_[base,list(np.asarray(choices)/200.)+[0.]*(3-len(choices)),[1.]*len(choices)+[0.]*(3-len(choices))].astype(np.float32)

    ALTERNATIVE_RULE='farthest verified time; no outcome/rank oracle'
    def alternative_index(self,choices,idx):
        return max((j for j in range(len(choices)) if j!=idx),key=lambda j:abs(choices[j]-choices[idx]))

    def apply_intent(self,action):
        e,g=self.env,self.generator
        if self._continuing:return super().apply_intent(np.zeros(1))
        if g._descending(e,e.metrics()):return super().apply_intent(np.zeros(1))
        choices=self.support();raw=float(np.clip(np.asarray(action).reshape(-1)[0],-1,1));idx=choice_index(raw,len(choices))
        tau=choices[idx] if choices else 0.
        self._request_record={'mode':'verified_support' if choices else 'no_admissible','a':raw,'support_tau_s':choices,'candidate_index':idx,'tau_request_s':tau,'necessary_resource_checks':self._support_cache[2] if self._support_cache else []}
        return float(tau),1.,.02

    def step(self,action):
        before=self._observation();physical_due_before=self._due;choices=tuple(float(x) for x in self.support());active=bool(before[N_DUE_INDEX]>.5)
        raw=float(np.clip(np.asarray(action).reshape(-1)[0],-1,1));idx=choice_index(raw,len(choices))
        canonical=executed_coordinate(choices[idx]) if active else 0.
        payload=paired_payload(self,raw,choices,idx,self.alternative_index(choices,idx)) if self.feedback_enabled and active else None
        obs,r,t,x,info=super().step(np.array([raw],np.float32))
        info['physical_review']=deepcopy(info['review'])
        info['review']['reviewed']=active;info['review']['learning_due']=active
        info['executed_action']=np.array([canonical],np.float32)
        info['support_transition']={'raw_latent':raw,'executed_coordinate':canonical,'tau_candidates':choices,'candidate_index':idx,'learning_decision':active,'empty_support_event':bool(physical_due_before and not choices and not self.generator._descending(self.env,self.env.metrics())),'version':VERSION}
        if payload is not None:
            info['counterfactual_pending']=payload
            info['counterfactual']={'w':0,'kind':'pending_real_prefix_validation'}
        return obs,r,t,x,info

class FeasiblePredictor(IntervalMissionPredictor):
    def __init__(self,real):
        CounterfactualPredictor.__init__(self,shadow_factory=lambda:FeasibleArrivalEnv(config=real.config,env_config=real.env.config,mission=real.mission_distribution,mission_override=real.mission_override,earliest_summary=real.earliest_summary,feedback=False))
        old=self.shadow.env.controller.reference_model
        self.shadow.env.controller.reference_model=NominalRK4Model(target_parameters=old.target_parameters,chaser_parameters=old.chaser_parameters,dt_s=old.dt_s,gravity_options=old.gravity_options,solver_settings=old.solver_settings,unwrap_attitude=old.unwrap_attitude)
        self.interval_config=IntervalModelConfig()
    def sync(self,real,fresh_solvers=True):
        super().sync(real,fresh_solvers=fresh_solvers)
        self.shadow._support_cache=deepcopy(real._support_cache)

def paired_payload(real,raw,choices,idx,alt=None):
    if alt is None:alt=FeasibleArrivalEnv.alternative_index(real,choices,idx)
    pred=real.predictor;pred.sync(real);fact=pred.rollout(np.array([raw],np.float32))
    alt_raw=bin_center(alt,len(choices));pred.sync(real);counter=pred.rollout(np.array([alt_raw],np.float32))
    effect=float(np.linalg.norm(counter['q']-fact['q'])) if counter['steps']==fact['steps'] else 0.
    return {'a_tilde':np.array([executed_coordinate(choices[alt])],np.float32),'factual':fact,'alternative':counter,'physical_difference_m':effect,'first_reward_difference':counter['rewards'][0]-fact['rewards'][0],'config':asdict(pred.interval_config),'kind':'verified_support_paired_mpc_prefix','sampling':{'population_tau_s':choices,'selected_index':idx,'alternative_index':alt,'rule':real.ALTERNATIVE_RULE},'raw_alternative':alt_raw}
