"""Exploratory feasible arrival catalogue; no reward-based task choice by G*."""
from time import perf_counter
import numpy as np
import gymnasium as gym
from env.t1_b_task import T1BReviewEnv,OBSERVATION_NAMES
from env.t1_n_review import action_for_tau
from env.t1_gstar import free_body_rotations

ADMISSIBLE_VERSION='t1b-admissible-catalogue-probe-1'
CATALOGUE_OBSERVATIONS=OBSERVATION_NAMES+tuple(f'catalogue_tau_{i}_200s' for i in range(3))+tuple(f'catalogue_available_{i}' for i in range(3))

class T1BAdmissibleEnv(T1BReviewEnv):
    """Initial/plan-loss intents choose among feasible tau_min + {0,20,40}s.

    Feasibility is the existing SCP's, never a full-mission reward label. The
    returned choices are not guaranteed to capture under real MPC. Live-plan
    keep/shift, all truth safety, reward and mission budgets remain the base's.
    """
    def __init__(self,*args,**kwargs):
        self._catalogue_cache=None;self.catalogue_wall_s=0.;self.catalogue_solves=0
        super().__init__(*args,**kwargs)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(len(CATALOGUE_OBSERVATIONS),),dtype=np.float32)
    def reset(self,**kwargs):
        self._catalogue_cache=None
        _,info=super().reset(**kwargs)
        self._catalogue_cache=None # Base reset installs the mission's deadline margin last.
        return self._observation(),info
    def catalogue(self):
        e,g=self.env,self.generator
        q=e.chaser_state.position-e.target_state.position
        v=e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity
        key=(float(e.time_s),float(e.config.max_time_s),float(g.deadline_margin_s),q.tobytes(),v.tobytes(),e.target_state.rotation.tobytes(),e.target_state.omega.tobytes())
        if self._catalogue_cache is not None and self._catalogue_cache[0]==key:return self._catalogue_cache[1]
        start=perf_counter();tau,first=g.earliest_feasible(e,g.entry_speed_m_s);choices=[]
        if tau is not None and first is not None:
            choices.append(float(tau))
            for offset in [20.,40.]:
                t=float(tau+offset)
                if t>min(200.,e.config.max_time_s-g.deadline_margin_s-e.time_s)+1e-8:continue
                history=free_body_rotations(e.target_state.rotation,e.target_state.omega,e.target_parameters.inertia,t+1.)
                result=g.planner.solve(q,v,t,history,None,entry_speed_m_s=g.entry_speed_m_s)
                self.catalogue_solves+=1
                if result['feasible']:choices.append(t)
        self.catalogue_wall_s+=perf_counter()-start
        self._catalogue_cache=(key,tuple(choices))
        return tuple(choices)
    def _observation(self):
        base=super()._observation();g=self.generator
        if self._due and g.remaining_s(self.env)<=0 and not g._descending(self.env,self.env.metrics()):choices=self.catalogue()
        else:choices=()
        taus=list(choices)+[0.]*(3-len(choices));mask=[1.]*len(choices)+[0.]*(3-len(choices))
        return np.r_[base,np.asarray(taus)/200.,mask].astype(np.float32)
    def apply_intent(self,action):
        g,e=self.generator,self.env;raw=float(np.clip(np.asarray(action).reshape(-1)[0],-1,1))
        if self._continuing or g.remaining_s(e)>0 or g._descending(e,e.metrics()):return super().apply_intent(action)
        choices=self.catalogue();index=min(len(choices)-1,int(np.floor(.5*(raw+1)*len(choices)))) if choices else None
        tau=float(choices[index]) if choices else 0.
        mapped=action_for_tau(tau,e.time_s,e.config.max_time_s,g.deadline_margin_s)
        _,kappa,speed=super().apply_intent(np.array([mapped],np.float32))
        self._request_record.update(a=raw,tau_request_s=tau,admissible_version=ADMISSIBLE_VERSION,
            catalogue_tau_s=list(choices),catalogue_index=index,base_physical_action=float(mapped))
        return tau,kappa,speed
