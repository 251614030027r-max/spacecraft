"""One-QP MPC-window feedback, gated against actual receding execution.

This trades prediction fidelity for coverage/cost, without changing real
physics or control. The local window is 2 s, within the h35 MPC horizon.
It supplies alternative Bellman transitions, never long-horizon rankings.
"""
from time import perf_counter
import numpy as np
from controllers.mpc.prediction import relative_to_vector
from dynamics.relative import relative_state
from env.t1_b_cf import MissionPredictor,T1BFeedbackEnv,ModelFeedbackConfig

class FastMissionPredictor(MissionPredictor):
    def __init__(self,real):
        super().__init__(real);self._cache={};self._stamp=None
    def sync(self,real,fresh_solvers=True):
        stamp=(real.current_opening,real.env.time_s,real.env._delta_v,real._review_index,
               real.env.chaser_state.position.tobytes())
        if stamp!=self._stamp:self._stamp=stamp;self._cache={}
        super().sync(real,fresh_solvers=fresh_solvers)
    def rollout(self,action,h=1):
        if h!=1:raise ValueError('Fast feedback is exactly one real 2 s macro-step')
        key=np.asarray(action,dtype=np.float64).tobytes()
        if key in self._cache:return self._cache[key]|{'wall_s':0.,'solves':0,'cache_hit':True}
        start=perf_counter();sh=self.shadow;e=sh.env;g=sh.generator;c=e.config
        reviewed=sh._due;sh._continuing=not reviewed
        tau,_,_=sh.apply_intent(action if reviewed else sh.continue_action())
        g.decide(e,tau);sh._last_request=tau
        outcome=g.last_outcome;effective_tau=g.remaining_s(e)
        if reviewed:sh._last_outcome=outcome;sh._review_request=tau
        m=e.metrics();waypoint,path=g(e,m);controller=e.controller;model=controller.reference_model
        x=relative_to_vector(relative_state(e.target_state,e.chaser_state))
        first,_=controller.command(x,target_state=e.target_state,time_seconds=e.time_s,
            external_reference=waypoint,external_reference_path=path,terminal_latched=bool(m.mouth_region))
        controls=controller._nominal_controls.copy();controls[0]=first
        r=0.;event=None
        limits=np.array([*[c.max_torque_per_axis_nm]*3,*[c.max_force_per_axis_n]*3])
        for j in range(c.decision_steps):
            remaining=max(0.,sh.mission['budget_m_s']-e._delta_v)
            if remaining<=1e-10:event='resource_exhausted';r-=c.reward.timeout_penalty;break
            u=np.clip(controls[j],-limits,limits)
            dv=float(np.linalg.norm(u[3:]))*c.dt_s/e.chaser_parameters.mass
            if dv>remaining:u[3:]*=remaining/dv;dv=remaining
            x=controller._predict(x,u)
            target_next=model.propagate_target(e.target_state,e.time_s)
            e.target_state=target_next;e.chaser_state=model.reconstruct_chaser(target_next,x)
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
        e.decision+=1;sh._continuing=False;sh.advance_review(reviewed)
        q=e.chaser_state.position-e.target_state.position
        v=e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity
        result={'rewards':[r],'return':r,'discount':c.reward.gamma,'done':bool(event),'event':event,
                'observation':sh._observation(),'q':q,'v':v,'tau':effective_tau,'outcome':outcome,
                'solves':1,'wall_s':perf_counter()-start,'cache_hit':False}
        self._cache[key]=result
        return result

class T1BFastFeedbackEnv(T1BFeedbackEnv):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('feedback',ModelFeedbackConfig(horizon_steps=1,every_reviews=1))
        super().__init__(*args,**kwargs)
        self.predictor=FastMissionPredictor(self)
