"""Short MPC model transitions, not G* cost rankings or long-term oracle labels.

The real branch remains RK45. Shadow states advance with the MPC local
SE(3) predictor and its affine drift; the nominal drift/target model uses
fixed-step RK4, never the truth environment or its RK45 integrator.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from controllers.mpc.prediction import RelativePredictionModel, relative_to_vector, _unwrap_prediction
from dynamics.relative import relative_state
from dynamics.spacecraft import pack_state, unpack_state, pack_derivative, state_derivative
from dynamics.types import GeneralizedForce
from env.t1_counterfactual import CounterfactualPredictor
from env.t1_b_task import T1BReviewEnv
from env.t1_n_review import action_for_tau, neutral_tau

def nominal_rk4(state, parameters, force, dt, gravity):
    y=pack_state(state)
    def rhs(v):
        return pack_derivative(state_derivative(unpack_state(v),parameters,force,
                                               GeneralizedForce.zeros(),gravity))
    k1=rhs(y);k2=rhs(y+dt*k1/2);k3=rhs(y+dt*k2/2);k4=rhs(y+dt*k3)
    return unpack_state(y+dt*(k1+2*k2+2*k3+k4)/6,project_rotation=True)

class NominalRK4Model(RelativePredictionModel):
    def propagate_target(self,target,time_seconds):
        return nominal_rk4(target,self.target_parameters,GeneralizedForce.zeros(),self.dt_s,self.gravity_options)
    def predict_given_target_next(self,relative_vector,wrench_vector,target,target_next,time_seconds):
        x=np.asarray(relative_vector)
        chaser=self.reconstruct_chaser(target,x)
        chaser_next=nominal_rk4(chaser,self.chaser_parameters,GeneralizedForce.from_vector(wrench_vector),
                              self.dt_s,self.gravity_options)
        result=relative_to_vector(relative_state(target_next,chaser_next))
        return _unwrap_prediction(result,x) if self.unwrap_attitude else result

@dataclass(frozen=True)
class ModelFeedbackConfig:
    horizon_steps: int = 2
    every_reviews: int = 2
    max_reward_error: float = .25
    max_position_error_m: float = .05
    max_velocity_error_m_s: float = .01
    max_observation_error: float = .25
    min_tau_difference_s: float = 4.
    min_position_difference_m: float = .005
    min_reward_difference: float = .01

class MissionPredictor(CounterfactualPredictor):
    def __init__(self,real):
        super().__init__(shadow_factory=lambda:T1BReviewEnv(
            config=real.config,env_config=real.env.config,mission=real.mission_distribution,
            mission_override=real.mission_override,earliest_summary=real.earliest_summary))
        old=self.shadow.env.controller.reference_model
        self.shadow.env.controller.reference_model=NominalRK4Model(
            target_parameters=old.target_parameters,chaser_parameters=old.chaser_parameters,
            dt_s=old.dt_s,gravity_options=old.gravity_options,solver_settings=old.solver_settings,
            unwrap_attitude=old.unwrap_attitude)

    def sync(self,real,fresh_solvers=True):
        self.shadow.mission=deepcopy(real.mission)
        self.shadow.env.config=real.env.config
        super().sync(real,fresh_solvers=fresh_solvers)
        self.shadow.mission=deepcopy(real.mission)
        self.shadow.generator.last_outcome=real.generator.last_outcome

    def rollout(self,action,h=2):
        start=perf_counter();sh=self.shadow;e=sh.env;g=sh.generator;c=e.config
        controller=e.controller;model=controller.reference_model
        rewards=[];event=None;solves=0;first_outcome=None;first_tau=None
        for k in range(h):
            reviewed=sh._due
            if k and reviewed:
                break  # stop at a genuine new decision, never fabricate an Actor action
            sh._continuing=not reviewed
            tau,_,_=sh.apply_intent(action if reviewed else sh.continue_action())
            g.decide(e,tau);sh._last_request=tau
            if reviewed:
                sh._last_outcome=g.last_outcome;sh._review_request=tau
            if k==0:
                first_outcome=g.last_outcome;first_tau=g.tau_effective_s
            r=0.
            for _ in range(c.decision_steps):
                if e._delta_v>=sh.mission["budget_m_s"]-1e-10:
                    event="resource_exhausted";r-=c.reward.timeout_penalty;break
                m=e.metrics();waypoint,path=g(e,m)
                x=relative_to_vector(relative_state(e.target_state,e.chaser_state))
                u,_=controller.command(x,target_state=e.target_state,time_seconds=e.time_s,
                    external_reference=waypoint,external_reference_path=path,terminal_latched=bool(m.mouth_region))
                solves+=1
                x_next=controller._predict(x,u)
                target_next=model.propagate_target(e.target_state,e.time_s)
                e.target_state=target_next;e.chaser_state=model.reconstruct_chaser(target_next,x_next)
                e.time_s=round(e.time_s+c.dt_s,10);e._control_steps+=1
                dv=float(np.linalg.norm(np.asarray(u)[3:]))*c.dt_s/e.chaser_parameters.mass
                e._delta_v+=dv
                r-=c.reward.time_cost_per_s*c.dt_s+c.reward.delta_v_cost_per_m_s*dv
                m=e.metrics();e._track(m)
                e._capture_streak=e._capture_streak+1 if m.instantaneous_capture and not m.hard_violation else 0
                if m.hard_violation or m.range_m>c.task.max_distance_m:
                    event="unsafe";r-=c.reward.unsafe_penalty;break
                if e._capture_streak>=c.task.completion_required_steps(c.dt_s):
                    event="success";r+=c.reward.capture_reward;break
                if e.time_s>=c.max_time_s-1e-9:
                    event="timeout";r-=c.reward.timeout_penalty;break
                if e._delta_v>=sh.mission["budget_m_s"]-1e-10:
                    event="resource_exhausted";r-=c.reward.timeout_penalty;break
            e.decision+=1;sh._continuing=False;sh.advance_review(reviewed);rewards.append(r)
            if event:break
        q=e.chaser_state.position-e.target_state.position
        v=e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity
        return {"rewards":rewards,"return":sum(c.reward.gamma**k*r for k,r in enumerate(rewards)),
                "discount":c.reward.gamma**len(rewards),"done":bool(event),"event":event,
                "observation":sh._observation(),"q":q,"v":v,"tau":first_tau,
                "outcome":first_outcome,"solves":solves,"wall_s":perf_counter()-start}

def alternate_action(real,action):
    g=real.generator;e=real.env
    if g._descending(e,e.metrics()):return None
    a=float(np.asarray(action).reshape(-1)[0]);remaining=g.remaining_s(e)
    if remaining>0:
        if abs(real.shift_of(a))>real.config.keep_tolerance_s:return np.zeros(1)
        shift=15. if real._review_index%4==0 else -15.
        if remaining+shift<15.:shift=15.
        return np.array([real.action_for_shift(shift)],np.float32)
    tau=neutral_tau(a,e.time_s,e.config.max_time_s,g.deadline_margin_s)
    alt=action_for_tau(tau+30,e.time_s,e.config.max_time_s,g.deadline_margin_s)
    if alt<=a+1e-6:alt=action_for_tau(tau-30,e.time_s,e.config.max_time_s,g.deadline_margin_s)
    return np.array([alt],np.float32)

class T1BFeedbackEnv(T1BReviewEnv):
    def __init__(self,*args,feedback:ModelFeedbackConfig|None=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.feedback=feedback or ModelFeedbackConfig();self.predictor=MissionPredictor(self)
    def step(self,action):
        cfg=self.feedback
        if not self._due or self._review_index%cfg.every_reviews:
            return super().step(action)
        alt=alternate_action(self,action)
        if alt is None:return super().step(action)
        pred=self.predictor;pred.sync(self);fact=pred.rollout(action,1)
        pred.sync(self);counter=pred.rollout(alt,cfg.horizon_steps)
        # Match both alternatives at the same local length for physical separation only.
        pred.sync(self);fact_full=pred.rollout(action,cfg.horizon_steps)
        obs,r,t,x,info=super().step(action)
        q=self.env.chaser_state.position-self.env.target_state.position
        v=self.env.chaser_state.rotation@self.env.chaser_state.velocity-self.env.target_state.rotation@self.env.target_state.velocity
        errors={"reward":abs(fact["rewards"][0]-r),"position_m":float(np.linalg.norm(fact["q"]-q)),
                "velocity_m_s":float(np.linalg.norm(fact["v"]-v)),
                "observation":float(np.linalg.norm(fact["observation"]-obs)),
                "terminal_match":fact["event"]==info["event"]}
        reasons=[]
        for name,limit in (("reward",cfg.max_reward_error),("position_m",cfg.max_position_error_m),
                           ("velocity_m_s",cfg.max_velocity_error_m_s),("observation",cfg.max_observation_error)):
            if errors[name]>limit:reasons.append(name)
        if not errors["terminal_match"]:reasons.append("terminal_mismatch")
        if counter["outcome"]!="accepted":reasons.append("alternative_not_accepted")
        if fact["outcome"]!="accepted":reasons.append("factual_not_accepted")
        if counter["done"]:
            # A small factual state error does not validate a discontinuous
            # alternative capture/unsafe/resource label. Keep it in the audit,
            # but do not inject unvalidated terminal rewards into the critic.
            reasons.append("predicted_terminal_unvalidated")
        if abs((counter["tau"] or 0)-(fact["tau"] or 0))<cfg.min_tau_difference_s:reasons.append("same_tau")
        dp=float(np.linalg.norm(counter["q"]-fact_full["q"]))
        dr=float(counter["return"]-fact_full["return"])
        if dp<cfg.min_position_difference_m and abs(dr)<cfg.min_reward_difference:reasons.append("no_physical_difference")
        if t or x:reasons.append("factual_ended")
        weight=0. if reasons else max(0.,1.-max(errors["reward"]/cfg.max_reward_error,
                                             errors["position_m"]/cfg.max_position_error_m,
                                             errors["velocity_m_s"]/cfg.max_velocity_error_m_s))
        info["counterfactual"]={"w":int(weight>0),"rank_weight":weight,"a_tilde":alt,
            "r_hat":counter["return"],"z_hat":counter["observation"],
            "cf_discount":counter["discount"],"cf_done":counter["done"],"cf_event":counter["event"],
            "errors_real":errors,"reasons":reasons,"physical_difference_m":dp,
            "local_return_difference":dr,"tau_real":fact["tau"],"tau_alt":counter["tau"],
            "solves":fact["solves"]+fact_full["solves"]+counter["solves"],
            "overhead_wall_s":fact["wall_s"]+fact_full["wall_s"]+counter["wall_s"]}
        return obs,r,t,x,info
