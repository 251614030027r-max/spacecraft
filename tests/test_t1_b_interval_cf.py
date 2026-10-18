import copy
import gymnasium as gym
import numpy as np
from unittest.mock import patch
from dataclasses import asdict
from env.t1_b_task import T1BReviewEnv
from env.t1_b_interval_cf import IntervalMissionPredictor,IntervalModelConfig
from train.t1_b_interval_sac import DelayedModelReplayBuffer
from env.t1_n_review import N_DUE_INDEX

def payload(steps=3,event=None):
    z=np.zeros(46,np.float32);z[N_DUE_INDEX]=1
    ret=-.1-.999*.2-.999**2*.3 if steps==3 else -100.
    fact={'observation':z.copy(),'return':ret,'discount':.999**steps,'done':bool(event),
          'event':event,'steps':steps,'tau':110.,'outcome':'accepted'}
    alt=copy.deepcopy(fact);alt.update(return_ignored=0.,tau=140.,done=False,event=None,steps=3,discount=.999**3)
    alt['return']=-.4;alt['observation'][0]=.01
    return {'factual':fact,'alternative':alt,'a_tilde':np.array([.6],np.float32),
            'physical_difference_m':.15,'first_reward_difference':.02,'config':asdict(IntervalModelConfig())}

def buffer(n_envs=1):
    return DelayedModelReplayBuffer(30,gym.spaces.Box(-np.inf,np.inf,shape=(46,),dtype=np.float32),
        gym.spaces.Box(-1.,1.,shape=(1,),dtype=np.float32),device='cpu',n_envs=n_envs)

def test_prediction_stops_at_next_review_without_truth_or_future_actor():
    e=T1BReviewEnv(openings=[292000]);e.reset();p=IntervalMissionPredictor(e);p.sync(e)
    position=e.env.chaser_state.position.copy()
    with patch('controllers.mpc.prediction.propagate_rk45',side_effect=AssertionError('truth forbidden')), \
         patch('env.t1_env.propagate_rk45',side_effect=AssertionError('truth forbidden')):
        r=p.rollout(np.zeros(1))
    assert r['steps']==3 and r['observation'][N_DUE_INDEX]==1
    assert r['discount']==.999**3 and r['solves']<60
    assert e.env.time_s==0 and np.array_equal(position,e.env.chaser_state.position)

def test_confidence_is_delayed_until_real_interval_closes():
    b=buffer();p=payload();start=np.zeros((1,46),np.float32);start[:,N_DUE_INDEX]=1
    cont=np.zeros((1,46),np.float32);end=start.copy();a=np.zeros((1,1),np.float32)
    b.add(start,cont,a,np.array([-.1]),np.array([False]),[{'counterfactual_pending':p}])
    assert b.cf_weights[0,0,0]==0
    b.add(cont,cont,a,np.array([-.2]),np.array([False]),[{}])
    b.add(cont,end,a,np.array([-.3]),np.array([False]),[{}])
    assert b.cf_weights[0,0,0]==0 and not b.interval_ready[0,0]
    b.add(end,cont,a,np.array([-.1]),np.array([False]),[{}])
    assert b.interval_ready[0,0] and b.cf_weights[0,0,0]>.99
    assert b.feedback_stats['valid']==1
    assert np.isclose(b.cf_discount[0,0,0],.999**3)

def test_terminal_same_slot_release_survives_base_buffer_write():
    b=buffer();p=payload(1,'resource_exhausted');z=np.zeros((1,46),np.float32);z[:,N_DUE_INDEX]=1
    b.add(z,z,np.zeros((1,1)),np.array([-100.]),np.array([True]),
          [{'event':'resource_exhausted','counterfactual_pending':p}])
    assert b.interval_ready[0,0] and b.interval_done[0,0,0]==1
    assert b.cf_weights[0,0,0]>.99

def test_unvalidated_alternative_terminal_never_trains():
    b=buffer();p=payload(1,'resource_exhausted');p['alternative']['done']=True
    z=np.zeros((1,46),np.float32);z[:,N_DUE_INDEX]=1
    b.add(z,z,np.zeros((1,1)),np.array([-100.]),np.array([True]),
          [{'event':'resource_exhausted','counterfactual_pending':p}])
    assert b.cf_weights[0,0,0]==0
    assert b.feedback_stats['reasons']['alternative_terminal_not_validated']==1

def test_interval_feedback_never_changes_real_controls_or_state():
    from env.t1_b_interval_cf import T1BIntervalFeedbackEnv
    a=T1BReviewEnv(openings=[292040]);b=T1BIntervalFeedbackEnv(openings=[292040])
    oa,_=a.reset();ob,_=b.reset();assert np.array_equal(oa,ob)
    for _ in range(4):
        action=np.zeros(1,np.float32)
        ra=a.step(action);rb=b.step(action)
        assert np.array_equal(ra[0],rb[0])
        assert ra[1:4]==rb[1:4]
        assert ra[4]['resource_control_trace']==rb[4]['resource_control_trace']
        assert np.array_equal(a.env.chaser_state.position,b.env.chaser_state.position)
        assert np.array_equal(a.env.controller._nominal_controls,b.env.controller._nominal_controls)

def test_model_qp_fallback_is_not_valid_learning_supervision():
    from env.t1_b_interval_cf import calibrate_interval
    p=payload();p['alternative']['qp_zero_fallbacks']=1
    gate=calibrate_interval(p,p['factual']['return'],p['factual']['observation'],3,None)
    assert gate['weight']==0 and 'model_qp_zero_fallback' in gate['reasons']
