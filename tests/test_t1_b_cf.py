import numpy as np
from unittest.mock import patch
from env.t1_b_task import T1BReviewEnv
from env.t1_b_cf import MissionPredictor,T1BFeedbackEnv,ModelFeedbackConfig

def make():
    return T1BReviewEnv(openings=[292000],mission_override={"deadline_s":300.,"budget_m_s":100.,"urgent":False})

def test_shadow_uses_nominal_model_without_rk45_and_does_not_pollute():
    e=make();obs,_=e.reset();p=MissionPredictor(e);p.sync(e)
    original=e.env.chaser_state.copy();clock=e.env.time_s;budget=e.env._delta_v
    with patch("controllers.mpc.prediction.propagate_rk45",side_effect=AssertionError("truth RK45 forbidden")), \
         patch("env.t1_env.propagate_rk45",side_effect=AssertionError("truth RK45 forbidden")):
        r=p.rollout(np.zeros(1),2)
    assert len(r["rewards"])==2 and r["solves"]==40
    assert np.array_equal(original.position,e.env.chaser_state.position)
    assert e.env.time_s==clock and e.env._delta_v==budget
    assert r["observation"].shape==obs.shape
    assert np.isclose(r["discount"],.999**2)

def test_factual_model_against_real_two_second_continuation():
    e=make();e.reset();p=MissionPredictor(e);p.sync(e)
    r=p.rollout(np.zeros(1),1)
    obs,reward,t,x,info=e.step(np.zeros(1))
    q=e.env.chaser_state.position-e.env.target_state.position
    assert abs(r["return"]-reward)<.25
    assert np.linalg.norm(r["q"]-q)<.05
    assert np.linalg.norm(r["observation"]-obs)<.25

def test_model_resource_end_has_no_bootstrap():
    e=T1BReviewEnv(openings=[292000],mission_override={"deadline_s":300.,"budget_m_s":.00001,"urgent":False})
    e.reset();p=MissionPredictor(e);p.sync(e);r=p.rollout(np.zeros(1),2)
    assert r["done"] and r["event"]=="resource_exhausted"
    assert p.shadow.env._delta_v<=.00001+1e-12

def test_feedback_records_independent_model_transition():
    e=T1BFeedbackEnv(openings=[292000],mission_override={"deadline_s":300.,"budget_m_s":100.,"urgent":False})
    e.reset();obs,r,t,x,i=e.step(np.zeros(1));cf=i["counterfactual"]
    assert cf["z_hat"].shape==obs.shape
    assert "signal" not in cf  # no old long-term ranking proxy
    assert cf["physical_difference_m"]>0
    assert cf["cf_discount"]==.999**2

def test_feedback_preserves_the_real_controller_and_trajectory_bitwise():
    options={"deadline_s":300.,"budget_m_s":100.,"urgent":False}
    plain=T1BReviewEnv(openings=[292000],mission_override=options)
    feedback=T1BFeedbackEnv(openings=[292000],mission_override=options)
    plain.reset();feedback.reset()
    for a in [np.zeros(1),np.zeros(1),np.zeros(1),np.array([.8])]:
        o1,r1,t1,x1,i1=plain.step(a)
        o2,r2,t2,x2,i2=feedback.step(a)
        assert np.array_equal(o1,o2) and r1==r2 and (t1,x1)==(t2,x2)
        assert np.array_equal(plain.env.controller._nominal_controls,feedback.env.controller._nominal_controls)
