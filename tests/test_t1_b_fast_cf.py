import numpy as np
from unittest.mock import patch
from env.t1_b_task import T1BReviewEnv
from env.t1_b_fast_cf import FastMissionPredictor,T1BFastFeedbackEnv

def test_fast_window_one_qp_no_truth_and_cache_reuse():
    e=T1BReviewEnv(openings=[292000]);e.reset();p=FastMissionPredictor(e);p.sync(e)
    with patch('controllers.mpc.prediction.propagate_rk45',side_effect=AssertionError('RK45 truth forbidden')):
        r=p.rollout(np.zeros(1),1)
    assert r['solves']==1 and len(r['rewards'])==1 and np.isfinite(r['return'])
    p.sync(e);cached=p.rollout(np.zeros(1),1)
    assert cached['cache_hit'] and cached['solves']==0
    assert np.array_equal(cached['observation'],r['observation'])
    assert e.env.time_s==0

def test_fast_feedback_real_path_is_bitwise_unchanged():
    p=T1BReviewEnv(openings=[292000]);f=T1BFastFeedbackEnv(openings=[292000])
    p.reset();f.reset()
    for a in [np.zeros(1),np.zeros(1),np.zeros(1),np.array([.8])]:
        o1,r1,t1,x1,_=p.step(a);o2,r2,t2,x2,i=f.step(a)
        assert np.array_equal(o1,o2) and r1==r2 and (t1,x1)==(t2,x2)

