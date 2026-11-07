import numpy as np
from env.t1_b_task import T1BReviewEnv, MissionDistribution, OBSERVATION_NAMES
from env.t1_n_review import T1NReviewEnv

def test_mission_is_deterministic_and_does_not_use_policy():
    d=MissionDistribution()
    assert d.sample(292000)==d.sample(292000)
    assert d.sample(292000)!=d.sample(292001)

def test_unlimited_mission_preserves_truth_trajectory():
    a=T1NReviewEnv(openings=[292000])
    b=T1BReviewEnv(openings=[292000],mission_override={"deadline_s":300.,"budget_m_s":100.,"urgent":False})
    oa,_=a.reset();ob,_=b.reset()
    assert np.array_equal(oa,ob[:41])
    for _ in range(2):
        oa,ra,ta,xa,ia=a.step(np.zeros(1))
        ob,rb,tb,xb,ib=b.step(np.zeros(1))
        assert np.array_equal(a.env.chaser_state.position,b.env.chaser_state.position)
        assert np.array_equal(a.env.chaser_state.rotation,b.env.chaser_state.rotation)
        assert abs(ra-rb)<1e-10
        assert ta==tb and xa==xb
        assert abs(a.env._delta_v-b.env._delta_v)<1e-12
    assert ob.shape==(len(OBSERVATION_NAMES),)

def test_resource_ends_at_control_boundary_without_overspend():
    b=T1BReviewEnv(openings=[292000],mission_override={"deadline_s":300.,"budget_m_s":0.00001,"urgent":False})
    b.reset()
    _,r,t,x,i=b.step(np.zeros(1))
    assert t and not x and i["event"]=="resource_exhausted"
    assert b.env._delta_v<=b.mission["budget_m_s"]+1e-12
    assert b.env.time_s==0.1
    assert r < -100

def test_deadline_is_terminal_not_time_limit_bootstrap():
    b=T1BReviewEnv(openings=[292000],mission_override={"deadline_s":0.2,"budget_m_s":100.,"urgent":True})
    b.reset()
    _,_,t,x,i=b.step(np.zeros(1))
    assert t and not x and i["event"]=="timeout"
    assert b.env.time_s==0.2
