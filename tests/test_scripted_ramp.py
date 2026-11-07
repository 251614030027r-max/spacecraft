from dataclasses import replace
import numpy as np
from env.hybrid_env import PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config
from experiments.evaluate_hybrid_scripted import ScriptedWaypointPolicy


def test_ramp_waits_then_advances_a_persistent_radius_and_stops_at_goal():
    env=PrecaptureHybridEnv(environment_config=replace(precapture_planning_environment_config(),cache_target_trajectory=False))
    env.reset(seed=262006)
    policy=ScriptedWaypointPolicy(env,'ramp_in',40,60,None,.4)
    _,started=policy.act()
    assert not started and policy.ramp_radius is None
    radius=np.linalg.norm(env._current_position())
    env.env.time_seconds=60
    _,started=policy.act()
    assert started and np.isclose(policy.ramp_radius,radius-.4)
    env.env.time_seconds=62
    policy.act()
    assert np.isclose(policy.ramp_radius,radius-.8)
    for _ in range(100): action,_=policy.act()
    assert np.allclose(env.waypoint_from_action(action),policy.task.desired_position)
    env.close()
