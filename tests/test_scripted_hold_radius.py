"""A hold is inertially frozen, and adding its radius preserves the control."""
from types import SimpleNamespace

import numpy as np

from dynamics.lie import so3_exp
from env.hybrid_env import PrecaptureHybridEnv
from experiments.evaluate_hybrid_scripted import ScriptedWaypointPolicy


def test_hold_radius_preserves_inertial_direction_until_commit():
    env=PrecaptureHybridEnv()
    env.reset(seed=262005)
    policy=ScriptedWaypointPolicy(env,'hold_radius',40.0,30.0,12.0)
    initial_rotation=env.env.target_state.rotation.copy()
    action,committed=policy.act()
    held=initial_rotation @ env.waypoint_from_action(action)
    assert not committed
    assert np.isclose(np.linalg.norm(held),12.0)
    assert np.allclose(held/12.0, initial_rotation @ env._current_position()/np.linalg.norm(env._current_position()))
    rotated=so3_exp(np.array([.2,-.1,.3])) @ initial_rotation
    env.env.target_state=SimpleNamespace(rotation=rotated)
    env.env.time_seconds=28.0
    action,committed=policy.act()
    assert not committed
    assert np.allclose(rotated @ env.waypoint_from_action(action),held)
    env.env.time_seconds=30.0
    action,committed=policy.act()
    assert committed
    assert np.allclose(env.waypoint_from_action(action),policy.task.desired_position)
    env.close()


def test_initial_radius_and_zero_time_cells_equal_existing_control():
    env=PrecaptureHybridEnv()
    env.reset(seed=262006)
    control=ScriptedWaypointPolicy(env,'commit_at',40.0,30.0)
    initial=ScriptedWaypointPolicy(env,'hold_radius',40.0,30.0,None)
    for t in [0.0,2.0,28.0,30.0,32.0]:
        env.env.time_seconds=t
        a,ca=control.act(); b,cb=initial.act()
        assert np.array_equal(a,b) and ca==cb
    env.env.time_seconds=0.0
    a,_=ScriptedWaypointPolicy(env,'commit_at',40.0,0.0).act()
    for radius in [12,14,16,18]:
        b,committed=ScriptedWaypointPolicy(env,'hold_radius',40.0,0.0,radius).act()
        assert committed and np.array_equal(a,b)
    env.close()
