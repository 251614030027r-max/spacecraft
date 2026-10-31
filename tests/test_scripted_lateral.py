from dataclasses import replace
import numpy as np
from env.hybrid_env import PrecaptureHybridConfig,PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config
from experiments.evaluate_hybrid_scripted import ScriptedWaypointPolicy

def test_lateral_control_is_exact_and_perturbations_preserve_radial_action():
    env=PrecaptureHybridEnv(environment_config=replace(precapture_planning_environment_config(),cache_target_trajectory=False),hybrid_config=PrecaptureHybridConfig(waypoint_parametrization='radial_local'))
    env.reset(seed=262006)
    base,_=ScriptedWaypointPolicy(env,'desired_pose',40).act()
    zero,_=ScriptedWaypointPolicy(env,'lateral_adjust',40,lateral_angle_deg=0).act()
    assert np.array_equal(base,zero)
    for axis in [0,1]:
        for angle in [-30,-20,-10,10,20,30]:
            action,_=ScriptedWaypointPolicy(env,'lateral_adjust',40,lateral_angle_deg=angle,lateral_axis=axis).act()
            assert np.isclose(action[0],base[0],rtol=0,atol=1e-12)
            assert np.all(np.isfinite(action)) and np.all(np.abs(action)<=1)
    env.close()
