from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from env.hybrid_env import PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config


@pytest.mark.parametrize('valid_steps',[0,10])
def test_feedback_masks_stale_slack_and_uses_actual_commands(valid_steps):
    cfg=replace(precapture_planning_environment_config(),cache_target_trajectory=False)
    env=PrecaptureHybridEnv(environment_config=cfg)
    env.reset(seed=262000)
    calls=0
    def command(*args,**kwargs):
        nonlocal calls
        valid=calls < valid_steps
        calls+=1
        wrench=np.zeros(6)
        if valid:
            wrench[0]=.5*cfg.max_torque_per_axis_nm
            wrench[3]=.5*cfg.max_force_per_axis_n
        return wrench,SimpleNamespace(used_zero_fallback=not valid,maximum_slack=.2 if valid else 99.0)
    env.controller.command=command
    observation,_,_,_,info=env.step(np.zeros(3))
    assert observation.shape==(24,)
    assert info['hybrid_control_steps']==20
    assert info['hybrid_feedback_valid_solve_steps']==valid_steps
    assert info['hybrid_feedback_fallback_fraction']==1-valid_steps/20
    assert info['hybrid_feedback_slack_max']==(.2 if valid_steps else None)
    assert np.isclose(info['hybrid_feedback_force_mean'],.5*valid_steps/20)
    assert np.isclose(info['hybrid_feedback_torque_mean'],.5*valid_steps/20)
    assert info['hybrid_feedback_force_peak']==(.5 if valid_steps else 0)
    env.close()
