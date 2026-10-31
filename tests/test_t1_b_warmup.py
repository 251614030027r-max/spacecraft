"""Fair option exploration must preserve deterministic 0k and both-arm sampling."""
import numpy as np
from unittest.mock import patch
from env.t1_n_review import N_DUE_INDEX,N_PLAN_REMAINING_INDEX,N_DESCENT_INDEX
from train.t1_b_interval_sac import T1BIntervalSAC

def test_warmup_keeps_accepted_options_without_changing_new_requests_or_post_warmup():
    model=T1BIntervalSAC.__new__(T1BIntervalSAC)
    model.preserve_warmup_options=True;model.warmup_options_preserved=0;model.num_timesteps=0
    model._last_obs=np.zeros((4,46),np.float32);model._last_obs[:,N_DUE_INDEX]=1
    model._last_obs[0,N_PLAN_REMAINING_INDEX]=.5
    model._last_obs[1,N_DUE_INDEX]=0
    model._last_obs[2,N_DESCENT_INDEX]=1
    def original(*args,**kwargs):return np.full((4,1),.8,np.float32),np.full((4,1),.8,np.float32)
    with patch('stable_baselines3.SAC._sample_action',side_effect=original) as source:
        actions,stored=model._sample_action(2000,n_envs=4)
        assert source.call_count==1
        assert np.array_equal(actions,stored)
        assert np.all(actions[:3]==0) and actions[3,0]==np.float32(.8)
        model.num_timesteps=2000
        actions,stored=model._sample_action(2000,n_envs=4)
        assert np.all(actions==np.float32(.8))

def test_option_exploration_flag_never_changes_normal_initial_policy_weights():
    import torch
    from env.t1_b_task import T1BReviewEnv
    def create(flag):
        return T1BIntervalSAC('MlpPolicy',T1BReviewEnv(),seed=0,device='cpu',buffer_size=20,
            learning_starts=2000,ent_coef=.005,mc_lambda=0.,eq_lambda=.1,delta_lambda=0.,
            policy_kwargs={'net_arch':[128,128]},preserve_warmup_options=flag,verbose=0)
    a=create(False);b=create(True)
    assert all(torch.equal(x,y) for x,y in zip(a.policy.parameters(),b.policy.parameters()))
