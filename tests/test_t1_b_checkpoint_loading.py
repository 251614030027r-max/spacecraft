import pytest
import numpy as np
import torch
from env.t1_b_task import T1BReviewEnv
from train.t1_b_sac import T1BSAC
from train.t1_b_interval_sac import T1BIntervalSAC
from experiments.t1b_evaluate import load_policy_checkpoint

@pytest.mark.parametrize('interval',[False,True])
def test_evaluation_reloads_correct_learner_without_changing_policy(interval,tmp_path):
    cls=T1BIntervalSAC if interval else T1BSAC
    model=cls('MlpPolicy',T1BReviewEnv(),seed=0,device='cpu',buffer_size=20,
        ent_coef=.005,mc_lambda=0.,eq_lambda=.1,delta_lambda=0.,policy_kwargs={'net_arch':[128,128]},verbose=0)
    model.interval_feedback_method=interval;path=tmp_path/'checkpoint';model.save(path)
    loaded=load_policy_checkpoint(path)
    assert type(loaded) is cls
    assert all(torch.equal(a,b) for a,b in zip(model.policy.parameters(),loaded.policy.parameters()))
    obs=np.zeros(46,np.float32)
    assert np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
