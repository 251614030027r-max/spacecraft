import numpy as np
import torch as th
import gymnasium as gym
from train.t1_b_sac import MissionReplayBuffer,T1BSAC

def test_model_terminal_and_variable_discount_enter_the_same_critic_target():
    b=MissionReplayBuffer(10,gym.spaces.Box(-np.inf,np.inf,shape=(46,),dtype=np.float32),
                          gym.spaces.Box(-1,1,shape=(1,),dtype=np.float32),device='cpu')
    z=np.zeros((1,46),np.float32)
    for done in (True,False):
        b.add(z,z,np.zeros((1,1)),np.array([-.1]),np.array([False]),
              [{'counterfactual':{'w':1,'rank_weight':.8,'a_tilde':np.array([.5]),'r_hat':-100.,
                 'z_hat':z[0],'cf_discount':.999**2,'cf_done':done}}])
    b._last=(np.array([0,1]),np.array([0,0]))
    cf=b.counterfactual()
    class M:
        replay_buffer=b
        def target_value(self,z):return th.full((len(z),1),100.)
    y=T1BSAC._delta_target(M(),None,cf,None)
    assert th.allclose(y,th.tensor([[-100.],[-100.+.999**2*100]]),atol=1e-5)

def test_model_loss_is_absolute_bellman_regression_not_old_ranking():
    class M:delta_huber=10.
    qa=[th.tensor([[1.]],requires_grad=True)]
    qr=[th.tensor([[100.]],requires_grad=True)]
    loss=T1BSAC._relative_loss(M(),qa,qr,th.tensor([[2.]]),th.ones(1,1))
    loss.backward()
    assert qa[0].grad.item()<0
    assert qr[0].grad is None
