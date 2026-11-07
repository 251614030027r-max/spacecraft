"""Same interval SAC critic, plus short model Bellman transitions (no rank labels)."""
import numpy as np
import torch as th
import torch.nn.functional as F
from train.t1_n_sac import T1NSAC,T1NReplayBuffer

class MissionReplayBuffer(T1NReplayBuffer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        shape=(self.buffer_size,self.n_envs,1)
        self.cf_discount=np.zeros(shape,np.float32)
        self.cf_done=np.zeros(shape,np.float32)
    def add(self,obs,next_obs,action,reward,done,infos):
        i=self.pos
        super().add(obs,next_obs,action,reward,done,infos)
        for e in range(self.n_envs):
            cf=infos[e].get("counterfactual",{})
            self.cf_discount[i,e]=cf.get("cf_discount",0.)
            self.cf_done[i,e]=cf.get("cf_done",True)
    def model_end(self):
        i,e=self._last
        return self.to_torch(self.cf_discount[i,e]),self.to_torch(self.cf_done[i,e])

class T1BSAC(T1NSAC):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault("replay_buffer_class",MissionReplayBuffer)
        super().__init__(*args,**kwargs)
    def _delta_target(self,data,cf,v_next):
        discount,done=self.replay_buffer.model_end()
        return cf.rewards+discount*(1-done)*self.target_value(cf.next_observations)
    def _relative_loss(self,alt_q,current_q,delta,weights):
        return .5*sum((weights*F.huber_loss(qa,delta,reduction="none",delta=self.delta_huber)).mean()
                      for qa in alt_q)
    def _audit(self,*args):
        # The inherited ranking-agreement statistic would have the wrong semantics.
        pass
