"""T1-B V2 learner: V1 (exact Gaussian-induced categorical SAC, real SMDP TD, shared 20 % expert
Replay, paired MPC relative Bellman in the same critic) with two learning-layer repairs.

Audit of V1 (evidence/T1B_V1_AUDIT):

1. Temperature. V1 inherited alpha = 0.005 x review gap (0.015) from the 2 s continuous-control
   line. For a once-per-review choice among <= 3 verified arrival times the soft-optimal policy is
   p_i ~ exp(Q_i / alpha); with alpha = 0.015 a Q gap of 0.5 already gives p_other ~ e^-33, so the
   policy was argmax(Q) from 2k on (first-review Gaussian mean 5-9 sigma beyond the bin edge, actor
   gradient 1e-5..1e-6 of its 0k value). V2: one fixed temperature per task decision,
   ``candidate_alpha`` (0.5), in the actor loss and the soft value alike (no gap scaling, no
   automatic tuning, DEAD_ENDS D02).
2. Coverage. After 1k online steps V1 executed the earliest candidate at 100 % of learning decisions
   (non-earliest: 74 online + 11 teacher choices in 10k), so Q of the other candidates was MLP
   extrapolation and the ranking could never be corrected. V2: during training only, the behaviour
   action at a learning decision is, with probability ``behaviour_epsilon`` (0.10), a uniformly
   drawn verified candidate (bin centre). Off-policy: the critic still evaluates the actor's
   categorical policy; deterministic evaluation is unchanged (tanh(mean) -> bin).

Everything else is V1: same network, init, optimiser, trainer, expert Replay, CF gate and the
5 % / 10 % relative-gradient share.
"""
import numpy as np
import torch as th

from env.t1_b_feasible_v1 import bin_center
from env.t1_n_review import N_DUE_INDEX
from train.t1_b_feasible_sac import FeasibleSupportSAC

CANDIDATE_ALPHA = 0.5
BEHAVIOUR_EPSILON = 0.10


class FeasibleSupportSACV2(FeasibleSupportSAC):
    def __init__(self, *args, candidate_alpha=CANDIDATE_ALPHA, behaviour_epsilon=BEHAVIOUR_EPSILON, **kwargs):
        self.candidate_alpha = float(candidate_alpha)
        self.behaviour_epsilon = float(behaviour_epsilon)
        self.behaviour_counts = {'policy': 0, 'epsilon': 0}
        super().__init__(*args, **kwargs)

    def alpha_of(self, observations: th.Tensor) -> th.Tensor:
        return th.full((observations.shape[0], 1), self.candidate_alpha, dtype=observations.dtype,
                       device=observations.device)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, stored = super()._sample_action(learning_starts, action_noise, n_envs)
        if self.num_timesteps >= learning_starts:
            for k, z in enumerate(self._last_obs):
                n = int((z[49:52] > .5).sum())
                if z[N_DUE_INDEX] < .5 or n < 2:
                    continue
                if np.random.random() < self.behaviour_epsilon:
                    raw = bin_center(int(np.random.randint(n)), n)
                    action[k] = raw
                    stored[k] = raw
                    self.behaviour_counts['epsilon'] += 1
                else:
                    self.behaviour_counts['policy'] += 1
        return action, stored

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'behaviour_counts']
