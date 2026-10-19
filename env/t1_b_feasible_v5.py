"""T1-B V5 interface (E-TCPF arm only): a full-horizon paired MPC counterfactual of the initial intent.

V4 sandbox (evidence/T1B_V5_PRECHECK): with coverage fixed, the critic still ranked the earliest
candidate first everywhere at 5k. The state-dependent value of the arrival time is mostly a
+2..+3 return difference (later arrival spends less total delta-v when the deadline slack allows;
it accrues in the descent, after arrival) plus rare catastrophes, buried under the +-100..200
outcome variance of single real returns; a 6 s MPC prefix cannot see it. The same frozen MPC run
on its nominal model to the end of the episode reproduces the truth ranking (15 dev states:
best candidate 11/15, pairwise sign 31/38, outcome event 37/45).

V5: at the first learning decision of each episode the factual intent and one other verified
candidate (fixed hash, V2 rule) are rolled out with the MPC on its model until the episode ends or
a new decision would be due (QP every 5 controls, <= 200 macro steps). The relative target is then
the paired model return difference (no critic bootstrap when both branches end); it is admitted
only if the factual model branch reproduces the real outcome (same terminal event, |return error|
<= 5). Learning decisions, task, D/B, G*, MPC, truth: V4's. E-SAC is V4 unchanged.
"""
from dataclasses import replace

import numpy as np

from env.t1_b_feasible_v1 import choice_index
from env.t1_b_feasible_v2 import FeasibleArrivalEnvV2
from env.t1_b_feasible_v3 import FeasiblePredictorV3
from env.t1_b_feasible_v4 import FeasibleArrivalEnvV4
from env.t1_n_review import N_DUE_INDEX

VERSION = 't1b-committed-intent-full-horizon-mpc-cf-v5'
FULL_HORIZON_MACRO_STEPS = 200


class FeasiblePredictorV5(FeasiblePredictorV3):
    """Rolls the shadow through physical reviews that are not learning decisions (a live plan is kept,
    as in the real V3/V4 environment) until a terminal, a new learning decision or the horizon."""

    def rollout(self, action, h=None):
        h = self.interval_config.maximum_macro_steps if h is None else h
        parts, a = [], action
        while True:
            r = super().rollout(a, h - sum(p['steps'] for p in parts))
            parts.append(r)
            total = sum(p['steps'] for p in parts)
            if r['done'] or r['steps'] == 0 or total >= h or self.shadow._observation()[N_DUE_INDEX] > .5:
                break
            a = np.zeros(1, np.float32)   # physical review with a live plan: the only admissible request is keep
        gamma = self.shadow.env.config.reward.gamma
        rewards = [x for p in parts for x in p['rewards']]
        out = dict(parts[-1])
        out.update(rewards=rewards, return_=None, steps=len(rewards), discount=gamma ** len(rewards),
                   observations=[o for p in parts for o in p['observations']], tau=parts[0]['tau'], outcome=parts[0]['outcome'],
                   solves=sum(p['solves'] for p in parts), qp_zero_fallbacks=sum(p['qp_zero_fallbacks'] for p in parts),
                   wall_s=sum(p['wall_s'] for p in parts))
        out['return'] = sum(gamma ** k * x for k, x in enumerate(rewards)); out.pop('return_')
        return out


class FeasibleArrivalEnvV5(FeasibleArrivalEnvV4):
    ALTERNATIVE_RULE = 'one other verified candidate by sha256(opening, review index), first decision of the episode only'
    alternative_index = FeasibleArrivalEnvV2.alternative_index

    def __init__(self, *args, feedback=False, **kwargs):
        super().__init__(*args, feedback=feedback, **kwargs)
        if feedback:
            self.predictor = FeasiblePredictorV5(self)
            self.predictor.interval_config = replace(self.predictor.interval_config, maximum_macro_steps=FULL_HORIZON_MACRO_STEPS)
        self._episode_cf_done = False

    def reset(self, **kwargs):
        self._episode_cf_done = False
        return super().reset(**kwargs)

    def step(self, action):
        before = self._observation(); active = bool(before[N_DUE_INDEX] > .5)
        choices = tuple(float(x) for x in self.support())
        raw = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1)); idx = choice_index(raw, len(choices))
        payloads = None
        if self.feedback_enabled and active and not self._episode_cf_done:
            alt = self.alternative_index(choices, idx)
            payloads = self._paired(raw, choices, idx, alt)
            self._episode_cf_done = True
        enabled = self.feedback_enabled; self.feedback_enabled = False
        try:
            obs, r, t, x, info = FeasibleArrivalEnvV2.step(self, action)
        finally:
            self.feedback_enabled = enabled
        if payloads:
            info['counterfactual_pending'] = payloads
            info['counterfactual'] = {'w': 0, 'kind': 'pending_real_outcome_validation', 'alternatives': len(payloads)}
        return obs, r, t, x, info

    def _paired(self, raw, choices, idx, alt):
        from dataclasses import asdict
        from env.t1_b_feasible_v1 import bin_center, executed_coordinate
        pred = self.predictor; pred.sync(self); fact = pred.rollout(np.array([raw], np.float32))
        alt_raw = bin_center(alt, len(choices)); pred.sync(self); counter = pred.rollout(np.array([alt_raw], np.float32))
        return [{'a_tilde': np.array([executed_coordinate(choices[alt])], np.float32), 'factual': fact, 'alternative': counter,
                 'physical_difference_m': float(abs(counter['return'] - fact['return'])), 'first_reward_difference': counter['rewards'][0] - fact['rewards'][0],
                 'config': asdict(pred.interval_config), 'kind': 'full_horizon_paired_mpc_model',
                 'sampling': {'population_tau_s': choices, 'selected_index': idx, 'alternative_index': alt, 'rule': self.ALTERNATIVE_RULE},
                 'raw_alternative': alt_raw}]
