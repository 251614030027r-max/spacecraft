"""T1-B V4 interface: V3 committed intents; the paired MPC counterfactual covers every unchosen candidate.

V3 midterm (evidence/T1B_V3_MIDTERM_20261019, evidence/T1B_V4_PRECHECK): with one alternative per
decision and 11 % non-earliest real decisions the critic's Q was linear in the arrival coordinate
and extrapolated "later is worse" from a handful of urgent timeouts; E-TCPF's single counterfactual
already flattened that slope eightfold but could not reverse it. V4 evaluates **all** other
verified candidates at each learning decision with the same factual MPC prefix (one factual and
<= 2 alternative shadow rollouts of <= 6 s), so the same critic receives a model-based relative
target for every unchosen intent at every decision state. Task, D/B, G*, MPC, truth unchanged.
"""
from dataclasses import asdict

import numpy as np

from env.t1_b_feasible_v1 import bin_center, choice_index, executed_coordinate
from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3
from env.t1_n_review import N_DUE_INDEX

VERSION = 't1b-committed-intent-all-alternatives-v4'


def all_alternative_payloads(real, raw, choices, idx):
    pred = real.predictor; pred.sync(real); fact = pred.rollout(np.array([raw], np.float32))
    payloads = []
    for alt in range(len(choices)):
        if alt == idx:
            continue
        alt_raw = bin_center(alt, len(choices)); pred.sync(real); counter = pred.rollout(np.array([alt_raw], np.float32))
        effect = float(np.linalg.norm(counter['q'] - fact['q'])) if counter['steps'] == fact['steps'] else 0.
        payloads.append({'a_tilde': np.array([executed_coordinate(choices[alt])], np.float32), 'factual': fact, 'alternative': counter,
                         'physical_difference_m': effect, 'first_reward_difference': counter['rewards'][0] - fact['rewards'][0],
                         'config': asdict(pred.interval_config), 'kind': 'verified_support_paired_mpc_prefix_all_alternatives',
                         'sampling': {'population_tau_s': choices, 'selected_index': idx, 'alternative_index': alt,
                                      'rule': real.ALTERNATIVE_RULE}, 'raw_alternative': alt_raw})
    return payloads


class FeasibleArrivalEnvV4(FeasibleArrivalEnvV3):
    ALTERNATIVE_RULE = 'every other verified candidate (no sampling, no outcome/rank oracle)'
    MAX_ALTERNATIVES = 2

    def step(self, action):
        before = self._observation(); active = bool(before[N_DUE_INDEX] > .5)
        choices = tuple(float(x) for x in self.support())
        raw = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1)); idx = choice_index(raw, len(choices))
        payloads = all_alternative_payloads(self, raw, choices, idx) if self.feedback_enabled and active else None
        enabled = self.feedback_enabled; self.feedback_enabled = False
        try:
            obs, r, t, x, info = super().step(action)
        finally:
            self.feedback_enabled = enabled
        if payloads:
            info['counterfactual_pending'] = payloads
            info['counterfactual'] = {'w': 0, 'kind': 'pending_real_prefix_validation', 'alternatives': len(payloads)}
        return obs, r, t, x, info
