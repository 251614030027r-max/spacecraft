"""T1-B V7 interface (E-TCPF arm): the full-horizon MPC counterfactual runs the *real* controller.

V6 (evidence/T1B_V6_ROUND/final_review_20261020_2230, evidence/T1B_V7_PRECHECK): the gated
full-horizon pairs were wrong exactly at the decisive boundaries (292140: 140 s really captures,
+56, the model branch exhausted the budget, -129; 292145: +3.3 really, -183 in the model). Cause:
the counterfactual MPC solved its QP only every 5th control (V1's ``qp_stride_controls = 5``, a
cost shortcut) and played stale nominal controls in between, overestimating delta-v by
0.1-0.25 m/s - a different controller from the one that flies. With the QP every control (the real
MPC) the same four boundary openings are reproduced (292140 -129.8 / -129.6 / +56.0 vs truth
-129.8 / -129.6 / +56.0; 292141, 292145 correct; 292146 one error); stride 2 still fails 292145.

A stride-1 branch costs about one real episode, so the branches run asynchronously: at the first
learning decision of an episode the environment emits a snapshot of the real state (plain
numpy/dataclass fields, the same fields ``CounterfactualPredictor.sync`` copies; bitwise-equal
rollouts, evidence/T1B_V7_PRECHECK) and the learner (``train.t1_b_v7_sac``) rolls the factual
intent and every other verified candidate out in a worker pool of the training process while the
real episode continues; the payloads are attached to the episode's terminal transition (waiting
there if a branch is still running), so training is deterministic. Gate, learner: V6.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

from env.t1_b_feasible_v1 import bin_center, choice_index, executed_coordinate
from env.t1_b_feasible_v2 import FeasibleArrivalEnvV2
from env.t1_b_feasible_v5 import FULL_HORIZON_MACRO_STEPS, FeasibleArrivalEnvV5, FeasiblePredictorV5
from env.t1_counterfactual import _CONTROLLER_FIELDS, _ENV_FIELDS, _GENERATOR_FIELDS
from env.t1_n_review import N_DUE_INDEX

VERSION = 't1b-full-horizon-real-controller-cf-v7'
CF_QP_STRIDE = 1           # the real MPC solves its QP every 0.1 s control
CF_WORKERS = 6             # worker processes of the training process (4 environments x <= 3 branches queue on them)


def snapshot(real):
    e, c, g = real.env, real.env.controller, real.generator
    return {'opening': real.current_opening, 'episode': real.episode,
            'target_state': e.target_state.copy(), 'chaser_state': e.chaser_state.copy(),
            'env_fields': {n: deepcopy(getattr(e, n)) for n in _ENV_FIELDS},
            'ctrl_fields': {n: deepcopy(getattr(c, n)) for n in _CONTROLLER_FIELDS},
            'ctrl_x': None if c._x.value is None else np.array(c._x.value),
            'ctrl_u': None if c._u.value is None else np.array(c._u.value),
            'ctrl_slack': None if c._slack is None or c._slack.value is None else np.array(c._slack.value),
            'gen_fields': {n: deepcopy(getattr(g, n)) for n in _GENERATOR_FIELDS},
            'plan': deepcopy(g.plan), 'hold_position': None if g.hold_position is None else np.array(g.hold_position),
            'descent': deepcopy(g._descent), 'gen_last_outcome': g.last_outcome,
            'last_request': real._last_request, 'last_kappa': real._last_kappa,
            'review_state': deepcopy(real.review_state()), 'mission': deepcopy(real.mission),
            'support_cache': deepcopy(real._support_cache), 'env_config': e.config}


def proxy(snap):
    ctrl = SimpleNamespace(_problem=SimpleNamespace(_solver_cache=None), _x=SimpleNamespace(value=snap['ctrl_x']),
                           _u=SimpleNamespace(value=snap['ctrl_u']),
                           _slack=SimpleNamespace(value=snap['ctrl_slack']), **snap['ctrl_fields'])
    env = SimpleNamespace(target_state=snap['target_state'], chaser_state=snap['chaser_state'], controller=ctrl,
                          config=snap['env_config'], **snap['env_fields'])
    gen = SimpleNamespace(plan=snap['plan'], hold_position=snap['hold_position'], _descent=snap['descent'],
                          last_outcome=snap['gen_last_outcome'], **snap['gen_fields'])
    rs = snap['review_state']
    return SimpleNamespace(env=env, generator=gen, _last_request=snap['last_request'], _last_kappa=snap['last_kappa'],
                           current_opening=snap['opening'], episode=snap['episode'], review_state=lambda: rs,
                           mission=snap['mission'], _support_cache=snap['support_cache'])


def make_predictor(spec):
    real_like = SimpleNamespace(config=spec['config'], env=SimpleNamespace(config=spec['env_config']),
                                mission_distribution=spec['mission_distribution'], mission_override=spec['mission_override'],
                                earliest_summary=spec['earliest_summary'])
    pred = FeasiblePredictorV5(real_like)
    pred.interval_config = replace(pred.interval_config, maximum_macro_steps=FULL_HORIZON_MACRO_STEPS, qp_stride_controls=CF_QP_STRIDE)
    return pred


_WORKER = {}


def _init_worker(spec):
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[k] = '1'
    _WORKER['pred'] = make_predictor(spec)


def _run_branch(args):
    snap, raw = args
    pred = _WORKER['pred']; pred.sync(proxy(snap)); r = pred.rollout(np.array([raw], np.float32))
    r = dict(r); r['observations'] = []   # the per-step observations are not needed downstream
    return r


class FeasibleArrivalEnvV7(FeasibleArrivalEnvV5):
    ALTERNATIVE_RULE = 'every other verified candidate, first learning decision of the episode only; real-controller (QP every control) model branches, run asynchronously'

    def __init__(self, *args, feedback=False, **kwargs):
        FeasibleArrivalEnvV2.__init__(self, *args, feedback=False, **kwargs)
        self.feedback_enabled = feedback
        self.predictor = None
        self._episode_cf_done = False
        self._spec = {'config': self.config, 'env_config': self.env.config, 'mission_distribution': self.mission_distribution,
                      'mission_override': self.mission_override, 'earliest_summary': self.earliest_summary}

    def step(self, action):
        before = self._observation(); active = bool(before[N_DUE_INDEX] > .5)
        choices = tuple(float(x) for x in self.support())
        raw = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1)); idx = choice_index(raw, len(choices))
        job = None
        if self.feedback_enabled and active and not self._episode_cf_done and len(choices) >= 2:
            alts = [j for j in range(len(choices)) if j != idx]
            job = {'snap': snapshot(self), 'raws': [raw] + [bin_center(j, len(choices)) for j in alts], 'choices': choices,
                   'idx': idx, 'alts': alts, 'spec': self._spec, 'rule': self.ALTERNATIVE_RULE}
            self._episode_cf_done = True
        enabled = self.feedback_enabled; self.feedback_enabled = False
        try:
            obs, r, t, x, info = FeasibleArrivalEnvV2.step(self, action)
        finally:
            self.feedback_enabled = enabled
        if job is not None:
            info['counterfactual_snapshot'] = job
            info['counterfactual'] = {'w': 0, 'kind': 'async_full_horizon_pending', 'alternatives': len(job['alts'])}
        return obs, r, t, x, info


def payloads_from(job, results):
    fact = results[0]; out = []
    for alt, counter in zip(job['alts'], results[1:]):
        out.append({'a_tilde': np.array([executed_coordinate(job['choices'][alt])], np.float32), 'factual': fact, 'alternative': counter,
                    'physical_difference_m': float(abs(counter['return'] - fact['return'])),
                    'first_reward_difference': counter['rewards'][0] - fact['rewards'][0],
                    'config': {'qp_stride_controls': CF_QP_STRIDE, 'maximum_macro_steps': FULL_HORIZON_MACRO_STEPS},
                    'kind': 'full_horizon_paired_real_controller_model',
                    'sampling': {'population_tau_s': job['choices'], 'selected_index': job['idx'], 'alternative_index': alt, 'rule': job['rule']},
                    'raw_alternative': bin_center(alt, len(job['choices']))})
    return out
