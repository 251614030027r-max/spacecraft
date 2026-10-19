"""T1-B V5 learner (E-TCPF arm): V4 learner, with the full-horizon paired MPC counterfactual of
``env.t1_b_feasible_v5`` admitted by the real outcome.

Gate: either the factual model branch ends in a terminal event and the real episode ends with the
same event within ``END_SLACK_STEPS`` of the predicted end, or it ends at a new learning decision
(plan lost) and the real interval ends the same way at the same step; in both cases the discounted
returns must agree within ``MAX_RETURN_ERROR``; weight = 1 - error / MAX_RETURN_ERROR. The relative target is the
V1 factual-anchored one; when both model branches end at a terminal it is the pure paired model
return difference (no critic bootstrap). Same 5 % calibrated / 10 % capped gradient share.
"""
from env.t1_n_review import N_DUE_INDEX
from train.t1_b_v4_sac import AllAlternativesReplay, FeasibleSupportSACV4

END_SLACK_STEPS = 10
MAX_RETURN_ERROR = 5.0


class FullHorizonReplay(AllAlternativesReplay):
    def close_after(self, pay):
        f = pay['factual']
        return f['steps'] + END_SLACK_STEPS if f['done'] else f['steps']

    def calibrate(self, pay, ret, end_obs, d, event, truncated):
        f, a = pay['factual'], pay['alternative']
        error = abs(float(f['return']) - float(ret)); reasons = []
        if truncated: reasons.append('actual_numerical_truncation')
        if f['done']:
            if f['event'] != event: reasons.append('factual_event_mismatch')
            if abs(f['steps'] - d) > END_SLACK_STEPS: reasons.append('factual_end_time_mismatch')
        else:
            # the model branch ended at a new learning decision (plan lost): the real interval must
            # end the same way at the same step; the target then bootstraps at both decision states
            if event is not None or end_obs is None or float(end_obs[N_DUE_INDEX]) < .5 or f['steps'] != d:
                reasons.append('factual_plan_loss_not_reproduced')
        if error > MAX_RETURN_ERROR: reasons.append('return')
        if f.get('outcome') != 'accepted' or a.get('outcome') != 'accepted': reasons.append('intent_not_accepted')
        if f.get('qp_zero_fallbacks', 0) or a.get('qp_zero_fallbacks', 0): reasons.append('model_qp_zero_fallback')
        weight = 0. if reasons else max(0., 1. - error / MAX_RETURN_ERROR)
        return {'weight': weight, 'reasons': reasons, 'errors': {'return': error, 'position_m': None}}


class FeasibleSupportSACV5(FeasibleSupportSACV4):
    def _setup_model(self):
        self.replay_buffer_class = FullHorizonReplay
        super(FeasibleSupportSACV4, self)._setup_model()
