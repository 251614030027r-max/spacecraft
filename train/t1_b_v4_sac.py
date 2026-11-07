"""T1-B V4 learner: the V2/V3 learner with (i) the paired MPC relative target for **every** unchosen
verified candidate at a learning decision (two counterfactual slots per Replay entry, each with its
own factual-prefix gate) and (ii) behaviour exploration 0.5 at learning decisions (both arms).

Why (evidence/T1B_V4_PRECHECK): under V3 committed intents a decision occurs ~1.5 times per
episode; with epsilon 0.10 only 11 % of 10k-step real decisions were non-earliest (economy x latest:
7), so Q was linear in the arrival coordinate and extrapolated urgent timeouts to every state.
Off-policy exploration at these rare decisions is cheap (deterministic evaluation is unchanged);
the full counterfactual gives the critic a model-based target for each unchosen intent.

The relative loss of the slots is summed and enters the critic with V1's calibrated 5 % share,
capped at 10 % of the real TD gradient (unchanged). E-SAC has no counterfactual (unchanged).
"""
import json

import numpy as np
import torch as th
from stable_baselines3.common.utils import polyak_update

from env.t1_b_interval_cf import calibrate_interval
from train.t1_b_feasible_sac import ExpertPrefixReplay
from train.t1_b_interval_sac import DelayedModelReplayBuffer, json_default
from train.t1_b_v2_sac import FeasibleSupportSACV2

V4_BEHAVIOUR_EPSILON = 0.5
CF_ARRAYS = ('cf_actions', 'cf_rewards', 'cf_next_observations', 'cf_weights', 'cf_discount', 'cf_done',
             'cf_fact_rewards', 'cf_fact_discount', 'cf_fact_done', 'cf_fact_obs')


class AllAlternativesReplay(ExpertPrefixReplay):
    SLOTS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sets = [{k: getattr(self, k) for k in CF_ARRAYS}]
        self._sets += [{k: np.zeros_like(getattr(self, k)) for k in CF_ARRAYS} for _ in range(self.SLOTS - 1)]
        self.slot_valid = [0] * self.SLOTS

    def use(self, k):
        for name, array in self._sets[k].items():
            setattr(self, name, array)

    def add(self, obs, next_obs, action, reward, done, infos):
        self.use(0)
        slot, index = self.pos, self.n_adds
        payloads = [i.get('counterfactual_pending') for i in infos]
        clean = [{k: v for k, v in i.items() if k not in ('counterfactual_pending', 'counterfactual')} for i in infos]
        for k in range(1, self.SLOTS):
            self._sets[k]['cf_weights'][slot] = 0.
        DelayedModelReplayBuffer.add(self, obs, next_obs, action, reward, done, clean)
        for e, payload in enumerate(payloads):
            if payload is not None:
                if self._prefix[e] is not None:
                    self._release(e, self._prefix[e], obs[e], None, False, early=True)
                self._prefix[e] = [slot, index, 0., 0, list(payload)]
            p = self._prefix[e]
            if p is None:
                continue
            p[2] += self.gamma ** p[3] * float(reward[e]); p[3] += 1
            if p[3] >= self.close_after(p[4][0]) or bool(done[e]):
                self._prefix[e] = None
                event = infos[e].get('event') if bool(done[e]) else None
                self._release(e, p, np.asarray(next_obs[e]), event, infos[e].get('TimeLimit.truncated', False))
        self.use(0)

    def close_after(self, pay):
        return pay['factual']['steps']

    def calibrate(self, pay, ret, end_obs, d, event, truncated):
        gate = calibrate_interval(pay, ret, end_obs, d, event, truncated)
        if pay['alternative']['steps'] != d:
            gate['reasons'].append('unequal_prefix_horizon'); gate['weight'] = 0.
        return gate

    def _release(self, e, p, end_obs, event, truncated, early=False):
        i, ix, ret, d, pays = p
        if self.add_index[i, e] != ix:
            return
        for k, pay in enumerate(pays[:self.SLOTS]):
            if early:
                gate = {'weight': 0., 'reasons': ['real_early_review_before_predicted_horizon'], 'errors': {'return': None, 'position_m': None}}
            else:
                gate = self.calibrate(pay, ret, end_obs, d, event, truncated)
            w = gate['weight']; s = self._sets[k]; s['cf_weights'][i, e] = w
            if w > 0:
                f, a = pay['factual'], pay['alternative']
                s['cf_actions'][i, e] = pay['a_tilde']; s['cf_rewards'][i, e] = a['return']; s['cf_discount'][i, e] = a['discount']
                s['cf_done'][i, e] = a['done']; s['cf_next_observations'][i, e] = a['observation']
                s['cf_fact_rewards'][i, e] = f['return']; s['cf_fact_discount'][i, e] = f['discount']
                s['cf_fact_done'][i, e] = f['done']; s['cf_fact_obs'][i, e] = f['observation']
                self.slot_valid[k] += 1
            st = self.feedback_stats; st['closed'] += 1; st['valid'] += int(w > 0); st['weights'].append(w); st['reasons'].update(gate['reasons'])
            if gate['errors'].get('return') is not None:
                st['return_errors'].append(gate['errors']['return']); st['position_errors'].append(gate['errors']['position_m'])
            st['review_steps'].append(d)
            if self.feedback_audit_path:
                with self.feedback_audit_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'slot': int(i), 'env': e, 'add_index': int(ix), 'cf_slot': k, 'actual_return': ret,
                                             'actual_steps': d, 'actual_event': event, 'actual_end_observation': np.asarray(end_obs).tolist(),
                                             'gate': gate, 'payload': pay}, default=json_default) + '\n')


class FeasibleSupportSACV4(FeasibleSupportSACV2):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('behaviour_epsilon', V4_BEHAVIOUR_EPSILON)
        super().__init__(*args, **kwargs)
        # FeasibleSupportSAC fixes its replay class; V4 replaces it before the buffer is built
    def _setup_model(self):
        self.replay_buffer_class = AllAlternativesReplay
        super()._setup_model()

    def train(self, gradient_steps, batch_size=256):
        self.policy.set_training_mode(True); self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        owed = max((self.num_timesteps - self.learning_starts) // 3 - self.actor_updates, 0)
        params = list(self.critic.parameters()); norm = lambda gs: float(th.sqrt(sum((g * g).sum() for g in gs if g is not None)))
        b = self.replay_buffer
        for step in range(gradient_steps):
            b.use(0); data = b.sample(batch_size); self._n_updates_in_call = self._n_updates + step
            target, mask = self._td_targets(data, self.target_value(data.next_observations)); current = self.critic(data.observations, data.actions)
            td = self._td_loss(current, target, mask); loss = td
            applied = share = gtd = gcf = relative = 0.; valid = 0.; phase = self.delta_phase(); lam = self.current_delta_lambda()
            if self.delta_enabled() and phase in ('calibrating', 'on'):
                rels = []
                for k in range(b.SLOTS):
                    b.use(k); cf = b.counterfactual(); w = self._cf_weights(cf); vk = float(w.sum())
                    if vk > 0:
                        with th.no_grad():
                            delta = self._delta_target(data, cf, None)
                        rels.append(self._relative_loss(self.critic(data.observations, cf.actions), current, delta, w)); valid += vk
                b.use(0)
                if rels:
                    rel = sum(rels)
                    gtd = norm(th.autograd.grad(td, params, retain_graph=True, allow_unused=True))
                    gcf = norm(th.autograd.grad(rel, params, retain_graph=True, allow_unused=True)); relative = float(rel.item())
                    if phase == 'calibrating' and self._n_updates_in_call % 5 == 0 and gtd > 1e-8:
                        self.delta_calibration['ratios'].append(gcf / gtd); self.delta_calibration['decisions'].append(int(self.num_timesteps))
                    if phase == 'on' and gtd > 1e-8 and gcf > 0:
                        applied = min(lam, self.delta_cap_share * gtd / gcf); share = applied * gcf / gtd; loss = td + applied * rel
            self.critic.optimizer.zero_grad(); loss.backward(); self.critic.optimizer.step()
            aloss = qgrad = 0.
            if step >= gradient_steps - owed:
                obs = b.sample_closed_review(batch_size)
                if obs is not None:
                    p = self.support_probabilities(obs); q = th.minimum(*self.candidate_q(obs, self.critic)).detach()
                    qterm = -(p * q).sum(1).mean(); entropy = (self.alpha_of(obs) * p * p.clamp_min(1e-8).log()).sum(1).mean(); actor_loss = qterm + entropy
                    qgrad = norm(th.autograd.grad(qterm, list(self.actor.parameters()), retain_graph=True, allow_unused=True))
                    self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step(); self.actor_updates += 1; aloss = float(actor_loss.item())
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            if self._n_updates_in_call % 20 in (0, 3):
                self.support_stats.append({'steps': int(self.num_timesteps), 'update': int(self._n_updates_in_call), 'actor_updates': int(self.actor_updates),
                                           'real_huber_td': float(td.item()), 'relative_loss': relative, 'valid_weight_mean': valid / batch_size,
                                           'g_td': gtd, 'g_mpc_relative': gcf, 'applied_lambda': applied, 'applied_gradient_share': share,
                                           'actor_loss': aloss, 'actor_gradient_via_same_Q': qgrad, 'phase': phase,
                                           'expert_sampling': dict(b.sample_counts), 'cf_slot_valid': list(getattr(b, 'slot_valid', []))})
        self._n_updates += gradient_steps
