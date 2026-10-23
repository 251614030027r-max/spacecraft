"""T1-B V6 learner: intent-level replay with an MPC-paired outcome advantage (both arms share the replay).

Why (evidence/T1B_V6_PRECHECK): a learning decision is ~2 % of the stored transitions, so a uniform
256 batch holds ~5 decisions; the critic's gradient is spent on the continuation values of
committed states and the state x intent interaction at decisions is hardly trained, while the
MPC labels attached to decisions are diluted by the same 2 % (V5.1: share pinned at the cap). And
one episode yields one real outcome (~150 in 10k steps): real returns can set the value *level*
but cannot resolve +2..+3 intent differences under +-100 outcome variance.

Each critic update therefore has two parts on the same twin critic:

* the V5.1 transition batch: real (interval) TD on all transitions, unchanged;
* an **intent batch** of closed learning decisions (``sample_decisions``): the real interval
  target (the Monte Carlo return of the committed intent when it ends at a terminal) anchors
  Q(s, a_taken); in E-TCPF the gated full-horizon MPC pairs of the same decisions give the intent
  advantage, Q(s, a~) - Q(s, a) -> Y, averaged over the admitted pairs with weight ``delta_lambda`` = 1
  (the same footing as the real target, itself a mean over the decisions of the batch; no
  gradient-share calibration or cap: the label is independent model information admitted only
  when the factual branch reproduced the real outcome).

E-SAC uses the same intent batch with the real target only, so the ablation isolates the MPC
feedback. Actor, temperature, exploration, interface (V5.1), task, G*, MPC and truth unchanged.
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update

from train.t1_b_v4_sac import FeasibleSupportSACV4
from train.t1_b_v5_sac import FeasibleSupportSACV5, FullHorizonReplay

INTENT_BATCH = 256
CF_WEIGHT = 1.0


class IntentReplay(FullHorizonReplay):
    def sample_decisions(self, batch_size):
        """A batch of closed learning decisions; sets ``_last`` so interval()/counterfactual() follow it."""
        upper = self.buffer_size if self.full else self.pos
        flat = np.flatnonzero((self.due[:upper] & self.interval_ready[:upper]).reshape(-1))
        if flat.size == 0:
            return None
        i, e = np.divmod(flat[np.random.randint(0, flat.size, size=batch_size)], self.n_envs)
        self._last = (i, e)
        data = (self._normalize_obs(self.observations[i, e, :], None), self.actions[i, e, :],
                self._normalize_obs(self.next_observations[i, e, :], None),
                (self.dones[i, e] * (1 - self.timeouts[i, e])).reshape(-1, 1),
                self._normalize_reward(self.rewards[i, e].reshape(-1, 1), None))
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class FeasibleSupportSACV6(FeasibleSupportSACV5):
    def __init__(self, *args, intent_batch=INTENT_BATCH, **kwargs):
        self.intent_batch = int(intent_batch)
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        self.replay_buffer_class = IntentReplay
        super(FeasibleSupportSACV4, self)._setup_model()

    def _relative_loss(self, alt_q, current_q, delta, weights):
        """Weighted mean over the admitted pairs of the intent batch (not over the whole batch): the
        MPC advantage and the real intent target are each a mean over their own supervised samples."""
        denom = weights.sum().clamp(min=1e-6)
        return .5 * sum((weights * F.huber_loss(qa - qr, delta, reduction='none', delta=self.delta_huber)).sum() / denom
                        for qa, qr in zip(alt_q, current_q))

    def train(self, gradient_steps, batch_size=256):
        self.policy.set_training_mode(True); self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        owed = max((self.num_timesteps - self.learning_starts) // 3 - self.actor_updates, 0)
        params = list(self.critic.parameters()); norm = lambda gs: float(th.sqrt(sum((g * g).sum() for g in gs if g is not None)))
        b = self.replay_buffer
        for step in range(gradient_steps):
            self._n_updates_in_call = self._n_updates + step
            b.use(0); data = b.sample(batch_size)
            target, mask = self._td_targets(data, self.target_value(data.next_observations))
            td = self._td_loss(self.critic(data.observations, data.actions), target, mask); loss = td
            td_i = rel = None; valid = 0.; lam = self.delta_lambda if self.delta_phase() == 'on' else 0.
            dec = b.sample_decisions(self.intent_batch)
            if dec is not None:
                t_i, m_i = self._td_targets(dec, self.target_value(dec.next_observations))
                current_i = self.critic(dec.observations, dec.actions)
                td_i = self._td_loss(current_i, t_i, m_i); loss = loss + td_i
                if lam > 0:
                    rels = []
                    for k in range(b.SLOTS):
                        b.use(k); cf = b.counterfactual(); w = self._cf_weights(cf); vk = float(w.sum())
                        if vk > 0:
                            with th.no_grad():
                                delta = self._delta_target(dec, cf, None)
                            rels.append(self._relative_loss(self.critic(dec.observations, cf.actions), current_i, delta, w)); valid += vk
                    b.use(0)
                    if rels:
                        rel = sum(rels); loss = loss + lam * rel
            probe = self._n_updates_in_call % 20 in (0, 3)
            g = {}
            if probe:
                for name, part in (('g_td', td), ('g_intent_td', td_i), ('g_mpc_advantage', rel)):
                    g[name] = norm(th.autograd.grad(part, params, retain_graph=True, allow_unused=True)) if part is not None else 0.
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
            if probe:
                self.support_stats.append({'steps': int(self.num_timesteps), 'update': int(self._n_updates_in_call), 'actor_updates': int(self.actor_updates),
                                           'real_huber_td': float(td.item()), 'intent_td': float(td_i.item()) if td_i is not None else None,
                                           'mpc_advantage_loss': float(rel.item()) if rel is not None else 0., 'valid_weight_mean_intent_batch': valid / self.intent_batch,
                                           'cf_weight': lam, **g, 'actor_loss': aloss, 'actor_gradient_via_same_Q': qgrad,
                                           'expert_sampling': dict(b.sample_counts), 'cf_slot_valid': list(b.slot_valid)})
        self._n_updates += gradient_steps
