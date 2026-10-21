"""T1-B V8 learner (ECPG): event-aware counterfactual policy guidance on a masked categorical intent head.

Why (evidence/T1B_V8_PRECHECK): V6/V7 regressed the MPC full-horizon advantage into the same twin
critic as the real intent target; the two conflict (gradient cosine -0.26) and the critic did not
carry the labels to unseen states. V8 moves the model information out of the critic:

* **Critic**: real transition TD + real intent-level SMDP target only (the V6 E-SAC path, both arms).
* **Intent head** (``policy.intent_head``): <= 3 logits aligned with the G* candidate indices, masked
  by the candidate-valid flags; it replaces the Gaussian-CDF induced categorical at learning decisions
  (training, target value and deployment). The last layer starts at zero, so the untrained policy is
  uniform and its deterministic choice is the earliest candidate (index 0): 0k == fixed:0.
* **Actor loss** in nats: KL-form discrete SAC, sum_j pi_j (log pi_j - Q_j / alpha), plus
  ``ce_weight`` x the gate-weighted cross-entropy to the event-aware model preference p_M on
  decisions whose every candidate has an admitted real-controller full-horizon branch (V7 labels,
  training openings only). p_M: predicted failures get zero mass when a predicted success exists;
  successes are soft-ranked by the model return / ``PREFERENCE_TEMPERATURE``; a decision whose
  candidates are all predicted to fail is skipped. E-SAC: ``ce_weight`` = 0 and no branches.

The model branch continues with the earliest candidate at its re-decisions (``G^fallback``), so p_M
is a preference supervision of the first intent, never a Bellman target. Task, interface (V7),
G*, MPC, truth, exploration (epsilon 0.5 at decisions) and the gate (V5.1) are unchanged.
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

from env.t1_b_feasible_v1 import bin_center
from env.t1_n_review import N_DUE_INDEX
from train.t1_b_v4_sac import FeasibleSupportSACV4
from train.t1_b_v7_sac import AsyncIntentReplay, FeasibleSupportSACV7

PREFERENCE_TEMPERATURE = 2.0   # return units; the admitted label error is <= 5 (V5.1 gate), typical 0.1-0.6
CE_WEIGHT = 1.0                # nats per nat of the KL-form SAC actor loss
PREFERENCE_BATCH = 64


def event_aware_targets(G, S, M, temperature=PREFERENCE_TEMPERATURE):
    """p_M over the candidates and a 0/1 usability flag per row.

    G: model returns, S: predicted success (bool), M: candidate labelled (bool), all (B, 3) tensors.
    """
    ok = S & M
    usable = ok.any(1)
    logits = th.where(ok, G / temperature, th.full_like(G, -1e9))
    p = th.softmax(logits, 1) * ok
    return p / p.sum(1, keepdim=True).clamp_min(1e-12), usable


class PreferenceReplay(AsyncIntentReplay):
    """V7 replay plus the per-candidate predicted event of every admitted branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = self.cf_weights.shape[:2]
        self.pref_fact_index = np.full(shape, -1, np.int8); self.pref_fact_success = np.zeros(shape, bool)
        self.pref_alt_index = [np.full(shape, -1, np.int8) for _ in range(self.SLOTS)]
        self.pref_alt_success = [np.zeros(shape, bool) for _ in range(self.SLOTS)]
        self.preference_counts = {'released': 0, 'complete': 0}

    def _release(self, e, p, end_obs, event, truncated, early=False):
        super()._release(e, p, end_obs, event, truncated, early)
        i, ix, _, _, pays = p
        if early or self.add_index[i, e] != ix:
            return
        self.preference_counts['released'] += 1
        for k, pay in enumerate(pays[:self.SLOTS]):
            self.pref_fact_index[i, e] = pay['sampling']['selected_index']
            self.pref_fact_success[i, e] = pay['factual'].get('event') == 'success'
            self.pref_alt_index[k][i, e] = pay['sampling']['alternative_index']
            self.pref_alt_success[k][i, e] = pay['alternative'].get('event') == 'success'
        if self._complete(np.array([i]), np.array([e]))[0]:
            self.preference_counts['complete'] += 1

    def _complete(self, i, e):
        n = (self.observations[i, e, 49:52] > .5).sum(-1)
        w = [self._sets[k]['cf_weights'][i, e, 0] > 0 for k in range(self.SLOTS)]
        return self.due[i, e] & (n >= 2) & w[0] & ((n == 2) | w[1])

    def sample_preferences(self, batch_size):
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            return None
        i_all, e_all = np.divmod(np.arange(upper * self.n_envs), self.n_envs)
        flat = np.flatnonzero(self._complete(i_all, e_all))
        if flat.size == 0:
            return None
        i, e = np.divmod(flat[np.random.randint(0, flat.size, size=batch_size)], self.n_envs)
        b = np.arange(batch_size)
        G = np.zeros((batch_size, 3), np.float32); S = np.zeros((batch_size, 3), bool); M = np.zeros((batch_size, 3), bool)
        W = np.ones(batch_size, np.float32)
        jf = self.pref_fact_index[i, e].astype(int)
        G[b, jf] = self._sets[0]['cf_fact_rewards'][i, e, 0]; S[b, jf] = self.pref_fact_success[i, e]; M[b, jf] = True
        for k in range(self.SLOTS):
            w = self._sets[k]['cf_weights'][i, e, 0]; valid = w > 0
            ja = self.pref_alt_index[k][i, e].astype(int)
            rows = b[valid]
            G[rows, ja[valid]] = self._sets[k]['cf_rewards'][i, e, 0][valid]
            S[rows, ja[valid]] = self.pref_alt_success[k][i, e][valid]; M[rows, ja[valid]] = True
            W[valid] = np.minimum(W[valid], w[valid])
        obs = self._normalize_obs(self.observations[i, e, :], None)
        return tuple(map(self.to_torch, (obs, G))) + (th.as_tensor(S), th.as_tensor(M), self.to_torch(W))


def _head(width=128):
    net = th.nn.Sequential(th.nn.Linear(52, width), th.nn.ReLU(), th.nn.Linear(width, width), th.nn.ReLU(), th.nn.Linear(width, 3))
    th.nn.init.zeros_(net[-1].weight); th.nn.init.zeros_(net[-1].bias)
    return net


class FeasibleSupportSACV8(FeasibleSupportSACV7):
    def __init__(self, *args, ce_weight=CE_WEIGHT, preference_batch=PREFERENCE_BATCH, **kwargs):
        self.ce_weight = float(ce_weight)
        self.preference_batch = int(preference_batch)
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        self.replay_buffer_class = PreferenceReplay
        super(FeasibleSupportSACV4, self)._setup_model()
        self.policy.intent_head = _head().to(self.device)
        self.intent_optimizer = th.optim.Adam(self.policy.intent_head.parameters(), lr=self.lr_schedule(1))

    # ---------------------------------------------------------------- categorical intent policy
    def intent_logits(self, obs):
        n = (obs[:, 49:52] > .5).sum(1).clamp(min=1)
        valid = th.arange(3, device=obs.device)[None, :] < n[:, None]
        return self.policy.intent_head(obs).masked_fill(~valid, -1e9)

    def support_probabilities(self, obs):
        return th.softmax(self.intent_logits(obs), 1)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action, state = super().predict(observation, state, episode_start, deterministic)
        z = np.asarray(observation, np.float32).reshape(-1, 52); a = np.asarray(action, np.float32).reshape(len(z), -1)
        due = (z[:, N_DUE_INDEX] > .5) & ((z[:, 49:52] > .5).sum(1) >= 2)
        if due.any():
            with th.no_grad():
                p = self.support_probabilities(th.as_tensor(z[due], device=self.device)).cpu().numpy()
            for k, row in zip(np.flatnonzero(due), p):
                n = int((z[k, 49:52] > .5).sum())
                j = int(np.argmax(row[:n])) if deterministic else int(np.random.choice(n, p=row[:n] / row[:n].sum()))
                a[k, 0] = bin_center(j, n)
        return a.reshape(np.shape(action)), state

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate([*optimizers, self.intent_optimizer])

    # ---------------------------------------------------------------- training
    def actor_objective(self, obs, pref=None):
        """KL-form discrete SAC on closed decisions + ce_weight x gate-weighted CE(p_M, pi) on labelled decisions."""
        alpha = self.alpha_of(obs)
        logp = th.log_softmax(self.intent_logits(obs), 1); p = logp.exp()
        q = th.minimum(*self.candidate_q(obs, self.critic)).detach()
        sac = (p * (logp - q / alpha)).sum(1).mean(); loss = sac; ce = None; used = 0
        if pref is not None:
            pobs, G, S, M, W = pref
            pm, usable = event_aware_targets(G, S, M)
            w = W * usable.float(); used = int(usable.sum())
            if float(w.sum()) > 0:
                lp = th.log_softmax(self.intent_logits(pobs), 1)
                ce = (w * -(pm * lp.clamp_min(-1e4)).sum(1)).sum() / w.sum(); loss = sac + self.ce_weight * ce
        return loss, sac, ce, used, p, logp

    def train(self, gradient_steps, batch_size=256):
        self.policy.set_training_mode(True); self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        owed = max((self.num_timesteps - self.learning_starts) // 3 - self.actor_updates, 0)
        norm = lambda gs: float(th.sqrt(sum((g * g).sum() for g in gs if g is not None)))
        head = list(self.policy.intent_head.parameters())
        b = self.replay_buffer
        for step in range(gradient_steps):
            self._n_updates_in_call = self._n_updates + step
            b.use(0); data = b.sample(batch_size)
            target, mask = self._td_targets(data, self.target_value(data.next_observations))
            td = self._td_loss(self.critic(data.observations, data.actions), target, mask); loss = td; td_i = None
            dec = b.sample_decisions(self.intent_batch)
            if dec is not None:
                t_i, m_i = self._td_targets(dec, self.target_value(dec.next_observations))
                td_i = self._td_loss(self.critic(dec.observations, dec.actions), t_i, m_i); loss = loss + td_i
            self.critic.optimizer.zero_grad(); loss.backward(); self.critic.optimizer.step()
            stats = {}
            if step >= gradient_steps - owed:
                obs = b.sample_closed_review(batch_size)
                if obs is not None:
                    pref = b.sample_preferences(self.preference_batch) if self.ce_weight > 0 else None
                    actor_loss, sac, ce, used, p, logp = self.actor_objective(obs, pref)
                    if self._n_updates_in_call % 20 in (0, 3):
                        stats['g_actor_sac'] = norm(th.autograd.grad(sac, head, retain_graph=True, allow_unused=True))
                        stats['g_actor_ce'] = norm(th.autograd.grad(ce, head, retain_graph=True, allow_unused=True)) if ce is not None else 0.
                    self.intent_optimizer.zero_grad(); actor_loss.backward(); self.intent_optimizer.step(); self.actor_updates += 1
                    stats.update(actor_loss=float(actor_loss.item()), actor_sac=float(sac.item()), actor_ce=float(ce.item()) if ce is not None else None,
                                 preference_rows_usable=used, policy_entropy=float(-(p * logp).sum(1).mean().item()))
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            if self._n_updates_in_call % 20 in (0, 3):
                self.support_stats.append({'steps': int(self.num_timesteps), 'update': int(self._n_updates_in_call), 'actor_updates': int(self.actor_updates),
                                           'real_huber_td': float(td.item()), 'intent_td': float(td_i.item()) if td_i is not None else None,
                                           'ce_weight': self.ce_weight, **stats, 'preference_counts': dict(b.preference_counts),
                                           'expert_sampling': dict(b.sample_counts), 'cf_slot_valid': list(b.slot_valid)})
        self._n_updates += gradient_steps

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'intent_optimizer']
