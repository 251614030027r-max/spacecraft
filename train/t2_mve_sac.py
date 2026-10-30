"""T2 learner: SAC on the 16 s stage decision, with Execution-Anchored Counterfactual Coupling (EACC, T2.3).

Real part (both arms): SB3 SAC (twin critic, fixed entropy coefficient, DEAD_ENDS D02) on the real
stage transitions; success / unsafe / timeout / resource exhaustion are terminal (no bootstrap).

Coupled part (``e_mve`` only). Per real transition the environment returns the executor model's raw
prediction of the executed intent (r^_a, s^'_a, d^_a) and K anchored candidate labels (a_k, r~_k, s~'_k, d~_k)
(``T2StageEnv._labels``: M1 executor option model, M2 hindsight execution anchoring). M3 robust precision
fusion, one rule in place of T2.2's gates and gradient cap:

    y(r, s', d)  = r + gamma (1 - d) V'(s'),   V'(s') = min_i Q'_i(s', a~) - alpha log pi(a~ | s'),  a~ ~ pi(.|s')
    e_a          = y(r, s', d) - y(r^_a, s^'_a, d^_a)          (the model's error on the executed intent, value units)
    w            = 1 / (1 + (e_a / sigma)^2)                    (sigma: running RMS of the real TD residual)
    L_Q          = L_real + beta * sum_{b,k} w_b 1/2 sum_i (Q_i(s_b, a_k) - y(r~_k, s~'_k, d~_k))^2 / #candidates

Every label is r + gamma (1 - d) V'(.) of a single next state, so the fused regression target is a
gamma-contraction; w |e_a| <= sigma / 2 bounds the influence of any model error (a Cauchy M-estimator).
The term is off until ``model_start`` real transitions are stored. beta = 0.5.

Collection (``train``): asynchronous -- each worker process steps its own environment at its own pace and
the learner (this process) answers every action request with the current policy and performs
``UPDATES_PER_TRANSITION`` updates per stored transition. T2.2's lock-step vector environment waited
for the slowest environment (a bridge + descent macro-step, ~50 s) at every decision.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update

from env.t2_stage import GAMMA_STAGE, OBS_DIM

K_CANDIDATES = 8          # 3 structured alternatives + 1 global + 4 local (T2StageEnv.candidate_actions)
MODEL_BETA = 0.5
MODEL_START = 300
ENT_COEF = 0.05
SIGMA_DECAY = 0.99        # running mean of the squared real TD residual
#: critic / actor updates per stored real transition
UPDATES_PER_TRANSITION = 2
#: T2.5 two-stage labels: the policy copy in each environment is refreshed every SYNC_UPDATES critic/actor updates;
#: a stage-2 part older than AGE_CAP stored transitions is not used in the Retrace correction (the label falls back to its
#: stage-1 target); its one-step stage-2 row (T2.5.1) has no age cap
SYNC_UPDATES = 100
AGE_CAP = 2000
#: learner version recorded in the manifest (T2.5.1: one-step stage-2 rows; T2.6: calibration shared by the workers;
#: T2.6.1: labels end at the deadline / budget, env t2-stage-0.4.2; env 0.4.3 adds the budget before a predicted entry)
LEARNER_VERSION = "t2-learner-2.6.1"
#: T2.6: every worker's executed stages and entries feed one stage / entry calibration (one executor, one model of it).
#: Per-worker calibrations were an artifact of the parallel collector: each worker waited for its own 20 entries, so the
#: labels started only after ~workers x 20 entries (T2.5.1: first model rows at 512 / 537 decisions with 3 workers).
POOL_CALIBRATION = True


class StageReplay(ReplayBuffer):
    def __init__(self, *args, k_candidates: int = K_CANDIDATES, **kwargs):
        super().__init__(*args, **kwargs)
        k, d, a = k_candidates, self.obs_shape[0], self.action_dim
        self.k = k
        shape = (self.buffer_size, self.n_envs)
        self.fact_r = np.zeros(shape, np.float32); self.fact_obs = np.zeros((*shape, d), np.float32)
        self.fact_done = np.zeros(shape, np.float32); self.fact_valid = np.zeros(shape, np.float32)
        self.cand_a = np.zeros((*shape, k, a), np.float32); self.cand_r = np.zeros((*shape, k), np.float32)
        self.cand_obs = np.zeros((*shape, k, d), np.float32); self.cand_done = np.zeros((*shape, k), np.float32)
        self.cand_valid = np.zeros((*shape, k), np.float32)
        #: label kind: 0 stage end, 1 entry with the executed entry's realized outcome anchored, 2 entry on the outcome model alone
        self.cand_kind = np.zeros((*shape, k), np.float32)
        #: stage 2 of a non-terminal candidate (T2.5): a2 ~ mu(.|o1), log mu, shaped r2, o2, terminal flag, kind
        self.c2_valid = np.zeros((*shape, k), np.float32); self.c2_a = np.zeros((*shape, k, a), np.float32)
        self.c2_logp = np.zeros((*shape, k), np.float32); self.c2_r = np.zeros((*shape, k), np.float32)
        self.c2_obs = np.zeros((*shape, k, d), np.float32); self.c2_done = np.zeros((*shape, k), np.float32)
        self.c2_kind = np.zeros((*shape, k), np.float32)
        self.created = np.zeros(shape, np.float32)             # stored transitions at label time (label age diagnostic)
        self.stored = 0
        self.model_counts = {'transitions': 0, 'with_candidates': 0, 'candidates': 0, 'entry_candidates': 0, 'reason': {}}
        self._last = None

    def add(self, obs, next_obs, action, reward, done, infos):
        i = self.pos
        self.fact_valid[i] = 0.0; self.cand_valid[i] = 0.0; self.cand_done[i] = 0.0; self.cand_kind[i] = 0.0
        self.c2_valid[i] = 0.0; self.c2_done[i] = 0.0; self.c2_kind[i] = 0.0
        self.created[i] = self.stored; self.stored += 1
        for e, info in enumerate(infos):
            model = info.get('model') or {}
            fact, cands = model.get('fact'), (model.get('cands') or [])[:self.k]
            reason = (model.get('diag') or {}).get('reason', 'none')
            c = self.model_counts; c['transitions'] += 1; c['reason'][reason] = c['reason'].get(reason, 0) + 1
            if fact is None or not cands:
                continue
            self.fact_r[i, e] = fact['reward']; self.fact_obs[i, e] = fact['next_obs']; self.fact_done[i, e] = float(fact['done'])
            self.fact_valid[i, e] = 1.0
            c['with_candidates'] += 1; c['candidates'] += len(cands)
            for j, x in enumerate(cands):
                self.cand_a[i, e, j] = x['a']; self.cand_r[i, e, j] = x['reward']; self.cand_obs[i, e, j] = x['next_obs']
                self.cand_done[i, e, j] = float(x['done']); self.cand_valid[i, e, j] = 1.0; self.cand_kind[i, e, j] = float(x.get('kind', 0))
                c['entry_candidates'] += int(x['done'] and x.get('kind', 0) > 0)
                c['cutoff_candidates'] = c.get('cutoff_candidates', 0) + int(x['done'] and x.get('kind', 0) == 0)   # T2.6: deadline / budget
                y2 = x.get('s2')
                if y2 is not None and not x['done']:
                    self.c2_valid[i, e, j] = 1.0; self.c2_a[i, e, j] = y2['a']; self.c2_logp[i, e, j] = y2['logp']
                    self.c2_r[i, e, j] = y2['reward']; self.c2_obs[i, e, j] = y2['next_obs']; self.c2_done[i, e, j] = float(y2['done'])
                    self.c2_kind[i, e, j] = float(y2.get('kind', 0))
                    c['two_stage'] = c.get('two_stage', 0) + 1; c['two_stage_entry'] = c.get('two_stage_entry', 0) + int(y2['done'] and y2.get('kind', 0) == 2)
                    c['two_stage_cutoff'] = c.get('two_stage_cutoff', 0) + int(y2['done'] and y2.get('kind', 0) == 0)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        self._last = (batch_inds, env_indices)
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (self._normalize_obs(self.observations[batch_inds, env_indices, :], env), self.actions[batch_inds, env_indices, :], next_obs,
                (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env))
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def model_batch(self):
        i, e = self._last
        return {k: self.to_torch(getattr(self, k)[i, e]) for k in
                ('fact_r', 'fact_obs', 'fact_done', 'fact_valid', 'cand_a', 'cand_r', 'cand_obs', 'cand_done', 'cand_valid', 'cand_kind', 'created',
                 'c2_valid', 'c2_a', 'c2_logp', 'c2_r', 'c2_obs', 'c2_done', 'c2_kind')}


class T2SAC(SAC):
    def __init__(self, *args, model_beta: float = 0.0, model_start: int = MODEL_START, **kwargs):
        self.model_beta, self.model_start = float(model_beta), int(model_start)
        self.model_stats = []
        self.td_ms = None          # running mean of the squared real TD residual (sigma^2)
        self.entry_bias_now = None # current entry outcome-model bias (mean over environments that know it), None = unknown
        kwargs.setdefault('replay_buffer_class', StageReplay)
        kwargs.setdefault('ent_coef', ENT_COEF)   # fixed (DEAD_ENDS D02)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'model_stats']

    def _soft_value(self, obs, ent_coef):
        a, logp = self.actor.action_log_prob(obs)
        return th.min(th.cat(self.critic_target(obs, a), dim=1), dim=1, keepdim=True).values - ent_coef * logp.reshape(-1, 1)

    def _log_pi(self, obs, actions):
        """log pi(a | obs) of given (squashed) actions under the current actor."""
        mean, log_std, kwargs = self.actor.get_action_dist_params(obs)
        dist = self.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
        return dist.log_prob(actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)).reshape(-1, 1)

    def _q_target_min(self, obs, actions):
        return th.min(th.cat(self.critic_target(obs, actions), dim=1), dim=1, keepdim=True).values

    def _candidate_targets(self, m, mask, err, ent_coef):
        """Targets and weights of the candidate labels selected by ``mask`` (see ``_model_loss``)."""
        k = mask.shape[1]
        o1 = m['cand_obs'][mask]; d1 = m['cand_done'][mask].reshape(-1, 1)
        y = m['cand_r'][mask].reshape(-1, 1) + (1.0 - d1) * self.gamma * self._soft_value(o1, ent_coef)
        age = (float(self.replay_buffer.stored) - m['created']).reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1)
        two = (m['c2_valid'][mask].reshape(-1, 1) > 0) & (d1 == 0) & (age <= AGE_CAP)
        c = th.zeros_like(y); delta = th.zeros_like(y)
        idx = two.reshape(-1)
        if bool(idx.any()):
            a2 = m['c2_a'][mask][idx]; o1s = o1[idx]
            c[idx] = th.exp(self._log_pi(o1s, a2) - m['c2_logp'][mask][idx].reshape(-1, 1)).clamp(max=1.0)
            d2 = m['c2_done'][mask][idx].reshape(-1, 1)
            delta[idx] = (m['c2_r'][mask][idx].reshape(-1, 1) + (1.0 - d2) * self.gamma * self._soft_value(m['c2_obs'][mask][idx], ent_coef)
                          - self._q_target_min(o1s, a2))
        y = y + self.gamma * c * delta
        kind1 = m['cand_kind'][mask].reshape(-1, 1); kind2 = m['c2_kind'][mask].reshape(-1, 1)
        b_now = float(np.sqrt(self.td_ms)) if self.entry_bias_now is None else abs(float(self.entry_bias_now))
        ea = err.abs().reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1)
        eps = ea + (kind1 == 2).float() * b_now + self.gamma * c * (ea + (kind2 == 2).float() * b_now)
        wk = 1.0 / (1.0 + eps ** 2 / self.td_ms)
        return y, c, delta, wk, idx, two, d1, age, kind1

    def _stage2_rows(self, m, mask, err, ent_coef):
        """T2.5.1 stage-2 rows: the stage-2 part of every non-terminal candidate as its own one-step label at the
        predicted stage-1 state, Q(o1, a2) <- r2 + gamma (1 - d2) V(o2). A one-step soft Bellman target is valid for any
        action, so these rows need no importance ratio and no age cap; their information reaches Q(s, a) through V(o1).
        Weight as every model row: eps2 = |e_a| + b2 (b2 the current entry bias for an unanchored entry, else 0).
        Also returns |Qbar(o1, a2) - Qbar(o1, a_pi)| / sigma, how far the label's action lies from the current policy's in value,
        d2 and the stage-2 kind (2 entry, 0 stage end or, T2.6.1, the deadline / budget end)."""
        k = mask.shape[1]
        d1 = m['cand_done'][mask].reshape(-1, 1)
        sel = ((m['c2_valid'][mask].reshape(-1, 1) > 0) & (d1 == 0)).reshape(-1)
        if not bool(sel.any()):
            return None
        o1 = m['cand_obs'][mask][sel]; a2 = m['c2_a'][mask][sel]; d2 = m['c2_done'][mask][sel].reshape(-1, 1)
        y2 = m['c2_r'][mask][sel].reshape(-1, 1) + (1.0 - d2) * self.gamma * self._soft_value(m['c2_obs'][mask][sel], ent_coef)
        b_now = float(np.sqrt(self.td_ms)) if self.entry_bias_now is None else abs(float(self.entry_bias_now))
        ea = err.abs().reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1)[sel]
        eps2 = ea + (m['c2_kind'][mask][sel].reshape(-1, 1) == 2).float() * b_now
        w2 = 1.0 / (1.0 + eps2 ** 2 / self.td_ms)
        a_pi, _ = self.actor.action_log_prob(o1)
        gap = (self._q_target_min(o1, a2) - self._q_target_min(o1, a_pi)).abs() / float(np.sqrt(self.td_ms))
        return o1, a2, y2, w2, gap, d2, m['c2_kind'][mask][sel].reshape(-1, 1)

    def _model_loss(self, data, target, ent_coef):
        """Counterfactual labels fused into the critic regression (module docstring).

        Stage-1 target (every candidate): y1 = r1 + gamma (1 - d1) V(o1), V the soft value of the target critic.
        Two-stage soft Retrace target (T2.5; non-terminal candidates with a stage 2):
            y = y1 + gamma c delta,  delta = r2 + gamma (1 - d2) V(o2) - Qbar(o1, a2),  c = min(1, pi(a2|o1) / mu(a2|o1)),
        a2 ~ mu the environment's policy copy at label time; c = 0 when the stage-2 part is older than AGE_CAP (y = y1).
        E_{a2~pi}[delta] = 0 when Qbar is the soft Q of pi, so the target is the soft Bellman target of pi; with stale
        mu the truncated ratio keeps it a contraction to Q^pi (Retrace, Munos et al. 2016).
        Weight: w_k = 1 / (1 + (eps_k / sigma)^2), eps_k = |e_a| + b1_k + gamma c (|e_a| + b2_k): the factual model
        error stands for each model stage's error, b the current entry outcome-model bias of entry parts resting on
        the outcome model alone (one sigma while unknown), 0 otherwise. eps_k is an estimate, not a proven bound.
        T2.5.1: the stage-2 part of every non-terminal candidate also enters as its own one-step row at the predicted
        stage-1 state (``_stage2_rows``), Q(o1, a2) <- r2 + gamma (1 - d2) V(o2). The Retrace trace c collapses once the
        narrow SAC policy has moved (measured c 0.01-0.55 at label ages 50-400), while the critic's value at the label's
        action differs from that at the current policy's action by a small fraction of sigma; the one-step row keeps
        that information without a ratio and passes it to Q(s, a) through V(o1). The loss is the mean over all model rows."""
        m = self.replay_buffer.model_batch()
        rows = m['fact_valid'] > 0
        if not bool(rows.any()) or self.td_ms is None:
            return None, {}
        with th.no_grad():
            y_fact = m['fact_r'].reshape(-1, 1) + (1.0 - m['fact_done'].reshape(-1, 1)) * self.gamma * self._soft_value(m['fact_obs'], ent_coef)
            err = (target - y_fact).reshape(-1)
            w = 1.0 / (1.0 + err ** 2 / self.td_ms)
            b, k = m['cand_valid'].shape
            mask = (m['cand_valid'] > 0) & rows.reshape(-1, 1)
            y, c, delta, wk, idx, two, d1, age, kind1 = self._candidate_targets(m, mask, err, ent_coef)
            entry = (kind1 > 0)
            rows2 = self._stage2_rows(m, mask, err, ent_coef)
        s = data.observations[:, None, :].expand(-1, k, -1)[mask]
        qs = self.critic(s, m['cand_a'][mask])
        total = (sum((wk * (q - y) ** 2).sum() for q in qs))
        n_rows = int(mask.sum())
        if rows2 is not None:                                        # every model row counts once (stage 1 and stage 2 alike)
            o1, a2, y2, w2, gap, d2, _ = rows2
            total = total + sum((w2 * (q - y2) ** 2).sum() for q in self.critic(o1, a2))
            n_rows += int(o1.shape[0])
        loss = 0.5 * total / max(n_rows, 1)
        e = err[rows]
        n2 = int(idx.sum())
        dec = (m['c2_done'][mask].reshape(-1, 1) > 0) & (m['c2_kind'][mask].reshape(-1, 1) == 2) & two   # stage 2 opens the entry
        return loss, {'model_rows': int(mask.sum()), 'w_mean': float(wk.mean()), 'abs_err_median': float(e.abs().median()),
                      'w_err_max': float((w[rows] * e.abs()).max()), 'entry_share': float(entry.float().mean()),
                      'entry_unanchored_share': float((kind1 == 2).float().mean()), 'entry_bias_now': self.entry_bias_now,
                      'label_age_mean': float(self.replay_buffer.stored - m['created'][rows].mean()),
                      'two_stage_share': n2 / max(int(mask.sum()), 1), 'decisive_share': float(dec.float().mean()),
                      'c_mean': float(c[idx].mean()) if n2 else None, 'c_low_share': float((c[idx] < 0.5).float().mean()) if n2 else None,
                      'increment_abs_mean': float((self.gamma * c[idx] * delta[idx]).abs().mean()) if n2 else None,
                      'age_capped_share': float(((m['c2_valid'][mask].reshape(-1, 1) > 0) & (d1 == 0) & (age > AGE_CAP)).float().mean()),
                      'stage2_rows': int(rows2[0].shape[0]) if rows2 is not None else 0,
                      'w2_mean': float(rows2[3].mean()) if rows2 is not None else None,
                      'pi_gap_sigma_median': float(rows2[4].median()) if rows2 is not None else None,
                      'stage2_entry_share': float((rows2[6] == 2).float().mean()) if rows2 is not None else None,
                      'stage2_cutoff_share': float(((m['c2_done'][mask] > 0) & (m['c2_kind'][mask] == 0) & (m['c2_valid'][mask] > 0)).float().mean()),
                      'stage1_cutoff_share': float(((m['cand_done'][mask] > 0) & (m['cand_kind'][mask] == 0)).float().mean())}

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        ent_coef = self.ent_coef_tensor
        stored = (self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos) * self.replay_buffer.n_envs
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actions_pi, log_prob = self.actor.action_log_prob(data.observations); log_prob = log_prob.reshape(-1, 1)
            with th.no_grad():
                target = data.rewards + (1 - data.dones) * self.gamma * self._soft_value(data.next_observations, ent_coef)
            current = self.critic(data.observations, data.actions)
            real = 0.5 * sum(F.mse_loss(q, target) for q in current)
            with th.no_grad():
                ms = float(sum(((q - target) ** 2).mean() for q in current) / len(current))
                self.td_ms = ms if self.td_ms is None else SIGMA_DECAY * self.td_ms + (1.0 - SIGMA_DECAY) * ms
            loss, mloss, info = real, None, {}
            if self.model_beta > 0 and stored >= self.model_start:
                mloss, info = self._model_loss(data, target, ent_coef)
                if mloss is not None:
                    loss = real + self.model_beta * mloss
                    info['loss_share'] = float(self.model_beta * mloss.item() / max(real.item(), 1e-9))
                    if (self._n_updates + step) % 100 == 0:   # read-only: do the two critic gradients agree in direction?
                        params = [p for p in self.critic.parameters() if p.requires_grad]
                        g1 = th.autograd.grad(real, params, retain_graph=True, allow_unused=True)
                        g2 = th.autograd.grad(mloss, params, retain_graph=True, allow_unused=True)
                        v1 = th.cat([g.reshape(-1) for g in g1 if g is not None]); v2 = th.cat([g.reshape(-1) for g in g2 if g is not None])
                        info['grad_cos'] = float(F.cosine_similarity(v1, v2, dim=0))
            self.critic.optimizer.zero_grad(); loss.backward(); self.critic.optimizer.step()
            q_pi = th.min(th.cat(self.critic(data.observations, actions_pi), dim=1), dim=1, keepdim=True).values
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            if (self._n_updates + step) % 25 == 0:
                self.model_stats.append({'steps': int(self.num_timesteps), 'update': int(self._n_updates + step), 'real_td': float(real.item()),
                                         'sigma': float(np.sqrt(self.td_ms)), 'model_td': float(mloss.item()) if mloss is not None else None,
                                         **info, 'actor_loss': float(actor_loss.item()), 'q_pi_mean': float(q_pi.mean().item())})
        self._n_updates += gradient_steps


def make_model(env, arm: str, seed: int, smoke: bool = False, updates_per_transition: float = UPDATES_PER_TRANSITION,
               model_beta: float = MODEL_BETA) -> T2SAC:
    return T2SAC('MlpPolicy', env, seed=seed, device='cpu', verbose=0, policy_kwargs={'net_arch': [256, 256]},
                 gamma=GAMMA_STAGE, learning_rate=3e-4, buffer_size=100_000, batch_size=64 if smoke else 256,
                 learning_starts=16 if smoke else 200, train_freq=(1, 'step'), gradient_steps=max(1, int(round(updates_per_transition))),
                 ent_coef=ENT_COEF, tau=0.005, model_beta=model_beta if arm == 'e_mve' else 0.0, model_start=16 if smoke else MODEL_START)


# ---------------------------------------------------------------------- asynchronous collection
class SpacesEnv(gym.Env):
    """Observation / action spaces of the stage decision, for the learner process (no physics)."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_DIM, np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, np.float32), 0.0, True, False, {}


def _slim(info: dict) -> dict:
    """What the learner needs from a step's info (the model labels without their audit fields)."""
    model = info.get('model') or {}
    cands = [{**{k: c.get(k, 0) for k in ('a', 'reward', 'next_obs', 'done', 'kind')}, 's2': c.get('s2')} for c in model.get('cands') or []]
    diag = model.get('diag') or {}
    return {'model': {'fact': model.get('fact'), 'cands': cands, 'diag': {k: diag.get(k) for k in ('reason', 'entry_bias', 'entry_records')}},
            **{k: info.get(k) for k in ('event', 'time_s', 'delta_v', 'mission')}}


class CalibrationPool:
    """Learner side of the shared calibration: rows reported by the workers, forwarded to every other worker once."""

    def __init__(self):
        self.rows, self.sent = [], {}

    def add(self, worker: int, rows) -> None:
        for kind in ('stage', 'entry'):
            for r in (rows or {}).get(kind, ()):
                self.rows.append((worker, kind, r))

    def pending(self, worker: int):
        new = [(k, r) for w, k, r in self.rows[self.sent.get(worker, 0):] if w != worker]
        self.sent[worker] = len(self.rows)
        if not new:
            return None
        return {kind: [r for k, r in new if k == kind] for kind in ('stage', 'entry')}


def share_calibration(env) -> dict:
    """Worker side: from now on every row this environment adds to its own stage / entry calibration is also put in
    the returned outbox (sent to the learner and emptied after each step)."""
    outbox = {'stage': [], 'entry': []}
    for kind, obj in (('stage', env.calibration), ('entry', env.entry_calibration)):
        def add(*a, _orig=obj.add, _obj=obj, _kind=kind, **k):
            _orig(*a, **k); outbox[_kind].append(_obj.rows[-1])
        obj.add = add
    return outbox


def merge_calibration(env, rows) -> None:
    """Worker side: the other workers' rows, appended in arrival order (the same sliding window as the own rows)."""
    from env.t2_stage import CALIBRATION_WINDOW
    for kind, obj in (('stage', env.calibration), ('entry', env.entry_calibration)):
        if rows and rows.get(kind):
            obj.rows = (obj.rows + list(rows[kind]))[-CALIBRATION_WINDOW:]


def worker(conn, rank: int, seed: int, candidates: int, entry_stages: int, window: bool, urgent_deadline=None) -> None:
    """One environment: send (transition or None, observation to act on), receive the action; until 'stop'."""
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    th.set_num_threads(1)
    from env.t2_stage import T2Mission, T2StageEnv
    mission = T2Mission() if urgent_deadline is None else T2Mission(urgent_deadline_s=tuple(float(x) for x in urgent_deadline))
    env = T2StageEnv(seed_base=1_000_000 * (seed + 1) + rank * 100_000, candidates=candidates, shaping=True, seed=seed * 100 + rank,
                     entry_stages=entry_stages, window=window, mission=mission)
    obs, _ = env.reset(); transition = None
    actor = None
    outbox = share_calibration(env) if (POOL_CALIBRATION and candidates > 0) else None

    def label_policy(o):          # mu: this environment's copy of the policy (T2.5 stage 2 of a counterfactual label)
        with th.no_grad():
            a, logp = actor.action_log_prob(th.as_tensor(np.asarray(o, np.float32)[None]))
        return a[0].numpy(), float(logp.reshape(-1)[0])
    while True:
        conn.send((transition, obs, {k: list(v) for k, v in outbox.items()} if outbox is not None else None))
        if outbox is not None:
            outbox['stage'].clear(); outbox['entry'].clear()
        msg = conn.recv()
        if isinstance(msg, str):
            break
        action, payload, shared = msg
        if shared is not None:
            merge_calibration(env, shared)
        if payload is not None:                   # ('actor', module) once, then ('state', state_dict) every SYNC_UPDATES
            if payload[0] == 'actor':
                actor = payload[1]; actor.set_training_mode(False); env.label_policy = label_policy
            else:
                actor.load_state_dict(payload[1])
        next_obs, reward, term, trunc, info = env.step(action)
        transition = (obs, next_obs, np.asarray(action, np.float32), float(reward), bool(term), bool(trunc), _slim(info))
        obs = env.reset()[0] if (term or trunc) else next_obs
    conn.close()


def train(out: Path, arm: str, seed: int, steps: int, workers: int, marks: list[int], smoke: bool = False,
          updates_per_transition: float = UPDATES_PER_TRANSITION, entry_stages: int | None = None, window: bool = True,
          model_beta: float = MODEL_BETA, urgent_deadline=None) -> None:
    import multiprocessing as mp
    from multiprocessing.connection import wait
    from stable_baselines3.common.logger import configure
    from env.t2_stage import ENTRY_STAGES, T2_VERSION
    entry_stages = ENTRY_STAGES if entry_stages is None else int(entry_stages)
    out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    th.set_num_threads(1)
    model = make_model(SpacesEnv(), arm, seed, smoke, updates_per_transition=updates_per_transition, model_beta=model_beta)
    model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
    candidates = K_CANDIDATES if arm == 'e_mve' else 0      # the SAC arm computes no model
    (out / 'manifest.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'steps': steps, 'workers': workers, 'git_head': head, 'smoke': smoke,
                                                   'model_beta': model.model_beta, 'model_start': model.model_start, 'ent_coef': ENT_COEF,
                                                   'k_candidates': candidates, 'collection': 'asynchronous', 'window': window,
                                                   'gradient_steps_per_transition': model.gradient_steps, 'entry_stages': entry_stages, 'sync_updates': SYNC_UPDATES, 'age_cap': AGE_CAP,
                                                   'pooled_calibration': bool(POOL_CALIBRATION and candidates > 0),
                                                   'urgent_deadline_s': list(urgent_deadline) if urgent_deadline is not None else [170.0, 210.0],
                                                   'env_version': T2_VERSION, 'learner_version': LEARNER_VERSION}, indent=1), encoding='utf-8')
    ctx = mp.get_context('spawn')
    pipes, procs = [], []
    for rank in range(workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, rank, seed, candidates, entry_stages, window, urgent_deadline), daemon=True)
        p.start(); pipes.append(a); procs.append(p)
    marks = sorted(set([m for m in marks if m <= steps] + [steps]))
    model.save(out / 'checkpoints' / 'step_0')
    episodes, order, t0 = [], [], time.time()
    bias_by_worker = {}
    synced = {}                                   # worker -> update count of its policy copy
    rng = np.random.default_rng(seed)
    pool = CalibrationPool() if (POOL_CALIBRATION and candidates > 0) else None
    active = list(pipes)
    while active:
        for conn in wait(active):
            transition, obs, rows = conn.recv()
            if pool is not None:
                pool.add(pipes.index(conn), rows)
            if transition is not None and model.num_timesteps < steps:
                o, o1, act, r, term, trunc, info = transition
                done = term or trunc
                info = dict(info, **({'TimeLimit.truncated': True} if trunc and not term else {}))
                model.replay_buffer.add(o[None], o1[None], act[None], np.array([r]), np.array([done]), [info])
                model.num_timesteps += 1; order.append(pipes.index(conn))
                b = (info['model']['diag'] or {}).get('entry_bias')
                if b is not None:                      # latest known outcome-model bias of each environment
                    bias_by_worker[pipes.index(conn)] = b
                    model.entry_bias_now = float(np.mean(list(bias_by_worker.values())))
                if done:
                    episodes.append({'steps': int(model.num_timesteps), 'worker': pipes.index(conn), **{k: info.get(k) for k in ('event', 'time_s', 'delta_v', 'mission')},
                                     'entry_bias': (info['model']['diag'] or {}).get('entry_bias')})
                if model.num_timesteps >= model.learning_starts:
                    model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)
                while marks and model.num_timesteps >= marks[0]:
                    m = marks.pop(0); model.save(out / 'checkpoints' / f'step_{m}')
                    (out / 'heartbeat.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'wall_s': time.time() - t0, 'episodes': len(episodes),
                                                                    'updates': int(model._n_updates), 'replay': model.replay_buffer.model_counts}), encoding='utf-8')
                    # logs at every checkpoint (a stopped run keeps them; T2.3 seed 1 lost its logs)
                    (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
                    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
            if model.num_timesteps >= steps:
                conn.send('stop'); active.remove(conn); continue
            if model.num_timesteps < model.learning_starts:
                action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                action = model.predict(obs, deterministic=False)[0]
            payload, w_ = None, pipes.index(conn)
            if candidates > 0 and entry_stages >= 2:     # refresh the environment's policy copy (stage 2 of the labels)
                if w_ not in synced:
                    payload = ('actor', copy.deepcopy(model.actor).cpu())
                elif model._n_updates - synced[w_] >= SYNC_UPDATES:
                    payload = ('state', {k: v.detach().clone() for k, v in model.actor.state_dict().items()})
                if payload is not None:
                    synced[w_] = model._n_updates
            conn.send((action, payload, pool.pending(w_) if pool is not None else None))
    for p in procs:
        p.join(timeout=60)
    model.save(out / 'final')
    (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
    (out / 'finished.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'updates': int(model._n_updates), 'wall_s': time.time() - t0,
                                                   'worker_order': order, 'replay': model.replay_buffer.model_counts}), encoding='utf-8')
