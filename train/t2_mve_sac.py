"""T2 learner: SAC on the 16 s stage decision, with an optional stage-scale model value expansion (MVE).

Real part (both arms): SB3 SAC (twin critic, fixed entropy coefficient, DEAD_ENDS D02) on the real
stage transitions; success / unsafe / timeout / resource exhaustion are terminal (no bootstrap).

Model part (``e_mve`` only), per sampled real transition s with candidates (a'_k, r^_k, s^'_k, w_k)
from ``T2StageEnv`` (local G* plan of a'_k, corrected by the environment's execution calibration;
usable only when the executed action's calibrated prediction matched the truth at the stage end):

    y_k = r^_k + gamma * ( min_i Q'_i(s^'_k, a~) - alpha log pi(a~ | s^'_k) ),   a~ ~ pi(. | s^'_k)
    L_model = sum_k w_k * 0.5 * sum_i (Q_i(s, a'_k) - y_k)^2 / sum_k w_k
    L_Q = L_real + lambda_eff * L_model,   lambda_eff = min(lambda, c * |grad L_real| / |grad L_model|)

Entry-level candidates (T2.2) are terminal: y_k = r^_k (the stage part up to the predicted entry plus
the calibrated entry outcome, no bootstrap); all others bootstrap as above. Every candidate lies
within 0.5 (action units) of the executed action. The term is
off until ``model_start`` real transitions are stored. lambda = 1, c = 0.5 (the model gradient is at
most half of the real TD gradient on the critic parameters).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update

from env.t2_stage import GAMMA_STAGE

K_CANDIDATES = 8          # 3 structured alternatives + 1 global + 4 local (T2StageEnv.candidate_actions)
MODEL_LAMBDA = 1.0
GRADIENT_CAP = 0.5
MODEL_START = 300
ENT_COEF = 0.05
#: critic / actor updates per stored real transition (gradient_steps = UPDATES_PER_TRANSITION * n_envs);
#: B ran 0.8 (4 per 5 transitions) and was still rising at 3000 decisions, T2.1 hard-coded 12 per step
UPDATES_PER_TRANSITION = 2


class StageReplay(ReplayBuffer):
    def __init__(self, *args, k_candidates: int = K_CANDIDATES, **kwargs):
        super().__init__(*args, **kwargs)
        k, d, a = k_candidates, self.obs_shape[0], self.action_dim
        self.k = k
        self.cand_a = np.zeros((self.buffer_size, self.n_envs, k, a), np.float32)
        self.cand_r = np.zeros((self.buffer_size, self.n_envs, k), np.float32)
        self.cand_obs = np.zeros((self.buffer_size, self.n_envs, k, d), np.float32)
        self.cand_w = np.zeros((self.buffer_size, self.n_envs, k), np.float32)
        self.cand_done = np.zeros((self.buffer_size, self.n_envs, k), np.float32)
        self.model_counts = {'transitions': 0, 'with_candidates': 0, 'candidates': 0, 'entry_candidates': 0, 'gate': {}}
        self._last = None

    def add(self, obs, next_obs, action, reward, done, infos):
        i = self.pos
        self.cand_w[i] = 0.0; self.cand_done[i] = 0.0
        for e, info in enumerate(infos):
            cands = (info.get('stage_model') or [])[:self.k]
            self.model_counts['transitions'] += 1
            reason = (info.get('gate') or {}).get('reason', 'none')
            self.model_counts['gate'][reason] = self.model_counts['gate'].get(reason, 0) + 1
            self.model_counts['with_candidates'] += int(bool(cands))
            self.model_counts['candidates'] += len(cands)
            for j, c in enumerate(cands):
                self.cand_a[i, e, j] = c['a']; self.cand_r[i, e, j] = c['reward']
                self.cand_obs[i, e, j] = c['next_obs']; self.cand_w[i, e, j] = c['weight']
                self.cand_done[i, e, j] = float(c.get('done', False)); self.model_counts['entry_candidates'] += int(c.get('done', False))
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

    def candidates(self):
        i, e = self._last
        return tuple(map(self.to_torch, (self.cand_a[i, e], self.cand_r[i, e], self.cand_obs[i, e], self.cand_w[i, e], self.cand_done[i, e])))


class T2SAC(SAC):
    def __init__(self, *args, model_lambda: float = 0.0, gradient_cap: float = GRADIENT_CAP, model_start: int = MODEL_START, **kwargs):
        self.model_lambda, self.gradient_cap, self.model_start = float(model_lambda), float(gradient_cap), int(model_start)
        self.model_stats = []
        kwargs.setdefault('replay_buffer_class', StageReplay)
        kwargs.setdefault('ent_coef', ENT_COEF)   # fixed (DEAD_ENDS D02)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'model_stats']

    def _model_loss(self, data, ent_coef):
        cand_a, cand_r, cand_obs, cand_w, cand_done = self.replay_buffer.candidates()
        mask = cand_w > 0
        if not bool(mask.any()):
            return None, 0
        b, k = cand_w.shape
        s = data.observations[:, None, :].expand(-1, k, -1)[mask]; a = cand_a[mask]; r = cand_r[mask].reshape(-1, 1)
        s1 = cand_obs[mask]; w = cand_w[mask].reshape(-1, 1); d = cand_done[mask].reshape(-1, 1)
        with th.no_grad():
            a1, logp1 = self.actor.action_log_prob(s1)
            q1 = th.min(th.cat(self.critic_target(s1, a1), dim=1), dim=1, keepdim=True).values - ent_coef * logp1.reshape(-1, 1)
            y = r + (1.0 - d) * self.gamma * q1
        qs = self.critic(s, a)
        loss = 0.5 * sum((w * (q - y) ** 2).sum() for q in qs) / w.sum().clamp_min(1e-6)
        return loss, int(mask.sum())

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        params = list(self.critic.parameters())
        norm = lambda gs: float(th.sqrt(sum((g * g).sum() for g in gs if g is not None)))
        ent_coef = self.ent_coef_tensor
        stored = (self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos) * self.replay_buffer.n_envs
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actions_pi, log_prob = self.actor.action_log_prob(data.observations); log_prob = log_prob.reshape(-1, 1)
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                nq = th.min(th.cat(self.critic_target(data.next_observations, next_actions), dim=1), dim=1, keepdim=True).values
                target = data.rewards + (1 - data.dones) * self.gamma * (nq - ent_coef * next_log_prob.reshape(-1, 1))
            current = self.critic(data.observations, data.actions)
            real = 0.5 * sum(F.mse_loss(q, target) for q in current)
            loss = real; applied = g_real = g_model = 0.; used = 0; mloss = None
            if self.model_lambda > 0 and stored >= self.model_start:
                mloss, used = self._model_loss(data, ent_coef)
                if mloss is not None:
                    g_real = norm(th.autograd.grad(real, params, retain_graph=True, allow_unused=True))
                    g_model = norm(th.autograd.grad(mloss, params, retain_graph=True, allow_unused=True))
                    applied = min(self.model_lambda, self.gradient_cap * g_real / max(g_model, 1e-12))
                    loss = real + applied * mloss
            self.critic.optimizer.zero_grad(); loss.backward(); self.critic.optimizer.step()
            q_pi = th.min(th.cat(self.critic(data.observations, actions_pi), dim=1), dim=1, keepdim=True).values
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            if (self._n_updates + step) % 25 == 0:
                self.model_stats.append({'steps': int(self.num_timesteps), 'update': int(self._n_updates + step), 'real_td': float(real.item()),
                                         'model_td': float(mloss.item()) if mloss is not None else None, 'model_rows': used,
                                         'lambda_eff': applied, 'g_real': g_real, 'g_model': g_model, 'actor_loss': float(actor_loss.item()),
                                         'q_pi_mean': float(q_pi.mean().item())})
        self._n_updates += gradient_steps


class Checkpoints(BaseCallback):
    def __init__(self, out: Path, marks: list[int]):
        super().__init__(); self.out, self.marks = out, sorted(marks); self.episodes = []; self.t0 = time.time()

    def _on_training_start(self) -> None:
        self.model.save(self.out / 'checkpoints' / 'step_0')

    def _on_step(self) -> bool:
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self.episodes.append({'steps': int(self.num_timesteps), 'event': info.get('event'), 'time_s': info.get('time_s'),
                                      'delta_v': info.get('delta_v'), 'mission': info.get('mission')})
        while self.marks and self.num_timesteps >= self.marks[0]:
            m = self.marks.pop(0); self.model.save(self.out / 'checkpoints' / f'step_{m}')
            (self.out / 'heartbeat.json').write_text(json.dumps({'steps': int(self.num_timesteps), 'wall_s': time.time() - self.t0,
                                                                 'episodes': len(self.episodes), 'replay': self.model.replay_buffer.model_counts}), encoding='utf-8')
        return True


def make_model(env, arm: str, seed: int, smoke: bool = False, gradient_cap: float = GRADIENT_CAP,
               updates_per_transition: float = UPDATES_PER_TRANSITION) -> T2SAC:
    n_envs = getattr(env, 'num_envs', 1)
    return T2SAC('MlpPolicy', env, seed=seed, device='cpu', verbose=0, policy_kwargs={'net_arch': [256, 256]},
                 gamma=GAMMA_STAGE, learning_rate=3e-4, buffer_size=100_000, batch_size=64 if smoke else 256,
                 learning_starts=16 if smoke else 200, train_freq=(1, 'step'), gradient_steps=max(1, int(round(updates_per_transition * n_envs))), ent_coef=ENT_COEF, tau=0.005,
                 model_lambda=MODEL_LAMBDA if arm == 'e_mve' else 0.0, model_start=16 if smoke else MODEL_START, gradient_cap=gradient_cap)


def train(out: Path, arm: str, seed: int, steps: int, n_envs: int, marks: list[int], smoke: bool = False,
          gradient_cap: float = GRADIENT_CAP, updates_per_transition: float = UPDATES_PER_TRANSITION, entry_stages: int | None = None) -> None:
    from env.t2_stage import ENTRY_STAGES
    entry_stages = ENTRY_STAGES if entry_stages is None else int(entry_stages)
    from functools import partial
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from env.t2_stage import T2StageEnv
    out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    th.set_num_threads(1)
    def make(rank):
        return T2StageEnv(seed_base=1_000_000 * (seed + 1) + rank * 100_000, candidates=K_CANDIDATES, shaping=True, seed=seed * 100 + rank,
                          entry_stages=entry_stages)
    env = SubprocVecEnv([partial(make, k) for k in range(n_envs)], start_method='spawn')
    model = make_model(env, arm, seed, smoke, gradient_cap=gradient_cap, updates_per_transition=updates_per_transition)
    (out / 'manifest.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'steps': steps, 'envs': n_envs, 'git_head': head, 'smoke': smoke,
                                                   'model_lambda': model.model_lambda, 'gradient_cap': model.gradient_cap, 'model_start': model.model_start,
                                                   'ent_coef': ENT_COEF, 'k_candidates': K_CANDIDATES, 'candidates_computed_both_arms': True,
                                                   'gradient_steps': model.gradient_steps, 'updates_per_transition': updates_per_transition, 'entry_stages': entry_stages,
                                                   'env_version': __import__('env.t2_stage', fromlist=['T2_VERSION']).T2_VERSION}, indent=1), encoding='utf-8')
    cb = Checkpoints(out, [m for m in marks if m <= steps] + [steps])
    model.learn(total_timesteps=steps, callback=cb)
    model.save(out / 'final')
    (out / 'episodes.json').write_text(json.dumps(cb.episodes), encoding='utf-8')
    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
    (out / 'finished.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'updates': int(model._n_updates), 'replay': model.replay_buffer.model_counts}), encoding='utf-8')
    env.close()
