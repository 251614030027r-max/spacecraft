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
        self.cand_bias = np.zeros((*shape, k), np.float32)     # systematic error of the label's outcome model (entry labels)
        self.created = np.zeros(shape, np.float32)             # stored transitions at label time (label age diagnostic)
        self.stored = 0
        self.model_counts = {'transitions': 0, 'with_candidates': 0, 'candidates': 0, 'entry_candidates': 0, 'reason': {}}
        self._last = None

    def add(self, obs, next_obs, action, reward, done, infos):
        i = self.pos
        self.fact_valid[i] = 0.0; self.cand_valid[i] = 0.0; self.cand_done[i] = 0.0; self.cand_bias[i] = 0.0
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
                self.cand_done[i, e, j] = float(x['done']); self.cand_valid[i, e, j] = 1.0; self.cand_bias[i, e, j] = float(x.get('bias', 0.0))
                c['entry_candidates'] += int(x['done'])
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
                ('fact_r', 'fact_obs', 'fact_done', 'fact_valid', 'cand_a', 'cand_r', 'cand_obs', 'cand_done', 'cand_valid', 'cand_bias', 'created')}


class T2SAC(SAC):
    def __init__(self, *args, model_beta: float = 0.0, model_start: int = MODEL_START, **kwargs):
        self.model_beta, self.model_start = float(model_beta), int(model_start)
        self.model_stats = []
        self.td_ms = None          # running mean of the squared real TD residual (sigma^2)
        kwargs.setdefault('replay_buffer_class', StageReplay)
        kwargs.setdefault('ent_coef', ENT_COEF)   # fixed (DEAD_ENDS D02)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'model_stats']

    def _soft_value(self, obs, ent_coef):
        a, logp = self.actor.action_log_prob(obs)
        return th.min(th.cat(self.critic_target(obs, a), dim=1), dim=1, keepdim=True).values - ent_coef * logp.reshape(-1, 1)

    def _model_loss(self, data, target, ent_coef):
        """M3 robust precision fusion of the anchored candidate labels (module docstring)."""
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
            s1 = m['cand_obs'][mask]
            y = m['cand_r'][mask].reshape(-1, 1) + (1.0 - m['cand_done'][mask].reshape(-1, 1)) * self.gamma * self._soft_value(s1, ent_coef)
            # candidate-level reliability (T2.4): w_k = 1 / (1 + (eps_k / sigma)^2) with the candidate's error estimate
            # eps_k = |e_a| + |b_k|: the factual model error (transferable part, executed intent) plus the label's own
            # outcome-model bias (entry labels: the entry outcome model's predict-then-observe bias on this environment's
            # executed entries, at label time; stage labels: 0). No distance term: the T2.3 audit found no dependence of
            # the label error on the candidate's distance from the executed intent.
            eps = err.abs().reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1) + m['cand_bias'][mask].reshape(-1, 1).abs()
            wk = 1.0 / (1.0 + eps ** 2 / self.td_ms)
            entry = m['cand_done'][mask].reshape(-1, 1) > 0
            rel = wk / w.reshape(-1, 1).expand(-1, k)[mask].reshape(-1, 1).clamp_min(1e-12)   # reported only
        s = data.observations[:, None, :].expand(-1, k, -1)[mask]
        qs = self.critic(s, m['cand_a'][mask])
        loss = 0.5 * sum((wk * (q - y) ** 2).sum() for q in qs) / max(int(mask.sum()), 1)
        e = err[rows]
        return loss, {'model_rows': int(mask.sum()), 'w_mean': float(wk.mean()), 'abs_err_median': float(e.abs().median()),
                      'w_err_max': float((w[rows] * e.abs()).max()), 'entry_share': float(entry.float().mean()),
                      'entry_rel_mean': float(rel[entry].mean()) if bool(entry.any()) else None,
                      'entry_bias_mean': float(m['cand_bias'][mask].reshape(-1, 1)[entry].mean()) if bool(entry.any()) else None,
                      'label_age_mean': float(self.replay_buffer.stored - m['created'][rows].mean())}

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
    cands = [{k: c.get(k, 0.0) for k in ('a', 'reward', 'next_obs', 'done', 'bias')} for c in model.get('cands') or []]
    diag = model.get('diag') or {}
    return {'model': {'fact': model.get('fact'), 'cands': cands, 'diag': {k: diag.get(k) for k in ('reason', 'entry_bias', 'entry_records')}},
            **{k: info.get(k) for k in ('event', 'time_s', 'delta_v', 'mission')}}


def worker(conn, rank: int, seed: int, candidates: int, entry_stages: int, window: bool) -> None:
    """One environment: send (transition or None, observation to act on), receive the action; until 'stop'."""
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    th.set_num_threads(1)
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(seed_base=1_000_000 * (seed + 1) + rank * 100_000, candidates=candidates, shaping=True, seed=seed * 100 + rank,
                     entry_stages=entry_stages, window=window)
    obs, _ = env.reset(); transition = None
    while True:
        conn.send((transition, obs))
        action = conn.recv()
        if isinstance(action, str):
            break
        next_obs, reward, term, trunc, info = env.step(action)
        transition = (obs, next_obs, np.asarray(action, np.float32), float(reward), bool(term), bool(trunc), _slim(info))
        obs = env.reset()[0] if (term or trunc) else next_obs
    conn.close()


def train(out: Path, arm: str, seed: int, steps: int, workers: int, marks: list[int], smoke: bool = False,
          updates_per_transition: float = UPDATES_PER_TRANSITION, entry_stages: int | None = None, window: bool = True,
          model_beta: float = MODEL_BETA) -> None:
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
                                                   'gradient_steps_per_transition': model.gradient_steps, 'entry_stages': entry_stages,
                                                   'env_version': T2_VERSION}, indent=1), encoding='utf-8')
    ctx = mp.get_context('spawn')
    pipes, procs = [], []
    for rank in range(workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, rank, seed, candidates, entry_stages, window), daemon=True)
        p.start(); pipes.append(a); procs.append(p)
    marks = sorted(set([m for m in marks if m <= steps] + [steps]))
    model.save(out / 'checkpoints' / 'step_0')
    episodes, order, t0 = [], [], time.time()
    rng = np.random.default_rng(seed)
    active = list(pipes)
    while active:
        for conn in wait(active):
            transition, obs = conn.recv()
            if transition is not None and model.num_timesteps < steps:
                o, o1, act, r, term, trunc, info = transition
                done = term or trunc
                info = dict(info, **({'TimeLimit.truncated': True} if trunc and not term else {}))
                model.replay_buffer.add(o[None], o1[None], act[None], np.array([r]), np.array([done]), [info])
                model.num_timesteps += 1; order.append(pipes.index(conn))
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
            conn.send(action)
    for p in procs:
        p.join(timeout=60)
    model.save(out / 'final')
    (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
    (out / 'finished.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'updates': int(model._n_updates), 'wall_s': time.time() - t0,
                                                   'worker_order': order, 'replay': model.replay_buffer.model_counts}), encoding='utf-8')
