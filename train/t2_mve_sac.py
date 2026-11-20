"""Learner of the ECC method (``docs/paper/facts/06``, ``docs/final_method/ROUND2.md``): standard SAC (and the TD3
baseline) on the 16 s stage decision. Actor and critics read the same 17-value observation.

Algorithms (``make_model``):
* ``sac`` -- SAC: twin critic, fixed entropy coefficient ENT_COEF (automatic temperature drove critic overestimation,
  DEAD_ENDS D02), [256, 256], learning rate 3e-4, replay 1e5, batch 256, 200 uniform random decisions before learning,
  tau 0.005, gamma = GAMMA_STAGE per decision, UPDATES_PER_TRANSITION updates per stored transition.
* ``td3`` -- SB3 TD3 with matched settings (target smoothing 0.2 / 0.5, policy delay 2, Gaussian collection noise 0.1).

Success / unsafe / timeout / resource exhaustion are terminal (no bootstrap); a numerical failure is a truncation. The
stored action is the policy's proposal; reward and successor come from the (coordinated) execution.

Collection (``train``): asynchronous -- each worker process steps its own environment and the learner answers every
action request with the current policy. With more than one worker the arrival order of transitions depends on the
workers' timing (recorded as ``worker_order``), so multi-worker runs are not bitwise repeatable; single-worker runs are.
Every run writes TensorBoard logs (``<run>/tb``; ``docs/final_method/ROUND2.md`` section 1.5).
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
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update

from env.t2_stage import GAMMA_STAGE, OBS_DIM

ENT_COEF = 0.05
SIGMA_DECAY = 0.99        # running mean of the squared real TD residual (diagnostic)
#: critic / actor updates per stored real transition
UPDATES_PER_TRANSITION = 2
LEARNER_VERSION = "t2-learner-9.1"  # 9.1: value arbitration requests and execution-equivalence experience (FEEDBACK_DEV)
ALGOS = ('sac', 'td3')
#: episodes in the TensorBoard moving capture rate
TB_WINDOW = 100
#: decisions between progress records (heartbeat, episodes.json)
HEARTBEAT_EVERY = 250


class StageReplay(ReplayBuffer):
    """The SB3 replay buffer with a transition counter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_counts = {'transitions': 0}

    def add(self, obs, next_obs, action, reward, done, infos):
        self.model_counts['transitions'] += len(infos)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (self._normalize_obs(self.observations[batch_inds, env_indices, :], env), self.actions[batch_inds, env_indices, :], next_obs,
                (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env))
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class T2SAC(SAC):
    """SB3 SAC with a fixed entropy coefficient and a read-only training record (``model_stats``)."""

    def __init__(self, *args, **kwargs):
        self.model_stats = []
        self.td_ms = None          # running mean of the squared real TD residual (sigma^2)
        #: execution-equivalence experience (``docs/final_method/FEEDBACK_DEV.md`` 2.2): critic regression only
        self.eq_buffer = None
        kwargs.setdefault('replay_buffer_class', StageReplay)
        kwargs.setdefault('ent_coef', ENT_COEF)   # fixed (DEAD_ENDS D02)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'model_stats', 'eq_buffer']

    def enable_equivalence(self) -> None:
        self.eq_buffer = StageReplay(self.buffer_size, self.observation_space, self.action_space, device=self.device,
                                     n_envs=1, optimize_memory_usage=False)

    def equivalence_batch(self, batch_size: int) -> int:
        """k = min(batch, round(batch * N_eq / N_real)): equivalence samples added to a critic batch."""
        if self.eq_buffer is None or self.eq_buffer.size() == 0:
            return 0
        n_real = max(self.replay_buffer.model_counts['transitions'], 1)
        return int(min(batch_size, round(batch_size * self.eq_buffer.model_counts['transitions'] / n_real)))

    def _soft_value(self, obs, ent_coef):
        a, logp = self.actor.action_log_prob(obs)
        return th.min(th.cat(self.critic_target(obs, a), dim=1), dim=1, keepdim=True).values - ent_coef * logp.reshape(-1, 1)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        ent_coef = self.ent_coef_tensor
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actions_pi, log_prob = self.actor.action_log_prob(data.observations); log_prob = log_prob.reshape(-1, 1)
            obs_c, act_c, next_c, rew_c, done_c = data.observations, data.actions, data.next_observations, data.rewards, data.dones
            k = self.equivalence_batch(batch_size)
            if k:                                  # the critic also regresses on execution-equivalent proposals
                eq = self.eq_buffer.sample(k, env=self._vec_normalize_env)
                obs_c, act_c = th.cat([obs_c, eq.observations]), th.cat([act_c, eq.actions])
                next_c, rew_c, done_c = th.cat([next_c, eq.next_observations]), th.cat([rew_c, eq.rewards]), th.cat([done_c, eq.dones])
            with th.no_grad():
                target = rew_c + (1 - done_c) * self.gamma * self._soft_value(next_c, ent_coef)
            current = self.critic(obs_c, act_c)
            real = 0.5 * sum(F.mse_loss(q, target) for q in current)
            with th.no_grad():
                ms = float(sum(((q - target) ** 2).mean() for q in current) / len(current))
                self.td_ms = ms if self.td_ms is None else SIGMA_DECAY * self.td_ms + (1.0 - SIGMA_DECAY) * ms
            self.critic.optimizer.zero_grad(); real.backward(); self.critic.optimizer.step()
            q_pi = th.min(th.cat(self.critic(data.observations, actions_pi), dim=1), dim=1, keepdim=True).values
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            if (self._n_updates + step) % 25 == 0:
                self.model_stats.append({'steps': int(self.num_timesteps), 'update': int(self._n_updates + step), 'real_td': float(real.item()),
                                         'sigma': float(np.sqrt(self.td_ms)), 'actor_loss': float(actor_loss.item()),
                                         'q_pi_mean': float(q_pi.mean().item()), **({'equivalence_batch': k} if k else {})})
        self._n_updates += gradient_steps


class TD3Replay(ReplayBuffer):
    """Real transitions and the read-only counters expected by the collector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_counts = {'transitions': 0}

    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        self.model_counts['transitions'] += len(infos)


class T2TD3(TD3):
    """SB3 TD3 (matched baseline) with explicit Gaussian collection noise (``td3_exploration_action``)."""

    exploration_sigma = 0.1

    def __init__(self, *args, **kwargs):
        self.model_stats = []
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'model_stats']

    def train(self, gradient_steps, batch_size=100):
        super().train(gradient_steps, batch_size)
        values = self.logger.name_to_value
        self.model_stats.append({
            'steps': self.num_timesteps, 'update': self._n_updates,
            'real_td': float(values['train/critic_loss']),
            'actor_loss': float(values['train/actor_loss']) if 'train/actor_loss' in values else None,
        })


def td3_exploration_action(model, obs, noise_rng):
    intent = model.predict(obs, deterministic=True)[0]
    noise = noise_rng.normal(0.0, model.exploration_sigma, size=intent.shape)
    return np.clip(intent + noise, -1.0, 1.0).astype(np.float32)


def make_model(env, algo: str, seed: int, smoke: bool = False, updates_per_transition: float = UPDATES_PER_TRANSITION) -> T2SAC | T2TD3:
    if algo not in ALGOS:
        raise ValueError(f'unknown algorithm {algo}: {ALGOS}')
    common = dict(seed=seed, device='cpu', verbose=0, policy_kwargs={'net_arch': [256, 256]}, gamma=GAMMA_STAGE, tau=0.005,
                  learning_rate=3e-4, buffer_size=100_000, batch_size=64 if smoke else 256, learning_starts=16 if smoke else 200,
                  train_freq=(1, 'step'))
    if algo == 'td3':
        return T2TD3('MlpPolicy', env, gradient_steps=2, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5,
                     replay_buffer_class=TD3Replay, **common)
    return T2SAC('MlpPolicy', env, gradient_steps=max(1, int(round(updates_per_transition))), ent_coef=ENT_COEF, **common)


# ---------------------------------------------------------------------- asynchronous collection
class SpacesEnv(gym.Env):
    """Observation / action spaces of the stage decision, for the learner process (no physics)."""

    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, obs_dim: int = OBS_DIM):
        self.obs_dim = int(obs_dim)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(self.obs_dim, np.float32), {}

    def step(self, action):
        return np.zeros(self.obs_dim, np.float32), 0.0, True, False, {}


def _slim(info: dict) -> dict:
    """What the learner process keeps from a step's info (with the execution-equivalent grid intents, if any)."""
    out = {k: info[k] for k in ('event', 'time_s', 'delta_v', 'mission', 'true_reward', 'exec_stats', 'min_margins', 'terminal')
           if k in info}
    equivalent = (info.get('coord') or {}).get('equivalent')
    if equivalent:
        out['equivalent'] = equivalent
    return out


def critic_ranker(model):
    """ranker(observation, candidates) -> min(Q1, Q2) of each candidate under the model's current critic
    (``docs/final_method/FEEDBACK_DEV.md`` 2.1)."""
    def rank(obs, cands):
        cands = np.asarray(cands, np.float32).reshape(-1, 2)
        o = th.as_tensor(np.repeat(np.asarray(obs, np.float32).reshape(1, -1), len(cands), axis=0))
        with th.no_grad():
            q = th.min(th.cat(model.critic(o, th.as_tensor(cands)), dim=1), dim=1).values
        return q.numpy().astype(np.float64).tolist()
    return rank


def worker(conn, rank: int, seed: int, env_kwargs: dict, audit_logging: bool = False) -> None:
    """One environment: send (transition or None, observation to act on), receive the action; until 'stop'.
    ``env_kwargs``: geometry, window, coordination, correction, long_entry, potential of ``T2StageEnv``."""
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    th.set_num_threads(1)
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(seed_base=1_000_000 * (seed + 1) + rank * 100_000, shaping=True, **env_kwargs)
    env.audit_logging = audit_logging
    if env.arbitration == 'value':                 # the learner's current critic ranks the executor's executable set
        def ranker(obs, cands):
            conn.send(('rank', np.asarray(obs, np.float32), np.asarray(cands, np.float32)))
            return conn.recv()
        env.ranker = ranker
    obs, _ = env.reset(); transition = None
    while True:
        conn.send((transition, obs))
        action = conn.recv()
        if isinstance(action, str):
            break
        next_obs, reward, term, trunc, info = env.step(action)
        if term or trunc:                         # the episode's executor statistics (logging only)
            st = env.stats or {}
            info['exec_stats'] = {**{k: st.get(k, 0) for k in ('stages', 'holds', 'rejected', 'bridge_attempts', 'rate_limited',
                                                               'entries_lost', 'corrections', 'long_entries', 'value_arbitrations',
                                                               'arbitration_changed', 'equivalent_samples')},
                                  'window_recoveries': int(getattr(env.generator.bridge, 'recoveries', 0)) if env.generator is not None else 0}
        transition = (obs, next_obs, np.asarray(action, np.float32), float(reward), bool(term), bool(trunc), _slim(info))
        obs = env.reset()[0] if (term or trunc) else next_obs
    conn.close()


#: TensorBoard outcome code of an episode
OUTCOME_CODE = {'success': 1, 'timeout': 0, 'resource_exhausted': -1, 'unsafe': -2, 'numerical_failure': -3}


class TrainingLog:
    """TensorBoard record of a run (``<run>/tb``), x = stored transitions (decisions). Writing it consumes no random
    number and does not change the run."""

    def __init__(self, path: Path):
        from collections import deque
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=str(path))
        self.recent = deque(maxlen=TB_WINDOW)
        self.updates_logged = 0

    def episode(self, step: int, ep: dict) -> None:
        w, st = self.writer, ep.get('exec_stats') or {}
        captured = float(ep.get('event') == 'success')
        self.recent.append(captured)
        w.add_scalar('episode/capture', captured, step)
        w.add_scalar('episode/outcome', OUTCOME_CODE.get(ep.get('event'), -4), step)
        w.add_scalar('episode/return_true', float(ep['return_true']), step)
        w.add_scalar('episode/return_shaped', float(ep['return_shaped']), step)
        w.add_scalar('episode/time_s', float(ep.get('time_s') or 0.0), step)
        w.add_scalar('episode/delta_v', float(ep.get('delta_v') or 0.0), step)
        w.add_scalar('episode/corrections', float(st.get('corrections', 0)), step)
        w.add_scalar('episode/long_entries', float(st.get('long_entries', 0)), step)
        w.add_scalar('episode/equivalent_samples', float(st.get('equivalent_samples', 0)), step)
        w.add_scalar('episode/value_arbitrations', float(st.get('value_arbitrations', 0)), step)
        w.add_scalar('episode/arbitration_changed', float(st.get('arbitration_changed', 0)), step)
        w.add_scalar(f'rolling/capture_rate_{TB_WINDOW}', sum(self.recent) / len(self.recent), step)

    def updates(self, step: int, stats: list) -> None:
        for row in stats[self.updates_logged:]:
            for key in ('real_td', 'actor_loss', 'q_pi_mean', 'equivalence_batch'):
                if row.get(key) is not None:
                    self.writer.add_scalar(f'train/{key}', float(row[key]), step)
        self.updates_logged = len(stats)

    def close(self) -> None:
        self.writer.flush(); self.writer.close()


def train(out: Path, algo: str, seed: int, steps: int, workers: int, env_kwargs: dict, smoke: bool = False,
          updates_per_transition: float = UPDATES_PER_TRANSITION, manifest_extra: dict | None = None,
          audit_logging: bool = False) -> None:
    import multiprocessing as mp
    from multiprocessing.connection import wait
    from stable_baselines3.common.logger import configure
    from env.t2_stage import T2_VERSION, T2Mission
    out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    th.set_num_threads(1)
    model = make_model(SpacesEnv(OBS_DIM), algo, seed, smoke, updates_per_transition=updates_per_transition)
    model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
    algo_manifest = {'algorithm': algo.upper(), 'net_arch': [256, 256], 'learning_rate': model.learning_rate,
                     'buffer_size': model.buffer_size, 'batch_size': model.batch_size, 'learning_starts': model.learning_starts,
                     'gamma': model.gamma, 'tau': model.tau, 'gradient_steps_per_transition': model.gradient_steps}
    if algo == 'td3':
        algo_manifest.update(policy_delay=model.policy_delay, target_policy_noise=model.target_policy_noise,
                             target_noise_clip=model.target_noise_clip, exploration_sigma=model.exploration_sigma)
    else:
        algo_manifest['ent_coef'] = ENT_COEF
    mission = T2Mission()
    (out / 'manifest.json').write_text(json.dumps({'algo': algo, 'seed': seed, 'steps': steps, 'workers': workers, 'git_head': head,
                                                   'smoke': smoke, 'collection': 'asynchronous', **env_kwargs,
                                                   'mission': {'deadline_s': list(mission.deadline_s), 'budget_m_s': mission.budget_m_s,
                                                               'time_cost_per_s': mission.time_cost_per_s,
                                                               'fuel_cost_per_m_s': mission.fuel_cost_per_m_s},
                                                   'obs_dim': OBS_DIM, 'env_version': T2_VERSION, 'learner_version': LEARNER_VERSION,
                                                   **algo_manifest, **(manifest_extra or {})}, indent=1), encoding='utf-8')
    log = TrainingLog(out / 'tb')
    ranker = critic_ranker(model)
    if env_kwargs.get('equivalence'):
        if algo != 'sac':
            raise ValueError('execution-equivalence experience is defined for SAC only')
        model.enable_equivalence()
    ctx = mp.get_context('spawn')
    pipes, procs = [], []
    for rank in range(workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, rank, seed, dict(env_kwargs), audit_logging), daemon=True)
        p.start(); pipes.append(a); procs.append(p)
    model.save(out / 'checkpoints' / 'step_0')
    episodes, order, t0 = [], [], time.time()
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng([seed, 3003]) if algo == 'td3' else None
    ret_true, ret_shaped = {}, {}                 # each worker's current episode return (true task reward / shaped training reward)

    def progress():
        (out / 'heartbeat.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'wall_s': time.time() - t0, 'episodes': len(episodes),
                                                        'updates': int(model._n_updates), 'replay': model.replay_buffer.model_counts}), encoding='utf-8')
        (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
        (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')

    active = list(pipes)
    while active:
        for conn in wait(active):
            msg = conn.recv()
            if isinstance(msg, tuple) and len(msg) == 3 and isinstance(msg[0], str) and msg[0] == 'rank':
                conn.send(ranker(msg[1], msg[2]))          # value arbitration inside the worker's step
                continue
            transition, obs = msg
            w_ = pipes.index(conn)
            if transition is not None and model.num_timesteps < steps:
                o, o1, act, r, term, trunc, info = transition
                done = term or trunc
                info = dict(info, **({'TimeLimit.truncated': True} if trunc and not term else {}))
                model.replay_buffer.add(o[None], o1[None], act[None], np.array([r]), np.array([done]), [info])
                for b in info.get('equivalent') or []:  # execution-equivalence experience: the same real transition
                    model.eq_buffer.add(o[None], o1[None], np.asarray(b, np.float32)[None], np.array([r]), np.array([done]), [info])
                model.num_timesteps += 1; order.append(w_)
                ret_true[w_] = ret_true.get(w_, 0.0) + float(info.get('true_reward') or 0.0); ret_shaped[w_] = ret_shaped.get(w_, 0.0) + float(r)
                if done:
                    episodes.append({'steps': int(model.num_timesteps), 'worker': w_, **{k: info.get(k) for k in ('event', 'time_s', 'delta_v', 'mission')},
                                     'return_true': ret_true.pop(w_, 0.0), 'return_shaped': ret_shaped.pop(w_, 0.0),
                                     **({'exec_stats': info['exec_stats']} if info.get('exec_stats') is not None else {}),
                                     **{k: info[k] for k in ('min_margins', 'terminal') if k in info}})
                    log.episode(int(model.num_timesteps), episodes[-1])
                if model.num_timesteps >= model.learning_starts:
                    model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)
                    log.updates(int(model.num_timesteps), model.model_stats)
                if model.num_timesteps % HEARTBEAT_EVERY == 0:
                    progress()
            if model.num_timesteps >= steps:
                conn.send('stop'); active.remove(conn); continue
            if model.num_timesteps < model.learning_starts:
                action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            elif noise_rng is not None:
                action = td3_exploration_action(model, obs, noise_rng)
            else:
                action = model.predict(obs, deterministic=False)[0]
            conn.send(action)
    for p in procs:
        p.join(timeout=60)
    model.save(out / 'final')
    model.save(out / 'checkpoints' / f'step_{int(model.num_timesteps)}')
    progress()
    log.close()
    (out / 'finished.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'updates': int(model._n_updates), 'wall_s': time.time() - t0,
                                                   'worker_order': order, 'replay': model.replay_buffer.model_counts,
                                                   **({'equivalence_samples': model.eq_buffer.model_counts['transitions']}
                                                      if getattr(model, 'eq_buffer', None) is not None else {})}), encoding='utf-8')
