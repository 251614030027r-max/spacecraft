"""Learner of the ECC method (``docs/paper/METHOD.md``): SAC (and the TD3 baseline) on the 16 s stage decision.

Arms (``make_model``):
* ``e_priv`` -- asymmetric critic: the actor reads the policy view (first OBS_DIM values of the ``cut_priv``
  observation), the critic reads the policy view and the executor feedback z_E (OBS_DIM_PRIV values). With the
  environment's certification on, this is ECC (``ECC_CERT_LOOP``); without it, "ECC w/o certification" (``EACH_C``).
* ``e_sac`` -- SAC whose critic reads the policy view only (``SAC_C``; with certification ``SAC_CERT_LOOP``).
* ``e_td3`` -- SB3 TD3 on the policy view, matched settings (``TD3_C``).

SAC: twin critic, fixed entropy coefficient ENT_COEF (automatic temperature drove critic overestimation, DEAD_ENDS
D02), [256, 256], learning rate 3e-4, replay 1e5, batch 256, 200 uniform random decisions before learning, tau 0.005,
gamma = GAMMA_STAGE per decision, UPDATES_PER_TRANSITION updates per stored transition. Success / unsafe / timeout /
resource exhaustion are terminal (no bootstrap); a numerical failure is a truncation. The stored action is the
policy's proposal; reward and successor come from the (certified) execution.

Collection (``train``): asynchronous -- each worker process steps its own environment and the learner answers every
action request with the current policy. With more than one worker the arrival order of transitions depends on the
workers' timing (recorded as ``worker_order``), so multi-worker runs are not bitwise repeatable; single-worker runs are.
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

from env.t2_stage import GAMMA_STAGE, OBS_DIM, OBS_DIM_PRIV, OBS_PROFILE

ENT_COEF = 0.05
SIGMA_DECAY = 0.99        # running mean of the squared real TD residual (diagnostic)
#: critic / actor updates per stored real transition
UPDATES_PER_TRANSITION = 2
LEARNER_VERSION = "t2-learner-8.0"
#: arms and whether the critic reads the executor feedback view
ARMS = {'e_priv': True, 'e_sac': False, 'e_td3': False}


class StageReplay(ReplayBuffer):
    """The SB3 replay buffer with a transition counter (``k_candidates`` is accepted for loading older checkpoints)."""

    def __init__(self, *args, k_candidates: int = 0, **kwargs):
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
        kwargs.setdefault('replay_buffer_class', StageReplay)
        kwargs.setdefault('ent_coef', ENT_COEF)   # fixed (DEAD_ENDS D02)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'model_stats']

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
            with th.no_grad():
                target = data.rewards + (1 - data.dones) * self.gamma * self._soft_value(data.next_observations, ent_coef)
            current = self.critic(data.observations, data.actions)
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
                                         'q_pi_mean': float(q_pi.mean().item())})
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


def make_model(env, arm: str, seed: int, smoke: bool = False, updates_per_transition: float = UPDATES_PER_TRANSITION) -> T2SAC | T2TD3:
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm}: {sorted(ARMS)}')
    obs_dim = int(env.observation_space.shape[0])
    if arm == 'e_td3':
        return T2TD3('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                     policy_kwargs={'net_arch': [256, 256], 'features_extractor_class': SliceExtractor,
                                    'features_extractor_kwargs': {'dim': OBS_DIM}},
                     gamma=GAMMA_STAGE, tau=0.005, learning_rate=3e-4, buffer_size=100_000,
                     batch_size=64 if smoke else 256, learning_starts=16 if smoke else 200,
                     train_freq=(1, 'step'), gradient_steps=2, policy_delay=2,
                     target_policy_noise=0.2, target_noise_clip=0.5, replay_buffer_class=TD3Replay)
    kw = {'net_arch': [256, 256], 'actor_dim': OBS_DIM, 'critic_dim': obs_dim if ARMS[arm] else OBS_DIM}
    return T2SAC(SplitViewPolicy, env, seed=seed, device='cpu', verbose=0, policy_kwargs=kw,
                 gamma=GAMMA_STAGE, learning_rate=3e-4, buffer_size=100_000, batch_size=64 if smoke else 256,
                 learning_starts=16 if smoke else 200, train_freq=(1, 'step'), gradient_steps=max(1, int(round(updates_per_transition))),
                 ent_coef=ENT_COEF, tau=0.005)


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


class SliceExtractor(BaseFeaturesExtractor):
    """The first ``dim`` values of the observation (T2.11: the actor's reduced interface, or the critic's view)."""

    def __init__(self, observation_space, dim: int):
        super().__init__(observation_space, features_dim=int(dim))
        self.dim = int(dim)

    def forward(self, obs):
        return obs[..., :self.dim]


class SplitViewPolicy(SACPolicy):
    """T2.11: one observation (OBS_DIM_PRIV: reduced interface followed by the executor's judgements), two views. The
    actor reads the first ``actor_dim`` values -- the deployed policy never sees the executor's information; the critic
    reads the first ``critic_dim`` -- all of it for EACC-A (the executor's judgements as training-time information), the
    actor's view for the SAC baseline (so the baseline is SAC on the reduced interface, the same environment process)."""

    def __init__(self, *args, actor_dim: int = OBS_DIM, critic_dim: int = OBS_DIM, **kwargs):
        self.actor_dim, self.critic_dim = int(actor_dim), int(critic_dim)
        kwargs['share_features_extractor'] = False
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        return super().make_actor(SliceExtractor(self.observation_space, self.actor_dim))

    def make_critic(self, features_extractor=None):
        return super().make_critic(SliceExtractor(self.observation_space, self.critic_dim))

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'actor_dim': self.actor_dim, 'critic_dim': self.critic_dim}


def _slim(info: dict) -> dict:
    """What the learner process keeps from a step's info."""
    return {k: info[k] for k in ('event', 'time_s', 'delta_v', 'mission', 'true_reward', 'exec_stats', 'min_margins', 'terminal')
            if k in info}


def worker(conn, rank: int, seed: int, window: bool, mission: str, geometry: bool = True, certify: bool = False,
           audit_logging: bool = False) -> None:
    """One environment: send (transition or None, observation to act on), receive the action; until 'stop'."""
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    th.set_num_threads(1)
    from env.t2_stage import T2StageEnv, mission_from_name
    env = T2StageEnv(seed_base=1_000_000 * (seed + 1) + rank * 100_000, shaping=True, window=window,
                     mission=mission_from_name(mission), geometry=geometry, certify=certify)
    env.audit_logging = audit_logging
    obs, _ = env.reset(); transition = None
    while True:
        conn.send((transition, obs))
        action = conn.recv()
        if isinstance(action, str):
            break
        next_obs, reward, term, trunc, info = env.step(action)
        if term or trunc:                         # the episode's executor statistics (logging only)
            st = env.stats or {}
            info['exec_stats'] = {**{k: st.get(k, 0) for k in ('stages', 'holds', 'rejected', 'bridge_attempts', 'rate_limited', 'entries_lost')},
                                  'window_recoveries': int(getattr(env.generator.bridge, 'recoveries', 0)) if env.generator is not None else 0,
                                  **({'cert_interventions': st['cert_interventions']} if 'cert_interventions' in st else {})}
        transition = (obs, next_obs, np.asarray(action, np.float32), float(reward), bool(term), bool(trunc), _slim(info))
        obs = env.reset()[0] if (term or trunc) else next_obs
    conn.close()


def train(out: Path, arm: str, seed: int, steps: int, workers: int, marks: list[int], smoke: bool = False,
          updates_per_transition: float = UPDATES_PER_TRANSITION, window: bool = True, mission: str = 'urgent_mix:0.75',
          geometry: bool = True, certify: bool = False, manifest_extra: dict | None = None) -> None:
    import multiprocessing as mp
    from multiprocessing.connection import wait
    from stable_baselines3.common.logger import configure
    from env.t2_stage import T2_VERSION
    out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    th.set_num_threads(1)
    model = make_model(SpacesEnv(OBS_DIM_PRIV), arm, seed, smoke, updates_per_transition=updates_per_transition)
    model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
    td3_manifest = {}
    if arm == 'e_td3':
        td3_manifest = {
            'algorithm': 'TD3', 'net_arch': [256, 256], 'learning_rate': model.learning_rate,
            'buffer_size': model.buffer_size, 'batch_size': model.batch_size,
            'learning_starts': model.learning_starts, 'gamma': model.gamma, 'tau': model.tau,
            'policy_delay': model.policy_delay, 'target_policy_noise': model.target_policy_noise,
            'target_noise_clip': model.target_noise_clip, 'exploration_sigma': model.exploration_sigma,
        }
    (out / 'manifest.json').write_text(json.dumps({'arm': arm, 'seed': seed, 'steps': steps, 'workers': workers, 'git_head': head, 'smoke': smoke,
                                                   'ent_coef': None if arm == 'e_td3' else ENT_COEF, 'collection': 'asynchronous',
                                                   'window': window, 'geometry': bool(geometry), 'certify': bool(certify),
                                                   'gradient_steps_per_transition': model.gradient_steps, 'mission': mission,
                                                   'obs_profile': OBS_PROFILE, 'actor_view': OBS_DIM,
                                                   'critic_view': int(getattr(model.policy, 'critic_dim', OBS_DIM)),
                                                   'env_version': T2_VERSION, 'learner_version': LEARNER_VERSION,
                                                   **td3_manifest, **(manifest_extra or {})}, indent=1), encoding='utf-8')
    audit_logging = (manifest_extra or {}).get('final_stage') == 'formal'
    ctx = mp.get_context('spawn')
    pipes, procs = [], []
    for rank in range(workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, rank, seed, window, mission, bool(geometry), bool(certify), audit_logging), daemon=True)
        p.start(); pipes.append(a); procs.append(p)
    marks = sorted(set([m for m in marks if m <= steps] + [steps]))
    model.save(out / 'checkpoints' / 'step_0')
    episodes, order, t0 = [], [], time.time()
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng([seed, 3003]) if arm == 'e_td3' else None
    ret_true, ret_shaped = {}, {}                 # each worker's current episode return (true task reward / shaped training reward)
    active = list(pipes)
    while active:
        for conn in wait(active):
            transition, obs = conn.recv()
            w_ = pipes.index(conn)
            if transition is not None and model.num_timesteps < steps:
                o, o1, act, r, term, trunc, info = transition
                done = term or trunc
                info = dict(info, **({'TimeLimit.truncated': True} if trunc and not term else {}))
                model.replay_buffer.add(o[None], o1[None], act[None], np.array([r]), np.array([done]), [info])
                model.num_timesteps += 1; order.append(w_)
                ret_true[w_] = ret_true.get(w_, 0.0) + float(info.get('true_reward') or 0.0); ret_shaped[w_] = ret_shaped.get(w_, 0.0) + float(r)
                if done:
                    episodes.append({'steps': int(model.num_timesteps), 'worker': w_, **{k: info.get(k) for k in ('event', 'time_s', 'delta_v', 'mission')},
                                     'return_true': ret_true.pop(w_, 0.0), 'return_shaped': ret_shaped.pop(w_, 0.0),
                                     **({'exec_stats': info['exec_stats']} if info.get('exec_stats') is not None else {}),
                                     **{k: info[k] for k in ('min_margins', 'terminal') if k in info}})
                if model.num_timesteps >= model.learning_starts:
                    model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)
                while marks and model.num_timesteps >= marks[0]:
                    m = marks.pop(0); model.save(out / 'checkpoints' / f'step_{m}')
                    (out / 'heartbeat.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'wall_s': time.time() - t0, 'episodes': len(episodes),
                                                                    'updates': int(model._n_updates), 'replay': model.replay_buffer.model_counts}), encoding='utf-8')
                    # logs at every checkpoint (a stopped run keeps them)
                    (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
                    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
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
    (out / 'episodes.json').write_text(json.dumps(episodes, default=str), encoding='utf-8')
    (out / 'model_stats.json').write_text(json.dumps(model.model_stats), encoding='utf-8')
    (out / 'finished.json').write_text(json.dumps({'steps': int(model.num_timesteps), 'updates': int(model._n_updates), 'wall_s': time.time() - t0,
                                                   'worker_order': order, 'replay': model.replay_buffer.model_counts}), encoding='utf-8')
