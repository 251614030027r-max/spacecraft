"""M1: SAC whose critic also regresses one gated MPC counterfactual target.

Design: "T1 SAC–MPC coupling design, upper hand-off final version, 2026-10-12",
section 5.4. Everything is SB3's SAC except the critic loss:

    L_Q = sum_j 1/B sum_i [ (Q_j(z_i, a_i) - y_i^real)^2 + lam w_i (Q_j(z_i, a~_i) - y_i^cf)^2 ] (x 0.5, SB3's factor)
    y^real = r + gamma (1 - d) V(z')                      (SB3's own target, unchanged)
    y^cf   = sg[ r^(a~) + gamma V(z^'(a~)) ]              (gate excludes predicted terminals: no mask)
    V(z)   = min_j Qbar_j(z, a') - alpha log pi(a'|z),  a' ~ pi(.|z)

The auxiliary term is averaged over the whole minibatch B, never over sum(w):
few valid samples weaken it, and a batch with sum(w) = 0 takes exactly the M0
code path (no extra forward pass, no extra random draws). lam is 0 for the
first ``warmup_fraction`` of the training budget. The actor and the
controller are untouched; the counterfactual reaches the actor only through
the critic.

The replay buffer stores, per transition, the cached counterfactual
``(a~[2], r^[1], z^'[22], w[1])`` written from ``info["counterfactual"]``; no
G* or MPC call is made on replay batches.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update


class CounterfactualBatch(NamedTuple):
    actions: th.Tensor
    rewards: th.Tensor
    next_observations: th.Tensor
    weights: th.Tensor


class M1ReplayBuffer(ReplayBuffer):
    """SB3 replay buffer plus the cached counterfactual per transition, for any number of envs."""

    def __init__(self, buffer_size, observation_space, action_space, device="auto", n_envs=1,
                 optimize_memory_usage=False, handle_timeout_termination=True) -> None:
        if optimize_memory_usage:
            raise ValueError("M1ReplayBuffer does not support memory optimisation")
        super().__init__(buffer_size, observation_space, action_space, device, n_envs,
                         optimize_memory_usage, handle_timeout_termination)
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.cf_actions = np.zeros((self.buffer_size, self.n_envs, act_dim), dtype=np.float32)
        self.cf_rewards = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.cf_next_observations = np.zeros((self.buffer_size, self.n_envs, obs_dim), dtype=np.float32)
        self.cf_weights = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self._last: tuple[np.ndarray, np.ndarray] | None = None

    def add(self, obs, next_obs, action, reward, done, infos: list[dict[str, Any]]) -> None:
        i = self.pos
        for e in range(self.n_envs):
            cf = infos[e].get("counterfactual") if e < len(infos) else None
            valid = bool(cf and cf.get("w") == 1 and cf.get("a_tilde") is not None)
            self.cf_weights[i, e] = 1.0 if valid else 0.0
            if valid:
                self.cf_actions[i, e] = np.asarray(cf["a_tilde"], dtype=np.float32)
                self.cf_rewards[i, e] = np.float32(cf["r_hat"])
                self.cf_next_observations[i, e] = np.asarray(cf["z_hat"], dtype=np.float32)
            else:
                self.cf_actions[i, e] = 0.0
                self.cf_rewards[i, e] = 0.0
                self.cf_next_observations[i, e] = 0.0
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env=None):
        # SB3 2.7.1 ReplayBuffer._get_samples, keeping the env indices it draws (same
        # random draws, so the real batch is exactly SB3's).
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        self._last = (np.asarray(batch_inds), env_indices)
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def counterfactual(self) -> CounterfactualBatch:
        """The cached counterfactual of the transitions of the last ``sample``."""

        assert self._last is not None
        i, e = self._last
        return CounterfactualBatch(
            self.to_torch(self.cf_actions[i, e]), self.to_torch(self.cf_rewards[i, e]),
            self.to_torch(self.cf_next_observations[i, e]), self.to_torch(self.cf_weights[i, e]),
        )


def auxiliary_critic_loss(q_values: list[th.Tensor], target: th.Tensor, weights: th.Tensor, lam: float) -> th.Tensor:
    """0.5 * sum_j mean_over_B( lam * w * (Q_j - y)^2 ) -- denominator B, not sum(w)."""

    return 0.5 * sum((lam * weights * (q - target) ** 2).mean() for q in q_values)


class M1SAC(SAC):
    def __init__(self, *args: Any, cf_lambda: float = 0.5, cf_warmup_fraction: float = 0.15,
                 cf_warmup_decisions: int | None = None, cf_diagnostics_every: int = 50, **kwargs: Any) -> None:
        self.cf_lambda = float(cf_lambda)
        self.cf_warmup_fraction = float(cf_warmup_fraction)
        #: absolute warm-up (decisions); when set it replaces the fraction of the budget
        self.cf_warmup_decisions = cf_warmup_decisions
        self.cf_diagnostics_every = int(cf_diagnostics_every)
        kwargs.setdefault("replay_buffer_class", M1ReplayBuffer)
        super().__init__(*args, **kwargs)
        self.cf_stats: dict[str, list[float]] = {
            "valid_fraction": [], "aux_loss": [], "real_loss": [],
            "grad_norm_real": [], "grad_norm_aux": [], "grad_update": [],
        }

    def current_lambda(self) -> float:
        if getattr(self, "cf_warmup_decisions", None) is not None:
            return 0.0 if self.num_timesteps < self.cf_warmup_decisions else self.cf_lambda
        total = getattr(self, "_total_timesteps", 0) or 0
        return 0.0 if self.num_timesteps < self.cf_warmup_fraction * total else self.cf_lambda

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # SB3 2.7.1 SAC.train with the critic loss of the module docstring.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        lam = self.current_lambda()

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            cf = self.replay_buffer.counterfactual()  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            valid = float(cf.weights.sum().item())
            self.cf_stats["valid_fraction"].append(valid / batch_size)
            self.cf_stats["real_loss"].append(critic_loss.item())
            if lam > 0.0 and valid > 0.0:
                with th.no_grad():
                    cf_next_actions, cf_next_log_prob = self.actor.action_log_prob(cf.next_observations)
                    cf_next_q = th.cat(self.critic_target(cf.next_observations, cf_next_actions), dim=1)
                    cf_next_q, _ = th.min(cf_next_q, dim=1, keepdim=True)
                    cf_next_q = cf_next_q - ent_coef * cf_next_log_prob.reshape(-1, 1)
                    cf_target = cf.rewards + self.gamma * cf_next_q
                aux = auxiliary_critic_loss(list(self.critic(replay_data.observations, cf.actions)), cf_target,
                                            cf.weights, lam)
                self.cf_stats["aux_loss"].append(aux.item())
                if self.cf_diagnostics_every and self._n_updates % self.cf_diagnostics_every == 0:
                    params = [p for p in self.critic.parameters() if p.requires_grad]
                    g_real = th.autograd.grad(critic_loss, params, retain_graph=True, allow_unused=True)
                    g_aux = th.autograd.grad(aux, params, retain_graph=True, allow_unused=True)
                    norm = lambda gs: float(th.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))  # noqa: E731
                    self.cf_stats["grad_norm_real"].append(norm(g_real))
                    self.cf_stats["grad_norm_aux"].append(norm(g_aux))
                    self.cf_stats["grad_update"].append(float(self._n_updates))
                critic_loss = critic_loss + aux
            else:
                self.cf_stats["aux_loss"].append(0.0)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/cf_lambda", lam)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


__all__ = ["CounterfactualBatch", "M1ReplayBuffer", "M1SAC", "auxiliary_critic_loss"]
