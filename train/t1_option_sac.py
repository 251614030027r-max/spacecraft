"""Phase-aware SAC for the T1-L review interface (plan 2026-10-15, modules C and D).

Every real 2 s transition ``(z, a_executed, r, z', terminal)`` is stored; ``a_executed`` is the
review action, or at a continue step the continue action ``a_c(z)`` (the environment reports
it in ``info["executed_action"]``; the policy's proposal there is discarded, never stored).
With the due flag of the observation (``env.t1_task_review.DUE_INDEX``):

    V(z')  = E_{a'~pi}[ min_j Qbar_j(z', a') - alpha(z') log pi(a'|z') ]   if due(z') = 1
           = min_j Qbar_j(z', a_c(z'))                                       if due(z') = 0
    y      = r + gamma (1 - terminal) V(z')                (success / unsafe / timeout: no bootstrap)
    L_TD   = 0.5 sum_j mean_B (Q_j(z, a_executed) - y)^2   (all transitions, every 2 s)
    actor  : mean over *review* states only (a separate batch of due transitions)
             of  alpha(z) log pi(a|z) - min_j Q_j(z, a),  a ~ pi(.|z)
    alpha(z) = alpha_0 * gap(z),  gap = 2 s steps the review closes (1..3; B2: 1, N0 regular: 3)

``alpha_0`` is the fixed 0.005 of the project (no automatic temperature): per unit of physical
time the 6 s arms' policy entropy is charged as the 2 s arm's (alpha_0 per 2 s step). The
shared critic fixes are those of ``train.t1_m1.M1SAC`` (MC labels of normally ended episodes;
keep-band equivalence, here on review states with a live plan only); the absolute MPC
counterfactual target of M1/M2 is off (``cf_lambda`` = 0).

NDelta adds the relative term on the cached alternative ``(a~, r^, z^', w)`` of a review state:

    delta = (r^(a~) - r(a)) + gamma (V(z^'(a~)) - V(z'(a)))           (stop-gradient)
    L_delta = 0.5 sum_j mean_B[ w Huber_10( (Q_j(z, a~) - Q_j(z, a)) - delta ) ]

anchored on the real transition (r, z' are the real ones; only the alternative is predicted);
V is the same mode-aware target value (the successor's own due flag and plan memory). It is
a relative short-window label that still bootstraps on the target critic; it is not an
unbiased long-run advantage. With ``delta_lambda`` = 0 nothing extra is computed or drawn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

from env.t1_gym import T1GymConfig
from env.t1_task_review import DUE_INDEX, ENTRY_INDEX, GAP_INDEX, HAS_PLAN_INDEX, KAPPA_INDEX
from train.t1_m1 import M1ReplayBuffer, M1SAC, eq_mask


class OptionReplayBuffer(M1ReplayBuffer):
    """M1 buffer plus the review flag of each stored state, and sampling of review states."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.due = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        self.due[self.pos] = np.asarray(obs).reshape(self.n_envs, -1)[:, DUE_INDEX] > 0.5
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_due(self, batch_size: int, env=None) -> th.Tensor | None:
        upper = self.buffer_size if self.full else self.pos
        flat = np.flatnonzero(self.due[:upper].reshape(-1))
        if flat.size == 0:
            return None
        pick = flat[np.random.randint(0, flat.size, size=batch_size)]
        i, e = np.divmod(pick, self.n_envs)
        return self.to_torch(self._normalize_obs(self.observations[i, e, :], env))


class OptionSAC(M1SAC):
    def __init__(self, *args: Any, delta_lambda: float = 0.0, delta_warmup_decisions: int = 3000,
                 delta_huber: float = 10.0, delta_diagnostics_every: int = 50, **kwargs: Any) -> None:
        self.delta_lambda = float(delta_lambda)
        self.delta_warmup_decisions = int(delta_warmup_decisions)
        self.delta_huber = float(delta_huber)
        self.delta_diagnostics_every = int(delta_diagnostics_every)
        kwargs.setdefault("replay_buffer_class", OptionReplayBuffer)
        kwargs["cf_lambda"] = 0.0  # the absolute M1/M2 counterfactual target is not used here
        super().__init__(*args, **kwargs)
        if self.ent_coef_optimizer is not None:
            raise ValueError("OptionSAC needs a fixed ent_coef (alpha_0)")
        gym = T1GymConfig()
        self._kappa_range = (gym.kappa_min, gym.kappa_max)
        self.delta_stats: dict[str, list[float]] = {key: [] for key in (
            "update", "decisions", "valid_fraction", "loss", "g_td", "g_delta", "abs_target_mean", "abs_dq_mean")}

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "delta_stats"]

    # ------------------------------------------------------------------ #
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        # the executed action (review action or continue action), never an unexecuted proposal
        executed = np.stack([np.asarray(info.get("executed_action", buffer_action[k]), dtype=np.float32)
                             for k, info in enumerate(infos)])
        super()._store_transition(replay_buffer, executed, new_obs, reward, dones, infos)

    def _eq_mask(self, observations: th.Tensor) -> th.Tensor:
        return eq_mask(observations) * (observations[:, DUE_INDEX:DUE_INDEX + 1] > 0.5).float()

    def alpha_of(self, observations: th.Tensor) -> th.Tensor:
        gap = th.clamp(th.round(observations[:, GAP_INDEX:GAP_INDEX + 1] * 3.0), 1.0, 3.0)
        return self.ent_coef_tensor * gap

    def continue_actions(self, observations: th.Tensor) -> th.Tensor:
        lo, hi = self._kappa_range
        a1 = 2.0 * (observations[:, KAPPA_INDEX:KAPPA_INDEX + 1] - lo) / (hi - lo) - 1.0
        plan = observations[:, HAS_PLAN_INDEX:HAS_PLAN_INDEX + 1] > 0.5
        a2 = th.where(plan, observations[:, ENTRY_INDEX:ENTRY_INDEX + 1], th.zeros_like(a1))
        return th.clamp(th.cat([th.zeros_like(a1), a1, a2], dim=1), -1.0, 1.0)

    def target_value(self, observations: th.Tensor) -> th.Tensor:
        """Mode-aware soft value (module docstring); no gradient."""

        with th.no_grad():
            actions, log_prob = self.actor.action_log_prob(observations)
            q = th.min(th.cat(self.critic_target(observations, actions), dim=1), dim=1, keepdim=True)[0]
            v_review = q - self.alpha_of(observations) * log_prob.reshape(-1, 1)
            q_continue = th.min(th.cat(self.critic_target(observations, self.continue_actions(observations)), dim=1),
                                dim=1, keepdim=True)[0]
            due = observations[:, DUE_INDEX:DUE_INDEX + 1] > 0.5
            return th.where(due, v_review, q_continue)

    def current_delta_lambda(self) -> float:
        return 0.0 if self.num_timesteps < self.delta_warmup_decisions else self.delta_lambda

    # ------------------------------------------------------------------ #
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        actor_losses, critic_losses = [], []
        lam = self.current_delta_lambda()
        buffer = self.replay_buffer
        for gradient_step in range(gradient_steps):
            self._n_updates_in_call = self._n_updates + gradient_step
            data = buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            cf = buffer.counterfactual()  # type: ignore[union-attr]
            v_next = self.target_value(data.next_observations)
            target = data.rewards + (1 - data.dones) * self.gamma * v_next
            current_q = self.critic(data.observations, data.actions)
            td = 0.5 * sum(F.mse_loss(q, target) for q in current_q)
            loss = self._fix_terms(td, data, current_q)
            valid = float(cf.weights.sum().item())
            if lam > 0.0 and valid > 0.0:
                with th.no_grad():
                    v_hat = self.target_value(cf.next_observations)
                    delta = (cf.rewards - data.rewards) + self.gamma * (v_hat - v_next)
                alt_q = self.critic(data.observations, cf.actions)
                rel = 0.5 * sum((cf.weights * F.huber_loss(qa - qr, delta, reduction="none", delta=self.delta_huber)).mean()
                                for qa, qr in zip(alt_q, current_q))
                n = self._n_updates_in_call
                if self.delta_diagnostics_every and n % self.delta_diagnostics_every == 0:
                    params = [p for p in self.critic.parameters() if p.requires_grad]
                    norm = lambda gs: float(th.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))  # noqa: E731
                    s = self.delta_stats
                    s["update"].append(float(n))
                    s["decisions"].append(float(self.num_timesteps))
                    s["valid_fraction"].append(valid / batch_size)
                    s["loss"].append(float(rel.item()))
                    s["g_td"].append(norm(th.autograd.grad(td, params, retain_graph=True, allow_unused=True)))
                    s["g_delta"].append(norm(th.autograd.grad(rel, params, retain_graph=True, allow_unused=True)))
                    w = cf.weights[:, 0] > 0
                    s["abs_target_mean"].append(float(delta[w].abs().mean().item()))
                    s["abs_dq_mean"].append(float((alt_q[0] - current_q[0]).detach()[w].abs().mean().item()))
                loss = loss + lam * rel
            critic_losses.append(float(loss.item()))
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()

            review = buffer.sample_due(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            if review is not None:
                actions_pi, log_prob = self.actor.action_log_prob(review)
                q_pi = th.min(th.cat(self.critic(review, actions_pi), dim=1), dim=1, keepdim=True)[0]
                actor_loss = (self.alpha_of(review) * log_prob.reshape(-1, 1) - q_pi).mean()
                actor_losses.append(float(actor_loss.item()))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/delta_lambda", lam)


__all__ = ["OptionReplayBuffer", "OptionSAC"]
