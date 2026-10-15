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

Learnability fixes (2026-10-15, shared by M0 and M2; M0 is this class with lam_cf = 0 and an
env that produces no counterfactual, so both use one update path):

    L_Q += lam_mc * 0.5 sum_j mean_B[ m * Huber_10(Q_j(z, a) - G) ]
         + lam_eq * 0.5 sum_j mean_B[ e * (Q_j(z, (u, a1)) - sg Q_j(z, (0, a1)))^2 ],  u ~ U[-0.5, 0.5]

G is the exact discounted return of the transition's own episode, written into the buffer
when that episode ends normally (success / unsafe / timeout; a numerical-failure truncation
leaves m = 0), used only while the transition is among the last ``mc_window`` collected (an
off-policy, entropy-free auxiliary label, not the soft Q). e = 1 when the state has a live
plan (``remaining_160s`` > 0, not descending): there |a0| <= 0.5 requests exactly the remaining time, so
every such a0 with the same remaining action components (kappa and, interface v4, the entry
speed, which are never varied here) yields the same transition and next observation (the soft
Q is equal). With lam_mc = lam_eq = 0 and the probe off, no extra computation or random draw
happens: the update is the previous one bitwise.

The replay buffer stores, per transition, the cached counterfactual
``(a~[act_dim], r^[1], z^'[obs_dim], w[1])`` written from ``info["counterfactual"]``; no
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


#: observation indices (env.t1_gym.OBSERVATION_NAMES): remaining_160s, descent
PLAN_REMAINING_INDEX = 16
DESCENT_INDEX = 19


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
        # exact discounted episode returns (written when the episode ends normally)
        self.gamma = 0.999
        self.mc_window = 10_000
        self.mc_returns = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.mc_mask = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.add_index = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.n_adds = 0
        self._episode: list[list[tuple[int, int, float]]] = [[] for _ in range(self.n_envs)]
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
        rewards = np.asarray(reward, dtype=np.float64).reshape(-1)
        dones = np.asarray(done).reshape(-1)
        for e in range(self.n_envs):
            self.mc_mask[i, e] = 0.0  # this slot now holds a new, unfinished transition
            self.mc_returns[i, e] = 0.0
            self.add_index[i, e] = self.n_adds
            self._episode[e].append((i, self.n_adds, float(rewards[e])))
            if bool(dones[e]):
                info = infos[e] if e < len(infos) else {}
                if not info.get("TimeLimit.truncated", False):  # normal end: success/unsafe/timeout
                    g = 0.0
                    for slot, index, r in reversed(self._episode[e]):
                        g = r + self.gamma * g
                        if self.add_index[slot, e] == index:  # not overwritten since
                            self.mc_returns[slot, e] = g
                            self.mc_mask[slot, e] = 1.0
                self._episode[e] = []
        self.n_adds += 1
        super().add(obs, next_obs, action, reward, done, infos)

    def monte_carlo(self) -> tuple[th.Tensor, th.Tensor]:
        """(G, mask) of the transitions of the last ``sample``; mask excludes stale samples."""

        assert self._last is not None
        i, e = self._last
        recent = ((self.n_adds - self.add_index[i, e]) * self.n_envs <= self.mc_window).astype(np.float32)
        return self.to_torch(self.mc_returns[i, e]), self.to_torch(self.mc_mask[i, e] * recent[:, None])

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


def eq_mask(observations: th.Tensor) -> th.Tensor:
    """1 where the keep band is exactly equivalent: a live plan (remaining > 0), not descending."""

    plan = observations[:, PLAN_REMAINING_INDEX:PLAN_REMAINING_INDEX + 1] > 0.0
    descent = observations[:, DESCENT_INDEX:DESCENT_INDEX + 1] > 0.5
    return (plan & ~descent).float()


def auxiliary_critic_loss(q_values: list[th.Tensor], target: th.Tensor, weights: th.Tensor, lam: float) -> th.Tensor:
    """0.5 * sum_j mean_over_B( lam * w * (Q_j - y)^2 ) -- denominator B, not sum(w)."""

    return 0.5 * sum((lam * weights * (q - target) ** 2).mean() for q in q_values)


class M1SAC(SAC):
    def __init__(self, *args: Any, cf_lambda: float = 0.5, cf_warmup_fraction: float = 0.15,
                 cf_warmup_decisions: int | None = None, cf_diagnostics_every: int = 50,
                 mc_lambda: float = 0.0, eq_lambda: float = 0.0, mc_window: int = 10_000, mc_delta: float = 10.0,
                 grad_probe_every: int = 0, audit_every: int = 0, **kwargs: Any) -> None:
        self.mc_lambda, self.eq_lambda = float(mc_lambda), float(eq_lambda)
        self.mc_window, self.mc_delta = int(mc_window), float(mc_delta)
        #: measure the unweighted auxiliary gradient norms every n updates (calibration); 0 = off
        self.grad_probe_every = int(grad_probe_every)
        #: critic audits (Q - G by state group, keep-band Q spread) every n updates; 0 = off
        self.audit_every = int(audit_every)
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
        self.fix_stats: dict[str, list[float]] = {key: [] for key in (
            "probe_update", "probe_decisions", "g_td", "g_mc", "g_eq", "g_mc_dot_eq", "mc_fraction", "eq_fraction",
            "audit_update", "audit_decisions", "q_minus_g_plan", "q_minus_g_no_plan", "q_minus_g_descent",
            "n_plan", "n_no_plan", "n_descent", "band_q_spread")}
        self._sync_buffer_settings()

    def _excluded_save_params(self) -> list[str]:
        # the private eq generator is not serialisable; fix_stats are written as JSON by the trainer
        return [*super()._excluded_save_params(), "_eq_gen", "fix_stats"]

    def _sync_buffer_settings(self) -> None:
        buffer = getattr(self, "replay_buffer", None)
        if buffer is not None and hasattr(buffer, "mc_window"):
            buffer.gamma, buffer.mc_window = self.gamma, self.mc_window

    def _setup_learn(self, *args: Any, **kwargs: Any):
        result = super()._setup_learn(*args, **kwargs)
        self._sync_buffer_settings()
        return result

    def current_lambda(self) -> float:
        if getattr(self, "cf_warmup_decisions", None) is not None:
            return 0.0 if self.num_timesteps < self.cf_warmup_decisions else self.cf_lambda
        total = getattr(self, "_total_timesteps", 0) or 0
        return 0.0 if self.num_timesteps < self.cf_warmup_fraction * total else self.cf_lambda

    # ------------------------------------------------------------------ #
    # learnability fixes (L_mc, L_eq), gradient probe and critic audits
    def _eq_generator(self) -> th.Generator:
        # a private generator: the eq draws never move the global torch RNG, so turning the
        # probe on does not change the training trajectory's random stream
        if getattr(self, "_eq_gen", None) is None:
            self._eq_gen = th.Generator(device=self.device)
            self._eq_gen.manual_seed(int(self.seed or 0) + 7919)
        return self._eq_gen

    def _mc_loss(self, replay_data, current_q_values) -> tuple[th.Tensor, float]:
        returns, mask = self.replay_buffer.monte_carlo()  # type: ignore[union-attr]
        loss = 0.5 * sum((mask * F.huber_loss(q, returns, reduction="none", delta=self.mc_delta)).mean()
                         for q in current_q_values)
        return loss, float(mask.mean().item())

    def _eq_mask(self, observations: th.Tensor) -> th.Tensor:
        """Where the keep band is exactly equivalent (overridden by the T1-L learner: reviews only)."""

        return eq_mask(observations)

    def _eq_loss(self, replay_data) -> tuple[th.Tensor, float]:
        obs, actions = replay_data.observations, replay_data.actions
        mask = self._eq_mask(obs)
        u = th.rand((actions.shape[0], 1), generator=self._eq_generator(), device=self.device) - 0.5
        a_band = th.cat([u, actions[:, 1:]], dim=1)
        a_zero = th.cat([th.zeros_like(u), actions[:, 1:]], dim=1)
        with th.no_grad():
            anchor = self.critic(obs, a_zero)
        moved = self.critic(obs, a_band)
        loss = 0.5 * sum((mask * (q - a.detach()) ** 2).mean() for q, a in zip(moved, anchor))
        return loss, float(mask.mean().item())

    def _fix_terms(self, td_loss: th.Tensor, replay_data, current_q_values) -> th.Tensor:
        """TD loss plus lam_mc L_mc + lam_eq L_eq; probe and audits are read-only."""

        n = self._n_updates_in_call  # update index of this gradient step
        probe = bool(self.grad_probe_every) and n % self.grad_probe_every == 0
        audit = bool(self.audit_every) and n % self.audit_every == 0
        loss = td_loss
        mc = eq = None
        if self.mc_lambda > 0.0 or probe:
            mc, mc_fraction = self._mc_loss(replay_data, current_q_values)
        if self.eq_lambda > 0.0 or probe:
            eq, eq_fraction = self._eq_loss(replay_data)
        if probe:
            params = [p for p in self.critic.parameters() if p.requires_grad]
            flat = lambda loss: th.cat([(g if g is not None else th.zeros_like(p)).flatten() for g, p in zip(  # noqa: E731
                th.autograd.grad(loss, params, retain_graph=True, allow_unused=True), params)])
            g_td, g_mc, g_eq = flat(td_loss), flat(mc), flat(eq)
            self.fix_stats["g_td"].append(float(g_td.norm()))
            self.fix_stats["g_mc"].append(float(g_mc.norm()))
            self.fix_stats["g_eq"].append(float(g_eq.norm()))
            self.fix_stats["g_mc_dot_eq"].append(float(g_mc @ g_eq))
            self.fix_stats["mc_fraction"].append(mc_fraction)
            self.fix_stats["eq_fraction"].append(eq_fraction)
            self.fix_stats["probe_update"].append(float(n))
            self.fix_stats["probe_decisions"].append(float(self.num_timesteps))
        if audit:
            self._audit(replay_data, current_q_values, n)
        if self.mc_lambda > 0.0:
            loss = loss + self.mc_lambda * mc
        if self.eq_lambda > 0.0:
            loss = loss + self.eq_lambda * eq
        return loss

    def _audit(self, replay_data, current_q_values, n: int) -> None:
        with th.no_grad():
            obs, actions = replay_data.observations, replay_data.actions
            returns, mask = self.replay_buffer.monte_carlo()  # type: ignore[union-attr]
            q = th.min(th.cat([c.detach() for c in current_q_values], dim=1), dim=1, keepdim=True)[0]
            plan = obs[:, PLAN_REMAINING_INDEX:PLAN_REMAINING_INDEX + 1] > 0.0
            descent = obs[:, DESCENT_INDEX:DESCENT_INDEX + 1] > 0.5
            band = self._eq_mask(obs)[:, 0] > 0.0
            groups = {"plan": plan & ~descent, "no_plan": ~plan & ~descent, "descent": descent}
            for name, g in groups.items():
                m = g.float() * mask
                count = float(m.sum().item())
                self.fix_stats[f"n_{name}"].append(count)
                self.fix_stats[f"q_minus_g_{name}"].append(
                    float(((q - returns) * m).sum().item() / count) if count else float("nan"))
            spread = float("nan")
            rows = band
            if bool(rows.any()):
                s, a1 = obs[rows], actions[rows, 1:]
                values = []
                for a0 in np.linspace(-0.5, 0.5, 11):
                    a = th.cat([th.full_like(a1[:, :1], float(a0)), a1], dim=1)
                    values.append(th.min(th.cat(self.critic(s, a), dim=1), dim=1, keepdim=True)[0])
                v = th.cat(values, dim=1)
                spread = float((v.max(dim=1)[0] - v.min(dim=1)[0]).mean().item())
            self.fix_stats["band_q_spread"].append(spread)
            self.fix_stats["audit_update"].append(float(n))
            self.fix_stats["audit_decisions"].append(float(self.num_timesteps))

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
            self._n_updates_in_call = self._n_updates + gradient_step
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
            critic_loss = self._fix_terms(critic_loss, replay_data, current_q_values)
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
        self.logger.record("train/mc_lambda", self.mc_lambda)
        self.logger.record("train/eq_lambda", self.eq_lambda)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


__all__ = ["DESCENT_INDEX", "PLAN_REMAINING_INDEX", "CounterfactualBatch", "eq_mask", "M1ReplayBuffer", "M1SAC", "auxiliary_critic_loss"]
