"""Learned stopping option: the V3e task policy plus a learned one-way handoff.

The method (``docs/STOPPING_METHOD_PREREGISTRATION_20261002.md``):

* the continuous 2D task policy is the unchanged V3e SAC actor on the
  unchanged interface;
* before every learned decision a **stopping head** gives
  ``beta(s) = P(hand off to Pure MPC now | s)``, a Bernoulli policy with its
  own entropy term (discrete soft actor-critic);
* the handoff value ``Q_H(s)`` is regressed on the **realized** discounted
  return of Pure MPC flying from ``s`` to the end of the episode -- the true
  MPC suffix return, never a bootstrap;
* the continuation critic is the usual SAC twin critic with the target
  ``Q_C(s, a) = r + gamma * V(s')``, where
  ``V(s) = beta Q_H + (1 - beta) V_C + alpha_stop H(beta)`` and
  ``V_C(s) = E_a[min Q_C(s, a) - alpha log pi(a|s)]``. At the stopping head's
  optimum ``beta = sigmoid((Q_H - V_C) / alpha_stop)`` this is the soft
  maximum ``alpha_stop * logsumexp(Q_H / alpha_stop, V_C / alpha_stop)``,
  i.e. ``V = max{Q_H, V_C}`` up to the entropy temperature.

Deployment is the mode of the Bernoulli: hand off at the first learned
decision with ``beta(s) >= 0.5``. That is the usual deterministic policy of a
discrete action, not a tuned threshold.

Two engineering consequences are handled here rather than left implicit:

* **Behaviour vs target stopping.** The value of continuing is only learnable
  from continue transitions. Early in training Pure MPC is far better than the
  immature task policy, so the rational stopping head hands off at the first
  decision everywhere, after which no continue transition is ever collected
  and the task policy can never improve -- a self-confirming collapse. During
  training the handoff is therefore *sampled* with
  ``clip(beta(s), h_min, h_max)``: the floor keeps labelling ``Q_H`` where
  the head says continue (the named "uncertain -> stay learned" defect), the
  cap keeps the task policy flying into the mid and late episode. Both critics
  are off-policy (``Q_H`` is a Monte Carlo regression on a fixed controller,
  ``Q_C`` bootstraps through the target ``V``), so the clip changes which data
  is collected, not what is estimated. Deployment uses ``beta`` alone.
* **Compute and update budget.** A handoff runs the Pure MPC suffix inside one
  environment step. ``num_timesteps`` counts *simulated* decisions (learned
  plus suffix) and one gradient update is made per simulated decision, so the
  training budget and the update count per unit of simulation are the same as
  V3e's 60k learned decisions.

The suffix also labels every state Pure MPC passes through on the way, with its
own discounted return to go. Those states are flown by an MPC that has not
been reset there; resetting it changes the trajectory only at solver-tolerance
level (probe: rewards differ by at most 2.2e-4 on seed 262004, same outcome),
so they are used as ``Q_H`` labels alongside the exact handoff-point label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.se3_rendezvous_env import SE3RendezvousConfig
from train.hybrid_configs import SAC_MPC_HYBRID, HybridSACConfig
from train.mainline import mainline_v3e_configs

#: The policy-independent observation blocks Pure MPC's future depends on (the
#: same blocks as stage C's estimator). ``Q_H`` sees only these.
HANDOFF_FEATURE_BLOCKS = ("core", "target_attitude", "remaining_time")


@dataclass(frozen=True)
class StoppingConfig:
    #: Decision discount for SAC, the stopping value and the potential
    #: shaping. 0.99 discounted a completion 150 decisions away to 0.22 of its
    #: value and, measured on the task reward, preferred a fast risky path
    #: (9.6) to a slow reliable one (7.0); at 0.999 the order is 14.7 < 17.2.
    #: Pressure to finish early comes from the task's own time cost.
    gamma: float = 0.999
    #: Initial stopping-head logit at every state (last layer zero weights):
    #: beta = 0.0067 per decision, P(no handoff in 100 decisions) = 0.51.
    #: An initialization only; the head is free to move anywhere.
    initial_handoff_logit: float = -5.0
    #: Training-time behaviour clip of the handoff probability (see the module
    #: docstring). Not used at deployment.
    behaviour_handoff_min: float = 0.002
    behaviour_handoff_max: float = 0.01
    #: Entropy temperature of the stopping head: the task policy's fixed
    #: ``ent_coef``, so no new temperature is introduced.
    stop_ent_coef: float = 0.005
    #: Deployment: the mode of the Bernoulli.
    deployment_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.behaviour_handoff_min <= self.behaviour_handoff_max <= 1.0:
            raise ValueError("behaviour clip must satisfy 0 <= min <= max <= 1")
        if self.stop_ent_coef < 0.0:
            raise ValueError("stop entropy coefficient must be non-negative")


STOPPING = StoppingConfig()

#: SAC hyperparameters: V3e's, with the decision discount moved to 0.999.
STOPPING_SAC: HybridSACConfig = replace(SAC_MPC_HYBRID, gamma=STOPPING.gamma)


def stopping_configs(
    config: StoppingConfig = STOPPING,
) -> tuple[SE3RendezvousConfig, PrecaptureHybridConfig]:
    """The V3e mainline task and interface, with the discount moved to ``gamma``.

    ``decision_discount_factor`` only scales the potential-shaping term; the
    dynamics, the MPC, the reference and Pure MPC are untouched.
    """

    environment_config, hybrid_config = mainline_v3e_configs()
    return environment_config, replace(hybrid_config, decision_discount_factor=config.gamma)


def handoff_feature_index(slices: dict[str, slice]) -> list[int]:
    index: list[int] = []
    for name in HANDOFF_FEATURE_BLOCKS:
        block = slices[name]
        index.extend(range(block.start, block.stop))
    return index


def discounted_returns_to_go(rewards: list[float] | np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for j in range(len(rewards) - 1, -1, -1):
        running = float(rewards[j]) + gamma * running
        returns[j] = running
    return returns


def soft_stopping_value(
    logit: th.Tensor, q_handoff: th.Tensor, v_continue: th.Tensor, ent_coef: float
) -> th.Tensor:
    """``beta Q_H + (1 - beta) V_C + ent_coef * H(beta)``, computed from the logit."""

    log_beta = F.logsigmoid(logit)
    log_continue = F.logsigmoid(-logit)
    beta = th.exp(log_beta)
    entropy = -(beta * log_beta + (1.0 - beta) * log_continue)
    return beta * q_handoff + (1.0 - beta) * v_continue + ent_coef * entropy


def _mlp(input_dim: int, net_arch: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in net_arch:
        layers += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class StopNetworks(nn.Module):
    """Stopping head on the full observation; twin ``Q_H`` on the handoff features."""

    def __init__(
        self,
        observation_dim: int,
        feature_index: list[int],
        net_arch: list[int],
        initial_logit: float,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_index", th.as_tensor(list(feature_index), dtype=th.long))
        self.stop_head = _mlp(observation_dim, net_arch)
        last = self.stop_head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(initial_logit))
        self.q_handoff_1 = _mlp(len(feature_index), net_arch)
        self.q_handoff_2 = _mlp(len(feature_index), net_arch)

    def logit(self, observation: th.Tensor) -> th.Tensor:
        return self.stop_head(observation)

    def q_handoff(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = observation.index_select(-1, self.feature_index)
        return self.q_handoff_1(features), self.q_handoff_2(features)

    def q_handoff_min(self, observation: th.Tensor) -> th.Tensor:
        return th.min(*self.q_handoff(observation))


class HandoffOptionEnv(gym.Wrapper):
    """Training environment: before each learned decision, maybe hand off for good.

    ``handoff_probability(observation) -> beta`` is the stopping head; the
    handoff is sampled with ``clip(beta, h_min, h_max)``. A handoff ignores the
    task action, flies Pure MPC to the end of the episode (the same
    ``step_with_branch(zero, "baseline")`` path as Pure MPC and every handoff
    scan, controller reset at the switch) and returns one terminal step whose
    ``info`` carries the suffix states and their discounted returns to go.
    """

    EPISODE_KEYS = (
        "handoff",
        "handoff_decision",
        "learned_decisions",
        "suffix_decisions",
        "simulated_decisions",
        "stop_beta_mean",
        "stop_beta_max",
        "stop_beta_last",
    )

    def __init__(self, env: PrecaptureHybridEnv, config: StoppingConfig, seed: int) -> None:
        super().__init__(env)
        self.config = config
        self.handoff_probability: Callable[[np.ndarray], float] | None = None
        self._rng = np.random.default_rng(int(seed))
        self._observation: np.ndarray | None = None
        self._learned_decisions = 0
        self._betas: list[float] = []

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._observation = observation
        self._learned_decisions = 0
        self._betas = []
        return observation, info

    def _summary(self, *, handoff: bool, suffix: int) -> dict[str, Any]:
        betas = self._betas or [0.0]
        return {
            "handoff": bool(handoff),
            "handoff_decision": self._learned_decisions if handoff else -1,
            "learned_decisions": self._learned_decisions,
            "suffix_decisions": suffix,
            "simulated_decisions": self._learned_decisions + suffix,
            "stop_beta_mean": float(np.mean(betas)),
            "stop_beta_max": float(np.max(betas)),
            "stop_beta_last": float(betas[-1]),
        }

    def step(self, action: np.ndarray):
        assert self._observation is not None, "reset before step"
        if self.handoff_probability is None:
            raise RuntimeError("attach the stopping head before stepping")
        beta = float(self.handoff_probability(self._observation))
        self._betas.append(beta)
        behaviour = min(max(beta, self.config.behaviour_handoff_min), self.config.behaviour_handoff_max)
        if self._rng.random() < behaviour:
            return self._hand_off()
        observation, reward, terminated, truncated, info = self.env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch="learned"
        )
        self._learned_decisions += 1
        self._observation = observation
        info = dict(info)
        info["handoff"] = False
        if terminated or truncated:
            info.update(self._summary(handoff=False, suffix=0))
        return observation, reward, terminated, truncated, info

    def _hand_off(self):
        zero = np.zeros(self.env.action_space.shape, dtype=np.float64)
        states = [np.asarray(self._observation, dtype=np.float32).copy()]
        rewards: list[float] = []
        while True:
            observation, reward, terminated, truncated, info = self.env.step_with_branch(
                zero, branch="baseline"
            )
            rewards.append(float(reward))
            if terminated or truncated:
                break
            states.append(np.asarray(observation, dtype=np.float32).copy())
        self._observation = observation
        info = dict(info)
        info.update(self._summary(handoff=True, suffix=len(rewards)))
        info["handoff_states"] = np.stack(states)
        info["handoff_returns"] = discounted_returns_to_go(rewards, self.config.gamma)
        # Monitor's episode return: the undiscounted sum, like a learned episode.
        return observation, float(np.sum(rewards)), terminated, truncated, info


class StoppingSAC(SAC):
    """SB3 SAC for the task action, plus the stopping head and ``Q_H``."""

    def __init__(
        self,
        policy: Any,
        env: Any,
        *,
        stopping: dict[str, Any] | None = None,
        handoff_feature_index: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.stopping = dict(stopping if stopping is not None else asdict(STOPPING))
        self.handoff_feature_index = list(handoff_feature_index or [])
        super().__init__(policy, env, **kwargs)

    # -- model setup and persistence ------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        if not self.handoff_feature_index:
            raise ValueError("handoff_feature_index is required")
        config = StoppingConfig(**self.stopping)
        if abs(config.gamma - self.gamma) > 0.0:
            raise ValueError("SAC gamma and the stopping gamma must agree")
        net_arch = list(self.policy_kwargs.get("net_arch", [256, 256]))
        observation_dim = int(np.prod(self.observation_space.shape))
        self.stop = StopNetworks(
            observation_dim, self.handoff_feature_index, net_arch, config.initial_handoff_logit
        ).to(self.device)
        lr = float(self.lr_schedule(1.0))
        self.stop_head_optimizer = th.optim.Adam(self.stop.stop_head.parameters(), lr=lr)
        self.q_handoff_optimizer = th.optim.Adam(
            list(self.stop.q_handoff_1.parameters()) + list(self.stop.q_handoff_2.parameters()), lr=lr
        )
        capacity = int(self.buffer_size)
        self._handoff_obs = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._handoff_returns = np.zeros(capacity, dtype=np.float32)
        self._handoff_pos = 0
        self._handoff_full = False
        self._handoff_rng = np.random.default_rng(None if self.seed is None else int(self.seed) + 7919)
        self._pending_gradient_steps = 0
        self.handoff_episodes = 0
        self.handoff_labels = 0

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts + ["stop", "stop_head_optimizer", "q_handoff_optimizer"], variables

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "stop", "stop_head_optimizer", "q_handoff_optimizer",
            "_handoff_obs", "_handoff_returns", "_handoff_rng",
        ]

    # -- the stopping policy --------------------------------------------------------

    @property
    def stopping_config(self) -> StoppingConfig:
        return StoppingConfig(**self.stopping)

    def handoff_probability(self, observation: np.ndarray) -> float:
        with th.no_grad():
            obs = th.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).reshape(1, -1)
            return float(th.sigmoid(self.stop.logit(obs)).item())

    def stopping_values(self, observation: np.ndarray) -> dict[str, float]:
        """Diagnostics at one state: beta, Q_H, and Q_C at the deterministic action."""

        with th.no_grad():
            obs = th.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).reshape(1, -1)
            action = self.actor(obs, deterministic=True)
            q_continue = th.min(*self.critic(obs, action))
            return {
                "beta": float(th.sigmoid(self.stop.logit(obs)).item()),
                "q_handoff": float(self.stop.q_handoff_min(obs).item()),
                "q_continue": float(q_continue.item()),
            }

    # -- data -----------------------------------------------------------------------

    @property
    def handoff_buffer_size(self) -> int:
        return len(self._handoff_returns) if self._handoff_full else self._handoff_pos

    def add_handoff_labels(self, states: np.ndarray, returns: np.ndarray) -> None:
        for state, value in zip(np.asarray(states), np.asarray(returns)):
            self._handoff_obs[self._handoff_pos] = state
            self._handoff_returns[self._handoff_pos] = value
            self._handoff_pos += 1
            if self._handoff_pos == len(self._handoff_returns):
                self._handoff_pos, self._handoff_full = 0, True
        self.handoff_labels += len(returns)

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos) -> None:
        info = infos[0]
        if not info.get("handoff"):
            super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
            return
        # The handoff step is not a continue transition: no task action was
        # executed. It contributes Q_H labels, and its extra simulated decisions
        # count toward the budget and earn their gradient updates.
        self.add_handoff_labels(info["handoff_states"], info["handoff_returns"])
        self.handoff_episodes += 1
        extra = int(info["suffix_decisions"]) - 1
        self.num_timesteps += extra
        # Only the suffix decisions past learning_starts earn an update.
        self._pending_gradient_steps += max(0, min(extra, self.num_timesteps - self.learning_starts))
        self._last_obs = new_obs

    # -- updates --------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        gradient_steps += self._pending_gradient_steps
        self._pending_gradient_steps = 0
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.stop_head_optimizer, self.q_handoff_optimizer]
        self._update_learning_rate(optimizers)
        config = self.stopping_config
        ent_coef = self.ent_coef_tensor
        if self.ent_coef_optimizer is not None:
            raise ValueError("the stopping method keeps the entropy coefficient fixed")

        actor_losses, critic_losses, q_handoff_losses, stop_losses, betas, gaps = [], [], [], [], [], []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            have_handoff = self.handoff_buffer_size > 0
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q = th.min(*self.critic_target(replay_data.next_observations, next_actions))
                next_v = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                if have_handoff:
                    next_v = soft_stopping_value(
                        self.stop.logit(replay_data.next_observations),
                        self.stop.q_handoff_min(replay_data.next_observations),
                        next_v,
                        config.stop_ent_coef,
                    )
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v

            current_q = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            min_qf_pi = th.min(*self.critic(replay_data.observations, actions_pi))
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if have_handoff:
                size = self.handoff_buffer_size
                index = self._handoff_rng.integers(0, size, size=batch_size)
                obs_h = th.as_tensor(self._handoff_obs[index], device=self.device)
                ret_h = th.as_tensor(self._handoff_returns[index], device=self.device).reshape(-1, 1)
                q1, q2 = self.stop.q_handoff(obs_h)
                q_handoff_loss = 0.5 * (F.mse_loss(q1, ret_h) + F.mse_loss(q2, ret_h))
                q_handoff_losses.append(q_handoff_loss.item())
                self.q_handoff_optimizer.zero_grad()
                q_handoff_loss.backward()
                self.q_handoff_optimizer.step()

                with th.no_grad():
                    v_continue = (min_qf_pi - ent_coef * log_prob).detach()
                    q_handoff_s = self.stop.q_handoff_min(replay_data.observations)
                logit = self.stop.logit(replay_data.observations)
                stop_loss = -soft_stopping_value(logit, q_handoff_s, v_continue, config.stop_ent_coef).mean()
                stop_losses.append(stop_loss.item())
                self.stop_head_optimizer.zero_grad()
                stop_loss.backward()
                self.stop_head_optimizer.step()
                betas.append(float(th.sigmoid(logit).mean().item()))
                gaps.append(float((q_handoff_s - v_continue).mean().item()))

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(ent_coef.item()))
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
        if q_handoff_losses:
            self.logger.record("stop/q_handoff_loss", np.mean(q_handoff_losses))
            self.logger.record("stop/stop_loss", np.mean(stop_losses))
            self.logger.record("stop/beta_mean", np.mean(betas))
            self.logger.record("stop/q_handoff_minus_v_continue", np.mean(gaps))
        self.logger.record("stop/handoff_episodes", self.handoff_episodes)
        self.logger.record("stop/handoff_labels", self.handoff_labels)


class SimulatedDecisionCheckpoint(BaseCallback):
    """Save the model each time the simulated-decision count crosses a multiple."""

    def __init__(self, every: int, directory: Any, prefix: str = "stopping") -> None:
        super().__init__()
        self.every = int(every)
        self.directory = directory
        self.prefix = prefix
        self.next = int(every)

    def _on_step(self) -> bool:
        while self.model.num_timesteps >= self.next:
            self.model.save(self.directory / f"{self.prefix}_{self.next}_decisions")
            self.next += self.every
        return True


__all__ = [
    "HANDOFF_FEATURE_BLOCKS",
    "STOPPING",
    "STOPPING_SAC",
    "HandoffOptionEnv",
    "SimulatedDecisionCheckpoint",
    "StopNetworks",
    "StoppingConfig",
    "StoppingSAC",
    "discounted_returns_to_go",
    "handoff_feature_index",
    "soft_stopping_value",
    "stopping_configs",
]
