"""T1-N learner (Research Plan R2 sec. 6-7): the TCPF learner on the T1-N interface.

Same update as ``train.t1_tcpf.TCPFSAC`` with ``uniform_eligibility`` (every arm trains only on
continue samples and on review samples whose interval has closed; the actor samples closed
reviews), adapted to the T1-N observation (``env.t1_n_review``) and its one-dimensional action:

* the continue action is ``[0]`` (keep); the keep-band equivalence term acts on review states
  with a live plan outside the descent (``|a| <= 0.5`` is executed identically there);
* the relative MPC term uses the *paired* label of the T1-N shadow (``env.t1_n_cf``):
      delta = (R^(a~) - R^(a)) + gamma^h [ Vbar(z^'(a~)) - Vbar(z^'(a)) ]
      L_delta = 0.5 sum_j mean_B[ w Huber_10( (Q_j(z, a~) - Q_j(z, a)) - delta ) ]
  with both h-step predictions made by the same shadow from the same state (stop-gradient);
  the weight of the term is calibrated once from gradients (5 % median share of the TD gradient
  after the warm-up) and capped at 10 % per update, as in ``train.t1_option_sac``.

The four R2 arms differ only in the target of an eligible review sample (one step or the closed
interval) and in whether the relative term is on (``train.train_t1_n``).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch as th

from env.t1_n_review import N_DESCENT_INDEX, N_DUE_INDEX, N_GAP_INDEX, N_PLAN_REMAINING_INDEX
from train.t1_tcpf import IntervalReplayBuffer, TCPFSAC


class PairedBatch(NamedTuple):
    actions: th.Tensor
    rewards: th.Tensor
    next_observations: th.Tensor
    weights: th.Tensor
    real_rewards: th.Tensor
    real_next_observations: th.Tensor
    discount: th.Tensor


class T1NReplayBuffer(IntervalReplayBuffer):
    """Interval buffer plus the predicted executed-action branch of each paired sample."""

    due_index = N_DUE_INDEX

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        n, e = self.buffer_size, self.n_envs
        obs_dim = int(np.prod(self.obs_shape))
        self.cf_real_rewards = np.zeros((n, e, 1), dtype=np.float32)
        self.cf_real_next_observations = np.zeros((n, e, obs_dim), dtype=np.float32)
        self.cf_discount = np.zeros((n, e, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        i = self.pos
        for e in range(self.n_envs):
            cf = infos[e].get("counterfactual") if e < len(infos) else None
            valid = bool(cf and cf.get("w") == 1 and cf.get("a_tilde") is not None)
            if valid:
                self.cf_real_rewards[i, e] = np.float32(cf["r_hat_real"])
                self.cf_real_next_observations[i, e] = np.asarray(cf["z_hat_real"], dtype=np.float32)
                self.cf_discount[i, e] = np.float32(self.gamma ** int(cf["h_eff"]))
            else:
                self.cf_real_rewards[i, e] = 0.0
                self.cf_real_next_observations[i, e] = 0.0
                self.cf_discount[i, e] = 0.0
        super().add(obs, next_obs, action, reward, done, infos)

    def counterfactual(self) -> PairedBatch:  # type: ignore[override]
        base = super().counterfactual()
        assert self._last is not None
        i, e = self._last
        return PairedBatch(base.actions, base.rewards, base.next_observations, base.weights,
                           self.to_torch(self.cf_real_rewards[i, e]),
                           self.to_torch(self.cf_real_next_observations[i, e]),
                           self.to_torch(self.cf_discount[i, e]))


class T1NSAC(TCPFSAC):
    due_index = N_DUE_INDEX
    gap_index = N_GAP_INDEX

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("replay_buffer_class", T1NReplayBuffer)
        kwargs["uniform_eligibility"] = True
        super().__init__(*args, **kwargs)

    def continue_actions(self, observations: th.Tensor) -> th.Tensor:
        return th.zeros((observations.shape[0], 1), dtype=observations.dtype, device=observations.device)

    def _eq_mask(self, observations: th.Tensor) -> th.Tensor:
        plan = observations[:, N_PLAN_REMAINING_INDEX:N_PLAN_REMAINING_INDEX + 1] > 0.0
        descent = observations[:, N_DESCENT_INDEX:N_DESCENT_INDEX + 1] > 0.5
        due = observations[:, N_DUE_INDEX:N_DUE_INDEX + 1] > 0.5
        return (plan & ~descent & due).float()

    def _delta_target(self, data, cf: PairedBatch, v_next: th.Tensor) -> th.Tensor:
        v_alt = self.target_value(cf.next_observations)
        v_real = self.target_value(cf.real_next_observations)
        return (cf.rewards - cf.real_rewards) + cf.discount * (v_alt - v_real)

    def _audit(self, replay_data, current_q_values, n: int) -> None:
        """Q - G by mode and the keep-band Q spread (the T1-L audit on the T1-N indices)."""

        with th.no_grad():
            obs = replay_data.observations
            returns, mask = self.replay_buffer.monte_carlo()  # type: ignore[union-attr]
            q = th.min(th.cat([c.detach() for c in current_q_values], dim=1), dim=1, keepdim=True)[0]
            plan = obs[:, N_PLAN_REMAINING_INDEX:N_PLAN_REMAINING_INDEX + 1] > 0.0
            descent = obs[:, N_DESCENT_INDEX:N_DESCENT_INDEX + 1] > 0.5
            groups = {"plan": plan & ~descent, "no_plan": ~plan & ~descent, "descent": descent}
            for name, g in groups.items():
                m = g.float() * mask
                count = float(m.sum().item())
                self.fix_stats[f"n_{name}"].append(count)
                self.fix_stats[f"q_minus_g_{name}"].append(
                    float(((q - returns) * m).sum().item() / count) if count else float("nan"))
            spread = float("nan")
            rows = self._eq_mask(obs)[:, 0] > 0.0
            if bool(rows.any()):
                s = obs[rows]
                values = [th.min(th.cat(self.critic(s, th.full((s.shape[0], 1), float(a0))), dim=1), dim=1,
                                 keepdim=True)[0] for a0 in np.linspace(-0.5, 0.5, 11)]
                v = th.cat(values, dim=1)
                spread = float((v.max(dim=1)[0] - v.min(dim=1)[0]).mean().item())
            self.fix_stats["band_q_spread"].append(spread)
            self.fix_stats["audit_update"].append(float(n))
            self.fix_stats["audit_decisions"].append(float(self.num_timesteps))


__all__ = ["PairedBatch", "T1NReplayBuffer", "T1NSAC"]
