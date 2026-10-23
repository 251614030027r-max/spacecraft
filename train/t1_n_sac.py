"""T1-N learner (Research Plan R2; research-layer ruling of 2026-10-18): the TCPF learner on the
T1-N interface with a cross-timescale ranking term.

Same update as ``train.t1_tcpf.TCPFSAC`` with ``uniform_eligibility`` (every arm trains only on
continue samples and on review samples whose interval has closed; the actor samples closed
reviews), adapted to the T1-N observation (``env.t1_n_review``) and its one-dimensional action:

* the continue action is ``[0]`` (keep); the keep-band equivalence term acts on review states
  with a live plan outside the descent (``|a| <= 0.5`` is executed identically there);
* the auxiliary term (arms D1 / TCPF) is a **confidence-weighted ranking constraint** on the
  same critic, from the cached signal D of ``env.t1_n_cf`` (G* plan value with the MPC 2 s
  closed-loop correction, alternative minus executed action) and its confidence w:
      s = sign(D),  L_rank = 0.5 sum_j mean_B[ w max(0, m - s (Q_j(z, a~) - Q_j(z, a))) ],  m = 0.5
  Only the direction of D is used (a finite margin, never a fit of its magnitude), and nothing
  is bootstrapped from a predicted state. The real long-horizon Q is learned by the interval
  (R3 / TCPF) or one-step (R1 / D1) Bellman target; the ranking term's weight is calibrated once
  from gradients to a 5 % median share of the TD gradient and capped at 10 % per update
  (``train.t1_option_sac``).

    L_Q = L_TD (interval or one step) + lambda_eq L_eq + lambda_p L_rank
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th

from env.t1_n_cf import T1NShadowConfig
from env.t1_n_review import N_DESCENT_INDEX, N_DUE_INDEX, N_GAP_INDEX, N_PLAN_REMAINING_INDEX
from train.t1_tcpf import IntervalReplayBuffer, TCPFSAC

RANK_MARGIN = T1NShadowConfig().rank_margin


class T1NReplayBuffer(IntervalReplayBuffer):
    """Interval buffer; the cached alternative holds (a~, D, w) with a continuous confidence w."""

    due_index = N_DUE_INDEX

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        i = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        for e in range(self.n_envs):
            cf = infos[e].get("counterfactual") if e < len(infos) else None
            self.cf_weights[i, e] = float(cf.get("rank_weight", 0.0)) if cf and cf.get("w") == 1 else 0.0


class T1NSAC(TCPFSAC):
    due_index = N_DUE_INDEX
    gap_index = N_GAP_INDEX

    def __init__(self, *args: Any, rank_margin: float = RANK_MARGIN, **kwargs: Any) -> None:
        self.rank_margin = float(rank_margin)
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

    def _delta_target(self, data, cf, v_next: th.Tensor) -> th.Tensor:
        """The cached signal D (no successor value: nothing predicted is bootstrapped)."""

        return cf.rewards

    def _relative_loss(self, alt_q, current_q, delta: th.Tensor, weights: th.Tensor) -> th.Tensor:
        direction = th.sign(delta)
        return 0.5 * sum((weights * th.relu(self.rank_margin - direction * (qa - qr))).mean()
                         for qa, qr in zip(alt_q, current_q))

    def _audit(self, replay_data, current_q_values, n: int) -> None:
        """Q - G by mode, the keep-band Q spread and the ranking agreement of the critic."""

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
            # does the critic order the cached pairs as the signal does? (all arms, read only)
            cf = self.replay_buffer.counterfactual()  # type: ignore[union-attr]
            valid = cf.weights[:, 0] > 0.0
            agree = float("nan")
            if bool(valid.any()):
                qa = th.min(th.cat(self.critic(obs[valid], cf.actions[valid]), dim=1), dim=1)[0]
                qr = th.min(th.cat(self.critic(obs[valid], replay_data.actions[valid]), dim=1), dim=1)[0]
                agree = float((th.sign(qa - qr) == th.sign(cf.rewards[valid, 0])).float().mean().item())
            self.fix_stats.setdefault("rank_agreement", []).append(agree)
            self.fix_stats["audit_update"].append(float(n))
            self.fix_stats["audit_decisions"].append(float(self.num_timesteps))


__all__ = ["RANK_MARGIN", "T1NReplayBuffer", "T1NSAC"]
