"""T1-P learner (research plan T1-P sec. 4-5): TCPF on the 3-D stage interface.

``train.t1_tcpf.TCPFSAC`` with ``uniform_eligibility`` on the T1-P observation and action:

* the continue action is ``[0, 0, 0]`` (the committed stage, the bridge or the descent fly on);
* the entropy coefficient is 0.005 x the review gap in 2 s steps (1..6);
* **no keep-band equivalence term** (lambda_eq = 0: a continuous 3-D stage action has no
  executed-equivalent band) and no MC label term;
* the relative term (arms D1 / TCPF) is the stage counterfactual of ``env.t1_p_cf`` anchored on the
  *real* closed stage (the interval fields of the same review sample):
      Y = [R^(a~) - R_real] + gamma^d [Vbar(z^(a~)) - Vbar(z_real)],   d = 6 (only six-step stages carry weight)
      L_cf = 0.5 sum_j mean_B[ w Huber_10( (Q_j(z, a~) - Q_j(z, a)) - sg(Y) ) ]
  with the 5 % / 10 % gradient-share weight of ``train.t1_option_sac``; its weight is masked by the
  sample eligibility (a review whose interval has not closed contributes nothing).
"""

from __future__ import annotations

from typing import Any

import torch as th

from env.t1_p_review import P_DUE_INDEX, P_GAP_INDEX, P_MODE_INDEX, STAGE_STEPS
from train.t1_tcpf import IntervalReplayBuffer, TCPFSAC


class T1PReplayBuffer(IntervalReplayBuffer):
    """Interval buffer; a stage pair is stored at its review with weight 0 ("pending") and gets its
    final weight when the environment closes the stage (``info["cf_final"]``)."""

    due_index = P_DUE_INDEX

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending: list[tuple[int, int] | None] = [None] * self.n_envs
        self.cf_counts = {"pairs": 0, "weighted": 0, "cancelled_overwritten": 0}

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        i = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        for e in range(self.n_envs):
            info = infos[e] if e < len(infos) else {}
            cf = info.get("counterfactual")
            if cf and cf.get("w") == 1:
                self.cf_weights[i, e] = 0.0  # pending until the stage closes
                self._pending[e] = (i, int(self.add_index[i, e]))
                self.cf_counts["pairs"] += 1
            final = info.get("cf_final")
            if final is not None and self._pending[e] is not None:
                slot, index = self._pending[e]
                if self.add_index[slot, e] == index:
                    self.cf_weights[slot, e] = float(final["weight"])
                    self.cf_counts["weighted"] += int(final["weight"] > 0.0)
                else:
                    self.cf_counts["cancelled_overwritten"] += 1
                self._pending[e] = None


class T1PSAC(TCPFSAC):
    due_index = P_DUE_INDEX
    gap_index = P_GAP_INDEX

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("replay_buffer_class", T1PReplayBuffer)
        kwargs["uniform_eligibility"] = True
        if kwargs.get("eq_lambda", 0.0) != 0.0:
            raise ValueError("T1-P has no keep-band equivalence (lambda_eq must be 0)")
        super().__init__(*args, **kwargs)

    def alpha_of(self, observations: th.Tensor) -> th.Tensor:
        gap = th.clamp(th.round(observations[:, self.gap_index:self.gap_index + 1] * STAGE_STEPS), 1.0, float(STAGE_STEPS))
        return self.ent_coef_tensor * gap

    def continue_actions(self, observations: th.Tensor) -> th.Tensor:
        return th.zeros((observations.shape[0], 3), dtype=observations.dtype, device=observations.device)

    def _eq_mask(self, observations: th.Tensor) -> th.Tensor:
        return th.zeros((observations.shape[0], 1), dtype=observations.dtype, device=observations.device)

    def _delta_target(self, data, cf, v_next: th.Tensor) -> th.Tensor:
        """Y of the module docstring: the real closed stage is the anchor."""

        _, ret, discount, end_obs, done = self.replay_buffer.interval()  # type: ignore[union-attr]
        v_alt = self.target_value(cf.next_observations)
        v_real = self.target_value(end_obs)
        return (cf.rewards - ret) + discount * (1.0 - done) * (v_alt - v_real)

    def _audit(self, replay_data, current_q_values, n: int) -> None:
        with th.no_grad():
            obs = replay_data.observations
            returns, mask = self.replay_buffer.monte_carlo()  # type: ignore[union-attr]
            q = th.min(th.cat([c.detach() for c in current_q_values], dim=1), dim=1, keepdim=True)[0]
            for name, index in P_MODE_INDEX.items():
                m = (obs[:, index:index + 1] > 0.5).float() * mask
                count = float(m.sum().item())
                self.fix_stats.setdefault(f"n_{name}", []).append(count)
                self.fix_stats.setdefault(f"q_minus_g_{name}", []).append(
                    float(((q - returns) * m).sum().item() / count) if count else float("nan"))
            self.fix_stats["audit_update"].append(float(n))
            self.fix_stats["audit_decisions"].append(float(self.num_timesteps))


__all__ = ["T1PReplayBuffer", "T1PSAC"]
