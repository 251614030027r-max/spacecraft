"""T1-L-S: task-consistent predictive feedback learner (plan 2026-10-16, sections 2-4).

Builds on ``train.t1_option_sac.OptionSAC`` (phase-aware SAC on the T1-L review interface, MC
and keep-band terms, NDelta relative MPC term) and changes only:

1. **Review-interval target** (``interval_bellman=True``). For a review at real 2 s step k,
   d_k is the number of real steps until the next real review state or the true end of the
   episode (usually 3; 1 or 2 after a plan event; 1 for B2). The critic target of that review
   sample is
       y = R_k + gamma^d (1 - D) Vbar(z_{k+d}),   R_k = sum_{i<d} gamma^i r_{k+i},
   with Vbar the mode-aware value of ``OptionSAC.target_value`` (the successor's own due flag).
   D = 1 for success / unsafe / timeout (no bootstrap); a numerical-failure truncation
   bootstraps from its terminal observation, as SB3 does for one step. Continue samples keep the
   one-step target. Each sample gets exactly one target; a review whose interval has not closed
   yet (the latest one of an environment) is masked out of the TD mean (mean over the remaining
   samples). The NDelta relative term keeps the *one-step* real transition (r_k, z_{k+1}) and
   never reads the interval fields. This is a multi-step target consistent with the executed
   interval: it removes the intermediate bootstraps through continue states; it does not remove
   function-approximation, bootstrap or policy-shift errors and is not claimed to be unbiased.

2. **Trainer configuration** (chosen per run, frozen before the formal round):
   * A: the 166244b trainer -- ``gradient_steps = 4 * n_envs``, one actor update per critic
     update, constant learning rate 1e-4;
   * B: ``gradient_steps = 1 * n_envs`` (one critic update per real transition), one actor
     update per ``actor_every_transitions`` = 3 real transitions on the *global* transition
     clock (``num_timesteps`` counts every environment's transitions; the updates owed are
     floor((num_timesteps - learning_starts) / 3) minus those done), cosine learning rate
     1e-4 -> 1e-5 over the real-transition progress (``CosineLearningRate``).
   With interval targets the actor samples only reviews whose interval has closed.

Replay (``IntervalReplayBuffer``): per environment, the open review interval accumulates the
discounted rewards of its own transitions; it closes when that environment's next observation
is a review state (end = that state, D = 0) or when that environment's episode ends (end = the
terminal observation SB3 stores, D = terminal and not truncated). A closed interval is written
to its start slot only if the slot still holds the same transition (``add_index``); otherwise
it is dropped and counted. A slot that is overwritten loses its interval fields.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch as th

from env.t1_task_review import DUE_INDEX
from train.t1_option_sac import OptionReplayBuffer, OptionSAC


class CosineLearningRate:
    """SB3 schedule (argument: progress remaining, 1 -> 0): lr = lo + (hi - lo)(1 + cos(pi p)) / 2,
    p = 1 - progress remaining = real transitions / total."""

    def __init__(self, high: float = 1e-4, low: float = 1e-5) -> None:
        self.high, self.low = float(high), float(low)

    def __call__(self, progress_remaining: float) -> float:
        p = min(max(1.0 - float(progress_remaining), 0.0), 1.0)
        return self.low + 0.5 * (self.high - self.low) * (1.0 + math.cos(math.pi * p))


class IntervalReplayBuffer(OptionReplayBuffer):
    """Option buffer plus the closed review interval of each review sample (module docstring)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        n, e = self.buffer_size, self.n_envs
        obs_dim = int(np.prod(self.obs_shape))
        self.interval_return = np.zeros((n, e, 1), dtype=np.float32)
        self.interval_discount = np.zeros((n, e, 1), dtype=np.float32)
        self.interval_end_obs = np.zeros((n, e, obs_dim), dtype=np.float32)
        self.interval_done = np.zeros((n, e, 1), dtype=np.float32)
        self.interval_ready = np.zeros((n, e), dtype=bool)
        self.interval_steps = np.zeros((n, e), dtype=np.int16)
        #: per environment: [start slot, start add index, discounted return, steps] of the open interval
        self._open: list[list | None] = [None] * e
        self.interval_counts: dict[str, int] = {"closed_by_review": 0, "closed_by_terminal": 0,
                                                "closed_by_truncation": 0, "dropped_overwritten": 0}
        self.interval_lengths: dict[int, int] = {}

    def _close(self, e: int, end_obs: np.ndarray, terminal: bool, how: str) -> None:
        slot, index, ret, steps = self._open[e]  # type: ignore[misc]
        self._open[e] = None
        # (an interval opened and closed within the current add: its slot is being written now)
        if slot != self.pos and self.add_index[slot, e] != index:
            self.interval_counts["dropped_overwritten"] += 1
            return
        self.interval_return[slot, e] = ret
        self.interval_discount[slot, e] = self.gamma ** steps
        self.interval_end_obs[slot, e] = end_obs
        self.interval_done[slot, e] = float(terminal)
        self.interval_ready[slot, e] = True
        self.interval_steps[slot, e] = steps
        self.interval_counts[how] += 1
        self.interval_lengths[steps] = self.interval_lengths.get(steps, 0) + 1

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        i = self.pos
        o = np.asarray(obs).reshape(self.n_envs, -1)
        nxt = np.asarray(next_obs).reshape(self.n_envs, -1)
        rewards = np.asarray(reward, dtype=np.float64).reshape(-1)
        dones = np.asarray(done).reshape(-1)
        index = self.n_adds  # the add index this transition receives in M1ReplayBuffer.add
        for e in range(self.n_envs):
            self.interval_ready[i, e] = False  # this slot now holds a new transition
            self.interval_steps[i, e] = 0
            if o[e, DUE_INDEX] > 0.5:
                if self._open[e] is not None:  # the previous interval ends at this review state
                    self._close(e, o[e], terminal=False, how="closed_by_review")
                self._open[e] = [i, index, 0.0, 0]
            if self._open[e] is not None:
                interval = self._open[e]
                interval[2] += (self.gamma ** interval[3]) * float(rewards[e])
                interval[3] += 1
                if bool(dones[e]):
                    info = infos[e] if e < len(infos) else {}
                    truncated = bool(info.get("TimeLimit.truncated", False))
                    self._close(e, nxt[e], terminal=not truncated,
                                how="closed_by_truncation" if truncated else "closed_by_terminal")
        super().add(obs, next_obs, action, reward, done, infos)

    def interval(self) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """(ready, return, discount, end observation, done) of the transitions of the last ``sample``."""

        assert self._last is not None
        i, e = self._last
        return (self.to_torch(self.interval_ready[i, e].astype(np.float32)[:, None]),
                self.to_torch(self.interval_return[i, e]), self.to_torch(self.interval_discount[i, e]),
                self.to_torch(self.interval_end_obs[i, e]), self.to_torch(self.interval_done[i, e]))

    def sample_closed_review(self, batch_size: int, env=None) -> th.Tensor | None:
        """Observations of review samples whose interval has closed (never touches ``_last``)."""

        upper = self.buffer_size if self.full else self.pos
        flat = np.flatnonzero((self.due[:upper] & self.interval_ready[:upper]).reshape(-1))
        if flat.size == 0:
            return None
        pick = flat[np.random.randint(0, flat.size, size=batch_size)]
        i, e = np.divmod(pick, self.n_envs)
        return self.to_torch(self._normalize_obs(self.observations[i, e, :], env))


class TCPFSAC(OptionSAC):
    def __init__(self, *args: Any, interval_bellman: bool = True, actor_every_transitions: int | None = None,
                 **kwargs: Any) -> None:
        self.interval_bellman = bool(interval_bellman)
        #: None: one actor update per critic update (trainer A); k: one per k real transitions (B)
        self.actor_every_transitions = actor_every_transitions
        self.actor_updates = 0
        self._actor_owed = 0
        kwargs.setdefault("replay_buffer_class", IntervalReplayBuffer)
        super().__init__(*args, **kwargs)
        self.tcpf_stats: dict[str, list[float]] = {key: [] for key in (
            "decisions", "review_fraction", "interval_ready_fraction", "td_review", "td_continue",
            "actor_updates", "learning_rate")}

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "tcpf_stats"]

    # -- TD targets ---------------------------------------------------------- #
    def _td_targets(self, data, v_next: th.Tensor) -> tuple[th.Tensor, th.Tensor | None]:
        one_step = data.rewards + (1 - data.dones) * self.gamma * v_next
        if not self.interval_bellman:
            return one_step, None
        ready, ret, discount, end_obs, done = self.replay_buffer.interval()  # type: ignore[union-attr]
        review = (data.observations[:, DUE_INDEX:DUE_INDEX + 1] > 0.5).float()
        v_end = self.target_value(end_obs)
        folded = ret + discount * (1 - done) * v_end
        target = review * folded + (1 - review) * one_step
        mask = (1 - review) + review * ready  # a review with an open interval is not trained on
        self._last_target_parts = (review, ready, folded, one_step)
        return target, mask

    def _td_loss(self, current_q, target, mask):
        loss = OptionSAC._td_loss(current_q, target, mask)
        parts = getattr(self, "_last_target_parts", None)
        if parts is not None and self._n_updates_in_call % 50 == 0:
            review, ready, _, _ = parts
            with th.no_grad():
                q = current_q[0]
                rm, cm = review * ready, (1 - review)
                s = self.tcpf_stats
                s["decisions"].append(float(self.num_timesteps))
                s["review_fraction"].append(float(review.mean().item()))
                s["interval_ready_fraction"].append(float((rm.sum() / review.sum().clamp(min=1)).item()))
                s["td_review"].append(float(((rm * (q - target) ** 2).sum() / rm.sum().clamp(min=1)).item()))
                s["td_continue"].append(float(((cm * (q - target) ** 2).sum() / cm.sum().clamp(min=1)).item()))
                s["actor_updates"].append(float(self.actor_updates))
                s["learning_rate"].append(float(self.actor.optimizer.param_groups[0]["lr"]))
        return loss

    # -- actor clock ------------------------------------------------------------ #
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.actor_every_transitions is not None:
            owed_total = max(self.num_timesteps - self.learning_starts, 0) // int(self.actor_every_transitions)
            self._actor_owed = max(owed_total - self.actor_updates, 0)
        super().train(gradient_steps, batch_size)

    def _actor_update_due(self, gradient_step: int, gradient_steps: int) -> bool:
        if self.actor_every_transitions is None:
            return True
        # the owed updates go to the last gradient steps of this call
        return gradient_step >= gradient_steps - self._actor_owed

    def _review_batch(self, batch_size: int) -> th.Tensor | None:
        buffer = self.replay_buffer
        batch = (buffer.sample_closed_review(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                 if self.interval_bellman else super()._review_batch(batch_size))
        if batch is not None:
            self.actor_updates += 1
        return batch


__all__ = ["CosineLearningRate", "IntervalReplayBuffer", "TCPFSAC"]
