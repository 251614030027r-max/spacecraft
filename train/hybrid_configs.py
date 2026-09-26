"""SAC hyperparameters for the coupled run, and why they differ from Pure SAC.

Only two values move against ``train.configs.PURE_SAC``, and both move because
the *step* means something different, not because they were tuned.

``gamma`` 0.997 -> 0.99
    A Pure SAC step is one 0.1 s control step, so a 300 s episode is about 950
    of them and the completion bonus is worth ``0.997**950 = 0.057`` at reset
    -- the documented reason hovering captures most of the scripted return
    without ever finishing. A coupled step is one 2 s decision, so the same
    episode is about 150 steps. Holding 0.997 there would give an effective
    horizon of 333 decisions, longer than the whole episode, which flattens
    the discount and removes any pressure to finish early. At 0.99 the horizon
    is 100 decisions -- 200 s, covering the episode -- and the bonus is worth
    ``0.99**150 = 0.22``, four times what Pure SAC ever sees. That is a
    structural consequence of the temporal abstraction and is the mechanism by
    which the coupling can represent a decision made 100 s before it pays.

``learning_starts`` 5000 -> 2000
    5000 coarse steps is 33 whole episodes of uniform random waypoints before
    the first gradient. 2000 is about 13, which is enough to seed the buffer.

``ent_coef`` stays pinned at 0.005. The entropy temperature is the one factor
this project has measured to destruction: automatic tuning raised alpha
monotonically in every run ever made here and drove the critic calibration
error to 45.7. It is not reopened.

``timeout_is_terminal`` True (2026-09-26, before v3d)
    The 300 s limit is part of the task and the normalised remaining time is
    in every coupled observation, so a state at the limit is terminal, not a
    cut-off of an episode that could go on. Stable-Baselines3 by default
    treats ``truncated`` as a time-limit cut-off and bootstraps
    ``gamma * Q(s_T)`` through it -- evaluating the critic at remaining time 0,
    a region it never trains on, and crediting a run-out-the-clock episode
    with the value of states that still had time to finish. With the time in
    the observation the correct target at the limit is the reward alone
    (Pardo et al., "Time Limits in Reinforcement Learning", ICML 2018). This
    also makes the SAC critic's target agree with the full-return regression
    the V3 value heads use. Timeout reward is unchanged (0).

Known deviation, recorded rather than fixed: at one update per decision step
the update-to-data ratio per second of simulated time is about five times
lower than Pure SAC's. Raising ``gradient_steps`` is the obvious lever and is
also the factor that made alpha diverge once, so it is left alone until there
is a measured reason to move it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class HybridSACConfig:
    learning_rate: float = 1.0e-4
    buffer_size: int = 300_000
    learning_starts: int = 2_000
    batch_size: int = 256
    tau: float = 0.001
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = 0.005
    target_entropy: float = -3.0
    target_update_interval: int = 1
    net_arch: tuple[int, ...] = (256, 256, 256, 256)
    timeout_is_terminal: bool = True


SAC_MPC_HYBRID = HybridSACConfig()


def hybrid_model_kwargs(config: HybridSACConfig = SAC_MPC_HYBRID) -> dict[str, Any]:
    values = asdict(config)
    net_arch = list(values.pop("net_arch"))
    values["policy_kwargs"] = {"net_arch": net_arch, "activation_fn": nn.ReLU}
    # SB3 bootstraps through a transition only when the replay buffer's
    # timeout handling marks it as a time-limit cut-off; turning the handling
    # off makes the task's time limit an ordinary terminal.
    values["replay_buffer_kwargs"] = {
        "handle_timeout_termination": not values.pop("timeout_is_terminal")
    }
    return values


def serializable_hybrid_hyperparameters(
    config: HybridSACConfig = SAC_MPC_HYBRID,
) -> dict[str, Any]:
    values = asdict(config)
    values["net_arch"] = list(values["net_arch"])
    values["activation_fn"] = "ReLU"
    return values


__all__ = [
    "SAC_MPC_HYBRID",
    "HybridSACConfig",
    "hybrid_model_kwargs",
    "serializable_hybrid_hyperparameters",
]
