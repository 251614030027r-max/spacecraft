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


SAC_MPC_HYBRID = HybridSACConfig()


def hybrid_model_kwargs(config: HybridSACConfig = SAC_MPC_HYBRID) -> dict[str, Any]:
    values = asdict(config)
    net_arch = list(values.pop("net_arch"))
    values["policy_kwargs"] = {"net_arch": net_arch, "activation_fn": nn.ReLU}
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
