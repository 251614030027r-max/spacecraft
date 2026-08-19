"""Single SAC configuration for the clean baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class SACHyperparameters:
    learning_rate: float = 2.0e-4
    buffer_size: int = 300_000
    learning_starts: int = 20_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.993
    train_freq: int = 4
    gradient_steps: int = 4
    ent_coef: str = "auto_0.01"
    target_update_interval: int = 1
    net_arch: tuple[int, ...] = (256, 256, 256, 256)


SAC_HYPERPARAMETERS = SACHyperparameters()
PHASE2_LEARNING_STARTS = 5_000
PHASE2_BEHAVIOR_INITIALIZED_LEARNING_STARTS = 1_000


def model_kwargs(
    *, phase2: bool = False, behavior_initialized: bool = False
) -> dict[str, Any]:
    values = asdict(SAC_HYPERPARAMETERS)
    if behavior_initialized:
        if not phase2:
            raise ValueError("behavior initialization is only defined for Phase-2")
        values["learning_starts"] = PHASE2_BEHAVIOR_INITIALIZED_LEARNING_STARTS
    elif phase2:
        values["learning_starts"] = PHASE2_LEARNING_STARTS
    net_arch = list(values.pop("net_arch"))
    values["policy_kwargs"] = {
        "net_arch": net_arch,
        "activation_fn": nn.ReLU,
    }
    return values


def serializable_hyperparameters(
    *, phase2: bool = False, behavior_initialized: bool = False
) -> dict[str, Any]:
    values = asdict(SAC_HYPERPARAMETERS)
    if behavior_initialized:
        if not phase2:
            raise ValueError("behavior initialization is only defined for Phase-2")
        values["learning_starts"] = PHASE2_BEHAVIOR_INITIALIZED_LEARNING_STARTS
    elif phase2:
        values["learning_starts"] = PHASE2_LEARNING_STARTS
    values["net_arch"] = list(values["net_arch"])
    values["activation_fn"] = "ReLU"
    return values
