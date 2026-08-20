"""Canonical Pure SAC configuration for Phase-2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class PureSACConfig:
    learning_rate: float = 1.0e-4
    buffer_size: int = 300_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.001
    gamma: float = 0.997
    train_freq: int = 4
    gradient_steps: int = 4
    ent_coef: str | float = "auto_0.005"
    target_entropy: float = -6.0
    target_update_interval: int = 1
    net_arch: tuple[int, ...] = (256, 256, 256, 256)


PURE_SAC = PureSACConfig()
# Compatibility name for manifests/tests; there is only one active value.
SAC_HYPERPARAMETERS = PURE_SAC


def model_kwargs(config: PureSACConfig = PURE_SAC) -> dict[str, Any]:
    values = asdict(config)
    net_arch = list(values.pop("net_arch"))
    values["policy_kwargs"] = {"net_arch": net_arch, "activation_fn": nn.ReLU}
    return values


def serializable_hyperparameters(
    config: PureSACConfig = PURE_SAC,
) -> dict[str, Any]:
    values = asdict(config)
    values["net_arch"] = list(values["net_arch"])
    values["activation_fn"] = "ReLU"
    return values


__all__ = [
    "PURE_SAC",
    "PureSACConfig",
    "model_kwargs",
    "serializable_hyperparameters",
]
