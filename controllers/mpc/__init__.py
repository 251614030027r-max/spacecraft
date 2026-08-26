"""MPC-only baseline for the SE(3) rendezvous task."""

from .config import (
    MPCConfig,
    constrained_mpc_nominal_config,
    corridor_tracking_mpc_config,
)
from .controller import MPCController, MPCStepDiagnostics
from .prediction import LocalRelativePredictionModel, RelativePredictionModel

__all__ = [
    "LocalRelativePredictionModel",
    "MPCConfig",
    "constrained_mpc_nominal_config",
    "corridor_tracking_mpc_config",
    "MPCController",
    "MPCStepDiagnostics",
    "RelativePredictionModel",
]
