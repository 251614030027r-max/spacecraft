"""MPC-only baseline for the SE(3) rendezvous task."""

from .config import (
    MPCConfig,
    constrained_mpc_nominal_config,
    corridor_tracking_mpc_config,
    learned_terminal_mpc_config,
)
from .controller import MPCController, MPCStepDiagnostics
from .prediction import LocalRelativePredictionModel, RelativePredictionModel
from .terminal_value import ConvexQuadraticTerminalValue, fit_convex_quadratic

__all__ = [
    "ConvexQuadraticTerminalValue",
    "LocalRelativePredictionModel",
    "MPCConfig",
    "constrained_mpc_nominal_config",
    "corridor_tracking_mpc_config",
    "fit_convex_quadratic",
    "learned_terminal_mpc_config",
    "MPCController",
    "MPCStepDiagnostics",
    "RelativePredictionModel",
]
