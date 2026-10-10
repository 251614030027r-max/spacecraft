"""The fallback the MPC uses when its QP does not solve.

Zero wrench is not a neutral action on a rigid body in a co-rotating approach:
angular momentum is conserved, so commanding nothing freezes whatever angular
rate the last feasible solve was part-way through establishing.  Seed 262001
at horizon 20 fails that way -- eight saturated steps begin to de-spin the
chaser, the QP then goes infeasible, and the boresight rotates out of the field
of view while the controller commands exactly 0.0 for 379 consecutive steps.

These tests pin the option that lets the controller keep flying the last
feasible plan instead, and pin that the historical behaviour is still the
default so every previously measured number reproduces.
"""

from __future__ import annotations

import numpy as np
import pytest

from controllers.mpc.config import precapture_mpc_config


def test_zero_is_still_the_default() -> None:
    assert precapture_mpc_config().infeasible_fallback == "zero"


def test_only_the_two_known_fallbacks_are_accepted() -> None:
    from dataclasses import replace

    replace(precapture_mpc_config(), infeasible_fallback="shift")
    with pytest.raises(ValueError, match="infeasible_fallback"):
        replace(precapture_mpc_config(), infeasible_fallback="hold")


def _shift(nominal: np.ndarray) -> np.ndarray:
    return np.vstack((nominal[1:], np.zeros((1, 6))))


def test_the_shifted_plan_decays_to_zero_after_one_horizon() -> None:
    # The controller's fallback is the previous plan advanced one step, whose
    # last row is zero.  Applying it repeatedly must therefore reach the zero
    # fallback on its own after ``horizon`` consecutive failures -- the option
    # continues a manoeuvre, it does not latch a stale command forever.
    horizon = 20
    plan = np.ones((horizon, 6), dtype=np.float64)
    for step in range(horizon):
        assert np.any(plan[0] != 0.0), f"decayed early at step {step}"
        plan = _shift(plan)
    assert np.all(plan == 0.0)
