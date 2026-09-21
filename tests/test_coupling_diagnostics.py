"""Diagnostic scaffolding for the adaptive coupling defect."""

from __future__ import annotations

import numpy as np

from experiments.evaluate_hybrid_policy import (
    arrival_blend_before_ratchet,
    staged_residual_action,
)


def test_arrival_blend_instrument_matches_residual_mapping() -> None:
    assert arrival_blend_before_ratchet(
        np.array([-1.0, 0.0]), baseline_anchored_residual=True
    ) == 0.0
    assert arrival_blend_before_ratchet(
        np.array([-0.5, 0.0]), baseline_anchored_residual=True
    ) == 0.5
    assert arrival_blend_before_ratchet(
        np.array([0.0, 0.0]), baseline_anchored_residual=True
    ) == 1.0


def test_staged_residual_holds_then_uses_zero_residual_nominal() -> None:
    np.testing.assert_array_equal(staged_residual_action(0, 5), [-1.0, 0.0])
    np.testing.assert_array_equal(staged_residual_action(4, 5), [-1.0, 0.0])
    np.testing.assert_array_equal(staged_residual_action(5, 5), [0.0, 0.0])
    # The collapsed positive half of the residual channel is made explicit.
    assert arrival_blend_before_ratchet(
        np.array([0.8, 0.0]), baseline_anchored_residual=True
    ) == 1.0
