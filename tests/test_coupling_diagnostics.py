"""Diagnostic scaffolding for the adaptive coupling defect."""

from __future__ import annotations

import numpy as np

from experiments.evaluate_hybrid_policy import arrival_blend_before_ratchet


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
    # The collapsed positive half of the residual channel is made explicit.
    assert arrival_blend_before_ratchet(
        np.array([0.8, 0.0]), baseline_anchored_residual=True
    ) == 1.0

