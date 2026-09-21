"""Diagnostic scaffolding for the adaptive coupling defect."""

from __future__ import annotations

import numpy as np
import pytest

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
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


def _residual_env() -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(
        hybrid_config=PrecaptureHybridConfig(
            waypoint_parametrization="arrival_condition",
            baseline_anchored_residual=True,
            runtime_diagnostics=False,
        )
    )


@pytest.mark.xfail(
    strict=True,
    reason="positive commit residual intervals collapse at the +1 nominal boundary",
)
def test_every_commit_residual_interval_changes_the_reference() -> None:
    references = []
    for residual in np.linspace(-1.0, 1.0, 21):
        env = _residual_env()
        try:
            env.reset(seed=262000)
            references.append(env.waypoint_from_action([residual, 0.0]))
        finally:
            env.close()
    unique = np.unique(np.round(np.asarray(references), decimals=9), axis=0)
    assert len(unique) == 21


@pytest.mark.xfail(
    strict=True,
    reason="a gate fallback currently mutates the irreversible commit latch",
)
def test_gate_fallback_does_not_advance_commit_latch() -> None:
    env = _residual_env()
    try:
        env.reset(seed=262000)
        before = env._commit_blend
        env.waypoint_from_action(np.zeros(2))
        assert env._commit_blend == before
    finally:
        env.close()


@pytest.mark.xfail(
    strict=True,
    reason="after full commit the current ratchet makes the entire action grid inert",
)
def test_latched_action_space_is_inert() -> None:
    env = _residual_env()
    try:
        env.reset(seed=262000)
        env.waypoint_from_action(np.zeros(2))  # latch full commit
        references = [
            env.waypoint_from_action([commit, radius])
            for commit in np.linspace(-1.0, 1.0, 5)
            for radius in np.linspace(-1.0, 1.0, 5)
        ]
        unique = np.unique(np.round(np.asarray(references), decimals=9), axis=0)
        assert len(unique) > 1
    finally:
        env.close()
    # The collapsed positive half of the residual channel is made explicit.
    assert arrival_blend_before_ratchet(
        np.array([0.8, 0.0]), baseline_anchored_residual=True
    ) == 1.0


def test_staged_residual_hold_level_selects_the_staging_radius() -> None:
    """The hold level decides *where* the arm stages, which is the question.

    -1.0 keeps the frozen start radius, i.e. sit still and do not approach at
    all; -0.265 maps to blend 0.735, about 7.4 m from a 20 m start, which is
    the radius the offline feasibility certificate stages at on the hard cases.
    An arm fixed at -1.0 only answers whether sitting at the start radius is
    worth anything, so a null result from it must not be read as staging having
    no value.
    """

    assert staged_residual_action(0, 5)[0] == pytest.approx(-1.0)
    assert staged_residual_action(0, 5, -0.265)[0] == pytest.approx(-0.265)
    # Past the staging phase every level releases to the same nominal.
    assert staged_residual_action(5, 5, -0.265) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0, -1.5])
def test_staged_residual_rejects_hold_levels_that_are_not_holds(
    level: float,
) -> None:
    # A non-negative commit residual maps to full commit, so it is not a hold;
    # below -1 is outside the action box.
    with pytest.raises(ValueError):
        staged_residual_action(0, 5, level)
