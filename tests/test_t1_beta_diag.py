"""Beta diagnostics: planner wiring only (no closed-loop simulation)."""

from __future__ import annotations

from diagnostics.t1_beta.beta_probe import RecordingPlanner, SmoothRendezvousPlanner, install
from env.t1_gym import T1GStarEnv


def test_beta_zero_wraps_the_frozen_planner_and_beta_positive_adds_one_term():
    env = T1GStarEnv(openings=[295009])
    env.reset()
    frozen = env.generator.planner
    install(env, 0.0)
    assert isinstance(env.generator.planner, RecordingPlanner) and env.generator.planner.inner is frozen
    assert env.generator.planner.nodes == frozen.nodes and env.generator.planner.mass == frozen.mass
    env.reset()
    install(env, 1.0)
    smooth = env.generator.planner.inner
    assert isinstance(smooth, SmoothRendezvousPlanner) and smooth.problem.is_dpp()
    assert smooth.max_force_n == env.generator.recovery_planner.max_force_n * 0.8
