"""Regression guards for the adaptive-mainline reward's physical units.

These tests pin the exact failure mode that invalidated the first adaptive
training round: every continuous cost is a rate integrated over ``dt`` while
terminal events are one-shot dimensionless rewards.
"""

from __future__ import annotations

from dataclasses import replace
import sys

import numpy as np
import pytest

from env.phase2_env import precapture_adaptive_capture_environment_config
from env.reward import PrecaptureReward
from env.se3_rendezvous_env import SE3RendezvousEnv
from env.task import compute_precapture_metrics
from train.train_hybrid import parse_args


def _outer_metrics():
    config = precapture_adaptive_capture_environment_config()
    env = SE3RendezvousEnv(config)
    env.reset(seed=262410)
    assert env.target_state is not None
    assert env.chaser_state is not None
    assert env.relative is not None
    metrics = compute_precapture_metrics(
        env.target_state,
        env.chaser_state,
        env.relative,
        config.precapture_task,
        terminal_region_active=False,
        staging_direction_inertial=env._staging_direction_inertial,
    )
    env.close()
    # Isolate one full-strength keep-out proximity warning. All other active
    # margins are placed well outside their 10% warning buffers.
    return config, replace(
        metrics,
        keepout_margin_m=0.0,
        fov_margin_rad=10.0,
        outer_inertial_speed_margin_m_s=10.0,
        outer_radial_margin_m_s=10.0,
        transition_speed_active=False,
    )


@pytest.mark.parametrize("dt_s", [0.05, 0.1, 0.2])
def test_all_continuous_reward_costs_scale_with_time_step(dt_s: float) -> None:
    config, metrics = _outer_metrics()
    reward = PrecaptureReward(
        task=config.precapture_task,
        settings=config.precapture_reward,
        time_step_s=dt_s,
    )
    reward.reset(metrics)
    breakdown = reward.compute(metrics, np.ones(6))

    assert breakdown.time_penalty == pytest.approx(
        -config.precapture_reward.time_weight * dt_s
    )
    assert breakdown.force_penalty == pytest.approx(
        -config.precapture_reward.force_weight * dt_s
    )
    assert breakdown.torque_penalty == pytest.approx(
        -config.precapture_reward.torque_weight * dt_s
    )
    assert breakdown.safety_penalty == pytest.approx(
        -config.precapture_reward.safety_weight * dt_s
    )


def test_terminal_events_keep_declared_sign_and_are_not_time_scaled() -> None:
    config, metrics = _outer_metrics()
    action = np.zeros(6)

    success = PrecaptureReward(
        task=config.precapture_task,
        settings=config.precapture_reward,
        time_step_s=config.dt_s,
    )
    success.reset(metrics)
    success_step = success.compute(
        metrics, action, event_reward=config.precapture_reward.success_reward
    )

    failure = PrecaptureReward(
        task=config.precapture_task,
        settings=config.precapture_reward,
        time_step_s=config.dt_s,
    )
    failure.reset(metrics)
    failure_step = failure.compute(
        metrics, action, event_reward=config.precapture_reward.failure_penalty
    )

    assert success_step.event_reward == 20.0
    assert failure_step.event_reward == -20.0
    assert success_step.total - failure_step.total == pytest.approx(40.0)
    assert success_step.total > 0.0 > failure_step.total


def test_hybrid_training_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.train_hybrid",
            "--steps",
            "1",
            "--seed",
            "262410",
            "--run-name",
            "preflight",
        ],
    )
    assert parse_args().device == "cpu"
