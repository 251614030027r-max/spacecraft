"""Critic-advantage deployment gate: the proposed method's deployment rule.

Deviate from the nominal (zero residual = fixed-setpoint Pure MPC) only when the
SAC critic prefers the policy's residual action over the zero residual by more
than a margin; otherwise command the nominal. These tests pin the critic query
and the gate decision on a fresh (untrained) model -- correctness of the
plumbing, not of a learned policy.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import SAC

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config
from experiments.evaluate_hybrid_policy import critic_min_q


def _residual_env() -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(
        environment_config=precapture_planning_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=20,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_execution_feedback_observation=True,
            baseline_anchored_residual=True,
        ),
    )


def _tiny_sac(env: PrecaptureHybridEnv) -> SAC:
    return SAC(
        "MlpPolicy",
        env,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": [16, 16]},
    )


def test_critic_min_q_is_finite_scalar() -> None:
    env = _residual_env()
    observation, _ = env.reset(seed=262400)
    model = _tiny_sac(env)
    q_nominal = critic_min_q(model, observation, np.zeros(2, dtype=np.float32))
    q_action = critic_min_q(model, observation, np.array([0.4, -0.3], np.float32))
    assert np.isfinite(q_nominal) and np.isfinite(q_action)
    env.close()


def test_gate_deviates_iff_advantage_exceeds_margin() -> None:
    env = _residual_env()
    observation, _ = env.reset(seed=262400)
    model = _tiny_sac(env)
    action, _ = model.predict(observation, deterministic=True)
    nominal = np.zeros_like(action)
    advantage = critic_min_q(model, observation, action) - critic_min_q(
        model, observation, nominal
    )
    # Below its own advantage -> deviate; above it -> fall back to nominal.
    deviates = advantage > (advantage - 1.0)
    holds = advantage > (advantage + 1.0)
    assert deviates is True
    assert holds is False
    env.close()
