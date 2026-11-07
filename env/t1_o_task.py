"""T1-O task profiles (research decision 2026-10-17): physical screening candidates P1 / P2.

Everything physical is T1's (``T1EnvConfig``): 6-DoF RK45 truth with J2, the shared MPC h35,
2 N / axis force, 0.6 N m / axis torque, the 300 s limit, the 30 m distance limit, the safety
rules and the capture condition, the initial inertial speed <= 0.10 m/s and the rest of the
T1 initial sampling. A profile changes only the initial range and tumble-rate intervals:

* ``p1``: 15-19 m, 2.1-2.7 deg/s (screened first);
* ``p2``: 14-18 m, 2.1-2.7 deg/s (screened only if P1 fails on far-field executability);
* ``o``: 15-19 m, 2.4-3.0 deg/s, the original T1-O (held: 6/20 unsafe in the exploratory check
  of 2026-10-17); kept only so that the record can be reproduced.

The task interface is the T1-L review wrapper with the first-plan delay range widened to
[0, 90] s after the feasible anchor (``T1O_DELAY_MAX_S``); the fuel weight of the reward is a
parameter (``fuel_cost``; T1's 5.0 by default) because the task Gate re-scores per-step
delta-v exactly for 10 / 15 / 20 (``info["delta_v_step_m_s"]``), it never changes the motion.
"""

from __future__ import annotations

from dataclasses import replace

from env.t1_env import T1EnvConfig
from env.t1_gym import T1GymConfig
from env.t1_task_review import T1LReviewEnv

T1O_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "p1": {"range_m": (15.0, 19.0), "tumble_rate_deg_s": (2.1, 2.7)},
    "p2": {"range_m": (14.0, 18.0), "tumble_rate_deg_s": (2.1, 2.7)},
    "o": {"range_m": (15.0, 19.0), "tumble_rate_deg_s": (2.4, 3.0)},
}
T1O_DELAY_MAX_S = 90.0


def t1o_env_config(profile: str, fuel_cost: float = 5.0) -> T1EnvConfig:
    spec = T1O_PROFILES[profile]
    base = T1EnvConfig()
    return replace(base, initial=replace(base.initial, **spec),
                   reward=replace(base.reward, delta_v_cost_per_m_s=float(fuel_cost)))


def t1o_review_env(profile: str, fuel_cost: float = 5.0, review_period_steps: int = 3,
                   gym_config: T1GymConfig | None = None, openings: list[int] | None = None) -> T1LReviewEnv:
    return T1LReviewEnv(gym_config, t1o_env_config(profile, fuel_cost), openings,
                        review_period_steps=review_period_steps, delay_max_s=T1O_DELAY_MAX_S)


__all__ = ["T1O_DELAY_MAX_S", "T1O_PROFILES", "t1o_env_config", "t1o_review_env"]
