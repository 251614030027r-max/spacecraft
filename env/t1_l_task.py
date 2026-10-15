"""T1-L task profile (plan 2026-10-15, module A): the T1 task with the initial range 12-18 m.

The only physical change against T1 is ``T1InitialDistribution.range_m`` = (12, 18) m (T1:
9-15 m). Truth (RK45, gravity, J2), tumbling, initial speed (<= 0.10 m/s), force / torque
bounds, every safety rule, the 300 s limit, the reward and the shared MPC are T1's, by
construction: the profile is a ``T1EnvConfig`` with a different ``initial`` only.
"""

from __future__ import annotations

from dataclasses import replace

from env.t1_env import T1EnvConfig
from env.t1_task import T1InitialDistribution

T1L_RANGE_M = (12.0, 18.0)


def t1l_env_config() -> T1EnvConfig:
    base = T1EnvConfig()
    return replace(base, initial=replace(base.initial, range_m=T1L_RANGE_M))


def task_env_config(task: str) -> T1EnvConfig:
    """``t1`` (original) or ``t1l``."""

    if task == "t1":
        return T1EnvConfig()
    if task == "t1l":
        return t1l_env_config()
    raise ValueError(f"unknown task profile {task!r}")


__all__ = ["T1L_RANGE_M", "T1InitialDistribution", "t1l_env_config", "task_env_config"]
