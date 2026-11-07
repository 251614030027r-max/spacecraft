"""T1-B: public mission time/resource conditions on the proven T1-N physics.

No per-opening policy outcomes enter the mission sampler. The G* SCP, MPC,
RK45 safety and capture tests are unchanged. A resource boundary limits only
the final 0.1 s force impulse, then ends the mission (not an unsafe event).
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Any
import gymnasium as gym
import numpy as np
from env.t1_n_review import T1NReviewEnv, T1N_OBSERVATION_NAMES

VERSION = "t1b-mission-v0"
EXTRA_OBSERVATIONS = ("mission_deadline_300s", "mission_budget_2m_s",
                      "time_remaining_300s", "budget_remaining_2m_s", "mission_class")
OBSERVATION_NAMES = T1N_OBSERVATION_NAMES + EXTRA_OBSERVATIONS

@dataclass(frozen=True)
class MissionDistribution:
    version: str = VERSION
    urgent_deadline: tuple[float, float] = (165., 210.)
    urgent_budget: tuple[float, float] = (1.55, 1.95)
    economy_deadline: tuple[float, float] = (230., 300.)
    economy_budget: tuple[float, float] = (1.15, 1.55)
    descent_reserve_s: float = 60.

    def sample(self, opening: int) -> dict[str, Any]:
        rng = np.random.default_rng(np.random.SeedSequence([int(opening), 190019]))
        urgent = bool(rng.integers(0, 2))
        d = rng.uniform(*(self.urgent_deadline if urgent else self.economy_deadline))
        b = rng.uniform(*(self.urgent_budget if urgent else self.economy_budget))
        # Exact control-grid deadline, independent of all policy outputs.
        return {"deadline_s": float(round(d, 1)), "budget_m_s": float(b),
                "urgent": urgent, "distribution": asdict(self)}

class T1BReviewEnv(T1NReviewEnv):
    def __init__(self, *args, mission: MissionDistribution | None = None,
                 mission_override: dict | None = None, **kwargs):
        self.mission_distribution = mission or MissionDistribution()
        self.mission_override = mission_override
        self.mission = {"deadline_s": 300., "budget_m_s": 100., "urgent": False}
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(len(OBSERVATION_NAMES),), dtype=np.float32)
        e = self.env
        original_step = e.step
        original_command = e.controller.command

        def limited_command(*args, **kwargs):
            wrench, diag = original_command(*args, **kwargs)
            limits = np.array([*[e.config.max_torque_per_axis_nm]*3,
                               *[e.config.max_force_per_axis_n]*3])
            wrench = np.clip(np.asarray(wrench, dtype=float), -limits, limits)
            remaining = max(0., self.mission["budget_m_s"] - e._delta_v)
            proposed = float(np.linalg.norm(wrench[3:])) * e.config.dt_s / e.chaser_parameters.mass
            if proposed > remaining and proposed > 0.:
                wrench[3:] *= remaining / proposed
            return wrench, diag
        e.controller.command = limited_command

        def resource_step(action=None, *, pure=False, reference=None):
            config = e.config
            initial_decision = e.decision
            e.config = replace(config, decision_steps=1)
            total_reward, total_dv, controls = 0., 0., []
            info, term, trunc = {"event": None}, False, False
            try:
                for _ in range(config.decision_steps):
                    if e._delta_v >= self.mission["budget_m_s"] - 1e-10:
                        info = {"event": "resource_exhausted"}
                        total_reward -= config.reward.timeout_penalty
                        term = True
                        break
                    r, term, trunc, info = original_step(action, pure=pure, reference=reference)
                    total_reward += r
                    total_dv += info["delta_v_step_m_s"]
                    controls.append([float(e.time_s), float(e._delta_v)])
                    if term or trunc:
                        break
                    if e._delta_v >= self.mission["budget_m_s"] - 1e-10:
                        info["event"] = "resource_exhausted"
                        total_reward -= config.reward.timeout_penalty
                        term = True
                        break
            finally:
                e.config = config
                e.decision = initial_decision + 1
            info.update(delta_v_step_m_s=total_dv, resource_control_trace=controls,
                        mission_deadline_s=self.mission["deadline_s"],
                        mission_budget_m_s=self.mission["budget_m_s"],
                        budget_remaining_m_s=max(0., self.mission["budget_m_s"]-e._delta_v))
            return float(total_reward), bool(term), bool(trunc), info
        e.step = resource_step

    def reset(self, *, seed=None, options=None):
        opening = int(options["opening"]) if options and "opening" in options else self.opening_seed()
        self.mission = dict(self.mission_override or self.mission_distribution.sample(opening))
        self.env.config = replace(self.env.config, max_time_s=self.mission["deadline_s"])
        obs, info = super().reset(seed=seed, options={"opening": opening})
        self.generator.deadline_margin_s = self.mission_distribution.descent_reserve_s
        self._earliest_cache = None
        return self._observation(), info | {"mission": self.mission}

    def _observation(self):
        base = super()._observation()
        d, b = self.mission["deadline_s"], self.mission["budget_m_s"]
        extra = [d/300., b/2., (d-self.env.time_s)/300.,
                 max(0., b-self.env._delta_v)/2., float(self.mission["urgent"])]
        return np.r_[base, extra].astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if info["event"] == "resource_exhausted":
            terminated, truncated = True, False
        return obs, reward, terminated, truncated, info

