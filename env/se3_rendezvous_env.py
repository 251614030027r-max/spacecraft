"""Nominal six-DOF tumbling-target rendezvous environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamics.constants import (
    CONTROL_DT_S,
    MAX_EPISODE_TIME_S,
    MAX_FORCE_PER_AXIS_N,
    MAX_RELATIVE_DISTANCE_M,
    MAX_TORQUE_PER_AXIS_NM,
)
from dynamics.disturbance import zero_disturbance
from dynamics.gravity import GravityOptions
from dynamics.integrator import IntegrationDiagnostics, RK45Settings, propagate_rk45
from dynamics.relative import RelativeState, relative_state
from dynamics.types import GeneralizedForce, SpacecraftParameters, SpacecraftState
from env.action import normalized_to_wrench
from env.observation import (
    PHASE2_MISSION_BODY_TRANSLATION_OBSERVATION_SCHEMA,
    PHASE2_MISSION_OBSERVATION_SCHEMA,
    PHASE2_MISSION_TARGET_TRANSLATION_OBSERVATION_SCHEMA,
    PHASE2_OBSERVATION_SCHEMA,
    build_observation,
    build_phase2_mission_observation,
    build_phase2_observation,
)
from env.reward import (
    Phase2MissionReward,
    Phase2MissionRewardBreakdown,
    Phase2RewardBreakdown,
    Phase2TaskReward,
    PhysicalStorageReward,
    RewardBreakdown,
)
from env.scenarios import (
    chaser_parameters,
    sample_chaser_state,
    sample_phase2_mission_chaser_state,
    sample_phase2_chaser_state,
    target_initial_state,
    target_parameters,
)
from env.task import (
    MissionMetrics,
    Phase2MissionConfig,
    Phase2TaskConfig,
    TaskMetrics,
    compute_mission_metrics,
    compute_task_metrics,
)
from env.termination import ErrorMetrics, SuccessThresholds, compute_error_metrics


@dataclass(frozen=True)
class SE3RendezvousConfig:
    dt_s: float = CONTROL_DT_S
    max_time_s: float = MAX_EPISODE_TIME_S
    max_distance_m: float = MAX_RELATIVE_DISTANCE_M

    shell_inner_m: float = 2.0
    shell_outer_m: float = 10.0
    max_relative_attitude_rad: float = float(np.deg2rad(75.0))
    inertial_velocity_component_limit_m_s: float = 0.125
    relative_angular_velocity_component_limit_rad_s: float = 0.03
    target_tumble_scale: float = 0.5

    curriculum_enabled: bool = False
    curriculum_start_step: int = 300_000
    curriculum_end_step: int = 1_300_000
    curriculum_rehearsal_end_step: int = 1_400_000
    curriculum_rotation_fraction: float = 0.45
    curriculum_frontier_width: float = 0.20
    curriculum_rehearsal_probability: float = 0.20
    curriculum_final_probability: float = 0.75
    target_tumble_increment: float = 0.05
    initial_shell_inner_m: float = 1.0
    initial_shell_outer_m: float = 3.0
    initial_max_relative_attitude_rad: float = float(np.deg2rad(30.0))
    initial_inertial_velocity_component_limit_m_s: float = 0.05
    initial_target_tumble_scale: float = 0.25

    observation_attitude_scale_rad: float = float(np.deg2rad(30.0))
    observation_distance_scale_m: float = 3.0
    observation_angular_velocity_scale_rad_s: float = 0.05
    observation_velocity_scale_m_s: float = 0.2
    observation_softsign_limit: float = 10.0

    max_force_per_axis_n: float = MAX_FORCE_PER_AXIS_N
    max_torque_per_axis_nm: float = MAX_TORQUE_PER_AXIS_NM
    storage_characteristic_length_m: float = 0.4482671036630004
    storage_time_constant_s: float = 30.0
    barrier_epsilon: float = 1.0e-6
    success_thresholds: SuccessThresholds = field(default_factory=SuccessThresholds)
    completion_required_steps: int = 5

    include_j2: bool = True
    solver_rtol: float = 1.0e-7
    solver_atol: float = 1.0e-9
    cache_target_trajectory: bool = True
    phase2_enabled: bool = False
    phase2_mission_enabled: bool = False
    phase2_training_mode: str = "full_mission"
    phase2_mission: Phase2MissionConfig = field(default_factory=Phase2MissionConfig)
    phase1_curriculum_enabled: bool = False
    phase1_curriculum_start_step: int = 5_000
    phase1_curriculum_end_step: int = 150_000
    phase1_curriculum_adaptive: bool = False
    phase1_curriculum_window_episodes: int = 20
    phase1_curriculum_difficulty_step: float = 0.005
    phase1_curriculum_promote_rate: float = 0.80
    phase1_curriculum_demote_rate: float = 0.50
    phase1_curriculum_easy_full_probability: float = 0.0
    phase1_curriculum_full_probability: float = 0.10
    phase1_curriculum_easy_distance_min_m: float = 9.02
    phase1_curriculum_easy_distance_max_m: float = 9.05
    phase1_curriculum_easy_direction_half_angle_rad: float = float(np.deg2rad(0.02))
    phase1_curriculum_easy_attitude_limit_rad: float = float(np.deg2rad(2.0))
    phase1_curriculum_easy_speed_limit_m_s: float = 0.03
    phase1_curriculum_easy_angular_velocity_limit_rad_s: float = 0.001
    phase2_task: Phase2TaskConfig = field(default_factory=Phase2TaskConfig)
    phase2_target_tumble_scale: float = 0.25
    phase2_axial_remaining_min_m: float = 3.0
    phase2_axial_remaining_max_m: float = 7.0
    phase2_max_relative_attitude_rad: float = float(np.deg2rad(60.0))
    phase2_near_boundary_probability: float = 0.25
    phase2_initial_total_speed_limit_m_s: float = 0.35
    phase2_initial_relative_angular_velocity_limit_rad_s: float = 0.03
    phase2_maximum_interior_radial_fraction: float = 0.75
    phase2_minimum_fov_margin_rad: float = 0.0
    phase2_warmup_steps: int = 0
    phase2_warmup_task: Phase2TaskConfig = field(
        default_factory=lambda: Phase2TaskConfig(
            corridor_half_angle_rad=float(np.deg2rad(45.0)),
            fov_half_angle_rad=float(np.deg2rad(60.0)),
        )
    )
    phase2_warmup_tumble_scale: float = 0.18
    phase2_warmup_axial_remaining_min_m: float = 2.0
    phase2_warmup_axial_remaining_max_m: float = 4.0
    phase2_warmup_max_relative_attitude_rad: float = float(np.deg2rad(35.0))
    phase2_warmup_near_boundary_probability: float = 0.0
    phase2_warmup_initial_total_speed_limit_m_s: float = 0.15
    phase2_warmup_initial_relative_angular_velocity_limit_rad_s: float = 0.018
    phase2_warmup_maximum_interior_radial_fraction: float = 0.65
    phase2_warmup_minimum_fov_margin_rad: float = float(np.deg2rad(5.0))
    phase2_observation_schema: str = PHASE2_OBSERVATION_SCHEMA
    phase2_observation_target_angular_velocity_scale_rad_s: float = 0.05
    phase2_distance_failure_penalty: float = -20.0


@lru_cache(maxsize=8)
def _cached_target_trajectory(
    max_time_s: float,
    dt_s: float,
    include_j2: bool,
    solver_rtol: float,
    solver_atol: float,
    tumble_scale: float,
) -> tuple[SpacecraftState, ...]:
    state = target_initial_state(tumble_scale=tumble_scale)
    parameters = target_parameters()
    gravity = GravityOptions(include_j2=include_j2)
    solver = RK45Settings(rtol=solver_rtol, atol=solver_atol, max_step=dt_s)
    maximum_steps = int(np.ceil(max_time_s / dt_s - 1.0e-12))
    states = [state.copy()]
    for step in range(maximum_steps):
        state, _ = propagate_rk45(
            state,
            parameters,
            step * dt_s,
            dt_s,
            control=GeneralizedForce.zeros(),
            disturbance=zero_disturbance,
            gravity_options=gravity,
            settings=solver,
        )
        states.append(state.copy())
    return tuple(states)


class SE3RendezvousEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, config: SE3RendezvousConfig | None = None) -> None:
        super().__init__()
        self.config = config or SE3RendezvousConfig()
        self._validate_config()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        limit = self.config.observation_softsign_limit
        observation_size = (
            24
            if self.config.phase2_mission_enabled
            else (23 if self.config.phase2_enabled else 12)
        )
        self.observation_space = spaces.Box(
            -limit, limit, shape=(observation_size,), dtype=np.float32
        )
        self._gravity = GravityOptions(include_j2=self.config.include_j2)
        self._solver = RK45Settings(
            rtol=self.config.solver_rtol,
            atol=self.config.solver_atol,
            max_step=self.config.dt_s,
        )
        self._reward = (
            Phase2MissionReward(
                task=self.config.phase2_task,
                phase1_soft_speed_m_s=(
                    self.config.phase2_mission.phase1_soft_speed_m_s
                ),
                phase1_cruise_speed_m_s=(
                    self.config.phase2_mission.phase1_cruise_speed_m_s
                ),
            )
            if self.config.phase2_mission_enabled
            else
            Phase2TaskReward(
                task=self.config.phase2_task,
                time_step_s=self.config.dt_s,
                time_constant_s=self.config.storage_time_constant_s,
                attitude_scale_rad=self.config.observation_attitude_scale_rad,
                distance_scale_m=self.config.observation_distance_scale_m,
                angular_velocity_scale_rad_s=(
                    self.config.observation_angular_velocity_scale_rad_s
                ),
                velocity_scale_m_s=self.config.observation_velocity_scale_m_s,
            )
            if self.config.phase2_enabled
            else PhysicalStorageReward(
                characteristic_length_m=self.config.storage_characteristic_length_m,
                time_constant_s=self.config.storage_time_constant_s,
                time_step_s=self.config.dt_s,
                max_distance_m=self.config.max_distance_m,
                barrier_epsilon=self.config.barrier_epsilon,
            )
        )
        self.target_state: SpacecraftState | None = None
        self.chaser_state: SpacecraftState | None = None
        self.target_parameters: SpacecraftParameters | None = None
        self.chaser_parameters: SpacecraftParameters | None = None
        self.relative: RelativeState | None = None
        self.time_seconds = 0.0
        self.step_count = 0
        self.total_transition_count = 0
        self._completion_streak = 0
        self._curriculum_progress = 1.0
        self._curriculum_ceiling = 1.0
        self._curriculum_ceiling_override: float | None = None
        self._curriculum_rehearsal = False
        self._curriculum_rotation_progress = 1.0
        self._curriculum_translation_progress = 1.0
        self._episode_shell = (self.config.shell_inner_m, self.config.shell_outer_m)
        self._episode_attitude_limit = self.config.max_relative_attitude_rad
        self._episode_velocity_limit = (
            self.config.inertial_velocity_component_limit_m_s
        )
        self._episode_tumble_scale = self.config.target_tumble_scale
        self._target_trajectory: tuple[SpacecraftState, ...] | None = None
        self._active_phase2_task = self.config.phase2_task
        self._phase2_stage = "nominal"
        self._phase1_curriculum_progress = 1.0
        self._phase1_curriculum_frontier = 0.0
        self._phase1_curriculum_difficulty = 1.0
        self._phase1_curriculum_full_sample = True
        self._phase1_curriculum_current_full_probability = 1.0
        self._phase1_curriculum_window_count = 0
        self._phase1_curriculum_window_successes = 0
        self._mission_phase = 0
        self._gate_reached = False
        self._constraint_success = True
        self._constraint_violation_steps = {
            name: 0 for name in ("corridor", "fov", "total_speed", "closing_speed")
        }
        self._constraint_max_violation = {
            name: 0.0 for name in self._constraint_violation_steps
        }

    def _validate_config(self) -> None:
        c = self.config
        positive = (
            c.dt_s,
            c.max_time_s,
            c.max_distance_m,
            c.shell_outer_m,
            c.max_relative_attitude_rad,
            c.inertial_velocity_component_limit_m_s,
            c.relative_angular_velocity_component_limit_rad_s,
            c.observation_attitude_scale_rad,
            c.observation_distance_scale_m,
            c.observation_angular_velocity_scale_rad_s,
            c.observation_velocity_scale_m_s,
            c.observation_softsign_limit,
            c.max_force_per_axis_n,
            c.max_torque_per_axis_nm,
            c.storage_characteristic_length_m,
            c.storage_time_constant_s,
        )
        if min(positive) <= 0.0:
            raise ValueError("physical and numerical scales must be positive")
        if not 0.0 <= c.shell_inner_m < c.shell_outer_m:
            raise ValueError("final shell requires 0 <= inner < outer")
        if not 0.0 <= c.initial_shell_inner_m < c.initial_shell_outer_m:
            raise ValueError("initial shell requires 0 <= inner < outer")
        if max(c.max_relative_attitude_rad, c.initial_max_relative_attitude_rad) > np.pi:
            raise ValueError("attitude limits cannot exceed pi")
        if c.curriculum_enabled and c.curriculum_end_step <= c.curriculum_start_step:
            raise ValueError("curriculum_end_step must exceed curriculum_start_step")
        if c.curriculum_rehearsal_end_step < c.curriculum_end_step:
            raise ValueError("curriculum rehearsal must not end before expansion")
        if c.curriculum_start_step < 0 or c.completion_required_steps <= 0:
            raise ValueError("step counts must be valid positive values")
        if min(c.target_tumble_scale, c.initial_target_tumble_scale) < 0.0:
            raise ValueError("tumble scales must be non-negative")
        probabilities = (
            c.curriculum_rehearsal_probability,
            c.curriculum_final_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("curriculum probabilities must lie in [0, 1]")
        if not 0.0 <= c.curriculum_frontier_width <= 1.0:
            raise ValueError("curriculum frontier width must lie in [0, 1]")
        if not 0.0 < c.curriculum_rotation_fraction < 1.0:
            raise ValueError("curriculum rotation fraction must lie in (0, 1)")
        if c.target_tumble_increment <= 0.0:
            raise ValueError("target tumble increment must be positive")
        if c.phase2_enabled and c.curriculum_enabled:
            raise ValueError("legacy curriculum cannot be enabled for Phase-2")
        if c.phase2_mission_enabled and not c.phase2_enabled:
            raise ValueError("Phase-2 mission requires phase2_enabled")
        if c.phase1_curriculum_enabled and not c.phase2_mission_enabled:
            raise ValueError("Phase-I curriculum requires the Phase-2 mission")
        if not 0 <= c.phase1_curriculum_start_step < c.phase1_curriculum_end_step:
            raise ValueError("Phase-I curriculum step interval is invalid")
        if c.phase1_curriculum_window_episodes <= 0:
            raise ValueError("Phase-I curriculum window must be positive")
        if not (
            0.0 < c.phase1_curriculum_difficulty_step <= 1.0
            and 0.0 <= c.phase1_curriculum_demote_rate
            < c.phase1_curriculum_promote_rate
            <= 1.0
        ):
            raise ValueError("Phase-I adaptive curriculum settings are invalid")
        if not (
            0.0 <= c.phase1_curriculum_easy_full_probability
            <= c.phase1_curriculum_full_probability
            <= 1.0
        ):
            raise ValueError("Phase-I curriculum full probability must lie in [0, 1]")
        if not (
            0.0 < c.phase1_curriculum_easy_distance_min_m
            < c.phase1_curriculum_easy_distance_max_m
            < c.phase2_mission.initial_distance_min_m
        ):
            raise ValueError("Phase-I easy distance envelope must precede full")
        if c.phase2_training_mode not in {"phase1_pretrain", "full_mission"}:
            raise ValueError("unsupported Phase-2 training mode")
        if c.phase2_target_tumble_scale < 0.0:
            raise ValueError("Phase-2 target tumble scale must be non-negative")
        if c.phase2_warmup_steps < 0:
            raise ValueError("Phase-2 warm-up steps cannot be negative")
        if not (
            0.0 <= c.phase2_near_boundary_probability <= 1.0
            and 0.0 <= c.phase2_warmup_near_boundary_probability <= 1.0
        ):
            raise ValueError("Phase-2 boundary probabilities must lie in [0, 1]")
        if not (
            0.0 < c.phase2_axial_remaining_min_m < c.phase2_axial_remaining_max_m
            and 0.0
            < c.phase2_warmup_axial_remaining_min_m
            < c.phase2_warmup_axial_remaining_max_m
        ):
            raise ValueError("Phase-2 axial sampling bounds are invalid")
        supported_schemas = (
            {
                PHASE2_MISSION_OBSERVATION_SCHEMA,
                PHASE2_MISSION_BODY_TRANSLATION_OBSERVATION_SCHEMA,
                PHASE2_MISSION_TARGET_TRANSLATION_OBSERVATION_SCHEMA,
            }
            if c.phase2_mission_enabled
            else {PHASE2_OBSERVATION_SCHEMA}
        )
        if c.phase2_enabled and c.phase2_observation_schema not in supported_schemas:
            raise ValueError("unsupported Phase-2 observation schema")
        if c.phase2_distance_failure_penalty > 0.0:
            raise ValueError("Phase-2 distance-failure penalty must be non-positive")

    def _select_phase2_stage(self) -> None:
        if not self.config.phase2_enabled or self.config.phase2_mission_enabled:
            return
        warmup = self.total_transition_count < self.config.phase2_warmup_steps
        self._phase2_stage = "warmup" if warmup else "nominal"
        self._active_phase2_task = (
            self.config.phase2_warmup_task if warmup else self.config.phase2_task
        )
        self._episode_tumble_scale = (
            self.config.phase2_warmup_tumble_scale
            if warmup
            else self.config.phase2_target_tumble_scale
        )
        self._reward = Phase2TaskReward(
            task=self._active_phase2_task,
            time_step_s=self.config.dt_s,
            time_constant_s=self.config.storage_time_constant_s,
            attitude_scale_rad=self.config.observation_attitude_scale_rad,
            distance_scale_m=self.config.observation_distance_scale_m,
            angular_velocity_scale_rad_s=(
                self.config.observation_angular_velocity_scale_rad_s
            ),
            velocity_scale_m_s=self.config.observation_velocity_scale_m_s,
        )

    def _require_state(self) -> None:
        if any(
            item is None
            for item in (
                self.target_state,
                self.chaser_state,
                self.target_parameters,
                self.chaser_parameters,
                self.relative,
            )
        ):
            raise RuntimeError("reset() must be called first")

    def _difficulty_ceiling(self) -> float:
        if not self.config.curriculum_enabled:
            return 1.0
        if self._curriculum_ceiling_override is not None:
            return self._curriculum_ceiling_override
        return float(
            np.clip(
                (self.total_transition_count - self.config.curriculum_start_step)
                / (
                    self.config.curriculum_end_step
                    - self.config.curriculum_start_step
                ),
                0.0,
                1.0,
            )
        )

    def _sample_difficulty(self) -> tuple[float, bool]:
        """Sample a seeded episode difficulty while retaining prior tasks."""

        if not self.config.curriculum_enabled:
            return 1.0, False
        ceiling = self._difficulty_ceiling()
        if ceiling <= 0.0:
            return 0.0, False
        if (
            self._curriculum_ceiling_override is None
            and self.total_transition_count
            >= self.config.curriculum_rehearsal_end_step
        ):
            return 1.0, False
        if ceiling >= 1.0:
            if self.np_random.random() < self.config.curriculum_final_probability:
                return 1.0, False
            lower = max(0.0, 1.0 - self.config.curriculum_frontier_width)
            return float(self.np_random.uniform(lower, 1.0)), True
        if self.np_random.random() < self.config.curriculum_rehearsal_probability:
            return float(self.np_random.uniform(0.0, ceiling)), True
        frontier_width = min(
            self.config.curriculum_frontier_width,
            self.config.curriculum_frontier_width * ceiling,
        )
        lower = max(0.0, ceiling - frontier_width)
        return float(self.np_random.uniform(lower, ceiling)), False

    def set_curriculum_ceiling(self, ceiling: float | None) -> None:
        """Override the open-loop ceiling for mastery-gated training."""

        if not self.config.curriculum_enabled:
            raise RuntimeError("curriculum ceiling requires an enabled curriculum")
        if ceiling is None:
            self._curriculum_ceiling_override = None
            return
        value = float(ceiling)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("curriculum ceiling must lie in [0, 1]")
        self._curriculum_ceiling_override = value

    @staticmethod
    def _lerp(start: float, end: float, progress: float) -> float:
        return (1.0 - progress) * start + progress * end

    def _resolve_episode_envelope(self) -> None:
        c = self.config
        self._curriculum_ceiling = self._difficulty_ceiling()
        p, rehearsal = self._sample_difficulty()
        self._curriculum_progress = p
        self._curriculum_rehearsal = rehearsal
        rotation_progress = min(1.0, p / c.curriculum_rotation_fraction)
        translation_progress = max(
            0.0,
            (p - c.curriculum_rotation_fraction)
            / (1.0 - c.curriculum_rotation_fraction),
        )
        self._curriculum_rotation_progress = rotation_progress
        self._curriculum_translation_progress = translation_progress
        self._episode_shell = (
            self._lerp(
                c.initial_shell_inner_m, c.shell_inner_m, translation_progress
            ),
            self._lerp(
                c.initial_shell_outer_m, c.shell_outer_m, translation_progress
            ),
        )
        self._episode_attitude_limit = self._lerp(
            c.initial_max_relative_attitude_rad,
            c.max_relative_attitude_rad,
            rotation_progress,
        )
        self._episode_velocity_limit = self._lerp(
            c.initial_inertial_velocity_component_limit_m_s,
            c.inertial_velocity_component_limit_m_s,
            translation_progress,
        )
        tumble = self._lerp(
            c.initial_target_tumble_scale,
            c.target_tumble_scale,
            rotation_progress,
        )
        self._episode_tumble_scale = float(
            np.clip(
                np.round(tumble / c.target_tumble_increment)
                * c.target_tumble_increment,
                c.initial_target_tumble_scale,
                c.target_tumble_scale,
            )
        )

    def _active_reference_position(self) -> np.ndarray:
        return (
            self.config.phase2_mission.gate_position
            if self._mission_phase == 0
            else self.config.phase2_task.desired_position
        )

    def _observation(self) -> np.ndarray:
        self._require_state()
        assert self.relative is not None
        builder = (
            build_phase2_mission_observation
            if self.config.phase2_mission_enabled
            else
            build_phase2_observation
            if self.config.phase2_enabled
            else build_observation
        )
        kwargs = dict(
            relative=self.relative,
            attitude_scale_rad=self.config.observation_attitude_scale_rad,
            distance_scale_m=self.config.observation_distance_scale_m,
            angular_velocity_scale_rad_s=(
                self.config.observation_angular_velocity_scale_rad_s
            ),
            velocity_scale_m_s=self.config.observation_velocity_scale_m_s,
            softsign_limit=self.config.observation_softsign_limit,
        )
        if self.config.phase2_enabled:
            kwargs["task"] = self._active_phase2_task
            assert self.target_state is not None
            kwargs["target_angular_velocity_rad_s"] = self.target_state.omega
            kwargs["target_angular_velocity_scale_rad_s"] = (
                self.config.phase2_observation_target_angular_velocity_scale_rad_s
            )
        if self.config.phase2_mission_enabled:
            active_reference = self._active_reference_position()
            kwargs["active_reference_position_m"] = active_reference
            kwargs["mission_phase"] = self._mission_phase
            kwargs["translational_observation_frame"] = (
                "chaser_body"
                if self.config.phase2_observation_schema
                in {
                    PHASE2_MISSION_BODY_TRANSLATION_OBSERVATION_SCHEMA,
                    PHASE2_MISSION_OBSERVATION_SCHEMA,
                }
                else "target"
            )
            velocity_tracking_error = (
                self.config.phase2_observation_schema
                == PHASE2_MISSION_OBSERVATION_SCHEMA
            )
            kwargs["translational_velocity_observation"] = (
                "tracking_error" if velocity_tracking_error else "actual"
            )
            if velocity_tracking_error:
                assert isinstance(self._reward, Phase2MissionReward)
                kwargs["active_reference_velocity_target_m_s"] = (
                    self._reward.active_desired_velocity(
                        self.relative,
                        active_reference,
                        terminal_constraints_active=(self._mission_phase == 1),
                    )
                )
        return builder(**kwargs)

    def scale_action(self, action: np.ndarray) -> GeneralizedForce:
        return normalized_to_wrench(
            action,
            max_torque_per_axis_nm=self.config.max_torque_per_axis_nm,
            max_force_per_axis_n=self.config.max_force_per_axis_n,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        self._resolve_episode_envelope()
        self._select_phase2_stage()
        if self.config.phase2_mission_enabled:
            self._mission_phase = 0
            self._gate_reached = False
            self._phase2_stage = "phase1"
            self._episode_tumble_scale = self.config.phase2_target_tumble_scale
            if self.config.phase1_curriculum_enabled:
                if self.config.phase1_curriculum_adaptive:
                    self._phase1_curriculum_progress = (
                        self._phase1_curriculum_frontier
                    )
                else:
                    self._phase1_curriculum_progress = float(
                        np.clip(
                            (
                                self.total_transition_count
                                - self.config.phase1_curriculum_start_step
                            )
                            / (
                                self.config.phase1_curriculum_end_step
                                - self.config.phase1_curriculum_start_step
                            ),
                            0.0,
                            1.0,
                        )
                    )
                self._phase1_curriculum_current_full_probability = self._lerp(
                    self.config.phase1_curriculum_easy_full_probability,
                    self.config.phase1_curriculum_full_probability,
                    self._phase1_curriculum_progress,
                )
                self._phase1_curriculum_full_sample = bool(
                    self.np_random.random()
                    < self._phase1_curriculum_current_full_probability
                )
                self._phase1_curriculum_difficulty = (
                    1.0
                    if self._phase1_curriculum_full_sample
                    else self._phase1_curriculum_progress
                )
            else:
                self._phase1_curriculum_progress = 1.0
                self._phase1_curriculum_difficulty = 1.0
                self._phase1_curriculum_full_sample = True
                self._phase1_curriculum_current_full_probability = 1.0
        supplied_target = options.get("target_state")
        supplied_target_parameters = options.get("target_parameters")
        if supplied_target is None:
            self.target_state = target_initial_state(
                tumble_scale=self._episode_tumble_scale
            )
            self.target_parameters = target_parameters()
            self._target_trajectory = (
                _cached_target_trajectory(
                    self.config.max_time_s,
                    self.config.dt_s,
                    self.config.include_j2,
                    self.config.solver_rtol,
                    self.config.solver_atol,
                    self._episode_tumble_scale,
                )
                if self.config.cache_target_trajectory
                else None
            )
        else:
            self.target_state = supplied_target.copy()
            self.target_parameters = supplied_target_parameters or target_parameters()
            self._target_trajectory = None
        supplied_chaser = options.get("chaser_state")
        self.chaser_state = (
            supplied_chaser.copy()
            if supplied_chaser is not None
            else (
                sample_phase2_mission_chaser_state(
                    self.np_random,
                    self.target_state,
                    mission=self.config.phase2_mission,
                    difficulty=self._phase1_curriculum_difficulty,
                    easy_distance_min_m=self.config.phase1_curriculum_easy_distance_min_m,
                    easy_distance_max_m=self.config.phase1_curriculum_easy_distance_max_m,
                    easy_direction_half_angle_rad=self.config.phase1_curriculum_easy_direction_half_angle_rad,
                    easy_attitude_limit_rad=self.config.phase1_curriculum_easy_attitude_limit_rad,
                    easy_speed_limit_m_s=self.config.phase1_curriculum_easy_speed_limit_m_s,
                    easy_angular_velocity_limit_rad_s=self.config.phase1_curriculum_easy_angular_velocity_limit_rad_s,
                )
                if self.config.phase2_mission_enabled
                else
                sample_phase2_chaser_state(
                    self.np_random,
                    self.target_state,
                    task=self._active_phase2_task,
                    axial_remaining_min_m=(
                        self.config.phase2_warmup_axial_remaining_min_m
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_axial_remaining_min_m
                    ),
                    axial_remaining_max_m=(
                        self.config.phase2_warmup_axial_remaining_max_m
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_axial_remaining_max_m
                    ),
                    max_relative_attitude_rad=(
                        self.config.phase2_warmup_max_relative_attitude_rad
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_max_relative_attitude_rad
                    ),
                    near_boundary_probability=(
                        self.config.phase2_warmup_near_boundary_probability
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_near_boundary_probability
                    ),
                    relative_angular_velocity_component_limit_rad_s=(
                        self.config.phase2_warmup_initial_relative_angular_velocity_limit_rad_s
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_initial_relative_angular_velocity_limit_rad_s
                    ),
                    initial_total_speed_limit_m_s=(
                        self.config.phase2_warmup_initial_total_speed_limit_m_s
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_initial_total_speed_limit_m_s
                    ),
                    maximum_interior_radial_fraction=(
                        self.config.phase2_warmup_maximum_interior_radial_fraction
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_maximum_interior_radial_fraction
                    ),
                    minimum_fov_margin_rad=(
                        self.config.phase2_warmup_minimum_fov_margin_rad
                        if self._phase2_stage == "warmup"
                        else self.config.phase2_minimum_fov_margin_rad
                    ),
                )
                if self.config.phase2_enabled
                else sample_chaser_state(
                    self.np_random,
                    self.target_state,
                    shell_inner_m=self._episode_shell[0],
                    shell_outer_m=self._episode_shell[1],
                    max_relative_attitude_rad=self._episode_attitude_limit,
                    inertial_velocity_component_limit_m_s=self._episode_velocity_limit,
                    relative_angular_velocity_component_limit_rad_s=(
                        self.config.relative_angular_velocity_component_limit_rad_s
                    ),
                )
            )
        )
        self.chaser_parameters = options.get("chaser_parameters") or chaser_parameters()
        self.relative = relative_state(self.target_state, self.chaser_state)
        self.time_seconds = 0.0
        self.step_count = 0
        self._completion_streak = 0
        task_metrics = (
            compute_task_metrics(self.relative, self._active_phase2_task)
            if self.config.phase2_enabled
            else None
        )
        self._constraint_success = bool(
            self.config.phase2_mission_enabled
            or task_metrics is None
            or task_metrics.constraints_satisfied
        )
        self._constraint_violation_steps = {
            name: 0 for name in self._constraint_violation_steps
        }
        self._constraint_max_violation = {
            name: 0.0 for name in self._constraint_max_violation
        }
        if isinstance(self._reward, Phase2MissionReward):
            self._reward.reset(self.relative, self._active_reference_position())
        else:
            self._reward.reset(self.relative)
        metrics = compute_error_metrics(
            self.relative, self.config.success_thresholds
        )
        mission_metrics = (
            compute_mission_metrics(self.relative, self.config.phase2_mission)
            if self.config.phase2_mission_enabled
            else None
        )
        info = self._info(
            metrics, task_metrics, mission_metrics, None, 0, 0
        )
        if self.config.phase2_mission_enabled:
            info.update(
                phase1_curriculum_progress=self._phase1_curriculum_progress,
                phase1_curriculum_frontier=self._phase1_curriculum_frontier,
                phase1_curriculum_difficulty=self._phase1_curriculum_difficulty,
                phase1_curriculum_full_sample=self._phase1_curriculum_full_sample,
                phase1_curriculum_full_probability=(
                    self._phase1_curriculum_current_full_probability
                ),
                constraint_feasible=True,
                constraint_success=True,
                completed=False,
                final_completed=False,
                gate_success=False,
                gate_transition=False,
                terminal_constraint_failure=False,
                phase1_speed_failure=False,
                premature_entry_failure=False,
                distance_failure=False,
                time_failure=False,
            )
        return self._observation(), info

    def _info(
        self,
        metrics: ErrorMetrics,
        task_metrics: TaskMetrics | None,
        mission_metrics: MissionMetrics | None,
        reward: RewardBreakdown | Phase2RewardBreakdown | Phase2MissionRewardBreakdown | None,
        target_nfev: int,
        chaser_nfev: int,
    ) -> dict:
        info = {
            "time_seconds": self.time_seconds,
            "step_count": self.step_count,
            "relative_distance_m": metrics.direct_relative_distance_m,
            "attitude_error_rad": metrics.attitude_rad,
            "angular_velocity_error_rad_s": metrics.angular_velocity_rad_s,
            "position_error_m": metrics.position_m,
            "translational_velocity_error_m_s": metrics.translational_velocity_m_s,
            "attitude_success": metrics.attitude_success,
            "position_success": metrics.position_success,
            "joint_success": metrics.joint_success,
            "completion_streak": self._completion_streak,
            "curriculum_progress": self._curriculum_progress,
            "curriculum_ceiling": self._curriculum_ceiling,
            "curriculum_rehearsal": self._curriculum_rehearsal,
            "curriculum_rotation_progress": self._curriculum_rotation_progress,
            "curriculum_translation_progress": self._curriculum_translation_progress,
            "phase1_curriculum_progress": self._phase1_curriculum_progress,
            "phase1_curriculum_frontier": self._phase1_curriculum_frontier,
            "phase1_curriculum_difficulty": self._phase1_curriculum_difficulty,
            "phase1_curriculum_window_count": self._phase1_curriculum_window_count,
            "phase1_curriculum_window_successes": self._phase1_curriculum_window_successes,
            "phase1_curriculum_full_sample": self._phase1_curriculum_full_sample,
            "phase1_curriculum_full_probability": self._phase1_curriculum_current_full_probability,
            "episode_shell_inner_m": self._episode_shell[0],
            "episode_shell_outer_m": self._episode_shell[1],
            "episode_max_relative_attitude_rad": self._episode_attitude_limit,
            "episode_inertial_velocity_limit_m_s": self._episode_velocity_limit,
            "episode_target_tumble_scale": self._episode_tumble_scale,
            "phase2_stage": self._phase2_stage,
            "mission_phase": self._mission_phase,
            "gate_reached": self._gate_reached,
            "target_nfev": target_nfev,
            "chaser_nfev": chaser_nfev,
        }
        if mission_metrics is not None:
            info.update(
                gate_position_error_m=mission_metrics.gate_position_error_m,
                target_center_distance_m=mission_metrics.target_center_distance_m,
                phase1_speed_limit_m_s=(
                    self.config.phase2_mission.phase1_speed_limit_m_s
                ),
                phase1_soft_speed_m_s=self.config.phase2_mission.phase1_soft_speed_m_s,
                phase1_catastrophic_speed_limit_m_s=(
                    self.config.phase2_mission.phase1_catastrophic_speed_limit_m_s
                ),
                active_reference_position_m=self._active_reference_position().copy(),
            )
        if task_metrics is not None:
            info.update(
                axial_remaining_m=task_metrics.axial_remaining_m,
                port_axial_distance_m=task_metrics.port_axial_distance_m,
                corridor_radial_distance_m=(
                    task_metrics.corridor_radial_distance_m
                ),
                corridor_axial_margin_m=task_metrics.corridor_axial_margin_m,
                corridor_lateral_margin_m=task_metrics.corridor_lateral_margin_m,
                fov_angle_rad=task_metrics.fov_angle_rad,
                fov_margin_rad=task_metrics.fov_margin_rad,
                total_speed_m_s=task_metrics.total_speed_m_s,
                total_speed_margin_m_s=task_metrics.total_speed_margin_m_s,
                closing_speed_m_s=task_metrics.closing_speed_m_s,
                closing_speed_limit_m_s=task_metrics.closing_speed_limit_m_s,
                closing_speed_margin_m_s=task_metrics.closing_speed_margin_m_s,
                position_error_m=task_metrics.position_error_m,
                attitude_error_rad=task_metrics.attitude_error_rad,
                angular_velocity_error_rad_s=(
                    task_metrics.angular_velocity_error_rad_s
                ),
                joint_success=task_metrics.instantaneous_completion,
                attitude_success=bool(
                    task_metrics.attitude_error_rad
                    <= self._active_phase2_task.completion_attitude_rad
                    and task_metrics.angular_velocity_error_rad_s
                    <= self._active_phase2_task.completion_angular_velocity_rad_s
                ),
                position_success=bool(
                    task_metrics.position_error_m
                    <= self._active_phase2_task.completion_position_m
                    and task_metrics.total_speed_m_s
                    <= self._active_phase2_task.completion_speed_m_s
                ),
                constraint_feasible=task_metrics.constraints_satisfied,
                constraint_success=self._constraint_success,
                **{
                    f"{name}_violation_steps": count
                    for name, count in self._constraint_violation_steps.items()
                },
                **{
                    f"{name}_max_violation": value
                    for name, value in self._constraint_max_violation.items()
                },
            )
        if isinstance(reward, RewardBreakdown):
            info.update(
                reward_storage_dissipation=reward.storage_dissipation,
                reward_storage_penalty=reward.storage_penalty,
                reward_actuation_penalty=reward.actuation_penalty,
                reward_storage_energy=reward.storage_energy,
                reward_storage_potential=reward.storage_potential,
                reward_safety_barrier=reward.safety_barrier,
            )
        elif isinstance(reward, Phase2RewardBreakdown):
            info.update(
                reward_task_dissipation=reward.task_dissipation,
                reward_task_penalty=reward.task_penalty,
                reward_actuation_penalty=reward.actuation_penalty,
                reward_corridor_penalty=reward.corridor_penalty,
                reward_fov_penalty=reward.fov_penalty,
                reward_total_speed_penalty=reward.total_speed_penalty,
                reward_closing_speed_penalty=reward.closing_speed_penalty,
                reward_constraint_penalty=reward.constraint_penalty,
                reward_task_potential=reward.task_potential,
            )
        elif isinstance(reward, Phase2MissionRewardBreakdown):
            info.update(
                reward_progress=reward.progress,
                reward_state_penalty=reward.state_penalty,
                reward_actuation_penalty=reward.actuation_penalty,
                reward_constraint_warning=reward.constraint_warning,
                reward_event=reward.event_reward,
                reward_potential=reward.potential,
            )
        return info

    def _update_phase1_curriculum(self, success: bool) -> None:
        """Move the persistent frontier only after a completed easy episode."""
        c = self.config
        if (
            not c.phase1_curriculum_enabled
            or not c.phase1_curriculum_adaptive
            or self.total_transition_count < c.phase1_curriculum_start_step
            or self._phase1_curriculum_full_sample
        ):
            return
        self._phase1_curriculum_window_count += 1
        self._phase1_curriculum_window_successes += int(success)
        if self._phase1_curriculum_window_count < c.phase1_curriculum_window_episodes:
            return
        success_rate = (
            self._phase1_curriculum_window_successes
            / self._phase1_curriculum_window_count
        )
        if success_rate >= c.phase1_curriculum_promote_rate:
            self._phase1_curriculum_frontier = min(
                1.0,
                self._phase1_curriculum_frontier
                + c.phase1_curriculum_difficulty_step,
            )
        elif success_rate < c.phase1_curriculum_demote_rate:
            self._phase1_curriculum_frontier = max(
                0.0,
                self._phase1_curriculum_frontier
                - c.phase1_curriculum_difficulty_step,
            )
        self._phase1_curriculum_window_count = 0
        self._phase1_curriculum_window_successes = 0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._require_state()
        assert self.target_state is not None
        assert self.chaser_state is not None
        assert self.target_parameters is not None
        assert self.chaser_parameters is not None
        control = self.scale_action(action)
        if self._target_trajectory is None:
            target_next, target_diag = propagate_rk45(
                self.target_state,
                self.target_parameters,
                self.time_seconds,
                self.config.dt_s,
                control=GeneralizedForce.zeros(),
                disturbance=zero_disturbance,
                gravity_options=self._gravity,
                settings=self._solver,
            )
        else:
            target_next = self._target_trajectory[self.step_count + 1].copy()
            target_diag = IntegrationDiagnostics(0, 0, 0, "cached target trajectory")
        chaser_next, chaser_diag = propagate_rk45(
            self.chaser_state,
            self.chaser_parameters,
            self.time_seconds,
            self.config.dt_s,
            control=control,
            disturbance=zero_disturbance,
            gravity_options=self._gravity,
            settings=self._solver,
        )
        self.target_state = target_next
        self.chaser_state = chaser_next
        self.step_count += 1
        self.total_transition_count += 1
        self.time_seconds = self.step_count * self.config.dt_s
        self.relative = relative_state(target_next, chaser_next)
        metrics = compute_error_metrics(
            self.relative, self.config.success_thresholds
        )
        task_metrics = (
            compute_task_metrics(self.relative, self._active_phase2_task)
            if self.config.phase2_enabled
            else None
        )
        if self.config.phase2_mission_enabled:
            assert task_metrics is not None
            assert isinstance(self._reward, Phase2MissionReward)
            mission = self.config.phase2_mission
            mission_metrics = compute_mission_metrics(self.relative, mission)
            gate_transition = bool(
                self._mission_phase == 0
                and mission.gate_satisfied(
                    mission_metrics,
                    fov_angle_rad=task_metrics.fov_angle_rad,
                )
            )
            phase1_pretrain_success = bool(
                gate_transition
                and self.config.phase2_training_mode == "phase1_pretrain"
            )
            terminal_active = bool(
                self._mission_phase == 1
                or (
                    gate_transition
                    and self.config.phase2_training_mode == "full_mission"
                )
            )
            terminal_constraint_failure = bool(
                terminal_active and not task_metrics.constraints_satisfied
            )
            phase1_speed_failure = bool(
                self._mission_phase == 0
                and mission_metrics.total_speed_m_s
                > mission.phase1_catastrophic_speed_limit_m_s
            )
            premature_entry_failure = bool(
                self._mission_phase == 0
                and not gate_transition
                and mission_metrics.target_center_distance_m
                < mission.premature_entry_distance_m
            )
            distance_failure = bool(
                metrics.direct_relative_distance_m > self.config.max_distance_m
            )
            if terminal_active:
                margins = {
                    "corridor": min(
                        task_metrics.corridor_axial_margin_m,
                        task_metrics.corridor_lateral_margin_m,
                    ),
                    "fov": task_metrics.fov_margin_rad,
                    "total_speed": task_metrics.total_speed_margin_m_s,
                    "closing_speed": task_metrics.closing_speed_margin_m_s,
                }
                for name, margin in margins.items():
                    violation = max(-margin, 0.0)
                    if violation > self._active_phase2_task.constraint_tolerance:
                        self._constraint_violation_steps[name] += 1
                        self._constraint_max_violation[name] = max(
                            self._constraint_max_violation[name], violation
                        )
                self._constraint_success = (
                    self._constraint_success and task_metrics.constraints_satisfied
                )
            final_candidate = bool(
                self._mission_phase == 1
                and task_metrics.instantaneous_completion
                and not terminal_constraint_failure
            )
            self._completion_streak = (
                self._completion_streak + 1 if final_candidate else 0
            )
            final_completed = bool(
                self._completion_streak
                >= self._active_phase2_task.completion_required_steps(
                    self.config.dt_s
                )
            )
            event_reward = 0.0
            if gate_transition:
                event_reward += mission.gate_reward
                self._gate_reached = True
            if final_completed:
                event_reward += mission.final_success_reward
            if terminal_constraint_failure:
                event_reward += mission.terminal_constraint_failure_penalty
            if phase1_speed_failure or premature_entry_failure:
                event_reward += mission.phase1_failure_penalty
            if distance_failure:
                event_reward += mission.catastrophic_failure_penalty
            reward = self._reward.compute(
                self.relative,
                action,
                terminal_constraints_active=(self._mission_phase == 1),
                event_reward=event_reward,
            )
            if gate_transition and not phase1_pretrain_success:
                self._mission_phase = 1
                self._phase2_stage = "phase2"
                self._completion_streak = 0
                self._reward.reset(
                    self.relative, self._active_reference_position()
                )
            completed = bool(phase1_pretrain_success or final_completed)
            time_failure = bool(
                self.time_seconds >= self.config.max_time_s and not completed
            )
            terminated = bool(
                completed
                or terminal_constraint_failure
                or phase1_speed_failure
                or premature_entry_failure
                or distance_failure
            )
            info = self._info(
                metrics,
                task_metrics,
                mission_metrics,
                reward,
                target_diag.nfev,
                chaser_diag.nfev,
            )
            info.update(
                completed=completed,
                final_completed=final_completed,
                gate_success=phase1_pretrain_success,
                gate_transition=gate_transition,
                constraint_feasible=(
                    task_metrics.constraints_satisfied if terminal_active else True
                ),
                terminal_constraint_failure=terminal_constraint_failure,
                phase1_speed_failure=phase1_speed_failure,
                premature_entry_failure=premature_entry_failure,
                distance_failure=distance_failure,
                time_failure=time_failure,
                terminal_failure_penalty=event_reward,
            )
            if terminated or time_failure:
                self._update_phase1_curriculum(phase1_pretrain_success)
            return (
                self._observation(),
                reward.total,
                terminated,
                time_failure,
                info,
            )
        if task_metrics is not None:
            margins = {
                "corridor": min(
                    task_metrics.corridor_axial_margin_m,
                    task_metrics.corridor_lateral_margin_m,
                ),
                "fov": task_metrics.fov_margin_rad,
                "total_speed": task_metrics.total_speed_margin_m_s,
                "closing_speed": task_metrics.closing_speed_margin_m_s,
            }
            for name, margin in margins.items():
                violation = max(-margin, 0.0)
                if violation > self._active_phase2_task.constraint_tolerance:
                    self._constraint_violation_steps[name] += 1
                    self._constraint_max_violation[name] = max(
                        self._constraint_max_violation[name], violation
                    )
            self._constraint_success = (
                self._constraint_success and task_metrics.constraints_satisfied
            )
        self._completion_streak = (
            self._completion_streak + 1
            if (
                task_metrics.instantaneous_completion
                if task_metrics is not None
                else metrics.joint_success
            )
            else 0
        )
        required_steps = (
            self._active_phase2_task.completion_required_steps(self.config.dt_s)
            if task_metrics is not None
            else self.config.completion_required_steps
        )
        completed = self._completion_streak >= required_steps
        distance_failure = metrics.direct_relative_distance_m > self.config.max_distance_m
        time_failure = self.time_seconds >= self.config.max_time_s and not completed
        reward = self._reward.compute(self.relative, action)
        terminal_failure_penalty = 0.0
        if distance_failure and isinstance(reward, Phase2RewardBreakdown):
            terminal_failure_penalty = self.config.phase2_distance_failure_penalty
        info = self._info(
            metrics, task_metrics, None, reward, target_diag.nfev, chaser_diag.nfev
        )
        info.update(
            completed=bool(completed),
            distance_failure=bool(distance_failure),
            time_failure=bool(time_failure),
            terminal_failure_penalty=terminal_failure_penalty,
        )
        return (
            self._observation(),
            reward.total + terminal_failure_penalty,
            bool(completed or distance_failure),
            bool(time_failure),
            info,
        )
