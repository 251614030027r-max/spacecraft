"""Gate-0 distribution audit for the Phase-2 V2.1 observation and Stage-A."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from dynamics.relative import relative_state
from env.observation import (
    PHASE2_OBSERVATION_SCHEMA,
    build_phase2_observation,
    build_phase2_observation_v1,
)
from env.reward import Phase2TaskReward
from env.scenarios import sample_phase2_chaser_state, target_initial_state
from env.task import compute_task_metrics
from train.train import training_environment_config


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def _channel_summaries(values: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {"channel": index, **_summary(values[:, index])}
        for index in range(values.shape[1])
    ]


def _sample(stage_a: bool, samples: int, seed: int) -> dict:
    config = training_environment_config(phase2=True, phase2_stage_a_only=True)
    task = config.phase2_warmup_task if stage_a else config.phase2_task
    tumble = (
        config.phase2_warmup_tumble_scale
        if stage_a
        else config.phase2_target_tumble_scale
    )
    target = target_initial_state(tumble_scale=tumble)
    rng = np.random.default_rng(seed)
    observations_v1 = []
    observations_v2 = []
    metrics_rows = []
    reward_rows = []
    reward_model = Phase2TaskReward(
        task=task,
        time_step_s=config.dt_s,
        time_constant_s=config.storage_time_constant_s,
        attitude_scale_rad=config.observation_attitude_scale_rad,
        distance_scale_m=config.observation_distance_scale_m,
        angular_velocity_scale_rad_s=config.observation_angular_velocity_scale_rad_s,
        velocity_scale_m_s=config.observation_velocity_scale_m_s,
    )
    for _ in range(samples):
        chaser = sample_phase2_chaser_state(
            rng,
            target,
            task=task,
            axial_remaining_min_m=(
                config.phase2_warmup_axial_remaining_min_m
                if stage_a
                else config.phase2_axial_remaining_min_m
            ),
            axial_remaining_max_m=(
                config.phase2_warmup_axial_remaining_max_m
                if stage_a
                else config.phase2_axial_remaining_max_m
            ),
            max_relative_attitude_rad=(
                config.phase2_warmup_max_relative_attitude_rad
                if stage_a
                else config.phase2_max_relative_attitude_rad
            ),
            relative_angular_velocity_component_limit_rad_s=(
                config.phase2_warmup_initial_relative_angular_velocity_limit_rad_s
                if stage_a
                else config.phase2_initial_relative_angular_velocity_limit_rad_s
            ),
            near_boundary_probability=(
                config.phase2_warmup_near_boundary_probability
                if stage_a
                else config.phase2_near_boundary_probability
            ),
            initial_total_speed_limit_m_s=(
                config.phase2_warmup_initial_total_speed_limit_m_s
                if stage_a
                else config.phase2_initial_total_speed_limit_m_s
            ),
            maximum_interior_radial_fraction=(
                config.phase2_warmup_maximum_interior_radial_fraction
                if stage_a
                else config.phase2_maximum_interior_radial_fraction
            ),
            minimum_fov_margin_rad=(
                config.phase2_warmup_minimum_fov_margin_rad
                if stage_a
                else config.phase2_minimum_fov_margin_rad
            ),
        )
        relative = relative_state(target, chaser)
        metrics = compute_task_metrics(relative, task)
        observations_v1.append(
            build_phase2_observation_v1(
                relative,
                task=task,
                attitude_scale_rad=config.observation_attitude_scale_rad,
                distance_scale_m=config.observation_distance_scale_m,
                angular_velocity_scale_rad_s=config.observation_angular_velocity_scale_rad_s,
                velocity_scale_m_s=config.observation_velocity_scale_m_s,
            )
        )
        observations_v2.append(
            build_phase2_observation(
                relative,
                task=task,
                attitude_scale_rad=config.observation_attitude_scale_rad,
                distance_scale_m=config.observation_distance_scale_m,
                angular_velocity_scale_rad_s=config.observation_angular_velocity_scale_rad_s,
                velocity_scale_m_s=config.observation_velocity_scale_m_s,
                target_angular_velocity_rad_s=target.omega,
                target_angular_velocity_scale_rad_s=(
                    config.phase2_observation_target_angular_velocity_scale_rad_s
                ),
            )
        )
        corridor_radius = metrics.port_axial_distance_m * np.tan(
            task.corridor_half_angle_rad
        )
        metrics_rows.append(
            [
                metrics.axial_remaining_m,
                metrics.position_error_m,
                metrics.total_speed_m_s,
                float(np.max(np.abs(relative.omega))),
                metrics.corridor_radial_distance_m / corridor_radius,
                metrics.corridor_lateral_margin_m,
                metrics.fov_margin_rad,
                metrics.total_speed_margin_m_s,
                metrics.closing_speed_margin_m_s,
            ]
        )
        reward_model.reset(relative)
        breakdown = reward_model.compute(relative, np.zeros(6))
        reward_rows.append(
            [
                breakdown.task_penalty,
                breakdown.corridor_penalty,
                breakdown.fov_penalty,
                breakdown.total_speed_penalty,
                breakdown.closing_speed_penalty,
            ]
        )
    observations_v1_array = np.asarray(observations_v1)
    observations_v2_array = np.asarray(observations_v2)
    metrics_array = np.asarray(metrics_rows)
    reward_array = np.asarray(reward_rows)
    metric_names = (
        "axial_remaining_m",
        "position_error_m",
        "total_speed_m_s",
        "max_relative_omega_component_rad_s",
        "corridor_radial_fraction",
        "corridor_margin_m",
        "fov_margin_rad",
        "total_speed_margin_m_s",
        "closing_speed_margin_m_s",
    )
    reward_names = (
        "task",
        "corridor",
        "fov",
        "total_speed",
        "closing_speed",
    )
    return {
        "stage": "stage_a" if stage_a else "nominal",
        "samples": samples,
        "seed": seed,
        "all_truth_feasible": bool(np.all(metrics_array[:, 5:] >= 0.0)),
        "metrics": {
            name: _summary(metrics_array[:, index])
            for index, name in enumerate(metric_names)
        },
        "observation_v1_channels": _channel_summaries(observations_v1_array),
        "observation_v2_channels": _channel_summaries(observations_v2_array),
        "near_constant_v2_channels_std_lt_1e_4": np.flatnonzero(
            np.std(observations_v2_array, axis=0) < 1.0e-4
        ).tolist(),
        "reward_components_at_reset_zero_action": {
            name: _summary(reward_array[:, index])
            for index, name in enumerate(reward_names)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=26_081_200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/phase2_v2_1_gate0_stats.json"),
    )
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    result = {
        "schema_version": 1,
        "observation_schema": PHASE2_OBSERVATION_SCHEMA,
        "stage_a": _sample(True, args.samples, args.seed),
        "nominal": _sample(False, args.samples, args.seed + 100_000),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
