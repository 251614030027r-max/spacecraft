"""One-shot semantic acceptance for the precapture-planning task."""

from __future__ import annotations

import json

import numpy as np

from dynamics.relative import relative_state
from env.phase2_env import precapture_planning_environment_config
from env.scenarios import sample_precapture_planning_chaser_state, target_initial_state
from env.task import compute_precapture_metrics


def main() -> None:
    config = precapture_planning_environment_config()
    task = config.precapture_task
    target = target_initial_state(tumble_scale=config.phase2_target_tumble_scale)
    rows = []
    for seed in range(100):
        chaser = sample_precapture_planning_chaser_state(
            np.random.default_rng(seed),
            target,
            task=task,
            initial_range_min_m=config.precapture_initial_range_min_m,
            initial_range_max_m=config.precapture_initial_range_max_m,
            initial_inertial_relative_speed_max_m_s=(
                config.precapture_initial_inertial_speed_max_m_s
            ),
            pointing_error_max_rad=config.precapture_initial_pointing_error_max_rad,
            chaser_angular_velocity_component_limit_rad_s=(
                config.precapture_initial_chaser_omega_component_limit_rad_s
            ),
        )
        metrics = compute_precapture_metrics(
            target, chaser, relative_state(target, chaser), task
        )
        rows.append(metrics)
    checks = {
        "finite": all(np.all(np.isfinite(row.position_target_m)) for row in rows),
        "range_15_to_20_m": all(
            15.0 <= row.target_center_distance_m <= 20.0 for row in rows
        ),
        "outside_terminal_region": all(
            row.target_center_distance_m > task.terminal_activation_range_m
            for row in rows
        ),
        "outside_corridor": all(row.corridor_lateral_margin_m < 0.0 for row in rows),
        "keepout_safe": all(row.keepout_margin_m > 0.0 for row in rows),
        "fov_safe": all(row.fov_margin_rad >= 0.0 for row in rows),
        "outer_inertial_speed_safe": all(
            row.outer_inertial_speed_margin_m_s >= 0.0 for row in rows
        ),
        "not_initialized_corotating": float(
            np.median([row.target_frame_speed_m_s for row in rows])
        )
        > 0.50,
        "not_initialized_rate_synchronized": float(
            np.median([row.angular_velocity_error_rad_s for row in rows])
        )
        > task.completion_angular_velocity_rad_s,
    }
    result = {
        "samples": len(rows),
        "checks": checks,
        "passed": all(checks.values()),
        "median_target_frame_speed_m_s": float(
            np.median([row.target_frame_speed_m_s for row in rows])
        ),
        "median_relative_omega_error_rad_s": float(
            np.median([row.angular_velocity_error_rad_s for row in rows])
        ),
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
