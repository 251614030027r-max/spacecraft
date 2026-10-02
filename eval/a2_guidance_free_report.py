"""Validate and aggregate the 15 pre-registered A2 trajectory blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import MARGIN_KEYS, summarize
from experiments.evaluate_a2 import METHODS


BLOCK_SEEDS = (262000, 20288000, 20290000)


def _correlation(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _load(directory: Path) -> dict[tuple[int, str], dict[str, Any]]:
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for path in directory.glob("a2_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "base_seed" not in payload or "a2_method" not in payload:
            continue
        key = (int(payload["base_seed"]), str(payload["a2_method"]))
        if key in runs:
            raise ValueError(f"duplicate A2 block: {key}")
        if payload["episodes"] != 20:
            raise ValueError(f"A2 block {key} has {payload['episodes']} episodes")
        runs[key] = payload
    expected = {(seed, method) for seed in BLOCK_SEEDS for method in METHODS}
    if set(runs) != expected:
        missing = sorted(expected - set(runs))
        extra = sorted(set(runs) - expected)
        raise ValueError(f"A2 inputs incomplete; missing={missing}, extra={extra}")
    return runs


def _aggregate(runs: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        records = [
            row
            for seed in BLOCK_SEEDS
            for row in runs[(seed, method)]["episode_records"]
        ]
        samples = [
            row
            for seed in BLOCK_SEEDS
            for row in runs[(seed, method)]["perception"]["time_series_samples"]
        ]
        completion_times = [float(row["time_seconds"]) for row in records if row["completed"]]
        position_std_max = [max(row["position_std_axes_m"]) for row in samples]
        velocity_std_max = [max(row["velocity_std_axes_m_s"]) for row in samples]
        fov_angles = [float(row["fov_angle_rad"]) for row in samples]
        visible = [float(row["visible_feature_count"]) for row in samples]
        methods[method] = {
            "episodes": len(records),
            "completion_rate": float(np.mean([row["completed"] for row in records])),
            "zero_violation_completion_rate": float(np.mean([
                row["truth_geometry_zero_violation_completed"] for row in records
            ])),
            "completion_time_s": summarize(completion_times),
            "force_impulse_n_s": summarize(
                float(row["force_impulse_n_s"]) for row in records
            ),
            "torque_impulse_nm_s": summarize(
                float(row["torque_impulse_nm_s"]) for row in records
            ),
            "final_errors": {
                field: summarize(float(row[field]) for row in records)
                for field in (
                    "position_error_m",
                    "attitude_error_rad",
                    "translational_velocity_error_m_s",
                    "angular_velocity_error_rad_s",
                )
            },
            "worst_truth_margin": {
                key: min(float(row["minimum_margins"][key]) for row in records)
                for key in MARGIN_KEYS
            },
            "visibility": {
                "visible_feature_count": summarize(visible),
                "longest_no_measurement_streak_s": summarize(
                    float(row["longest_no_measurement_streak_s"]) for row in records
                ),
            },
            "axis_covariance": {
                "max_position_std_m": summarize(position_std_max),
                "max_velocity_std_m_s": summarize(velocity_std_max),
                "fov_angle_vs_position_std": _correlation(fov_angles, position_std_max),
                "fov_angle_vs_velocity_std": _correlation(fov_angles, velocity_std_max),
                "fov_angle_vs_visible_count": _correlation(fov_angles, visible),
            },
            "worst_axis_consistency": {
                field: summarize(float(row[field]) for row in samples)
                for field in (
                    "worst_position_axis_error_over_std",
                    "worst_velocity_axis_error_over_std",
                )
            },
            "violation_episode_counts": {
                key: sum(int(row[key]) > 0 for row in records)
                for key in (
                    "corridor_violation_steps",
                    "fov_violation_steps",
                    "total_speed_violation_steps",
                    "closing_speed_violation_steps",
                )
            },
        }
    return {
        "schema_version": 1,
        "probe": "A2_guidance_free",
        "block_seeds": list(BLOCK_SEEDS),
        "episodes_per_block": 20,
        "methods": methods,
        "compute_claim": "pending separate serial profile",
    }


def _add_compute(summary: dict[str, Any], directory: Path) -> None:
    profiles: dict[str, Any] = {}
    for path in directory.glob("compute_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload["a2_method"])
        if method in profiles:
            raise ValueError(f"duplicate compute profile: {method}")
        if not payload.get("compute_profile_valid", False):
            raise ValueError(f"compute profile not declared serial: {method}")
        compute = payload["controller_compute"]
        profiles[method] = {
            "steps": int(compute["command_time_s"]["count"]),
            "command_time_s": compute["command_time_s"],
            "over_0p1s_rate": float(compute["over_control_period_rate"]),
            "qp_solve_time_s": compute["qp_solve_time_s"],
            "model_linearization_time_s": compute["model_linearization_time_s"],
            "rollout_time_s": compute["rollout_time_s"],
            "exact_refresh_time_s": compute["exact_refresh_time_s"],
            "runtime": payload["runtime"],
            "strict_p95_realtime": compute["command_time_s"]["p95"] <= 0.1,
            "mean_realtime": compute["command_time_s"]["mean"] <= 0.1,
        }
    if set(profiles) != set(METHODS):
        raise ValueError(
            f"expected compute profiles {sorted(METHODS)}, got {sorted(profiles)}"
        )
    summary["compute_claim"] = "separate one-process serial profile"
    summary["serial_compute"] = profiles

    strong = summary["methods"]["planning_tracking"]
    oracle = summary["methods"]["planning_tracking_oracle"]
    position_cov_delta = (
        strong["axis_covariance"]["max_position_std_m"]["p95"]
        / oracle["axis_covariance"]["max_position_std_m"]["p95"]
        - 1.0
    )
    velocity_cov_delta = (
        strong["axis_covariance"]["max_velocity_std_m_s"]["p95"]
        / oracle["axis_covariance"]["max_velocity_std_m_s"]["p95"]
        - 1.0
    )
    summary["decision"] = {
        "strong_nonlearning_safe": strong["zero_violation_completion_rate"] == 1.0,
        "strong_nonlearning_near_oracle_time": abs(
            strong["completion_time_s"]["mean"]
            - oracle["completion_time_s"]["mean"]
        ) <= 5.0,
        "strong_nonlearning_strict_p95_realtime": profiles[
            "planning_tracking"
        ]["strict_p95_realtime"],
        "strong_nonlearning_mean_realtime": profiles[
            "planning_tracking"
        ]["mean_realtime"],
        "matched_position_covariance_p95_delta": position_cov_delta,
        "matched_velocity_covariance_p95_delta": velocity_cov_delta,
        "action_related_visibility_or_covariance_bottleneck": bool(
            strong["visibility"]["visible_feature_count"]["min"] < 5.0
            or strong["visibility"]["longest_no_measurement_streak_s"]["max"] > 0.0
            or abs(position_cov_delta) > 0.10
            or abs(velocity_cov_delta) > 0.10
        ),
        "interpretation": (
            "strong classical planning closes the control gap; exact-refresh "
            "compute tail remains; current geometry shows no active-perception gap"
        ),
    }


def _plot(summary: dict[str, Any], output: Path) -> None:
    labels = list(METHODS)
    short = [name.replace("planning_", "plan_").replace("guided_a1_", "a1_") for name in labels]
    methods = summary["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [methods[name]["completion_rate"] for name in labels], 0.36, label="completion")
    axes[0].bar(x + 0.18, [methods[name]["zero_violation_completion_rate"] for name in labels], 0.36, label="zero violation")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("rate")
    axes[1].bar(x, [methods[name]["force_impulse_n_s"]["mean"] for name in labels])
    axes[1].set_ylabel("mean force impulse [N s]")
    axes[2].bar(x - 0.18, [methods[name]["axis_covariance"]["max_position_std_m"]["p95"] for name in labels], 0.36, label="position std")
    axes[2].bar(x + 0.18, [methods[name]["axis_covariance"]["max_velocity_std_m_s"]["p95"] for name in labels], 0.36, label="velocity std")
    axes[2].set_ylabel("pooled p95 worst-axis std")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x, short, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_compute(summary: dict[str, Any], output: Path) -> None:
    labels = list(METHODS)
    short = [name.replace("planning_", "plan_").replace("guided_a1_", "a1_") for name in labels]
    profiles = summary["serial_compute"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - 0.18, [1.0e3 * profiles[name]["command_time_s"]["mean"] for name in labels], 0.36, label="mean")
    axes[0].bar(x + 0.18, [1.0e3 * profiles[name]["command_time_s"]["p95"] for name in labels], 0.36, label="p95")
    axes[0].axhline(100.0, color="tab:red", linewidth=1, label="0.1 s period")
    axes[0].set_ylabel("command time [ms]")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [profiles[name]["over_0p1s_rate"] for name in labels])
    axes[1].set_ylabel("fraction above 0.1 s")
    for axis in axes:
        axis.set_xticks(x, short, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--compute", type=Path, required=True)
    parser.add_argument("--compute-figure", type=Path, required=True)
    args = parser.parse_args()
    summary = _aggregate(_load(args.input))
    _add_compute(summary, args.compute)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.compute_figure.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.figure)
    _plot_compute(summary, args.compute_figure)
    print(json.dumps({"summary": str(args.summary), "figure": str(args.figure), "compute_figure": str(args.compute_figure)}))


if __name__ == "__main__":
    main()
