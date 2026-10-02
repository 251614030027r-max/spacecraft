"""Aggregate the six pre-registered G0 blocks and render fixed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import MARGIN_KEYS, summarize
from experiments.evaluate_mpc import _perception_trend


BLOCK_SEEDS = (262000, 20288000, 20290000)
SOURCES = ("oracle", "estimate")
ERROR_FIELDS = (
    ("position_error_m", "position error [m]"),
    ("attitude_error_rad", "attitude error [rad]"),
    ("velocity_error_m_s", "velocity error [m/s]"),
    ("angular_velocity_error_rad_s", "angular velocity error [rad/s]"),
)


def _load_inputs(paths: list[Path]) -> dict[tuple[int, str], dict[str, Any]]:
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (int(payload["base_seed"]), payload["control_state_source"])
        if key in runs:
            raise ValueError(f"duplicate G0 block: {key}")
        if payload["episodes"] != 20:
            raise ValueError(f"G0 block {key} has {payload['episodes']} episodes")
        runs[key] = payload
    expected = {(seed, source) for seed in BLOCK_SEEDS for source in SOURCES}
    if set(runs) != expected:
        raise ValueError(f"expected {sorted(expected)}, got {sorted(runs)}")
    return runs


def _records(
    runs: dict[tuple[int, str], dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        record
        for seed in BLOCK_SEEDS
        for record in runs[(seed, source)]["episode_records"]
    ]


def _samples(
    runs: dict[tuple[int, str], dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        sample
        for seed in BLOCK_SEEDS
        for sample in runs[(seed, source)]["perception"]["time_series_samples"]
    ]


def _aggregate(runs: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    source_summary: dict[str, Any] = {}
    for source in SOURCES:
        records = _records(runs, source)
        samples = _samples(runs, source)
        command_times = [
            value
            for seed in BLOCK_SEEDS
            for value in runs[(seed, source)]["controller_compute"][
                "command_time_samples_s"
            ]
        ]
        source_summary[source] = {
            "episodes": len(records),
            "completion_rate": float(np.mean([row["completed"] for row in records])),
            "zero_violation_completion_rate": float(
                np.mean(
                    [
                        row["truth_geometry_zero_violation_completed"]
                        for row in records
                    ]
                )
            ),
            "violation_episode_counts": {
                key: sum(int(row[key]) > 0 for row in records)
                for key in (
                    "corridor_violation_steps",
                    "fov_violation_steps",
                    "total_speed_violation_steps",
                    "closing_speed_violation_steps",
                )
            },
            "estimation_error_over_time": _perception_trend(samples),
            "pooled_estimation": {
                field: summarize(float(row[field]) for row in samples)
                for field, _ in ERROR_FIELDS
            },
            "nees_12d": summarize(float(row["nees_12d"]) for row in samples),
            "visible_feature_count": summarize(
                float(row["visible_feature_count"]) for row in samples
            ),
            "longest_no_measurement_streak_s": summarize(
                float(row["longest_no_measurement_streak_s"])
                for row in records
            ),
            "controller_compute": {
                "command_time_s": summarize(command_times),
                "over_0p1s_rate": float(np.mean(np.asarray(command_times) > 0.1)),
            },
            "worst_truth_margin": {
                key: min(float(row["minimum_margins"][key]) for row in records)
                for key in MARGIN_KEYS
            },
        }

    oracle = source_summary["oracle"]
    estimate = source_summary["estimate"]
    per_block = []
    block_checks = []
    for seed in BLOCK_SEEDS:
        counts = {}
        for source in SOURCES:
            records = runs[(seed, source)]["episode_records"]
            counts[source] = sum(
                bool(row["truth_geometry_zero_violation_completed"])
                for row in records
            )
        check = counts["estimate"] >= counts["oracle"] - 1
        block_checks.append(check)
        per_block.append({"base_seed": seed, "counts": counts, "passed": check})

    violation_deltas = {
        key: estimate["violation_episode_counts"][key]
        - oracle["violation_episode_counts"][key]
        for key in estimate["violation_episode_counts"]
    }
    checks = {
        "aggregate_zero_violation_completion_gap_at_most_0p05": (
            oracle["zero_violation_completion_rate"]
            - estimate["zero_violation_completion_rate"]
            <= 0.05 + 1.0e-12
        ),
        "aggregate_raw_completion_gap_at_most_0p05": (
            oracle["completion_rate"] - estimate["completion_rate"]
            <= 0.05 + 1.0e-12
        ),
        "each_block_at_most_one_zero_violation_completion_worse": all(block_checks),
        "no_systematic_new_violation_type": all(
            delta < 2 for delta in violation_deltas.values()
        ),
    }
    return {
        "schema_version": 1,
        "probe": "G0_perception_closed_loop",
        "block_seeds": list(BLOCK_SEEDS),
        "episodes_per_block": 20,
        "sources": source_summary,
        "paired_block_decisions": per_block,
        "violation_episode_count_deltas_estimate_minus_oracle": violation_deltas,
        "acceptance": {"checks": checks, "passed": all(checks.values())},
    }


def _plot_errors(summary: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, (field, label) in zip(axes.flat, ERROR_FIELDS):
        for source, color in (("oracle", "black"), ("estimate", "tab:blue")):
            trend = summary["sources"][source]["estimation_error_over_time"]
            time_s = [point["time_seconds"] for point in trend]
            p50 = [point[field]["p50"] for point in trend]
            p95 = [point[field]["p95"] for point in trend]
            axis.plot(time_s, p50, color=color, label=f"{source} p50")
            axis.plot(time_s, p95, color=color, linestyle="--", label=f"{source} p95")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_consistency(summary: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for source, color in (("oracle", "black"), ("estimate", "tab:blue")):
        trend = summary["sources"][source]["estimation_error_over_time"]
        time_s = [point["time_seconds"] for point in trend]
        for axis, field in zip(axes, ("nees_12d", "visible_feature_count")):
            axis.plot(
                time_s,
                [point[field]["p50"] for point in trend],
                color=color,
                label=f"{source} p50",
            )
            axis.plot(
                time_s,
                [point[field]["p95"] for point in trend],
                color=color,
                linestyle="--",
                label=f"{source} p95",
            )
    axes[0].axhline(12.0, color="tab:red", linewidth=1, label="state dimension")
    axes[0].set_ylabel("12D NEES")
    axes[1].set_ylabel("visible features")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_margins_and_compute(
    runs: dict[tuple[int, str], dict[str, Any]], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(MARGIN_KEYS))
    width = 0.36
    for offset, source, color in ((-width / 2, "oracle", "black"), (width / 2, "estimate", "tab:blue")):
        records = _records(runs, source)
        values = [min(row["minimum_margins"][key] for row in records) for key in MARGIN_KEYS]
        axes[0].bar(x + offset, values, width, label=source, color=color)
        command = [
            value
            for seed in BLOCK_SEEDS
            for value in runs[(seed, source)]["controller_compute"]["command_time_samples_s"]
        ]
        ordered = np.sort(np.asarray(command))
        axes[1].plot(1.0e3 * ordered, np.linspace(0.0, 1.0, ordered.size), label=source, color=color)
    axes[0].axhline(0.0, color="tab:red", linewidth=1)
    axes[0].set_xticks(x, [key.replace("_margin_m_s", "").replace("_margin_m", "").replace("_margin_rad", "") for key in MARGIN_KEYS], rotation=25, ha="right")
    axes[0].set_ylabel("worst truth margin")
    axes[0].legend()
    axes[1].axvline(100.0, color="tab:red", linewidth=1, label="0.1 s budget")
    axes[1].set_xlabel("controller command time [ms]")
    axes[1].set_ylabel("empirical CDF")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_trajectories(
    runs: dict[tuple[int, str], dict[str, Any]], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, source in zip(axes, SOURCES):
        for seed in BLOCK_SEEDS:
            for trajectory in runs[(seed, source)]["representative_trajectories"]:
                points = np.asarray(
                    [row["truth_position_m"] for row in trajectory["samples"]]
                )
                if points.size:
                    axis.plot(points[:, 0], points[:, 1], alpha=0.55)
        axis.set_title(source)
        axis.set_xlabel("target-frame x [m]")
        axis.set_ylabel("target-frame y [m]")
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = _load_inputs(args.inputs)
    summary = _aggregate(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_errors(summary, args.figure_dir / "g0_estimation_errors.png")
    _plot_consistency(summary, args.figure_dir / "g0_nees_visibility.png")
    _plot_margins_and_compute(runs, args.figure_dir / "g0_margins_compute.png")
    _plot_trajectories(runs, args.figure_dir / "g0_trajectories.png")
    print(json.dumps(summary["acceptance"], indent=2))
    print(f"G0 summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
