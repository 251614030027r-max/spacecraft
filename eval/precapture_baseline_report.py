"""Aggregate the one-shot precapture baseline diagnosis into review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_CASES = (
    "fixed_h001",
    "fixed_h003",
    "fixed_h010",
    "fixed_h050",
    "fixed_h200",
    "two_stage_h050",
    "oracle_two_stage_h200",
)
CASE_SPEC = {
    "fixed_h001": (1, "fixed", None, "nominal"),
    "fixed_h003": (3, "fixed", None, "nominal"),
    "fixed_h010": (10, "fixed", None, "nominal"),
    "fixed_h050": (50, "fixed", None, "nominal"),
    "fixed_h200": (200, "fixed", None, "nominal"),
    "two_stage_h050": (50, "external_local", "two_stage", "nominal"),
    "oracle_two_stage_h200": (200, "external_local", "two_stage", "truth"),
}


def _average(values: Iterable[float]) -> float | None:
    rows = list(values)
    return float(mean(rows)) if rows else None


def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload["episode_records"]
    completed = [row for row in records if row["completed"]]
    entered = [row for row in records if "terminal_region_entry_time_s" in row]
    compute = payload["controller_compute"]["command_time_s"]
    return {
        "episodes": len(records),
        "completion_rate": _average(float(row["completed"]) for row in records),
        "zero_violation_completion_rate": _average(
            float(row["truth_geometry_zero_violation_completed"]) for row in records
        ),
        "constraint_violation_rate": _average(
            float(row["constraint_violated"]) for row in records
        ),
        "completion_time_s_mean": _average(
            float(row["time_seconds"]) for row in completed
        ),
        "force_impulse_n_s_mean": _average(
            float(row["force_impulse_n_s"]) for row in records
        ),
        "equivalent_delta_v_m_s_mean": _average(
            float(row["equivalent_delta_v_m_s"]) for row in records
        ),
        "terminal_entry_rate": len(entered) / len(records),
        "terminal_entry_time_s_mean": _average(
            float(row["terminal_region_entry_time_s"]) for row in entered
        ),
        "entry_speed_m_s_mean": _average(
            float(row["entry_target_frame_speed_m_s"]) for row in entered
        ),
        "entry_attitude_error_rad_mean": _average(
            float(row["entry_attitude_error_rad"]) for row in entered
        ),
        "entry_relative_omega_error_rad_s_mean": _average(
            float(row["entry_angular_velocity_error_rad_s"]) for row in entered
        ),
        "minimum_normalized_margin": min(
            float(row["minimum_normalized_margin"]) for row in records
        ),
        "maximum_axis_saturation_fraction": max(
            max(float(value) for value in row["saturation_fraction_per_axis"])
            for row in records
        ),
        "controller_compute_s_mean": float(compute["mean"]),
        "controller_compute_s_p95": float(compute["p95"]),
        "controller_over_budget_rate": float(
            payload["controller_compute"]["over_control_period_rate"]
        ),
    }


def _load(input_directory: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_CASES:
        path = input_directory / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing diagnosis case: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("environment", {}).get("precapture_planning_enabled") is not True:
            raise ValueError(f"not a precapture-planning result: {path}")
        actual = (
            int(payload["mpc_config"]["horizon_steps"]),
            str(payload["mpc_config"]["reference_source"]),
            payload.get("external_guidance"),
            str(payload["controller_model_source"]),
        )
        if actual != CASE_SPEC[name]:
            raise ValueError(
                f"diagnosis case {name} has settings {actual}, expected {CASE_SPEC[name]}"
            )
        runs[name] = payload
    episode_counts = {int(run["episodes"]) for run in runs.values()}
    seed_blocks = {int(run["base_seed"]) for run in runs.values()}
    if len(episode_counts) != 1 or len(seed_blocks) != 1:
        raise ValueError("all diagnosis cases must share episode count and base seed")
    return runs


def _plot_horizon_curve(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    horizons = [1, 3, 10, 50, 200]
    rows = [summaries[f"fixed_h{horizon:03d}"] for horizon in horizons]
    fields = (
        ("completion_rate", "Completion rate"),
        ("equivalent_delta_v_m_s_mean", "Equivalent delta-v (m/s)"),
        ("completion_time_s_mean", "Completion time (s)"),
        ("terminal_entry_time_s_mean", "Terminal entry time (s)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields):
        values = [np.nan if row[field] is None else row[field] for row in rows]
        axis.plot(horizons, values, marker="o")
        axis.set_xscale("log")
        axis.set_xticks(horizons, labels=[str(value) for value in horizons])
        axis.set_xlabel("MPC horizon steps")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_trajectories(runs: dict[str, dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    for case, color in (("fixed_h050", "tab:blue"), ("two_stage_h050", "tab:orange")):
        for index, trajectory in enumerate(runs[case]["representative_trajectories"]):
            target = np.asarray(
                [row["truth_position_m"] for row in trajectory["samples"]], dtype=float
            )
            inertial = np.asarray(
                [row["inertial_relative_position_m"] for row in trajectory["samples"]],
                dtype=float,
            )
            label = case if index == 0 else None
            axes[0].plot(target[:, 0], target[:, 1], color=color, alpha=0.7, label=label)
            axes[1].plot(inertial[:, 0], inertial[:, 1], color=color, alpha=0.7, label=label)
    axes[0].set_title("Target-frame trajectories")
    axes[1].set_title("Inertial relative trajectories")
    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.axis("equal")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Precapture baseline diagnosis",
        "",
        f"Episodes per case: {summary['episodes_per_case']}; base seed: {summary['base_seed']}.",
        "This report exposes the Commit-5 evidence and intentionally leaves the final Go/No-Go judgement to review of the complete trajectories.",
        "",
        "| case | success | zero-violation success | delta-v mean | completion time | entry time | min norm margin | compute p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in EXPECTED_CASES:
        row = summary["cases"][case]
        value = lambda key: "n/a" if row[key] is None else f"{row[key]:.4g}"
        lines.append(
            f"| {case} | {value('completion_rate')} | "
            f"{value('zero_violation_completion_rate')} | "
            f"{value('equivalent_delta_v_m_s_mean')} | "
            f"{value('completion_time_s_mean')} | "
            f"{value('terminal_entry_time_s_mean')} | "
            f"{value('minimum_normalized_margin')} | "
            f"{value('controller_compute_s_p95')} |"
        )
    lines.extend(
        (
            "",
            "## Review boundary",
            "",
            "Go requires stable oracle feasibility and a real planning-timescale effect: horizon-dependent fuel/time/entry behaviour or phase-dependent weakness of fixed guidance, with short MPC remaining basically safe. Pure MPC need not fail. If short and long horizons tie, trajectories are effectively identical, and hand guidance approaches the oracle, stop before hybrid and return to task definition.",
            "",
            "See `precapture_horizon_curve.png`, `precapture_trajectories.png`, and `precapture_baseline_summary.json` for the review evidence.",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    runs = _load(args.input_directory)
    summaries = {name: _run_summary(payload) for name, payload in runs.items()}
    first = runs[EXPECTED_CASES[0]]
    summary = {
        "schema_version": 1,
        "probe": "precapture_planning_commit5_baseline_diagnosis",
        "base_seed": int(first["base_seed"]),
        "episodes_per_case": int(first["episodes"]),
        "cases": summaries,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "precapture_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_directory / "PRECAPTURE_BASELINE_DIAGNOSIS.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    _plot_horizon_curve(
        summaries, args.output_directory / "precapture_horizon_curve.png"
    )
    _plot_trajectories(
        runs, args.output_directory / "precapture_trajectories.png"
    )
    print(f"Precapture diagnosis report: {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
