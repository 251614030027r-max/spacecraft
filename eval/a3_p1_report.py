"""Validate, aggregate and plot the pre-registered A3 P1 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import MARGIN_KEYS, summarize
from experiments.evaluate_a3 import METHODS


BLOCKS = (262000, 20288000, 20290000)


def _records(run: dict[str, Any]) -> list[dict[str, Any]]:
    return list(run["episode_records"])


def _row(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["completed"]]
    return {
        "episodes": len(records),
        "completion_rate": float(np.mean([row["completed"] for row in records])),
        "zero_violation_completion_rate": float(np.mean([
            row["truth_geometry_zero_violation_completed"] for row in records
        ])),
        "completion_time_s": summarize(row["time_seconds"] for row in completed),
        "force_impulse_completed_n_s": summarize(
            row["force_impulse_n_s"] for row in completed
        ),
        "force_impulse_all_n_s": summarize(
            row["force_impulse_n_s"] for row in records
        ),
        "torque_impulse_completed_nm_s": summarize(
            row["torque_impulse_nm_s"] for row in completed
        ),
        "torque_impulse_all_nm_s": summarize(
            row["torque_impulse_nm_s"] for row in records
        ),
        "worst_truth_margin": {
            key: min(float(row["minimum_margins"][key]) for row in records)
            for key in MARGIN_KEYS
        },
        "violation_episode_counts": {
            key: sum(int(row[key]) > 0 for row in records)
            for key in (
                "corridor_violation_steps", "fov_violation_steps",
                "total_speed_violation_steps", "closing_speed_violation_steps",
            )
        },
    }


def _load_formal(directory: Path) -> dict[tuple[int, str], dict[str, Any]]:
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for path in directory.glob("p1_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (int(payload["base_seed"]), str(payload["a3_method"]))
        if key in runs or payload["episodes"] != 20:
            raise ValueError(f"invalid or duplicate formal block: {key}")
        runs[key] = payload
    expected = {(seed, method) for seed in BLOCKS for method in METHODS}
    if set(runs) != expected:
        raise ValueError(f"formal inputs incomplete; missing={sorted(expected-set(runs))}")
    return runs


def _mean_completed(records: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in records if row["completed"]]
    return float(np.mean(values)) if values else float("nan")


def _median_completed(records: list[dict[str, Any]]) -> float:
    values = [float(row["time_seconds"]) for row in records if row["completed"]]
    return float(np.median(values)) if values else float("nan")


def aggregate(
    formal_dir: Path, sensitivity_dir: Path, compute_dir: Path
) -> dict[str, Any]:
    runs = _load_formal(formal_dir)
    methods = {
        method: _row([
            record for seed in BLOCKS for record in _records(runs[(seed, method)])
        ])
        for method in METHODS
    }
    block_comparison = []
    force_worse = []
    time_worse = []
    for seed in BLOCKS:
        online = _records(runs[(seed, "deployable_online")])
        bound = _records(runs[(seed, "offline_estimate_bound")])
        force_ratio = _mean_completed(online, "force_impulse_n_s") / _mean_completed(
            bound, "force_impulse_n_s"
        )
        time_ratio = _median_completed(online) / _median_completed(bound)
        force_worse.append(force_ratio > 1.0)
        time_worse.append(time_ratio > 1.0)
        block_comparison.append({
            "base_seed": seed,
            "force_ratio_online_over_bound": force_ratio,
            "median_time_ratio_online_over_bound": time_ratio,
        })
    aggregate_force_ratio = (
        methods["deployable_online"]["force_impulse_completed_n_s"]["mean"]
        / methods["offline_estimate_bound"]["force_impulse_completed_n_s"]["mean"]
    )
    aggregate_time_ratio = (
        methods["deployable_online"]["completion_time_s"]["median"]
        / methods["offline_estimate_bound"]["completion_time_s"]["median"]
    )

    sensitivity: dict[str, Any] = {}
    for path in sensitivity_dir.glob("sensitivity_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["episodes"] != 20 or payload["base_seed"] != 262000:
            raise ValueError(f"invalid sensitivity run: {path}")
        horizon = int(payload["mpc_config"]["horizon_steps"])
        sensitivity[f"h{horizon}"] = _row(_records(payload))
    if set(sensitivity) != {"h10", "h15"}:
        raise ValueError("sensitivity inputs must contain h10 and h15")

    compute: dict[str, Any] = {}
    for path in compute_dir.glob("compute_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload["a3_method"])
        if not payload.get("compute_profile_valid", False):
            raise ValueError(f"compute profile not declared serial: {method}")
        timing = payload["controller_compute"]
        compute[method] = {
            "command_time_s": timing["command_time_s"],
            "over_0p1s_rate": timing["over_control_period_rate"],
            "runtime": payload["runtime"],
        }
    if set(compute) != set(METHODS):
        raise ValueError("serial compute profiles incomplete")

    bounds_good = (
        methods["offline_estimate_bound"]["zero_violation_completion_rate"] >= 0.8
        and methods["offline_truth_bound"]["zero_violation_completion_rate"] >= 0.8
        and methods["long_exact_h50"]["zero_violation_completion_rate"] >= 0.8
    )
    safety_gap = (
        methods["deployable_online"]["zero_violation_completion_rate"]
        < methods["offline_estimate_bound"]["zero_violation_completion_rate"]
    )
    stable_force_gap = all(force_worse) and aggregate_force_ratio >= 1.05
    stable_time_gap = all(time_worse) and aggregate_time_ratio >= 1.05
    return {
        "schema_version": 1,
        "probe": "A3_P1_deployable_planning",
        "blocks": list(BLOCKS),
        "methods": methods,
        "sensitivity": sensitivity,
        "serial_compute": compute,
        "block_comparison": block_comparison,
        "decision": {
            "bounds_high_quality": bounds_good,
            "stable_force_gap": stable_force_gap,
            "stable_completion_time_gap": stable_time_gap,
            "completion_or_safety_gap": safety_gap,
            "p1_gap_established": bounds_good and (
                stable_force_gap or stable_time_gap or safety_gap
            ),
            "aggregate_force_ratio_online_over_bound": aggregate_force_ratio,
            "aggregate_median_time_ratio_online_over_bound": aggregate_time_ratio,
        },
    }


def plot_main(summary: dict[str, Any], output: Path) -> None:
    labels = list(METHODS)
    rows = summary["methods"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(x, [rows[name]["zero_violation_completion_rate"] for name in labels])
    axes[0, 0].set_ylabel("zero-violation completion rate")
    axes[0, 1].bar(x, [rows[name]["completion_time_s"]["median"] if rows[name]["completion_time_s"] else np.nan for name in labels])
    axes[0, 1].set_ylabel("median completion time [s]")
    axes[1, 0].bar(x, [rows[name]["force_impulse_completed_n_s"]["mean"] if rows[name]["force_impulse_completed_n_s"] else np.nan for name in labels])
    axes[1, 0].set_ylabel("completed force impulse [N s]")
    axes[1, 1].bar(x, [rows[name]["torque_impulse_completed_nm_s"]["mean"] if rows[name]["torque_impulse_completed_nm_s"] else np.nan for name in labels])
    axes[1, 1].set_ylabel("completed torque impulse [N m s]")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_frontier(summary: dict[str, Any], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        row = summary["methods"][method]
        timing = summary["serial_compute"][method]["command_time_s"]
        force = row["force_impulse_completed_n_s"]
        if force is not None:
            axis.scatter(1e3 * timing["p95"], force["mean"], s=80)
            axis.annotate(method, (1e3 * timing["p95"], force["mean"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.axvline(100.0, color="tab:red", linewidth=1, label="0.1 s reference")
    axis.set_xlabel("serial command p95 [ms]")
    axis.set_ylabel("completed force impulse mean [N s]")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--compute", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--main-figure", type=Path, required=True)
    parser.add_argument("--frontier-figure", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.formal, args.sensitivity, args.compute)
    for path in (args.summary, args.main_figure, args.frontier_figure):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_main(summary, args.main_figure)
    plot_frontier(summary, args.frontier_figure)
    print(json.dumps({"summary": str(args.summary), "p1_gap": summary["decision"]["p1_gap_established"]}))


if __name__ == "__main__":
    main()
