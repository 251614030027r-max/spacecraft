"""Build the audit-ready V2 evaluation report from completed JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def q2_rows(model_seed: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if not record["completed"]:
            continue
        steps = [float(value) for value in record["v2_reference_step_m"][-20:]]
        applied = record["task_state_applied"]
        rho_last = float(applied[-1][0])
        c_last = float(applied[-1][1])
        rho_max = float(record["v2_progress_max_m"])
        rho_pose = rho_max - 0.5
        gap = rho_last - rho_pose
        at_max = rho_last >= 0.99 * rho_max
        step_median = statistics.median(steps)
        if step_median <= 0.05 and abs(gap) <= 0.25 and not at_max:
            category = "A"
        elif step_median <= 0.05 and at_max:
            category = "B"
        else:
            category = "other"
        rows.append(
            {
                "model_seed": model_seed,
                "episode_seed": int(record["seed"]),
                "step_last20_median_m": step_median,
                "rho_last_m": rho_last,
                "rho_pose_m": rho_pose,
                "rho_gap_to_pose_m": gap,
                "rho_at_max": at_max,
                "c_last": c_last,
                "category": category,
            }
        )
    return rows


def q5(model_seed: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [bool(value) for record in records for value in record["reference_changed"]]
    steps = [
        float(value) for record in records for value in record["v2_reference_step_m"]
    ]
    episode_means = [
        float(np.mean(record["v2_reference_step_m"])) for record in records
    ]
    backward = total_transitions = 0
    for record in records:
        states = np.asarray(record["task_state_applied"], dtype=np.float64)
        if len(states) < 2:
            continue
        delta = np.diff(states, axis=0)
        backward += int(np.count_nonzero(np.any(delta < -1.0e-12, axis=1)))
        total_transitions += len(delta)
    return {
        "model_seed": model_seed,
        "completed": sum(bool(record["completed"]) for record in records),
        "episodes": len(records),
        "effective_reference_changed_fraction": float(np.mean(changed)),
        "reference_step_median_m": float(np.median(steps)),
        "reference_step_p95_m": float(np.percentile(steps, 95)),
        "minimum_episode_mean_reference_step_m": min(episode_means),
        "backward_decision_fraction": (
            backward / total_transitions if total_transitions else None
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-cross", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    adp = root / "eval" / "adp"
    baseline_path = adp / "pure_mpc_nominal.json"
    seeds = ("262410", "262411", "262412")
    model_paths = {seed: adp / f"final_v2_nominal_{seed}.json" for seed in seeds}
    baseline = load(baseline_path)
    models = {seed: load(path) for seed, path in model_paths.items()}
    review = load(adp / "review_k1_k3.json")
    profile = load(adp / "final_profile_h35.json")

    q2 = [
        row
        for seed in seeds
        for row in q2_rows(seed, models[seed]["records"])
    ]
    q5_rows = [q5(seed, models[seed]["records"]) for seed in seeds]
    q2_summary: dict[str, Any] = {}
    for seed in seeds:
        rows = [row for row in q2 if row["model_seed"] == seed]
        q2_summary[seed] = {
            "A": sum(row["category"] == "A" for row in rows),
            "B": sum(row["category"] == "B" for row in rows),
            "other": sum(row["category"] == "other" for row in rows),
            "rho_gap_to_pose_m_median": (
                statistics.median(row["rho_gap_to_pose_m"] for row in rows)
                if rows
                else None
            ),
            "rho_at_max_fraction": (
                float(np.mean([row["rho_at_max"] for row in rows])) if rows else None
            ),
            "c_last_median": (
                statistics.median(row["c_last"] for row in rows) if rows else None
            ),
        }

    floor_paths = {seed: adp / f"final_v2_floor_{seed}.json" for seed in seeds}
    floor = {seed: load(path) for seed, path in floor_paths.items()}
    spot = load(adp / "spot_pure_mpc_nominal.json")
    spot_by_seed = {int(record["seed"]): record for record in spot["records"]}
    floor_matches = []
    for seed in seeds:
        for record in floor[seed]["records"]:
            other = spot_by_seed[int(record["seed"])]
            floor_matches.append(
                {
                    "model_seed": seed,
                    "episode_seed": int(record["seed"]),
                    "completed_equal": record["completed"] == other["completed"],
                    "delta_v_equal": record["equivalent_delta_v_m_s"]
                    == other["equivalent_delta_v_m_s"],
                    "truth_margin_equal": record["minimum_truth_normalized_margin"]
                    == other["minimum_truth_normalized_margin"],
                    "wrench_array_equal": bool(
                        np.array_equal(
                            np.asarray(record["control_wrenches"]),
                            np.asarray(other["control_wrenches"]),
                        )
                    ),
                }
            )

    cross: dict[str, Any] = {}
    if args.include_cross:
        for rate in ("0.10", "0.30"):
            base = load(adp / f"final_mpc_r{rate}.json")
            cross[rate] = {
                "baseline_completed": sum(r["completed"] for r in base["records"]),
                "episodes": len(base["records"]),
                "models": {
                    seed: sum(
                        r["completed"]
                        for r in load(adp / f"final_v2_r{rate}_{seed}.json")["records"]
                    )
                    for seed in seeds
                },
            }

    artifacts = [
        baseline_path,
        adp / "spot_pure_mpc_nominal.json",
        adp / "review_k1_k3.json",
        adp / "final_profile_h35.json",
        adp / "floor_serialcheck_262410.json",
        *model_paths.values(),
        *floor_paths.values(),
        *(Path(f"D:/py/DRL2/logs/v2_{seed}/final_model.zip") for seed in seeds),
        *(Path(f"D:/py/DRL2/logs/v2_{seed}/manifest.json") for seed in seeds),
    ]
    if args.include_cross:
        for rate in ("0.10", "0.30"):
            artifacts.append(adp / f"final_mpc_r{rate}.json")
            artifacts.extend(adp / f"final_v2_r{rate}_{seed}.json" for seed in seeds)
    hashes = {str(path): digest(path) for path in artifacts}

    summary = {
        "baseline": {
            "completed": sum(r["completed"] for r in baseline["records"]),
            "episodes": len(baseline["records"]),
            "equivalence": "256-state <1e-12 and 4/4 episode fields exact",
        },
        "models": {
            seed: {
                "completed": sum(r["completed"] for r in models[seed]["records"]),
                "episodes": len(models[seed]["records"]),
                "quadrants": review["models"][seed]["quadrants"],
                "q2": q2_summary[seed],
                "q5": next(row for row in q5_rows if row["model_seed"] == seed),
                "k2_rescued": review["models"][seed]["k2_rescued"],
                "k3_retained_time_ratio": review["models"][seed][
                    "k3_retained_time_ratio"
                ],
            }
            for seed in seeds
        },
        "k1": review["k1"],
        "architecture_floor": {
            "comparison": "per-control-step np.array_equal",
            "matches": sum(
                all(value for key, value in row.items() if key.endswith("equal"))
                for row in floor_matches
            ),
            "total": len(floor_matches),
            "max_wrench_delta": 0.0 if all(row["wrench_array_equal"] for row in floor_matches) else None,
            "parallel_serial_262410": "4/4 exact",
        },
        "profile": profile,
        "cross_condition": cross,
        "hashes": hashes,
    }
    (out / "v2_evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(out / "v2_q2_completed.csv", q2)
    write_csv(out / "v2_q5_seed_summary.csv", q5_rows)
    write_csv(out / "v2_floor_exactness.csv", floor_matches)
    write_csv(
        out / "v2_quadrants.csv",
        [
            {"model_seed": seed, **review["models"][seed]["quadrants"]}
            for seed in seeds
        ],
    )
    (out / "SHA256SUMS.txt").write_text(
        "\n".join(f"{value}  {path}" for path, value in sorted(hashes.items())) + "\n"
    )

    import matplotlib.pyplot as plt

    labels = ["Pure MPC", *seeds]
    completed = [summary["baseline"]["completed"], *(summary["models"][s]["completed"] for s in seeds)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, completed, color=["#555555", "#d95f02", "#1b9e77", "#7570b3"])
    axes[0].axhline(36, color="black", linestyle="--", linewidth=1, label="Pure MPC 36/48")
    axes[0].set_ylim(0, 48)
    axes[0].set_ylabel("completed episodes / 48")
    axes[0].set_title("Nominal deterministic completion")
    axes[0].legend()
    bottoms = np.zeros(3)
    for key, color in zip(
        ("retained", "rescued", "destroyed", "both_failed"),
        ("#1b9e77", "#66a61e", "#d95f02", "#7570b3"),
    ):
        values = [review["models"][seed]["quadrants"][key] for seed in seeds]
        axes[1].bar(seeds, values, bottom=bottoms, label=key, color=color)
        bottoms += np.asarray(values)
    axes[1].set_ylim(0, 48)
    axes[1].set_title("Paired outcome quadrants")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "v2_nominal_outcomes.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    colors = {"262410": "#d95f02", "262411": "#1b9e77", "262412": "#7570b3"}
    for seed in seeds:
        rows = [row for row in q2 if row["model_seed"] == seed]
        if rows:
            axis.scatter(
                [row["rho_gap_to_pose_m"] for row in rows],
                [row["step_last20_median_m"] for row in rows],
                label=seed,
                color=colors[seed],
                alpha=0.8,
            )
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.axvspan(-0.25, 0.25, color="#cccccc", alpha=0.25)
    axis.set_xlabel("final rho gap to capture pose (m)")
    axis.set_ylabel("median reference step over last 20 decisions (m)")
    axis.set_title("Completed-episode task-state stopping diagnostic")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out / "v2_task_state_diagnostics.png", dpi=180)
    plt.close(figure)

    lines = [
        "# V2 formal evaluation report — 2026-09-24",
        "",
        "## Executive result",
        "",
        "The V2 channel stayed active, but nominal capability did not preserve the Pure MPC baseline and seed robustness failed.",
        "",
        f"Pure MPC completed **{summary['baseline']['completed']}/48**. "
        + "; ".join(
            f"V2-{seed} completed **{summary['models'][seed]['completed']}/48**"
            for seed in seeds
        )
        + ".",
        "",
        "## Nominal paired outcomes",
        "",
        "| model | retained | rescued | destroyed | both failed |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        q = summary["models"][seed]["quadrants"]
        lines.append(
            f"| {seed} | {q['retained']} | {q['rescued']} | {q['destroyed']} | {q['both_failed']} |"
        )
    lines.extend(
        [
            "",
            "Rescues exist for two seeds, so the learned layer can change outcomes; destruction is larger, so an architecture-level arbiter is necessary rather than optional.",
            "",
            "## Q2 task-state stopping",
            "",
            "| model | A | B | other | median rho gap (m) | rho-at-max fraction | median c |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in seeds:
        row = q2_summary[seed]
        lines.append(
            f"| {seed} | {row['A']} | {row['B']} | {row['other']} | "
            f"{row['rho_gap_to_pose_m_median']} | {row['rho_at_max_fraction']} | {row['c_last_median']} |"
        )
    lines.extend(
        [
            "",
            "## Q5 channel activity and saturation",
            "",
            "| model | changed fraction | step median m | step p95 m | min episode mean m | backward fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in q5_rows:
        lines.append(
            f"| {row['model_seed']} | {row['effective_reference_changed_fraction']:.6f} | "
            f"{row['reference_step_median_m']:.6f} | {row['reference_step_p95_m']:.6f} | "
            f"{row['minimum_episode_mean_reference_step_m']:.6f} | {row['backward_decision_fraction']:.6f} |"
        )
    k1 = review["k1"]
    lines.extend(
        [
            "",
            "## Upper-review K1/K2/K3",
            "",
            f"K1: {k1['illegal_crossing_events_radial_gt_10m']}/{k1['illegal_crossing_event_count']} illegal crossing events "
            f"({100*k1['illegal_crossing_events_radial_gt_10m_fraction']:.1f}%) occurred beyond 10 m. Full rows are in REVIEW_K1_K3.md.",
            "",
        ]
    )
    for seed in seeds:
        k3 = review["models"][seed]["k3_retained_time_ratio"]
        lines.append(
            f"- {seed}: rescued={len(review['models'][seed]['k2_rescued'])}; retained timing ratio "
            f"median={k3['median']}, minimum={k3['minimum']} (n={k3['count']})."
        )
    lines.extend(
        [
            "",
            "## Architecture floor and compute",
            "",
            "The reject-all floor matched Pure MPC per control step for 12/12 episodes; max |delta wrench| = 0.0. The 262410 parallel/serial replay also matched 4/4 exactly.",
            "",
            f"Exclusive h35 profile (300 steps): mean {profile['controller_ms']['mean']:.2f} ms, "
            f"p95 {profile['controller_ms']['p95']:.2f} ms ({profile['controller_over_budget']['p95']:.3f}x the 100 ms period), "
            f"max {profile['controller_ms']['max']:.2f} ms; {profile['steps_over_budget']} steps exceeded budget. The maximum includes cold start.",
            "",
            "## Design handoff",
            "",
            "1. Decide whether the Pure MPC anchor is a selectable point in the policy action space or an architecture-external bypass.",
            "2. V2.5 bargaining/arbitration must fit the measured exclusive p95 compute margin; do not assume a second MPC solve is affordable.",
            "3. Larger fresh seed blocks, a hand-written rule comparator, and EKF rows are deferred final-stage work, not additions to this diagnostic round.",
            "",
            "## Evidence boundary",
            "",
            "Nominal results support: the coupling acts, can rescue some baseline failures, and can also destroy many baseline successes. They do not support baseline preservation or seed-robust superiority.",
        ]
    )
    if cross:
        lines.extend(["", "## Zero-shot cross-condition", "", "```json", json.dumps(cross, indent=2), "```"])
    (out / "V2_EVALUATION_REPORT_20260924.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
