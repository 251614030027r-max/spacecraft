"""Answer the upper-review K1/K2/K3 questions from existing evaluation JSON.

This script is deliberately read-only with respect to evaluation runs: it never
constructs an environment or loads a model.  It aligns records by their stored
episode seed and writes one audit JSON plus a compact Markdown table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _records(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} has no records list")
    result = {int(record["seed"]): record for record in records}
    if len(result) != len(records):
        raise ValueError(f"{path} contains duplicate episode seeds")
    return result


def analyze(
    baseline_path: Path, model_paths: dict[str, Path]
) -> dict[str, Any]:
    baseline = _records(baseline_path)
    baseline_failures = [record for record in baseline.values() if not record["completed"]]
    k1_rows: list[dict[str, Any]] = []
    illegal_crossings: list[dict[str, Any]] = []
    for record in baseline_failures:
        crossings = list(record.get("entry_crossings", []))
        illegal_crossings.extend(c for c in crossings if not c["legal"])
        first = crossings[0] if crossings else None
        k1_rows.append(
            {
                "seed": int(record["seed"]),
                "time_s": None if first is None else first["time_s"],
                "legal": None if first is None else bool(first["legal"]),
                "radial_distance_m": (
                    None if first is None else first["radial_distance_m"]
                ),
                "final_position_error_m": record["final_position_error_m"],
            }
        )
    far_illegal = sum(c["radial_distance_m"] > 10.0 for c in illegal_crossings)

    models: dict[str, Any] = {}
    for model_seed, path in model_paths.items():
        model = _records(path)
        if set(model) != set(baseline):
            raise ValueError(f"seed block mismatch for model {model_seed}")
        rescued = [
            seed
            for seed in baseline
            if not baseline[seed]["completed"] and model[seed]["completed"]
        ]
        retained = [
            seed
            for seed in baseline
            if baseline[seed]["completed"] and model[seed]["completed"]
        ]
        destroyed = [
            seed
            for seed in baseline
            if baseline[seed]["completed"] and not model[seed]["completed"]
        ]
        both_failed = [
            seed
            for seed in baseline
            if not baseline[seed]["completed"] and not model[seed]["completed"]
        ]
        k2 = []
        for seed in rescued:
            baseline_illegal = [
                crossing
                for crossing in baseline[seed].get("entry_crossings", [])
                if not crossing["legal"]
            ]
            first_illegal = baseline_illegal[0] if baseline_illegal else None
            k2.append(
                {
                    "seed": seed,
                    "baseline_first_illegal_radial_distance_m": (
                        None if first_illegal is None else first_illegal["radial_distance_m"]
                    ),
                    "v2_illegal_entry_crossing_count": int(
                        model[seed].get("illegal_entry_crossing_count", 0)
                    ),
                }
            )
        ratios = [
            float(model[seed]["survival_s"]) / float(baseline[seed]["survival_s"])
            for seed in retained
        ]
        models[model_seed] = {
            "quadrants": {
                "retained": len(retained),
                "rescued": len(rescued),
                "destroyed": len(destroyed),
                "both_failed": len(both_failed),
            },
            "k2_rescued": k2,
            "k3_retained_time_ratio": {
                "count": len(ratios),
                "median": statistics.median(ratios) if ratios else None,
                "minimum": min(ratios) if ratios else None,
                "records": [
                    {
                        "seed": seed,
                        "baseline_survival_s": baseline[seed]["survival_s"],
                        "v2_survival_s": model[seed]["survival_s"],
                        "ratio": float(model[seed]["survival_s"])
                        / float(baseline[seed]["survival_s"]),
                    }
                    for seed in retained
                ],
            },
        }

    return {
        "inputs": {
            "baseline": str(baseline_path),
            "models": {key: str(value) for key, value in model_paths.items()},
        },
        "k1": {
            "baseline_failure_count": len(baseline_failures),
            "failure_first_crossing_rows": sorted(k1_rows, key=lambda row: row["seed"]),
            "illegal_crossing_event_count": len(illegal_crossings),
            "illegal_crossing_events_radial_gt_10m": far_illegal,
            "illegal_crossing_events_radial_gt_10m_fraction": (
                far_illegal / len(illegal_crossings) if illegal_crossings else None
            ),
        },
        "models": models,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# V2 upper-review readout: K1/K2/K3",
        "",
        "This report is derived only from existing evaluation JSON; it adds no runs.",
        "",
        "## K1 baseline scoring",
        "",
        "| seed | first crossing time (s) | legal | radial (m) | final position error (m) |",
        "|---:|---:|:---:|---:|---:|",
    ]
    for row in result["k1"]["failure_first_crossing_rows"]:
        lines.append(
            f"| {row['seed']} | {row['time_s']} | {row['legal']} | "
            f"{row['radial_distance_m']} | {row['final_position_error_m']} |"
        )
    k1 = result["k1"]
    lines.extend(
        [
            "",
            f"Illegal crossing events beyond 10 m: {k1['illegal_crossing_events_radial_gt_10m']} / "
            f"{k1['illegal_crossing_event_count']} = "
            f"{k1['illegal_crossing_events_radial_gt_10m_fraction']}",
            "",
            "## K2 rescued episodes and K3 retained timing",
            "",
        ]
    )
    for model_seed, model in result["models"].items():
        lines.extend(
            [
                f"### Model {model_seed}",
                "",
                f"Quadrants: {model['quadrants']}",
                "",
                "| rescued seed | baseline first illegal radial (m) | V2 illegal crossings |",
                "|---:|---:|---:|",
            ]
        )
        for row in model["k2_rescued"]:
            lines.append(
                f"| {row['seed']} | {row['baseline_first_illegal_radial_distance_m']} | "
                f"{row['v2_illegal_entry_crossing_count']} |"
            )
        k3 = model["k3_retained_time_ratio"]
        lines.extend(
            [
                "",
                f"Retained completion-time ratio: n={k3['count']}, "
                f"median={k3['median']}, minimum={k3['minimum']}.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("SEED", "JSON"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, {seed: Path(path) for seed, path in args.model})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    args.output_md.write_text(_markdown(result))


if __name__ == "__main__":
    main()
