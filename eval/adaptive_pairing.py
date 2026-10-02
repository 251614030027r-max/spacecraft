"""Paired audit for adaptive SAC-MPC against its fixed-setpoint anchor.

The comparison is seed-paired.  It separates baseline successes retained by the
learned layer from failures rescued and successes destroyed, then reports the
gate/action behaviour inside each quadrant.  This is deliberately independent
of controller timing: timing remains valid only in the serial source runs.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _norms(actions: list[list[float]] | None) -> list[float]:
    if actions is None:
        return []
    return [sqrt(sum(float(component) ** 2 for component in action)) for action in actions]


def _quadrant(baseline_completed: bool, candidate_completed: bool) -> str:
    if baseline_completed and candidate_completed:
        return "retained"
    if not baseline_completed and candidate_completed:
        return "rescued"
    if baseline_completed and not candidate_completed:
        return "destroyed"
    return "both_failed"


def compare_payloads(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_records = {int(record["seed"]): record for record in baseline["records"]}
    candidate_records = {int(record["seed"]): record for record in candidate["records"]}
    if baseline_records.keys() != candidate_records.keys():
        raise ValueError("baseline and candidate seed sets differ")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        name: [] for name in ("retained", "rescued", "destroyed", "both_failed")
    }
    for seed in sorted(baseline_records):
        base = baseline_records[seed]
        learned = candidate_records[seed]
        grouped[_quadrant(bool(base["completed"]), bool(learned["completed"]))].append(
            (base, learned)
        )

    def group_summary(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> dict[str, Any]:
        learned = [pair[1] for pair in pairs]
        raw_norms = [
            value
            for record in learned
            for value in _norms(record.get("policy_actions_raw"))
        ]
        applied_norms = [
            value
            for record in learned
            for value in _norms(record.get("applied_actions"))
        ]
        gate_decisions = sum(int(record.get("gate_decisions", 0)) for record in learned)
        gate_deviations = sum(int(record.get("gate_deviations", 0)) for record in learned)
        return {
            "count": len(pairs),
            "seeds": [int(record["seed"]) for record in learned],
            "candidate_legal_completions": sum(
                bool(record.get("truth_geometry_zero_violation_completed"))
                for record in learned
            ),
            "gate_decisions": gate_decisions,
            "gate_deviations": gate_deviations,
            "gate_deviation_fraction": (
                gate_deviations / gate_decisions if gate_decisions else None
            ),
            "raw_action_l2": _summary(raw_norms),
            "applied_action_l2": _summary(applied_norms),
            "candidate_minimum_truth_normalized_margin": _summary(
                [
                    float(record["minimum_truth_normalized_margin"])
                    for record in learned
                    if record.get("minimum_truth_normalized_margin") is not None
                ]
            ),
        }

    common_success = grouped["retained"]

    def paired_delta(key: str) -> dict[str, float | int] | None:
        return _summary(
            [float(candidate_row[key]) - float(base_row[key]) for base_row, candidate_row in common_success]
        )

    all_candidate = [candidate_records[seed] for seed in sorted(candidate_records)]
    gate_decisions = sum(int(record.get("gate_decisions", 0)) for record in all_candidate)
    gate_deviations = sum(int(record.get("gate_deviations", 0)) for record in all_candidate)
    return {
        "episodes": len(candidate_records),
        "seed_block": min(candidate_records) if candidate_records else None,
        "candidate_completed": sum(bool(record["completed"]) for record in all_candidate),
        "candidate_legal_completed": sum(
            bool(record.get("truth_geometry_zero_violation_completed"))
            for record in all_candidate
        ),
        "quadrants": {
            name: group_summary(pairs) for name, pairs in grouped.items()
        },
        "gate": {
            "decisions": gate_decisions,
            "deviations": gate_deviations,
            "deviation_fraction": (
                gate_deviations / gate_decisions if gate_decisions else None
            ),
        },
        "common_success_candidate_minus_baseline": {
            "survival_s": paired_delta("survival_s"),
            "force_impulse_n_s": paired_delta("force_impulse_n_s"),
            "equivalent_delta_v_m_s": paired_delta("equivalent_delta_v_m_s"),
            "minimum_truth_normalized_margin": paired_delta(
                "minimum_truth_normalized_margin"
            ),
        },
        "solver": {
            "qp_infeasible_steps_total": sum(
                int(record.get("qp_infeasible_steps_total", 0))
                for record in all_candidate
            ),
            "zero_fallback_steps_total": sum(
                int(record.get("zero_fallback_steps_total", 0))
                for record in all_candidate
            ),
            "max_consecutive_zero_wrench_steps": max(
                (
                    int(record.get("max_consecutive_zero_wrench_steps", 0))
                    for record in all_candidate
                ),
                default=0,
            ),
            "maximum_successful_slack": max(
                (
                    float(record["maximum_successful_slack"])
                    for record in all_candidate
                    if record.get("maximum_successful_slack") is not None
                ),
                default=None,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidates: dict[str, Any] = {}
    for entry in args.candidate:
        label, separator, raw_path = entry.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"--candidate must be LABEL=PATH, got {entry!r}")
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        candidates[label] = compare_payloads(baseline, payload)

    result = {
        "baseline": str(args.baseline),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
