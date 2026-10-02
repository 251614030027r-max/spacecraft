"""v3e 50k early read: one table from the evaluator JSONs (read-only, no decisions encoded).

Rows (all on the same 48-seed evaluation block):
    Pure MPC, smooth nominal (v3_nominal),
    v3d learned-only @50k (old reference realization), v3e learned-only @50k.

Per row: completion, zero-violation completion, episodes with a truth
violation, QP zero-fallback steps, and -- against Pure MPC -- rescued /
destroyed / both-failed plus time and delta-v differences on the common
successes. Also the pooled v3d-vs-v3e comparison at equal budget.

    python -B -m experiments.v3e_early_readout --pure-mpc P.json --nominal N.json --v3d a.json b.json c.json --v3e d.json e.json f.json --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from eval.adaptive_pairing import compare_payloads


def _row(pure: dict[str, Any], payload: dict[str, Any], label: str) -> dict[str, Any]:
    paired = compare_payloads(pure, payload)
    records = payload["records"]
    delta = paired["common_success_candidate_minus_baseline"]
    return {
        "label": label,
        "episodes": len(records),
        "completed": paired["candidate_completed"],
        "zero_violation_completed": paired["candidate_legal_completed"],
        "episodes_with_truth_violation": sum(bool(r.get("constraint_violated")) for r in records),
        "qp_zero_fallback_steps": paired["solver"]["zero_fallback_steps_total"],
        "retained": paired["quadrants"]["retained"]["count"],
        "rescued": paired["quadrants"]["rescued"]["count"],
        "destroyed": paired["quadrants"]["destroyed"]["count"],
        "both_failed": paired["quadrants"]["both_failed"]["count"],
        "rescued_seeds": paired["quadrants"]["rescued"]["seeds"],
        "common_success_time_delta_s_mean": (delta["survival_s"] or {}).get("mean"),
        "common_success_dv_delta_mean": (delta["equivalent_delta_v_m_s"] or {}).get("mean"),
        "mean_dv_completed": (
            mean(float(r["equivalent_delta_v_m_s"]) for r in records if r["completed"])
            if any(r["completed"] for r in records)
            else None
        ),
        "mean_time_completed_s": (
            mean(float(r["survival_s"]) for r in records if r["completed"])
            if any(r["completed"] for r in records)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-mpc", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--v3d", type=Path, nargs="+", default=[])
    parser.add_argument("--v3e", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pure = json.loads(args.pure_mpc.read_text())
    rows = [_row(pure, pure, "Pure MPC"), _row(pure, json.loads(args.nominal.read_text()), "v3_nominal")]
    for path in args.v3d:
        rows.append(_row(pure, json.loads(path.read_text()), f"v3d@50k {path.stem}"))
    for path in args.v3e:
        rows.append(_row(pure, json.loads(path.read_text()), f"v3e@50k {path.stem}"))
    args.output.write_text(json.dumps({"rows": rows}, indent=1))
    header = f"{'row':34s} {'done':>5s} {'0-viol':>6s} {'viol':>4s} {'qpfb':>5s} {'resc':>4s} {'destr':>5s} {'dt(s)':>7s} {'ddv':>6s}"
    print(header)
    for r in rows:
        dt = r["common_success_time_delta_s_mean"]
        ddv = r["common_success_dv_delta_mean"]
        print(
            f"{r['label'][:34]:34s} {r['completed']:>5d} {r['zero_violation_completed']:>6d} "
            f"{r['episodes_with_truth_violation']:>4d} {r['qp_zero_fallback_steps']:>5d} {r['rescued']:>4d} "
            f"{r['destroyed']:>5d} {dt if dt is None else round(dt, 1)!s:>7s} {ddv if ddv is None else round(ddv, 2)!s:>6s}"
        )


if __name__ == "__main__":
    main()
