"""Same policy, same seeds: the evaluator before and after the feedback fix.

Inputs are ``evaluate_hybrid_policy`` outputs produced from one checkout, one
with the pre-fix evaluator file (commit a438c61) and one with the fixed one.
Reports, per seed, both outcomes and the first decision at which the applied
reference differs.

    python -B -m experiments.compare_evaluator_feedback_fix --before a.json b.json --after c.json d.json --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _records(paths: list[Path]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in paths:
        for record in json.loads(path.read_text())["records"]:
            out[int(record["seed"])] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True)
    parser.add_argument("--after", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = _records(args.before), _records(args.after)
    if before.keys() != after.keys():
        raise ValueError("seed sets differ")
    rows = []
    for seed in sorted(before):
        old, new = before[seed], after[seed]
        w_old = np.asarray(old["waypoints_target_frame"])
        w_new = np.asarray(new["waypoints_target_frame"])
        n = min(len(w_old), len(w_new))
        diff = np.flatnonzero(np.any(w_old[:n] != w_new[:n], axis=1))
        rows.append(
            {
                "seed": seed,
                "completed_before_fix": bool(old["completed"]),
                "completed_after_fix": bool(new["completed"]),
                "survival_s_before_fix": float(old["survival_s"]),
                "survival_s_after_fix": float(new["survival_s"]),
                "first_differing_decision": int(diff[0]) if diff.size else None,
            }
        )
    summary = {
        "episodes": len(rows),
        "completed_before_fix": sum(r["completed_before_fix"] for r in rows),
        "completed_after_fix": sum(r["completed_after_fix"] for r in rows),
        "outcome_flips": sum(r["completed_before_fix"] != r["completed_after_fix"] for r in rows),
        "trajectories_differing": sum(r["first_differing_decision"] is not None for r in rows),
        "sources": {"before": [str(p) for p in args.before], "after": [str(p) for p in args.after]},
    }
    args.output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
