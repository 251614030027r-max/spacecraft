"""Summarise the two pre-v3e probes into one artifact.

- near-field floor on/off (experiments/probe_v3_nearfield_floor.py): QP zero
  fallbacks under a scripted "advance without committing" policy;
- target-phase sensitivity (experiments/probe_v3_phase_sensitivity.py): the same
  base start flown by Pure MPC and by the smooth nominal at 8 initial phases.

    python -B -m experiments.summarize_v3e_probes --floor DIR --phase DIR --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=Path, required=True)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    floor_rows = [json.loads(p.read_text()) for p in sorted(args.floor.glob("*.json"))]
    floor = {}
    for row in floor_rows:
        key = f"commit={row['commit']} floor={row['floor']}"
        entry = floor.setdefault(key, {"episodes": 0, "control_steps": 0, "fallbacks": 0, "episodes_with_fallback": 0})
        entry["episodes"] += 1
        entry["control_steps"] += 20 * row["decisions"]
        entry["fallbacks"] += row["fallbacks"]
        entry["episodes_with_fallback"] += int(row["fallbacks"] > 0)
    for entry in floor.values():
        entry["rate"] = entry["fallbacks"] / entry["control_steps"]
    phase_rows = [json.loads(p.read_text()) for p in sorted(args.phase.glob("*.json"))]
    phase: dict[str, dict] = {}
    for row in phase_rows:
        seed = phase.setdefault(str(row["seed"]), {})
        seed.setdefault(row["controller"], []).append(
            {k: row[k] for k in ("phase_deg", "zero_violation_completed", "survival_s", "equivalent_delta_v_m_s", "qp_zero_fallbacks", "failure")}
        )
    summary = {}
    for seed, controllers in phase.items():
        for controller, rows in controllers.items():
            rows.sort(key=lambda r: r["phase_deg"])
            done = [r for r in rows if r["zero_violation_completed"]]
            summary[f"{seed}/{controller}"] = {
                "completed_phases": f"{len(done)}/{len(rows)}",
                "dv_min_max_completed": [min(r["equivalent_delta_v_m_s"] for r in done), max(r["equivalent_delta_v_m_s"] for r in done)] if done else None,
                "time_min_max_completed": [min(r["survival_s"] for r in done), max(r["survival_s"] for r in done)] if done else None,
            }
        pure = {r["phase_deg"]: r for r in controllers.get("pure", [])}
        nominal = {r["phase_deg"]: r for r in controllers.get("direct", [])}
        summary[f"{seed}/which_completes"] = {
            str(int(p)): ("both" if pure[p]["zero_violation_completed"] and nominal[p]["zero_violation_completed"]
                         else "pure_only" if pure[p]["zero_violation_completed"]
                         else "nominal_only" if nominal[p]["zero_violation_completed"] else "neither")
            for p in sorted(set(pure) & set(nominal))
        }
    result = {"nearfield_floor": floor, "phase_summary": summary, "phase_rows": phase}
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({"nearfield_floor": floor, "phase_summary": summary}, indent=1))


if __name__ == "__main__":
    main()
