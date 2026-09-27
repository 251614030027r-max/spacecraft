"""Readout for docs/V3_HEADROOM_PREREGISTRATION_20260927.md.

Inputs: the Pure MPC evaluator outputs on the development block and the
per-episode JSON of ``experiments.probe_v3_staging_headroom`` (``fix_*`` on the
reference-path-fixed interface, ``old_*`` on the v3d interface).

    python -B -m experiments.analyze_v3_staging_headroom --pure DIR --grid DIR --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESCUE_BANDS = ((0.45, "A: ample headroom -> improve the method (+ sub-innovation)"),
                (0.20, "medium headroom -> keep the method; sub-innovation targets finding the rescuable states"),
                (0.0, "B: headroom too small -> physically motivated harder setting"))
FUEL_SAVING = 0.10
FUEL_BAND = 0.50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pure = {}
    for path in sorted(args.pure.glob("pure_*.json")):
        for record in json.loads(path.read_text())["records"]:
            pure[int(record["seed"])] = record
    runs: dict[str, dict[int, dict[str, dict]]] = {"fix": {}, "old": {}}
    for path in sorted(args.grid.glob("*.json")):
        interface = path.name.split("_", 1)[0]
        row = json.loads(path.read_text())
        runs[interface].setdefault(int(row["seed"]), {})[row["variant"]] = row
    fix = runs["fix"]
    failures = sorted(s for s in fix if not pure[s]["completed"])
    successes = sorted(s for s in fix if pure[s]["completed"])

    rescued = {
        s: sorted(v for v, r in fix[s].items() if r["zero_violation_completed"]) for s in failures
    }
    rate = sum(bool(v) for v in rescued.values()) / len(failures) if failures else 0.0
    band = next(label for threshold, label in RESCUE_BANDS if rate >= threshold)

    fuel = {}
    for s in successes:
        base = float(pure[s]["equivalent_delta_v_m_s"])
        options = [
            (r["equivalent_delta_v_m_s"], v, r["survival_s"])
            for v, r in fix[s].items()
            if r["zero_violation_completed"]
        ]
        best = min(options) if options else None
        fuel[s] = {
            "pure_mpc_dv": base,
            "pure_mpc_time_s": float(pure[s]["survival_s"]),
            "best_dv": best[0] if best else None,
            "best_variant": best[1] if best else None,
            "best_time_s": best[2] if best else None,
            "saving_fraction": (1.0 - best[0] / base) if best else None,
        }
    saving_share = (
        sum(1 for f in fuel.values() if f["saving_fraction"] is not None and f["saving_fraction"] >= FUEL_SAVING)
        / len(fuel)
        if fuel
        else 0.0
    )
    per_variant = {}
    variants = sorted({v for s in fix for v in fix[s]})
    for v in variants:
        rows = [fix[s][v] for s in fix if v in fix[s]]
        per_variant[v] = {
            "episodes": len(rows),
            "zero_violation_completed_on_failures": sum(fix[s][v]["zero_violation_completed"] for s in failures if v in fix[s]),
            "zero_violation_completed_on_successes": sum(fix[s][v]["zero_violation_completed"] for s in successes if v in fix[s]),
            "qp_zero_fallbacks": sum(r["qp_zero_fallbacks"] for r in rows),
        }
    direct = [
        {
            "seed": s,
            "time_delta_s": fix[s]["direct"]["survival_s"] - pure[s]["survival_s"],
            "dv_delta": fix[s]["direct"]["equivalent_delta_v_m_s"] - pure[s]["equivalent_delta_v_m_s"],
            "completed": fix[s]["direct"]["zero_violation_completed"],
        }
        for s in successes
        if "direct" in fix[s]
    ]
    old_vs_fix = [
        {
            "seed": s,
            "variant": v,
            "dv_old_interface": runs["old"][s][v]["equivalent_delta_v_m_s"],
            "dv_fixed_interface": fix[s][v]["equivalent_delta_v_m_s"],
            "completed_old": runs["old"][s][v]["zero_violation_completed"],
            "completed_fixed": fix[s][v]["zero_violation_completed"],
        }
        for s in sorted(runs["old"])
        for v in runs["old"][s]
        if v in fix.get(s, {})
    ]
    result = {
        "pure_mpc_completed": sum(bool(r["completed"]) for r in pure.values()),
        "pure_mpc_episodes": len(pure),
        "failures": failures,
        "rescued_by_variant": rescued,
        "rescue_rate": rate,
        "rescue_reading": band,
        "fuel": fuel,
        "fuel_saving_share": saving_share,
        "fuel_reading": "fuel story viable" if saving_share >= FUEL_BAND else "fuel is not a main available gain",
        "per_variant": per_variant,
        "direct_vs_pure_mpc_on_successes": direct,
        "interface_fix_effect": old_vs_fix,
    }
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("pure_mpc_completed", "rescue_rate", "rescue_reading", "fuel_saving_share", "fuel_reading")}, indent=1))
    print(json.dumps(rescued, indent=1))


if __name__ == "__main__":
    main()
