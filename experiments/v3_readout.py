"""M5 readout: the preregistered two-layer judgement, per model (never pooled).

Inputs are evaluator JSONs on the same 48-seed block, all produced on the same
commit and machine:

    --pure-mpc      Pure MPC row (``--control desired_pose``)
    --model LABEL=LEARNED_ONLY.json,ARBITRATED.json   (once per training seed)

Per model (docs/V3_METHOD_EXECUTION_ORDER_20260924.md section 6, with the
Pure MPC count taken from ``--pure-mpc`` instead of a hard-coded 36):

    layer 1, baseline protection (arbitrated row):
        completed >= Pure MPC completed
        destroyed <= 2
        episodes with a truth violation <= Pure MPC's
        mean learned-branch share >= 5%   (else "coupling inactive")
    layer 2, net improvement:  rescued > destroyed

    >= 2 models pass both layers        -> method holds
    else >= 2 models pass layer 1       -> baseline protected, no net gain
    otherwise                           -> does not hold

The learned-only row is the plain hierarchical SAC->MPC reference; it has no
pass condition and is reported alongside.

    python -B -m experiments.v3_readout --pure-mpc pure.json --model 262420=lo.json,arb.json --model ... --output readout.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from eval.adaptive_pairing import compare_payloads

DESTROYED_LIMIT = 2
LEARNED_SHARE_FLOOR = 0.05


def _violations(payload: dict[str, Any]) -> int:
    return sum(bool(record.get("constraint_violated")) for record in payload["records"])


def _row(pure: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    paired = compare_payloads(pure, candidate)
    records = candidate["records"]
    shares = [float(r.get("learned_branch_share", 0.0)) for r in records]
    quadrants = {name: paired["quadrants"][name]["count"] for name in paired["quadrants"]}
    return {
        "completed": paired["candidate_completed"],
        "zero_violation_completed": paired["candidate_legal_completed"],
        **quadrants,
        "rescued_seeds": paired["quadrants"]["rescued"]["seeds"],
        "destroyed_seeds": paired["quadrants"]["destroyed"]["seeds"],
        "episodes_with_truth_violation": _violations(candidate),
        "mean_learned_branch_share": mean(shares) if shares else 0.0,
        "episodes_started_on_learned": sum(bool(r.get("started_on_learned")) for r in records),
        "handbacks": sum(r.get("handback_decision") is not None for r in records),
        "mean_completion_time_s": (
            mean(r["survival_s"] for r in records if r["completed"])
            if any(r["completed"] for r in records)
            else None
        ),
        "mean_equivalent_delta_v_m_s": mean(float(r["equivalent_delta_v_m_s"]) for r in records),
        "common_success_candidate_minus_pure": paired["common_success_candidate_minus_baseline"],
        "solver": paired["solver"],
    }


def judge(pure_completed: int, pure_violations: int, arbitrated: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "completed_at_least_pure_mpc": arbitrated["completed"] >= pure_completed,
        "destroyed_at_most_2": arbitrated["destroyed"] <= DESTROYED_LIMIT,
        "violations_not_above_pure_mpc": arbitrated["episodes_with_truth_violation"] <= pure_violations,
        "learned_share_at_least_5pct": arbitrated["mean_learned_branch_share"] >= LEARNED_SHARE_FLOOR,
    }
    layer1 = all(checks.values())
    layer2 = arbitrated["rescued"] > arbitrated["destroyed"]
    return {
        "layer1_checks": checks,
        "layer1_baseline_protection": layer1,
        "layer2_net_improvement": layer2,
        "coupling_inactive": not checks["learned_share_at_least_5pct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-mpc", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="LABEL=learned_only.json,arbitrated.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pure = json.loads(args.pure_mpc.read_text())
    pure_completed = sum(bool(r["completed"]) for r in pure["records"])
    pure_violations = _violations(pure)
    episodes = len(pure["records"])
    models = {}
    for item in args.model:
        label, paths = item.split("=", 1)
        learned_only_path, arbitrated_path = (Path(p) for p in paths.split(","))
        learned_only = json.loads(learned_only_path.read_text())
        arbitrated = json.loads(arbitrated_path.read_text())
        if arbitrated.get("arbiter") != "one_way" or learned_only.get("arbiter", "off") != "off":
            raise ValueError(f"{label}: expected learned-only (arbiter off) then arbitrated (one_way)")
        row_arbitrated = _row(pure, arbitrated)
        models[label] = {
            "learned_only": _row(pure, learned_only),
            "arbitrated": row_arbitrated,
            "judgement": judge(pure_completed, pure_violations, row_arbitrated),
        }
    both = sum(m["judgement"]["layer1_baseline_protection"] and m["judgement"]["layer2_net_improvement"] for m in models.values())
    first_only = sum(m["judgement"]["layer1_baseline_protection"] and not m["judgement"]["layer2_net_improvement"] for m in models.values())
    verdict = (
        "method holds: baseline protected with net improvement"
        if both >= 2
        else "baseline protected, no net improvement"
        if both + first_only >= 2
        else "does not hold"
    )
    result = {
        "pure_mpc": {"file": str(args.pure_mpc), "completed": pure_completed, "episodes_with_truth_violation": pure_violations},
        "models": models,
        "models_passing_both_layers": both,
        "models_passing_layer1_only": first_only,
        "verdict": verdict,
    }
    args.output.write_text(json.dumps(result, indent=1))
    for label, m in models.items():
        a, lo = m["arbitrated"], m["learned_only"]
        print(
            f"{label}: learned-only {lo['completed']}/{episodes} (resc {lo['rescued']} destr {lo['destroyed']}) | "
            f"arbitrated {a['completed']}/{episodes} (resc {a['rescued']} destr {a['destroyed']}, share {a['mean_learned_branch_share']:.2f}) "
            f"L1={m['judgement']['layer1_baseline_protection']} L2={m['judgement']['layer2_net_improvement']}"
        )
    print(f"Pure MPC {pure_completed}/{episodes} -> {verdict}")


if __name__ == "__main__":
    main()
