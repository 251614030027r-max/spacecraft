"""Planned vs executed far-field delta-v of the earliest-feasible rule (no closed-loop simulation).

    python -B -m diagnostics.t1_beta.plan_gap --episodes evidence/T1_R1_pass2/episodes.jsonl.gz --output <json>

For each R1 rule episode (row rv:0:0.7): the first G* plan is re-solved from the logged
opening (reset + one planner decision, as in the logged run; tau is checked against the
trace), its pure delta-v is (dt/m) sum |f| (the planner's ``delta_v_m_s`` is its objective,
which includes the range penalty). The executed far-field delta-v is recovered from the
trace rewards of the non-descent decisions: r = -0.05*2 - 5 dv, so dv = (-r - 0.1) / 5.
Under the rule a live plan is only replaced when the state is more than 1 m off the plan
(or the plan expired), so the number of new plans counts tracking-driven re-plans.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from env.t1_env import T1Env
from env.t1_gstar import make_rendezvous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("evidence/T1_R1_pass2/episodes.jsonl.gz"))
    parser.add_argument("--row", default="rv:0:0.7")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.episodes, "rt") as f:
        rows = [r for r in (json.loads(line) for line in f) if r["row"] == args.row]
    env, out = T1Env(), []
    for r in rows:
        trace = r["trace"]
        phase_a = [d for d in trace if d["plan_status"] != "descent"]
        env.reset(r["seed"])
        g = make_rendezvous(env, 0.7)
        g.decide(env, 0.0)
        plan = g.plan
        assert plan is not None and abs(plan["tau_s"] - trace[0]["tau_effective_s"]) < 1e-6
        out.append({
            "opening": r["seed"], "event": r["event"],
            "new_plans_far_field": sum(d["plan_status"].startswith(("optimal", "recovery")) for d in phase_a),
            "first_plan_tau_s": plan["tau_s"],
            "first_plan_delta_v_m_s": float(plan["dt"] / env.chaser_parameters.mass * np.sum(np.linalg.norm(plan["f"], axis=1))),
            "first_plan_objective": plan["delta_v_m_s"],
            "executed_far_field_delta_v_m_s": float(sum((-d["reward"] - 0.1) / 5.0 for d in phase_a)),
            "episode_delta_v_m_s": r["delta_v_m_s"], "far_field_decisions": len(phase_a),
        })
    a = np.array([[o["new_plans_far_field"], o["first_plan_delta_v_m_s"], o["executed_far_field_delta_v_m_s"],
                   o["episode_delta_v_m_s"]] for o in out])
    groups = {}
    for name, lo, hi in (("1", 1, 1), ("2-3", 2, 3), (">=4", 4, 10_000)):
        m = (a[:, 0] >= lo) & (a[:, 0] <= hi)
        groups[name] = {"n": int(m.sum()), "plan_dv": float(a[m, 1].mean()), "executed_far_field_dv": float(a[m, 2].mean()),
                        "ratio_median": float(np.median(a[m, 2] / a[m, 1])), "episode_dv": float(a[m, 3].mean())}
    summary = {"episodes": len(out), "groups_by_far_field_new_plans": groups,
               "corr_new_plans_vs_excess": float(np.corrcoef(a[:, 0], a[:, 2] - a[:, 1])[0, 1]),
               "mean_excess_m_s": float(np.mean(a[:, 2] - a[:, 1]))}
    args.output.write_text(json.dumps({"summary": summary, "episodes": out}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
