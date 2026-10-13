"""Tables of the root-cause diagnosis from the raw outputs of interventions.py and critic_probe.py.

    python -B -m diagnostics.t1_rootcause.analyze --interventions <dir> --dev-eval <run>/dev_eval \
        --rule <rule_local dir> --probe <dir> --output <readout.json>

Mode A of a checkpoint is its development evaluation (same machine, deterministic); mode E
is the rule:earliest evaluation. Spot-check A files written by interventions.py, when
present, are compared with the development evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MARKS = {"0k": 0, "5k": 5000, "10k": 10000, "15k": 15000, "20k": 20000, "25k": 25000, "30k": 30000}


def _row(event, ret, time_s, dv, new_plans=None, requests=None):
    return {"event": event, "return": ret, "time_fail_as_300": time_s if event == "success" else 300.0,
            "delta_v": dv, "new_plans": new_plans, "requests": requests}


def p1(interventions: Path, dev_eval: Path, rule: Path) -> dict:
    runs: dict = defaultdict(dict)
    spot = []
    for f in sorted(interventions.glob("*.json")):
        r = json.loads(f.read_text())
        row = _row(r["event"], r["discounted_return_from_k"], r["time_s"], r["delta_v_m_s"], r["new_plans"], r["requests"])
        if r["mode"] == "A":
            d = json.loads((dev_eval / str(MARKS[r["label"]]) / f"{r['opening']}.json").read_text())
            spot.append({"label": r["label"], "opening": r["opening"], "event": (r["event"], d["event"]),
                         "return": (r["discounted_return_from_k"], d["discounted_return"]),
                         "identical": r["event"] == d["event"] and r["discounted_return_from_k"] == d["discounted_return"]})
            continue
        runs[(r["label"], r["mode"])][r["opening"]] = row
    openings = sorted({o for v in runs.values() for o in v})
    for label in sorted({k[0] for k in runs}):
        for o in openings:
            d = json.loads((dev_eval / str(MARKS[label]) / f"{o}.json").read_text())
            runs[(label, "A")][o] = _row(d["event"], d["discounted_return"], d["time_s"], d["delta_v_m_s"], d["new_plans"])
    for o in openings:
        d = json.loads((rule / f"{o}.json").read_text())
        runs[("rule", "E")][o] = _row(d["event"], d["discounted_return"], d["time_s"], d["delta_v_m_s"], d["new_plans"])
    table = {}
    for (label, mode), per in sorted(runs.items()):
        rows = [per[o] for o in openings if o in per]
        table[f"{label}_{mode}"] = {
            "n": len(rows), "success": sum(r["event"] == "success" for r in rows),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_time_fail_as_300": float(np.mean([r["time_fail_as_300"] for r in rows])),
            "mean_delta_v": float(np.mean([r["delta_v"] for r in rows])),
            "new_plans": sum((r["new_plans"] or 0) for r in rows),
            "per_opening": {o: (per[o]["event"], round(per[o]["return"], 2)) for o in openings if o in per},
        }
    return {"openings": openings, "table": table, "spot_check_A": spot}


def p2(probe: Path) -> dict:
    out = []
    for f in sorted(probe.glob("*.json")):
        r = json.loads(f.read_text())
        acts = ("keep", "earlier15", "later15")
        realized = {}
        for follow in ("rule", "traj_det", "traj_stoch"):
            key = next((k for k in r["rollouts"]["keep"] if k.startswith(follow)), None)
            if key is None:
                continue
            realized[follow] = {a: float(np.mean([x["discounted_return_from_k"] for x in r["rollouts"][a][key]]))
                                for a in acts}
            realized[follow + "_events"] = {a: [x["event"] for x in r["rollouts"][a][key]] for a in acts}
        critics = {}
        for label, c in r["critics"].items():
            q = {a: c["actions"][a]["q_min"] for a in acts}
            grid = np.array(c["q_min_on_a0_grid"])
            a0 = np.array(c["a0_grid"])
            band = np.abs(a0) <= 0.5
            critics[label] = {
                "q_min": q, "best": max(q, key=q.get), "q1_q2_gap": {a: c["actions"][a]["q1"] - c["actions"][a]["q2"] for a in acts},
                "own_a0": c["own_action"][0], "dq_da0_at_own": c["dq_da_at_own_action"][0],
                "q_range_inside_band": float(grid[band].max() - grid[band].min()),
                "q_range_outside_band": float(grid[~band].max() - grid[~band].min()),
                "argmax_a0_on_grid": float(a0[int(np.argmax(grid))]),
                "m1_y_cf": {a: r["m1_targets"][label][a]["y_cf"] for a in acts},
                "m1_r_hat": {a: r["m1_targets"][label][a]["r_hat"] for a in acts},
            }
        best_true = {k: max(v, key=v.get) for k, v in realized.items() if not k.endswith("_events")}
        out.append({"state": f.stem, "traj": r["traj"], "opening": r["opening"], "k": r["k"], "why": r["why"],
                    "replay_match": r["replay_check"]["match"], "time_s": r["time_s"], "remaining_s": r["remaining_s"],
                    "traj_action": r["traj_action"], "realized": realized, "best_realized": best_true,
                    "critics": critics, "verify_branch": r.get("verify_branch")})
    return {"states": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interventions", type=Path)
    parser.add_argument("--dev-eval", type=Path)
    parser.add_argument("--rule", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    if args.interventions:
        result["p1"] = p1(args.interventions, args.dev_eval, args.rule)
    if args.probe:
        result["p2"] = p2(args.probe)
    args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
