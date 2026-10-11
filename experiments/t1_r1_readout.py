"""R1 readout: pre-registered decomposition and GO / REVIEW / STOP for T1 + G* v0.3.

Written before any R1 row was run; thresholds are not to be edited after.

Rows (``experiments.t1_gstar_probe``): ``pure``; ``rv:0:K`` (Pure approach,
G* descent at pace K: the no-learning terminal fast-descent rule); G* rows
``rv:T:K`` (T > 0), ``rvgeo:K``, ``rvbest:K``.

Per opening, J is the discounted task return of the row (time, delta-v and
outcome in one scalar, gamma = 0.999 per 2 s); times are clean-capture times
(a non-capture counts as 300 s for the paired time statistics).

  descent gain   J(rv:0) - J(pure)
  route gain     J(oracle) - J(rv:0), oracle = best G* row of that opening
  rule capture   sum over openings of (J(rule) - J(rv:0)) / sum of route gains,
                 for the best single no-learning rule (highest mean J among
                 the G* rows; every G* row here is a no-learning rule)

Verdict:
  EXECUTOR   any G* row with more than 1 unsafe / numerical episode in the
             block, or more than 2 over all G* rows -> REVIEW (executor first)
  STOP       median route time gain < 5 s, or route gain J > 0 on < 16 openings
  GO         median route time gain >= 10 s, route gain J > 0 on >= 24
             openings, and rule capture < 0.8
  REVIEW     everything else (e.g. a single rule captures >= 80 % of the
             route gain, or the descent gain dominates)

    python -B -m experiments.t1_r1_readout --input-dir eval/t1/r1 --openings eval/t1/r1/openings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HORIZON_S = 300.0


def load(input_dir: Path, seeds: list[int]) -> dict[str, dict[int, dict]]:
    rows: dict[str, dict[int, dict]] = {}
    for path in sorted(input_dir.glob("*__*.json")):
        tag, seed = path.stem.rsplit("__", 1)
        if int(seed) in seeds:
            rows.setdefault(tag, {})[int(seed)] = json.loads(path.read_text())
    return rows


def capture_time(result: dict) -> float:
    return float(result["time_s"]) if result["event"] == "success" else HORIZON_S


def readout(rows: dict[str, dict[int, dict]], seeds: list[int], kappa: str = "0.7") -> dict:
    fast_tag = f"rv_0_{kappa}"
    gstar = sorted(t for t in rows if t.startswith("rv") and t != fast_tag)
    missing = {t: sorted(set(seeds) - set(rows[t])) for t in ["pure", fast_tag, *gstar] if t in rows}
    complete = [s for s in seeds if all(s in rows.get(t, {}) for t in ["pure", fast_tag, *gstar])]
    per_row = {}
    for tag, results in rows.items():
        r = [results[s] for s in complete if s in results]
        events = [x["event"] for x in r]
        success = [x for x in r if x["event"] == "success"]
        per_row[tag] = {
            "episodes": len(r),
            "clean_capture": sum(e == "success" for e in events),
            "unsafe": sum(e == "unsafe" for e in events),
            "timeout": sum(e == "timeout" for e in events),
            "numerical_failure": sum(e == "numerical_failure" for e in events),
            "violations": [x.get("violations") for x in r if x["event"] == "unsafe"],
            "median_time_s_success": float(np.median([x["time_s"] for x in success])) if success else None,
            "median_delta_v_success": float(np.median([x["delta_v_m_s"] for x in success])) if success else None,
            "mean_J": float(np.mean([x["discounted_return"] for x in r])) if r else None,
            "qp_zero_fallback_steps": int(sum(x["qp_zero_fallback_steps"] for x in r)),
        }
    J = {t: np.array([rows[t][s]["discounted_return"] for s in complete]) for t in ["pure", fast_tag, *gstar]}
    T = {t: np.array([capture_time(rows[t][s]) for s in complete]) for t in ["pure", fast_tag, *gstar]}
    G = np.stack([J[t] for t in gstar])
    oracle_index = np.argmax(G, axis=0)
    J_oracle = G[oracle_index, np.arange(len(complete))]
    T_oracle = np.stack([T[t] for t in gstar])[oracle_index, np.arange(len(complete))]
    route_J = J_oracle - J[fast_tag]
    route_T = T[fast_tag] - T_oracle
    descent_J = J[fast_tag] - J["pure"]
    descent_T = T["pure"] - T[fast_tag]
    best_rule = max(gstar, key=lambda t: float(np.mean(J[t])))
    total_route = float(np.sum(np.maximum(route_J, 0.0)))
    capture = float(np.sum(J[best_rule] - J[fast_tag]) / total_route) if total_route > 0 else float("nan")
    gstar_bad = {t: per_row[t]["unsafe"] + per_row[t]["numerical_failure"] for t in gstar}
    if max(gstar_bad.values(), default=0) > 1 or sum(gstar_bad.values()) > 2:
        verdict = "REVIEW_EXECUTOR"
    elif float(np.median(route_T)) < 5.0 or int(np.sum(route_J > 0)) < 16:
        verdict = "STOP"
    elif float(np.median(route_T)) >= 10.0 and int(np.sum(route_J > 0)) >= 24 and capture < 0.8:
        verdict = "GO"
    else:
        verdict = "REVIEW"
    return {
        "openings": len(complete),
        "missing": {t: m for t, m in missing.items() if m},
        "rows": per_row,
        "descent_gain": {"median_J": float(np.median(descent_J)), "median_time_s": float(np.median(descent_T)),
                          "positive_J": int(np.sum(descent_J > 0))},
        "route_gain": {"median_J": float(np.median(route_J)), "median_time_s": float(np.median(route_T)),
                        "positive_J": int(np.sum(route_J > 0))},
        "pure_vs_oracle_median_time_s": float(np.median(T["pure"] - T_oracle)),
        "oracle_row_counts": {t: int(np.sum(oracle_index == i)) for i, t in enumerate(gstar)},
        "best_single_rule": best_rule,
        "best_rule_capture_of_route_gain": capture,
        "gstar_unsafe_or_numerical": gstar_bad,
        "verdict": verdict,
        "per_opening": [
            {"seed": s, "oracle_row": gstar[int(oracle_index[i])], "route_J": float(route_J[i]),
             "route_time_s": float(route_T[i]), "descent_time_s": float(descent_T[i])}
            for i, s in enumerate(complete)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--kappa", default="0.7")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = json.loads(args.openings.read_text())["seeds"]
    result = readout(load(args.input_dir, seeds), seeds, args.kappa)
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text)
    summary = {k: v for k, v in result.items() if k not in ("per_opening", "rows")}
    print(json.dumps(summary, indent=1))
    for tag, row in result["rows"].items():
        print(tag, {k: v for k, v in row.items() if k != "violations"})


if __name__ == "__main__":
    main()
