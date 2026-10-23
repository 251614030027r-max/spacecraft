"""T1-O Gate 1 (discovery set): fuel weight choice and task-opportunity readout.

Runs: the 8 fixed candidates of ``experiments.t1o_screen.CANDIDATES`` on the 16 discovery
openings (``experiments.t1o_openings --set discovery``), one physical trajectory each; every
fuel weight in ``FUEL_WEIGHTS`` re-scores the same trajectories exactly from per-step delta-v.

Lexicographic rule per fuel weight (research plan 2026-10-17 sec. 4, Gate 1; the operational
constants below are the upper's pre-registration and are confirmed by the research layer
before the discovery runs):

1. exclusion: (a) a reward loophole -- some delayed candidate (F2-F4) has more timeouts than F1
   and a higher mean return; (b) the strongest fixed candidate (highest mean return over the 16)
   has < 15 / 16 captures or any unsafe episode;
2. opportunity on the *valid subset* (openings where the strongest fixed candidate captured
   safely and at least two candidates captured safely): mean_i max_{a safe} J_i(a) - max_a
   mean_i J_i(a) >= ``ORACLE_GAP``; and at least ``MIN_WINNERS`` valid openings are won by a
   delay-0 candidate (F1, F5, F7) and at least ``MIN_WINNERS`` by a delay >= 30 s candidate
   (F2, F3, F4, F6, F8), each by >= ``WIN_MARGIN`` over the best safe candidate of the other
   group;
3. another dimension: for kappa (pairs F1-F5, F3-F6) or the entry speed (F1-F7, F3-F8), at least
   ``MIN_DIM_OPENINGS`` openings where both members capture, differ in closed loop (time >= 2 s
   or delta-v >= 0.02 m/s apart) and in return by >= ``DIM_RETURN_DIFF``;
4. physical trade-off (reported for every weight; required): at least ``MIN_PARETO_OPENINGS``
   valid openings whose safe captures contain >= 2 time / delta-v non-dominated candidates.
The smallest passing weight is chosen (10 before 15 before 20); none passing is NO-GO.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.t1o_screen import CANDIDATES

FUEL_WEIGHTS = (10.0, 15.0, 20.0)
ORACLE_GAP = 3.0
MIN_WINNERS = 4
WIN_MARGIN = 0.5
MIN_DIM_OPENINGS = 4
DIM_RETURN_DIFF = 1.0
MIN_PARETO_OPENINGS = 4
DELAY0 = ("F1", "F5", "F7")
DELAYED = ("F2", "F3", "F4", "F6", "F8")
DIMENSIONS = {"kappa": (("F1", "F5"), ("F3", "F6")), "entry_speed": (("F1", "F7"), ("F3", "F8")),
              "delay": (("F1", "F2"), ("F1", "F3"), ("F1", "F4"), ("F2", "F3"), ("F3", "F4"))}


def load(root: Path, seeds: list[int]) -> dict[str, dict[int, dict]]:
    return {c: {s: json.loads((root / c / f"{s}.json").read_text()) for s in seeds} for c in CANDIDATES}


def J(row: dict, lam: float) -> float:
    return float(row["discounted_return_by_fuel_weight"][str(lam)])


def safe(row: dict) -> bool:
    return row["event"] == "success"


def distinct(a: dict, b: dict) -> bool:
    return abs(a["time_s"] - b["time_s"]) >= 2.0 or abs(a["delta_v_m_s"] - b["delta_v_m_s"]) >= 0.02


def pareto_size(rows: list[dict]) -> int:
    pts = [(r["time_s"], r["delta_v_m_s"]) for r in rows]
    return sum(not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts) for p in pts)


def evaluate(rows: dict[str, dict[int, dict]], seeds: list[int], lam: float) -> dict:
    means = {c: float(np.mean([J(rows[c][s], lam) for s in seeds])) for c in CANDIDATES}
    strong = max(means, key=means.get)
    out: dict = {"fuel_weight": lam, "mean_return": means, "strongest_fixed": strong}
    timeouts = {c: sum(rows[c][s]["event"] == "timeout" for s in seeds) for c in CANDIDATES}
    loophole = [c for c in ("F2", "F3", "F4") if timeouts[c] > timeouts["F1"] and means[c] > means["F1"]]
    s_rows = [rows[strong][s] for s in seeds]
    out["exclusion"] = {"loophole": loophole, "strong_captures": sum(map(safe, s_rows)),
                        "strong_unsafe": sum(r["event"] == "unsafe" for r in s_rows), "timeouts": timeouts}
    excluded = bool(loophole) or out["exclusion"]["strong_captures"] < 15 or out["exclusion"]["strong_unsafe"] > 0
    valid = [s for s in seeds if safe(rows[strong][s]) and sum(safe(rows[c][s]) for c in CANDIDATES) >= 2]
    out["valid_openings"] = valid
    out["excluded_openings"] = {s: {c: rows[c][s]["event"] for c in CANDIDATES} for s in seeds if s not in valid}
    best = {s: max((J(rows[c][s], lam) for c in CANDIDATES if safe(rows[c][s]))) for s in valid}
    fixed = max(float(np.mean([J(rows[c][s], lam) for s in valid])) for c in CANDIDATES) if valid else np.nan
    gap = float(np.mean(list(best.values())) - fixed) if valid else np.nan
    winners = {"delay0": 0, "delayed": 0}
    for s in valid:
        g0 = [J(rows[c][s], lam) for c in DELAY0 if safe(rows[c][s])]
        g1 = [J(rows[c][s], lam) for c in DELAYED if safe(rows[c][s])]
        if g0 and (not g1 or max(g0) >= max(g1) + WIN_MARGIN):
            winners["delay0"] += 1
        elif g1 and (not g0 or max(g1) >= max(g0) + WIN_MARGIN):
            winners["delayed"] += 1
    out["oracle_gap"], out["winners"] = gap, winners
    dims = {}
    for dim, pairs in DIMENSIONS.items():
        hit = [s for s in seeds if any(safe(rows[a][s]) and safe(rows[b][s]) and distinct(rows[a][s], rows[b][s])
                                       and abs(J(rows[a][s], lam) - J(rows[b][s], lam)) >= DIM_RETURN_DIFF
                                       for a, b in pairs)]
        dims[dim] = len(hit)
    out["dimension_openings"] = dims
    pareto = {s: pareto_size([rows[c][s] for c in CANDIDATES if safe(rows[c][s])]) for s in valid}
    out["pareto_nondominated"] = pareto
    flips = []
    for a, b in combinations(CANDIDATES, 2):
        sign = [np.sign(J(rows[a][s], lam) - J(rows[b][s], lam)) for s in valid if safe(rows[a][s]) and safe(rows[b][s])]
        if sign:
            flips.append(float(min(sum(x > 0 for x in sign), sum(x < 0 for x in sign)) / len(sign)))
    out["preference_flip_fraction_mean"] = float(np.mean(flips)) if flips else None
    checks = {
        "1_not_excluded": not excluded,
        "2_oracle_gap": bool(valid) and gap >= ORACLE_GAP,
        "2_winners": winners["delay0"] >= MIN_WINNERS and winners["delayed"] >= MIN_WINNERS,
        "3_second_dimension": max(dims["kappa"], dims["entry_speed"]) >= MIN_DIM_OPENINGS,
        "4_physical_tradeoff": sum(v >= 2 for v in pareto.values()) >= MIN_PARETO_OPENINGS,
    }
    out["checks"], out["passes"] = checks, all(checks.values())
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="directory with one sub-directory per candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    doc = json.loads(args.openings.read_text())
    rows = load(args.root, doc["seeds"])
    result = {"profile": doc["profile"], "per_weight": {str(l): evaluate(rows, doc["seeds"], l) for l in FUEL_WEIGHTS}}
    passing = [l for l in FUEL_WEIGHTS if result["per_weight"][str(l)]["passes"]]
    result["fuel_weight"] = passing[0] if passing else None
    result["verdict"] = "PASS" if passing else "NO-GO"
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"verdict": result["verdict"], "fuel_weight": result["fuel_weight"],
                      **{l: result["per_weight"][str(l)]["checks"] for l in map(str, FUEL_WEIGHTS)}}, indent=1))


if __name__ == "__main__":
    main()
