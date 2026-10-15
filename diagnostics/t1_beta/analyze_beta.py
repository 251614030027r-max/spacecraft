"""Pre-registered read-out of B1/B2 (reference-shaping beta). Upper analysis only.

    python -B -m diagnostics.t1_beta.analyze_beta --b1 <dir> --b2 <dir> --output <json>

B1 (8 openings x beta in {0, 0.3, 1, 3}):
  safety       a beta is vetoed if, on any opening and relative to beta = 0 there, it adds a task
               failure (unsafe or timeout), a numerical failure, a decision ending with no usable
               plan (G* hold) or a solver exception. Infeasible candidates inside G*'s normal
               search (earliest grid, slips) are reported, not vetoed. Margins are reported as
               absolute minima and differences (no percentage when the base is near 0).
  effect       per opening: the accepted first plans of beta and beta = 0 differ by more than
               0.1 m somewhere on their common 1 s time grid AND the episode delta-v differs by
               more than 0.02 m/s; a beta "acts" if this holds on >= 4 of 8 openings.
  frozen here  the best fixed beta (highest B1 mean return among the four, beta = 0 included)
               and one opening-state rule beta = (lo if x_f <= thr else hi), f one of
               STATE_FEATURES of the initial observation, thr a midpoint between consecutive
               B1 values, (lo, hi) an ordered pair of distinct betas; chosen by the highest B1
               mean return, ties to the larger distance of the B1 points from thr, then to the
               earlier feature. If no rule beats the best fixed beta on B1, the rule is the best
               fixed beta.
B2 (the other 8 openings, all four betas): no fitting. Rows: beta = 0, the B1 best fixed beta,
  the B1 rule (each opening's return taken from its fixed-beta episode: one choice at the
  opening, beta fixed for the episode -- not a dynamic SAC choice), and the per-opening
  oracle (observation only, an upper bound with selection bias).
Recommendation threshold: rule - best fixed >= 1.0 mean return on B2, and the rule adds no task
failure, numerical failure, final planning failure or solver exception relative to either the
best fixed beta or beta = 0 on any opening; every truth margin is reported per row with its
change against both.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

BETAS = (0.0, 0.3, 1.0, 3.0)
STATE_FEATURES = {0: "range_10m", 5: "rho_perp_10m", 6: "port_los_180deg", 7: "port_latitude_90deg",
                  8: "omega_2p7deg_s", 9: "fov_margin_50deg"}
REF_DIFF_M, DV_DIFF_M_S, ACTS_MIN = 0.1, 0.02, 4
GAIN_MIN = 1.0


def load(directory: Path) -> dict[tuple[float, int], dict]:
    runs = {}
    for f in sorted(directory.glob("beta*_*.json")):
        r = json.loads(f.read_text())
        runs[(float(r["beta"]), int(r["opening"]))] = r
    return runs


def _first_plan_difference(a: dict, b: dict) -> float:
    pa, pb = a["accepted_plans"][0]["position_on_grid_m"], b["accepted_plans"][0]["position_on_grid_m"]
    n = min(len(pa), len(pb))
    return float(np.max(np.linalg.norm(np.asarray(pa[:n]) - np.asarray(pb[:n]), axis=1)))


def _failed(r: dict) -> bool:
    return r["event"] != "success"


def _added_hard(r: dict, base: dict) -> bool:
    """Numerical failure, a decision with no usable plan, or a solver exception, beyond base."""

    return bool((r["numerical_failure"] and not base["numerical_failure"])
                or r["final_plan_failures"] > base["final_plan_failures"]
                or r["solver_exceptions"] > base["solver_exceptions"])


def _margins(per: dict) -> dict:
    keys = sorted({k for r in per.values() for k in r["min_margins"]})
    return {k: {"min": float(min(r["min_margins"][k] for r in per.values())),
                "mean": float(np.mean([r["min_margins"][k] for r in per.values()]))} for k in keys}


def b1_checks(runs: dict, openings: list[int]) -> dict:
    out = {}
    for beta in BETAS[1:]:
        per = {}
        for o in openings:
            r, base = runs[(beta, o)], runs[(0.0, o)]
            per[o] = {
                "reference_max_diff_m": _first_plan_difference(r, base),
                "first_tau_diff_s": r["accepted_plans"][0]["tau_s"] - base["accepted_plans"][0]["tau_s"],
                "delta_v_diff_m_s": r["delta_v_m_s"] - base["delta_v_m_s"],
                "return_diff": r["discounted_return"] - base["discounted_return"],
                "deviation_replans": (r["new_plan_triggers"]["deviation"], base["new_plan_triggers"]["deviation"]),
                "far_field_delta_v": (r["far_field_delta_v_m_s"], base["far_field_delta_v_m_s"]),
                "added_failure": _failed(r) and not _failed(base),
                "added_hard_failure": _added_hard(r, base),
                "candidate_infeasible": (r["candidate_infeasible"], base["candidate_infeasible"]),
                "min_margins": {k: (r["min_margins"].get(k), base["min_margins"].get(k)) for k in base["min_margins"]},
            }
        acts = sum(p["reference_max_diff_m"] > REF_DIFF_M and abs(p["delta_v_diff_m_s"]) > DV_DIFF_M_S for p in per.values())
        safe = not any(p["added_failure"] or p["added_hard_failure"] for p in per.values())
        out[beta] = {"per_opening": per, "acts_on": acts, "acts": acts >= ACTS_MIN, "safe": safe}
    return out


def mean_return(runs: dict, beta: float, openings: list[int]) -> float:
    return float(np.mean([runs[(beta, o)]["discounted_return"] for o in openings]))


def fit_rule(runs: dict, openings: list[int], allowed: list[float]) -> dict:
    best_fixed = max(allowed, key=lambda b: (mean_return(runs, b, openings), -b))
    best = {"kind": "fixed", "beta": best_fixed, "b1_mean_return": mean_return(runs, best_fixed, openings),
            "spacing": np.inf, "feature_order": -1}
    for order, (index, name) in enumerate(STATE_FEATURES.items()):
        values = sorted({runs[(0.0, o)]["initial_observation"][index] for o in openings})
        for thr in [(a + b) / 2.0 for a, b in zip(values[:-1], values[1:])]:
            spacing = min(abs(v - thr) for v in values)
            for lo, hi in itertools.permutations(allowed, 2):
                ret = float(np.mean([runs[(lo if runs[(0.0, o)]["initial_observation"][index] <= thr else hi, o)]
                                     ["discounted_return"] for o in openings]))
                key = (ret, spacing, -order)
                if best["kind"] == "fixed":
                    better = ret > best["b1_mean_return"]
                else:
                    better = key > (best["b1_mean_return"], best["spacing"], -best["feature_order"])
                if better:
                    best = {"kind": "rule", "feature_index": index, "feature": name, "threshold": thr,
                            "beta_low": lo, "beta_high": hi, "b1_mean_return": ret, "spacing": spacing,
                            "feature_order": order}
    return {"best_fixed_beta": best_fixed, "rule": best}


def apply_rule(rule: dict, runs: dict, o: int) -> float:
    if rule["kind"] == "fixed":
        return rule["beta"]
    x = runs[(0.0, o)]["initial_observation"][rule["feature_index"]]
    return rule["beta_low"] if x <= rule["threshold"] else rule["beta_high"]


def b2_validate(runs: dict, openings: list[int], frozen: dict) -> dict:
    rule = frozen["rule"]
    rows = {
        "beta_0": {o: runs[(0.0, o)] for o in openings},
        "best_fixed": {o: runs[(frozen["best_fixed_beta"], o)] for o in openings},
        "state_rule": {o: runs[(apply_rule(rule, runs, o), o)] for o in openings},
        "oracle_observation_only": {o: max((runs[(b, o)] for b in BETAS), key=lambda r: r["discounted_return"])
                                    for o in openings},
    }
    table = {name: {"mean_return": float(np.mean([r["discounted_return"] for r in per.values()])),
                    "success": sum(r["event"] == "success" for r in per.values()),
                    "mean_time_s": float(np.mean([r["time_s"] for r in per.values()])),
                    "mean_delta_v_m_s": float(np.mean([r["delta_v_m_s"] for r in per.values()])),
                    "beta_used": {o: r["beta"] for o, r in per.items()}} for name, per in rows.items()}
    for name, per in rows.items():
        table[name]["margins"] = _margins(per)
    safety = {}
    for ref in ("best_fixed", "beta_0"):
        safety[ref] = {
            "added_task_failure": [o for o in openings if _failed(rows["state_rule"][o]) and not _failed(rows[ref][o])],
            "added_hard_failure": [o for o in openings if _added_hard(rows["state_rule"][o], rows[ref][o])],
            "margin_change": {k: {o: rows["state_rule"][o]["min_margins"][k] - rows[ref][o]["min_margins"][k]
                                  for o in openings} for k in rows[ref][openings[0]]["min_margins"]},
        }
    gain = table["state_rule"]["mean_return"] - table["best_fixed"]["mean_return"]
    added = any(safety[ref]["added_task_failure"] or safety[ref]["added_hard_failure"] for ref in safety)
    return {"table": table, "rule_minus_best_fixed": gain, "rule_minus_beta_0":
            table["state_rule"]["mean_return"] - table["beta_0"]["mean_return"],
            "rule_safety": safety, "rule_added_failure": added,
            "recommend_sac_beta_design": bool(gain >= GAIN_MIN and not added)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b1 = load(args.b1)
    b1_openings = sorted({o for _, o in b1})
    checks = b1_checks(b1, b1_openings)
    allowed = [0.0] + [b for b in BETAS[1:] if checks[b]["safe"]]
    result = {"b1_openings": b1_openings, "b1_checks": checks,
              "b1_pass": any(c["safe"] and c["acts"] for c in checks.values()),
              "b1_mean_return": {b: mean_return(b1, b, b1_openings) for b in BETAS}}
    result["frozen_on_b1"] = fit_rule(b1, b1_openings, allowed)
    if args.b2 is not None:
        b2 = load(args.b2)
        result["b2"] = b2_validate(b2, sorted({o for _, o in b2}), result["frozen_on_b1"])
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ("b1_pass", "b1_mean_return", "frozen_on_b1")}, indent=1, default=float))


if __name__ == "__main__":
    main()
