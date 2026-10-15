"""Read-out of the T1-S1 screen (pre-registered; no fitting on the confirmation openings).

Options: E (earliest, kappa 0.7), T70 / T90 (first request 70 s / 90 s, kappa 0.7),
K04 / K10 (first request 70 s -- the tau baseline frozen on the screen openings -- with
kappa 0.4 / 1.0). On the screen openings (296000-296003): the best uniform option (highest
mean return) and one opening-state rule (one feature of the initial observation, one
threshold at a midpoint between screen values, two options) are frozen; on the
confirmation openings (296004-296007) both are applied unchanged. The per-opening oracle
is an exploration bound only.

Verdict (one of three):
  not_executable    the baseline (E) succeeds on fewer than 6 of 8 openings
  near_equivalent   on fewer than 2 openings does any option beat the opening's
                    second-best option by more than 1.0 return point (openings that every
                    option fails are excluded: a margin among failures is not a decision value)
  decision_value    otherwise; reported with whether the frozen rule beats the frozen best
                    uniform option on the confirmation openings by >= 1.0 with no added failure
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

SCREEN = [296000, 296001, 296002, 296003]
CONFIRM = [296004, 296005, 296006, 296007]
OPTIONS = {"E": ("earliest", 0.7), "T70": ("70", 0.7), "T90": ("90", 0.7), "K04": ("70", 0.4), "K10": ("70", 1.0)}
FEATURES = {0: "range_10m", 5: "rho_perp_10m", 6: "port_los_180deg", 7: "port_latitude_90deg",
            8: "omega_2p7deg_s", 9: "fov_margin_50deg"}


def load(directory: Path) -> dict:
    runs = {}
    for name, (tau, kappa) in OPTIONS.items():
        for o in SCREEN + CONFIRM:
            f = directory / f"S1_tau{tau}_k{kappa}_{o}.json"
            if f.exists():
                runs[(name, o)] = json.loads(f.read_text())
    return runs


def truncate(r: dict, deadline_s: float, gamma: float = 0.999) -> dict:
    """The same episode under a shorter deadline, exactly: fixed policies and G* do not see the
    deadline, so the trajectory is identical until it; the env checks success before timeout
    inside each control step, and the deadline is a decision boundary (2 s grid)."""

    if r["time_s"] <= deadline_s + 1e-9:
        return r
    log = [d for d in r["decision_log"] if d["t_s"] < deadline_s - 1e-9]
    ret = sum(gamma**k * d["reward"] for k, d in enumerate(log)) - gamma ** (len(log) - 1) * 100.0
    return dict(r, event="timeout", discounted_return=ret, time_s=deadline_s,
                delta_v_m_s=float(sum(d["delta_v_m_s"] for d in log)))


def mean(runs, name, openings):
    return float(np.mean([runs[(name, o)]["discounted_return"] for o in openings]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=None, help="offline re-scoring under a shorter deadline")
    args = parser.parse_args()
    runs = load(args.runs)
    if args.deadline is not None:
        runs = {key: truncate(r, args.deadline) for key, r in runs.items()}
    names = [n for n in OPTIONS if all((n, o) in runs for o in SCREEN + CONFIRM)]
    table = {n: {o: {k: runs[(n, o)][k] for k in ("event", "discounted_return", "time_s", "delta_v_m_s",
                                                  "first_tau_request_s", "first_tau_effective_s", "triggers",
                                                  "slips", "holds", "descent_decisions", "descent_delta_v_m_s")}
                 for o in SCREEN + CONFIRM} for n in names}
    best_uniform = max(names, key=lambda n: mean(runs, n, SCREEN))
    rule = {"kind": "uniform", "option": best_uniform, "screen_mean": mean(runs, best_uniform, SCREEN)}
    for index, fname in FEATURES.items():
        values = sorted({runs[("E", o)]["initial_observation"][index] for o in SCREEN})
        for thr in [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]:
            for lo, hi in itertools.permutations(names, 2):
                pick = lambda o: lo if runs[("E", o)]["initial_observation"][index] <= thr else hi  # noqa: E731
                m = float(np.mean([runs[(pick(o), o)]["discounted_return"] for o in SCREEN]))
                if m > rule["screen_mean"] + 1e-12:
                    rule = {"kind": "rule", "feature": fname, "index": index, "threshold": thr, "low": lo, "high": hi,
                            "screen_mean": m}

    def apply(o):
        if rule["kind"] == "uniform":
            return rule["option"]
        return rule["low"] if runs[("E", o)]["initial_observation"][rule["index"]] <= rule["threshold"] else rule["high"]

    confirm = {
        "best_uniform": {"option": best_uniform, "mean_return": mean(runs, best_uniform, CONFIRM),
                         "success": sum(runs[(best_uniform, o)]["event"] == "success" for o in CONFIRM)},
        "rule": {"choices": {o: apply(o) for o in CONFIRM},
                 "mean_return": float(np.mean([runs[(apply(o), o)]["discounted_return"] for o in CONFIRM])),
                 "success": sum(runs[(apply(o), o)]["event"] == "success" for o in CONFIRM)},
        "oracle_exploration_only": float(np.mean([max(runs[(n, o)]["discounted_return"] for n in names) for o in CONFIRM])),
    }
    confirm["rule_minus_best_uniform"] = confirm["rule"]["mean_return"] - confirm["best_uniform"]["mean_return"]
    added = [o for o in CONFIRM if runs[(apply(o), o)]["event"] != "success" and runs[(best_uniform, o)]["event"] == "success"]
    gaps = {}
    for o in SCREEN + CONFIRM:
        r = sorted(((runs[(n, o)]["discounted_return"], n) for n in names), reverse=True)
        gaps[o] = {"best": r[0][1], "margin_over_second": r[0][0] - r[1][0], "spread": r[0][0] - r[-1][0]}
    baseline_success = sum(runs[("E", o)]["event"] == "success" for o in SCREEN + CONFIRM)
    all_fail = [o for o in SCREEN + CONFIRM if all(runs[(n, o)]["event"] != "success" for n in names)]
    decisive = sum(g["margin_over_second"] > 1.0 for o, g in gaps.items() if o not in all_fail)
    winners = {g["best"] for o, g in gaps.items() if g["margin_over_second"] > 1.0 and o not in all_fail}
    verdict = ("not_executable" if baseline_success < 6 else "near_equivalent" if decisive < 2 else "decision_value")
    out = {"options": names, "table": table,
           "means": {n: {"screen": mean(runs, n, SCREEN), "confirm": mean(runs, n, CONFIRM), "all": mean(runs, n, SCREEN + CONFIRM)} for n in names},
           "success": {n: sum(runs[(n, o)]["event"] == "success" for o in SCREEN + CONFIRM) for n in names},
           "frozen_on_screen": {"best_uniform": best_uniform, "rule": rule}, "confirmation": confirm,
           "rule_added_failures": added, "per_opening_gaps": gaps, "openings_failed_by_every_option": all_fail, "decisive_openings": decisive,
           "decisive_winners": sorted(winners), "baseline_success": baseline_success, "verdict": verdict,
           "rule_beats_uniform_on_confirmation": bool(confirm["rule_minus_best_uniform"] >= 1.0 and not added)}
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("means", "success", "frozen_on_screen", "confirmation", "per_opening_gaps",
                                          "decisive_openings", "decisive_winners", "verdict",
                                          "rule_beats_uniform_on_confirmation")}, indent=1))


if __name__ == "__main__":
    main()
