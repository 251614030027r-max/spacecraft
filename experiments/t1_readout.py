"""T1 scripted-intent readout: per-opening table and the diagnostics R1 reports.

    python -B -m experiments.t1_readout --input-dir eval/t1/r1 --output eval/t1/r1/readout.json

Reports, without deciding GO/REVIEW/STOP (that is a written review):
coverage (openings with any clean capture), the per-opening best of the
scripted intents against C0 (Pure), the best single fixed intent, which
intent is best by latitude/phase bin, and the ranking under three time/delta-v
exchange rates recomputed offline from the logged time and delta-v (no new
simulation). Returns use the frozen reward except for the exchange rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.t1_env import T1Reward

INTENTS = ("C0", "C1", "C2", "C3")
RATES_S_PER_M_S = (50.0, 100.0, 200.0)


def episode_return(row: dict[str, Any], rate_s_per_m_s: float, reward: T1Reward) -> float:
    """Undiscounted-cost approximation of the frozen reward at a given exchange rate."""

    c_t = reward.time_cost_per_s
    c_v = c_t * rate_s_per_m_s
    value = -(c_t * row["time_s"] + c_v * row["delta_v_m_s"])
    event = row["event"]
    if event == "success":
        value += reward.capture_reward
    elif event == "timeout":
        value -= reward.timeout_penalty
    elif event == "unsafe":
        value -= reward.unsafe_penalty
    return float(value)


def load(input_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    rows: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("C*_*.json")):
        row = json.loads(path.read_text())
        rows.setdefault(int(row["seed"]), {})[row["intent"]] = row
    return rows


def readout(rows: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    reward = T1Reward()
    complete = {s: by for s, by in rows.items() if all(i in by for i in INTENTS)}
    n = len(complete)
    clean = {i: sum(by[i]["event"] == "success" for by in complete.values()) for i in INTENTS}
    covered = sum(any(by[i]["event"] == "success" for i in INTENTS) for by in complete.values())
    numerical = {i: sum(by[i]["event"] == "numerical_failure" for by in complete.values()) for i in INTENTS}
    by_rate: dict[str, Any] = {}
    for rate in RATES_S_PER_M_S:
        returns = {s: {i: episode_return(by[i], rate, reward) for i in INTENTS} for s, by in complete.items()}
        mean = {i: float(np.mean([r[i] for r in returns.values()])) if n else None for i in INTENTS}
        best_fixed = max(INTENTS, key=lambda i: mean[i]) if n else None
        oracle = {s: max(INTENTS, key=lambda i: r[i]) for s, r in returns.items()}
        oracle_mean = float(np.mean([returns[s][oracle[s]] for s in returns])) if n else None
        by_rate[f"{rate:g}"] = {
            "mean_return": mean,
            "best_single_intent": best_fixed,
            "oracle_mean_return": oracle_mean,
            "oracle_minus_best_single": (oracle_mean - mean[best_fixed]) if n else None,
            "oracle_minus_C0": (oracle_mean - mean["C0"]) if n else None,
            "best_intent_counts": {i: sum(v == i for v in oracle.values()) for i in INTENTS},
        }
    oracle_clean = covered
    best_single_clean = max(clean.values()) if n else 0
    paired = {}
    for i in INTENTS[1:]:
        both = [by for by in complete.values() if by[i]["event"] == "success" and by["C0"]["event"] == "success"]
        paired[i] = {
            "common_clean": len(both),
            "median_time_minus_C0_s": float(np.median([b[i]["time_s"] - b["C0"]["time_s"] for b in both])) if both else None,
            "median_dv_minus_C0_m_s": float(np.median([b[i]["delta_v_m_s"] - b["C0"]["delta_v_m_s"] for b in both])) if both else None,
        }
    return {
        "openings_complete": n,
        "clean_by_intent": clean,
        "openings_with_any_clean": covered,
        "oracle_clean_minus_C0": oracle_clean - clean.get("C0", 0),
        "oracle_clean_minus_best_single": oracle_clean - best_single_clean,
        "numerical_failures_by_intent": numerical,
        "paired_vs_C0": paired,
        "exchange_rate_sensitivity": by_rate,
        "qp_zero_fallback_steps_total": int(sum(by[i]["qp_zero_fallback_steps"] for by in complete.values() for i in INTENTS)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = readout(load(args.input_dir))
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
