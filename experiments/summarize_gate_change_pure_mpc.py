"""Summarise what a terminal-gate change does to the Pure MPC row.

Inputs: two directories of ``evaluate_hybrid_policy`` outputs on the same
seeds and machine (``--control desired_pose``, which stores every control
wrench), one produced before and one after the change, plus the
``*.gatecount.json`` files written by
``experiments/count_pure_mpc_gate_differences.py`` for the "after" run.

Per episode: completion before/after, whether the wrench sequence is bitwise
identical, the first differing control step, illegal-entry counts. Totals:
completions, identical episodes, and gate-decision differences.

    python -B -m experiments.summarize_gate_change_pure_mpc --before DIR --after DIR --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _records(directory: Path) -> dict[int, dict]:
    out = {}
    for path in sorted(directory.glob("pure_*.json")):
        if path.name.endswith(".gatecount.json"):
            continue
        for record in json.loads(path.read_text())["records"]:
            out[int(record["seed"])] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = _records(args.before)
    after = _records(args.after)
    if before.keys() != after.keys():
        raise ValueError("before/after seed sets differ")
    episodes = []
    for seed in sorted(before):
        old, new = before[seed], after[seed]
        w_old = np.asarray(old["control_wrenches"])
        w_new = np.asarray(new["control_wrenches"])
        identical = w_old.shape == w_new.shape and bool(np.array_equal(w_old, w_new))
        first = None
        if not identical:
            n = min(len(w_old), len(w_new))
            diff = np.flatnonzero(np.any(w_old[:n] != w_new[:n], axis=1))
            first = int(diff[0]) if diff.size else n
        episodes.append(
            {
                "seed": seed,
                "completed_before": bool(old["completed"]),
                "completed_after": bool(new["completed"]),
                "wrenches_bitwise_identical": identical,
                "first_differing_control_step": first,
                "illegal_entries_before": int(old["illegal_terminal_entry_count"]),
                "illegal_entries_after": int(new["illegal_terminal_entry_count"]),
                "survival_s_before": float(old["survival_s"]),
                "survival_s_after": float(new["survival_s"]),
                "zero_fallback_steps_before": int(old["zero_fallback_steps_total"]),
                "zero_fallback_steps_after": int(new["zero_fallback_steps_total"]),
            }
        )
    counts = {
        path.name: json.loads(path.read_text())
        for path in sorted(args.after.glob("*.gatecount.json"))
    }
    changed = [e for e in episodes if not e["wrenches_bitwise_identical"]]
    summary = {
        "episodes": len(episodes),
        "completed_before": sum(e["completed_before"] for e in episodes),
        "completed_after": sum(e["completed_after"] for e in episodes),
        "bitwise_identical": len(episodes) - len(changed),
        "changed_seeds": [e["seed"] for e in changed],
        "changed_and_completed_before": [e["seed"] for e in changed if e["completed_before"]],
        "changed_without_illegal_entry_before": [
            e["seed"] for e in changed if e["illegal_entries_before"] == 0
        ],
        "outcome_flips": [
            {"seed": e["seed"], "before": e["completed_before"], "after": e["completed_after"]}
            for e in episodes
            if e["completed_before"] != e["completed_after"]
        ],
        "gate_calls": sum(c["calls"] for c in counts.values()),
        "gate_differ_vs_at_7555185": sum(c["differ_vs_at_7555185"] for c in counts.values()),
        "gate_differ_vs_pre_7555185": sum(c["differ_vs_pre_7555185"] for c in counts.values()),
    }
    args.output.write_text(
        json.dumps({"summary": summary, "gate_counts": counts, "episodes": episodes}, indent=1)
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
