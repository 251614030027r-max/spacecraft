"""Upper-window review of the stopping-option formal round (no simulation).

Reads the lower window's review directory (``docs/reviews/stopping_20261006``
on ``review/stopping-20261006-audit``, commit 6c5cb97) and writes one JSON:

* where the stopping row's openings went (k=0 / mid-episode / never) against
  Pure MPC and the same policy flying alone;
* how well Q_H at the deployed handoff point separates clean from failed
  takeovers;
* whether the stopping head agrees with its own values: ``beta >= 0.5`` vs
  ``Q_H(s) >= Q_C(s, mu(s))`` on every learned-branch decision;
* for the destroyed Pure successes, the first decision at which the value
  comparison itself says "hand off" on the learned-only trace (same
  deterministic policy, so identical to the stopping trajectory up to its
  actual handoff).

    python -I experiments/stopping_formal_review.py REVIEW_DIR OUTPUT_JSON
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

MODELS = ("262430", "262431", "262432")


def load(directory: str) -> dict[int, dict]:
    return {int(os.path.basename(p)[5:11]): json.load(open(p)) for p in glob.glob(directory + "/seed_*.json")}


def first_value_handoff(trace: dict) -> int | None:
    return next((k for k, (qh, qc) in enumerate(zip(trace["q_handoff"], trace["q_continue"])) if qh >= qc), None)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main(review: str, output: str) -> None:
    pure = load(review + "/raw/formal/pure")
    result: dict = {"source": "docs/reviews/stopping_20261006 at 6c5cb97", "models": {}}
    mid_all = []
    for m in MODELS:
        stopping = load(f"{review}/raw/formal/stopping/{m}")
        learned = load(f"{review}/raw/formal/learned/{m}")
        buckets: dict[str, list[int]] = {"k0": [], "mid": [], "never": []}
        for s, r in stopping.items():
            k = r["handoff_k"]
            buckets["k0" if k == 0 else "never" if k is None else "mid"].append(s)
        clean = lambda rows, ss: sum(bool(rows[s]["clean_completion"]) for s in ss)
        agree = total = against = 0
        missed = 0
        for r in stopping.values():
            t = r["trace"]
            for b, qh, qc in zip(t["beta"], t["q_handoff"], t["q_continue"]):
                total += 1
                agree += (b >= 0.5) == (qh >= qc)
                missed += b < 0.5 and qh > qc + 1.0
            k = r["handoff_k"]
            if k is not None:
                against += t["q_handoff"][k] < t["q_continue"][k]
        for s in buckets["mid"]:
            r, k = stopping[s], stopping[s]["handoff_k"]
            mid_all.append({"model": m, "seed": s, "k": k, "clean": bool(r["clean_completion"]),
                            "q_handoff": r["trace"]["q_handoff"][k], "q_continue": r["trace"]["q_continue"][k]})
        value_rule = [first_value_handoff(r["trace"]) for r in learned.values()]
        learned_fail: dict[str, int] = {}
        for r in learned.values():
            if not r["clean_completion"]:
                key = "+".join(r["failure"]) or "violation"
                learned_fail[key] = learned_fail.get(key, 0) + 1
        result["models"][m] = {
            "buckets": {b: {"n": len(ss), "stopping_clean": clean(stopping, ss), "pure_clean": clean(pure, ss),
                            "learned_clean": clean(learned, ss)} for b, ss in buckets.items()},
            "learned_only_failures": learned_fail,
            "learned_only_clean_median_time_s": statistics.median(
                [r["survival_s"] for r in learned.values() if r["clean_completion"]]),
            "head_agrees_with_values_fraction": agree / total,
            "learned_branch_decisions": total,
            "handoffs": sum(r["handoff_k"] is not None for r in stopping.values()),
            "handoffs_fired_with_q_handoff_below_q_continue": against,
            "decisions_head_continues_though_q_handoff_exceeds_q_continue_by_1": missed,
            "value_rule_on_learned_trace": {"k0": sum(k == 0 for k in value_rule),
                                            "never": sum(k is None for k in value_rule),
                                            "mid": sum(k not in (0, None) for k in value_rule)},
        }
    pos = [a["q_handoff"] for a in mid_all if a["clean"]]
    neg = [a["q_handoff"] for a in mid_all if not a["clean"]]
    result["mid_episode_handoffs"] = {"n": len(mid_all), "clean": len(pos), "auc_q_handoff_clean_vs_failed": auc(pos, neg),
                                      "rows": mid_all}
    destroyed = []
    for case in json.load(open(review + "/DESTROYED_PURE_CASES.json")):
        trace = load(f"{review}/raw/formal/learned/{case['model']}")[case["seed"]]["trace"]
        destroyed.append({"model": case["model"], "seed": case["seed"], "actual_handoff_k": case["stopping"]["handoff_k"],
                          "value_rule_handoff_k": first_value_handoff(trace)})
    result["destroyed_pure_successes"] = destroyed
    with open(output, "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
