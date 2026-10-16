"""Pre-registered choice of the T1-L-S trainer (A or B) from the *old* T1-L N0 run (plan 2026-10-16).

Frozen before the old T1-L 40k results exist; applied mechanically, once, to the delivered N0 run
directory (and NDelta for the record). Inputs: ``dev_eval/<mark>/summary.json`` (16 development
openings) for 0, 25000, 30000, 35000, 40000; ``fix_stats.json`` (replay critic audits); and
``decisions.jsonl`` (behaviour requests).

N0 counts as degraded (v4-type) when any of:
  (a) at least 2 of the marks 25k / 30k / 35k / 40k have <= 13/16 captures;
  (b) the mean discounted return over those marks is below the 0k return minus 10;
  (c) over decisions >= 30000, the mean replay audit Q - G (live plan) exceeds +50 AND the
      share of "later" among requests with a live plan exceeds 0.30.
Choice: B (UTD 1, actor every 3 real transitions, cosine learning rate) if N0 is degraded,
otherwise A (the 166244b trainer). Fewer than 2 of the late marks available: "undetermined"
(stop and report). NDelta degraded while N0 is not: A, with ``ndelta_only_degraded`` recorded
(the relative term is then reviewed separately, not blamed on the trainer).

Reference points (not used by the rule): v4 M0/M2 at 25-40k had 8-14/16 captures, returns
50-110 below 0k, replay Q - G +70..+100 and later shares 0.4-0.6; T1-L N0 at 20k had 16/16 and
+2.07 over its 0k.

    python -B -m experiments.t1_l_s_trainer_choice --n0 <old n0_s0 dir> --ndelta <old ndelta_s0 dir> --output choice.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LATE_MARKS = (25_000, 30_000, 35_000, 40_000)
CAPTURE_LIMIT = 13
RETURN_DROP = 10.0
Q_LIMIT = 50.0
LATER_LIMIT = 0.30
AUDIT_FROM = 30_000


def _summary(run: Path, mark: int) -> dict | None:
    path = run / "dev_eval" / str(mark) / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def degraded(run: Path) -> dict:
    zero = _summary(run, 0)
    late = {m: s for m in LATE_MARKS if (s := _summary(run, m)) is not None}
    out: dict = {"run": str(run), "late_marks": sorted(late)}
    if zero is None or len(late) < 2:
        out["status"] = "undetermined"
        return out
    low = [m for m, s in late.items() if s["clean_capture"] <= CAPTURE_LIMIT]
    mean_late = float(np.mean([s["mean_discounted_return"] for s in late.values()]))
    out |= {"zero_return": zero["mean_discounted_return"], "late_mean_return": mean_late,
            "marks_with_low_capture": low,
            "a": len(low) >= 2, "b": mean_late < zero["mean_discounted_return"] - RETURN_DROP}
    q, later = None, None
    fix = run / "fix_stats.json"
    if fix.exists():
        s = json.loads(fix.read_text())
        at = np.asarray(s.get("audit_decisions", []), dtype=float)
        vals = np.asarray(s.get("q_minus_g_plan", []), dtype=float)
        sel = (at >= AUDIT_FROM) & np.isfinite(vals) if at.size else np.array([], dtype=bool)
        q = float(vals[sel].mean()) if sel.any() else None
    decisions = run / "decisions.jsonl"
    if decisions.exists():
        counts = {"keep": 0, "later": 0, "earlier": 0}
        with decisions.open() as handle:
            for line in handle:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    break
                rv = d.get("rv") or {}
                if d.get("t", 0) >= AUDIT_FROM and rv.get("due") and d.get("req") in counts:
                    counts[d["req"]] += 1
        total = sum(counts.values())
        later = counts["later"] / total if total else None
    out |= {"q_minus_g_plan_from_30k": q, "later_share_from_30k": later,
            "c": q is not None and later is not None and q > Q_LIMIT and later > LATER_LIMIT}
    out["status"] = "degraded" if (out["a"] or out["b"] or out["c"]) else "stable"
    return out


def choose(n0: Path, ndelta: Path | None) -> dict:
    result: dict = {"rule": __doc__.split("\n\n")[1], "n0": degraded(n0)}
    if ndelta is not None:
        result["ndelta"] = degraded(ndelta)
    status = result["n0"]["status"]
    if status == "undetermined":
        result["trainer"] = "undetermined"
    else:
        result["trainer"] = "B" if status == "degraded" else "A"
        result["ndelta_only_degraded"] = bool(ndelta is not None and status == "stable"
                                              and result["ndelta"].get("status") == "degraded")
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n0", type=Path, required=True)
    parser.add_argument("--ndelta", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = choose(args.n0, args.ndelta)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("trainer",)} | {"n0": result["n0"]["status"]}))
    return result


if __name__ == "__main__":
    main()
