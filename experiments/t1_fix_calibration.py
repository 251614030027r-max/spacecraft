"""Pre-registered one-shot weight calibration of the critic fixes (revision 2, 2026-10-15).

Reads ``fix_stats.json`` of a 2k-decision M0 calibration run trained with lam_mc = lam_eq = 0
and ``--grad-probe-every`` > 0 (the auxiliary terms only measured, never applied). Over the
probes of decisions [1000, 2000]:

* lam_eq = 0.1, fixed (the keep-band equivalence is exact where its mask is 1);
* lam_mc = the largest of {0.3, 0.1, 0.03} with lam * median(||g_mc|| / ||g_TD||) <= 0.3, the
  median taken over probes whose batch held MC labels; never below the floor 0.03 (a large
  early TD gradient does not switch the long-run return anchor off; Huber(10) bounds each
  label's gradient), and 0.03 when too few labelled probes (< 30) exist;
* protection: if median(||lam_mc g_mc + lam_eq g_eq|| / ||g_TD||) > 0.5, lam_mc steps down one
  candidate (not below 0.03);
* non-finite or zero gradients: a code defect, reported (``defect``), no value is frozen.

Revision 1 (science d8b185f) could set lam_mc to 0 when the term was weak, or to 0.2/median
when it was strong; revision 2 replaces it. Performance is never read.

    python -B -m experiments.t1_fix_calibration logs/t1/t1_cal_m0_s0/fix_stats.json --output calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WINDOW = (1_000, 2_000)
MIN_PROBES = 30
LAMBDA_EQ = 0.1
MC_CANDIDATES = (0.3, 0.1, 0.03)
MC_SHARE_MAX = 0.3
MC_FLOOR = 0.03
COMBINED_SHARE_MAX = 0.5


def calibrate(stats: dict) -> dict:
    """Revision 2 (module docstring)."""

    d = np.asarray(stats["probe_decisions"], dtype=float)
    sel = (d >= WINDOW[0]) & (d <= WINDOW[1])
    g_td = np.asarray(stats["g_td"], dtype=float)[sel]
    g_mc = np.asarray(stats["g_mc"], dtype=float)[sel]
    g_eq = np.asarray(stats["g_eq"], dtype=float)[sel]
    dot = np.asarray(stats.get("g_mc_dot_eq", [0.0] * len(d)), dtype=float)[sel]
    mc_fraction = np.asarray(stats["mc_fraction"], dtype=float)[sel]
    eq_fraction = np.asarray(stats["eq_fraction"], dtype=float)[sel]
    out: dict = {"revision": 2, "window_decisions": WINDOW, "probes": int(sel.sum()),
                 "mc_fraction_mean": float(mc_fraction.mean()) if sel.any() else None,
                 "eq_fraction_mean": float(eq_fraction.mean()) if sel.any() else None,
                 "g_td_median": float(np.median(g_td)) if sel.any() else None}
    finite = sel.any() and bool(np.all(np.isfinite(np.concatenate([g_td, g_mc, g_eq, dot])))) and bool(np.all(g_td > 0))
    if not finite:
        out["defect"] = "no probes in the window, or non-finite / zero gradients: report, do not train"
        return out
    labelled = mc_fraction > 0.0
    ratio_mc = g_mc[labelled] / g_td[labelled]
    ratio_eq = g_eq / g_td
    out["mc"] = {"labelled_probes": int(labelled.sum()),
                 "ratio_median": float(np.median(ratio_mc)) if ratio_mc.size else None,
                 "ratio_p10_p90": [float(np.percentile(ratio_mc, 10)), float(np.percentile(ratio_mc, 90))]
                 if ratio_mc.size else None}
    out["eq"] = {"ratio_median": float(np.median(ratio_eq)),
                 "ratio_p10_p90": [float(np.percentile(ratio_eq, 10)), float(np.percentile(ratio_eq, 90))]}
    if ratio_mc.size < MIN_PROBES:
        lam_mc, why = MC_FLOOR, f"too few labelled probes ({ratio_mc.size} < {MIN_PROBES}): floor"
    else:
        median = float(np.median(ratio_mc))
        fits = [lam for lam in MC_CANDIDATES if lam * median <= MC_SHARE_MAX]
        lam_mc, why = (fits[0], "largest candidate with share <= 0.3") if fits else (MC_FLOOR, "floor (0.03 exceeds 0.3)")

    def combined(lam: float) -> float:
        value = np.sqrt(np.maximum(lam ** 2 * g_mc ** 2 + LAMBDA_EQ ** 2 * g_eq ** 2 + 2 * lam * LAMBDA_EQ * dot, 0.0)) / g_td
        return float(np.median(value))

    share = combined(lam_mc)
    if share > COMBINED_SHARE_MAX and lam_mc > MC_FLOOR:
        lam_mc = MC_CANDIDATES[MC_CANDIDATES.index(lam_mc) + 1]
        why += f"; combined share {share:.3f} > {COMBINED_SHARE_MAX}: one step down"
        share = combined(lam_mc)
    out["mc"].update(lam=lam_mc, reason=why)
    out["eq"].update(lam=LAMBDA_EQ, reason="fixed")
    out["combined_share_median"] = share
    out["frozen"] = {"lambda_mc": lam_mc, "lambda_eq": LAMBDA_EQ}
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fix_stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = calibrate(json.loads(args.fix_stats.read_text()))
    result["fix_stats_sha256"] = __import__("hashlib").sha256(args.fix_stats.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
