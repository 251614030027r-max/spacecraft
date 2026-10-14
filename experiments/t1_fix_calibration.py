"""Pre-registered one-shot weight calibration of the critic fixes (task book 2026-10-15, sec. D).

Reads ``fix_stats.json`` of a 2k-decision M0 calibration run trained with lam_mc = lam_eq = 0
and ``--grad-probe-every`` > 0 (the auxiliary terms only measured, never applied). For each term,
over the probes of decisions [1000, 2000]:

    ratio = ||g_aux|| / ||g_TD||   (unweighted, critic parameters)

and lam is the largest of {0.03, 0.1, 0.3} with lam * median(ratio) in [0.1, 0.3]. If none fits:
too strong (0.03 * median > 0.3) -> the safe smaller value 0.2 / median; too weak
(0.3 * median < 0.1) -> 0 (an auxiliary term is not forced on). Too few probes (< 30), too few
MC labels (mean MC mask fraction < 0.02) or non-finite gradients -> 0. The combined share
median(||lam_mc g_mc + lam_eq g_eq|| / ||g_TD||) is checked against 0.4; above it both lams are
scaled down by the same factor. Performance is never read.

    python -B -m experiments.t1_fix_calibration logs/t1/t1_cal_m0_s0/fix_stats.json --output calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CANDIDATES = (0.3, 0.1, 0.03)
BAND = (0.1, 0.3)
WINDOW = (1_000, 2_000)
COMBINED_MAX = 0.4
MIN_PROBES = 30
MIN_MC_FRACTION = 0.02


def choose(median: float) -> tuple[float, str]:
    for lam in CANDIDATES:
        if BAND[0] <= lam * median <= BAND[1]:
            return lam, "largest candidate in band"
    if CANDIDATES[-1] * median > BAND[1]:
        return float(f"{0.2 / median:.2g}"), "too strong at 0.03: safe smaller value 0.2/median"
    return 0.0, "too weak at 0.3: not forced on"


def calibrate(stats: dict) -> dict:
    d = np.asarray(stats["probe_decisions"], dtype=float)
    sel = (d >= WINDOW[0]) & (d <= WINDOW[1])
    g_td = np.asarray(stats["g_td"], dtype=float)[sel]
    g_mc = np.asarray(stats["g_mc"], dtype=float)[sel]
    g_eq = np.asarray(stats["g_eq"], dtype=float)[sel]
    dot = np.asarray(stats.get("g_mc_dot_eq", [np.nan] * len(d)), dtype=float)[sel]
    mc_fraction = np.asarray(stats["mc_fraction"], dtype=float)[sel]
    eq_fraction = np.asarray(stats["eq_fraction"], dtype=float)[sel]
    out: dict = {"window_decisions": WINDOW, "probes": int(sel.sum()),
                 "mc_fraction_mean": float(mc_fraction.mean()) if sel.any() else None,
                 "eq_fraction_mean": float(eq_fraction.mean()) if sel.any() else None,
                 "g_td_median": float(np.median(g_td)) if sel.any() else None}
    finite = bool(np.all(np.isfinite(np.concatenate([g_td, g_mc, g_eq])))) and bool(np.all(g_td > 0))
    lams = {}
    for name, g, fraction in (("mc", g_mc, mc_fraction), ("eq", g_eq, eq_fraction)):
        ratio = g / g_td if sel.any() else np.array([])
        median = float(np.median(ratio)) if ratio.size else float("nan")
        entry = {"ratio_median": median,
                 "ratio_p10_p90": [float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90))] if ratio.size else None}
        if out["probes"] < MIN_PROBES:
            lam, why = 0.0, f"too few probes ({out['probes']} < {MIN_PROBES})"
        elif not finite:
            lam, why = 0.0, "non-finite or zero gradients"
        elif float(fraction.mean()) < MIN_MC_FRACTION:
            lam, why = 0.0, f"too few labelled samples (mean mask fraction {fraction.mean():.4f})"
        else:
            lam, why = choose(median)
        entry.update(lam=lam, reason=why, weighted_share_median=lam * median if np.isfinite(median) else None)
        out[name] = entry
        lams[name] = lam
    combined = np.sqrt(np.maximum(lams["mc"] ** 2 * g_mc ** 2 + lams["eq"] ** 2 * g_eq ** 2
                                  + 2 * lams["mc"] * lams["eq"] * np.nan_to_num(dot), 0.0)) / g_td
    share = float(np.median(combined)) if combined.size else 0.0
    scale = 1.0 if share <= COMBINED_MAX or share == 0.0 else COMBINED_MAX / share
    out["combined_share_median"] = share
    out["combined_scale"] = scale
    out["frozen"] = {"lambda_mc": float(f"{lams['mc'] * scale:.3g}"), "lambda_eq": float(f"{lams['eq'] * scale:.3g}")}
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
