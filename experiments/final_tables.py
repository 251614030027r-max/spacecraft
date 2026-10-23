"""Paper tables over the two formal blocks (271000 + 272000, 96 paired openings).

Rows: Pure MPC, nominal (handcrafted guidance + MPC; attribution, not a gate),
learned-only, value stopping. Everything is paired by opening. Descriptive:
the preregistered verdict of the round is the 271000 readout of
``FINAL_MAINLINE_20261006``; this tool adds no new pass/fail rule.

    python -B -m experiments.final_tables --pure DIR --pure DIR --nominal DIR --nominal DIR \
        --learned MODEL=DIR (one per model and block) --stopping MODEL=DIR (same) --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v3_common import parse_seed_range
from experiments.v3_stage_c import _scans, _write_json

FORMAL_SEEDS = "271000-271047,272000-272047"


def merge(directories: list[str]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for directory in directories:
        for seed, row in _scans(directory).items():
            if seed in rows:
                raise ValueError(f"opening {seed} appears twice")
            rows[seed] = row
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on b (row1 clean only) vs c (row2 clean only)."""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return float(min(1.0, 2.0 * tail))


def paired(a: dict[int, dict], b: dict[int, dict], seeds: list[int]) -> dict[str, Any]:
    """a vs b: openings only a completes cleanly (gained) and only b does (lost)."""

    clean = lambda rows, s: bool(rows[s]["clean_completion"])
    gained = [s for s in seeds if clean(a, s) and not clean(b, s)]
    lost = [s for s in seeds if clean(b, s) and not clean(a, s)]
    shared = [s for s in seeds if clean(a, s) and clean(b, s)]
    med = lambda key: float(np.median([a[s][key] - b[s][key] for s in shared])) if shared else None
    return {"clean_a": sum(clean(a, s) for s in seeds), "clean_b": sum(clean(b, s) for s in seeds),
            "a_only": gained, "b_only": lost, "mcnemar_p": mcnemar_exact(len(gained), len(lost)),
            "shared_clean": len(shared), "median_time_diff_s": med("survival_s"),
            "median_delta_v_diff_m_s": med("equivalent_delta_v_m_s")}


def row_summary(rows: dict[int, dict], seeds: list[int]) -> dict[str, Any]:
    clean = [s for s in seeds if rows[s]["clean_completion"]]
    out = {"clean": len(clean), "openings": len(seeds),
           "violation_episodes": sum(not rows[s]["zero_violation"] for s in seeds),
           "clean_median_time_s": float(np.median([rows[s]["survival_s"] for s in clean])) if clean else None,
           "clean_median_delta_v_m_s": float(np.median([rows[s]["equivalent_delta_v_m_s"] for s in clean])) if clean else None}
    if all("handoff_k" in rows[s] for s in seeds):
        ks = [rows[s]["handoff_k"] for s in seeds]
        out["handoff"] = {"k0": sum(k == 0 for k in ks), "never": sum(k is None for k in ks),
                          "mid": sum(k not in (0, None) for k in ks)}
    return out


def build(pure: dict, nominal: dict, learned: dict[str, dict], stopping: dict[str, dict], seeds: list[int]) -> dict:
    problems = []
    every = {"pure": pure, "nominal": nominal, **{f"learned {m}": r for m, r in learned.items()},
             **{f"stopping {m}": r for m, r in stopping.items()}}
    commits = set()
    for name, rows in every.items():
        missing = sorted(set(seeds) - set(rows))
        if missing:
            problems.append(f"{name}: missing {len(missing)} openings")
        expected_row = name.split()[0]
        for r in rows.values():
            commits.add(r.get("code_commit"))
            if r.get("code_dirty") or r.get("max_decisions") is not None or r.get("row") != expected_row:
                problems.append(f"{name}: dirty, truncated or wrong row")
                break
    for m in stopping:
        shas = {r.get("model_sha256") for r in stopping[m].values()} | {r.get("model_sha256") for r in learned.get(m, {}).values()}
        if len(shas) != 1:
            problems.append(f"{m}: more than one model file")
    if len(commits) != 1 or None in commits:
        problems.append(f"{len(commits)} code commits across rows")
    if problems:
        return {"problems": problems}
    out: dict[str, Any] = {"problems": [], "openings": len(seeds),
                           "pure": row_summary(pure, seeds), "nominal": row_summary(nominal, seeds),
                           "nominal_vs_pure": paired(nominal, pure, seeds), "models": {}}
    for m in sorted(stopping):
        out["models"][m] = {
            "learned": row_summary(learned[m], seeds),
            "stopping": row_summary(stopping[m], seeds),
            "stopping_vs_pure": paired(stopping[m], pure, seeds),
            "stopping_vs_nominal": paired(stopping[m], nominal, seeds),
            "stopping_vs_learned": paired(stopping[m], learned[m], seeds),
        }
    return out


def _pairs_multi(values: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for value in values:
        name, _, path = value.partition("=")
        out.setdefault(name, []).append(path)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pure", action="append", required=True)
    parser.add_argument("--nominal", action="append", required=True)
    parser.add_argument("--learned", action="append", required=True, help="MODEL=DIR, once per block")
    parser.add_argument("--stopping", action="append", required=True, help="MODEL=DIR, once per block")
    parser.add_argument("--seeds", default=FORMAL_SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    seeds = parse_seed_range(args.seeds)
    result = build(merge(args.pure), merge(args.nominal),
                   {m: merge(d) for m, d in _pairs_multi(args.learned).items()},
                   {m: merge(d) for m, d in _pairs_multi(args.stopping).items()}, seeds)
    result["note"] = "descriptive paper tables; the preregistered verdict is the 271000 readout"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    if result["problems"]:
        print(json.dumps({"problems": result["problems"]}, indent=1))
        return
    print(json.dumps({"pure": result["pure"]["clean"], "nominal": result["nominal"]["clean"],
                      "models": {m: {"learned": v["learned"]["clean"], "stopping": v["stopping"]["clean"],
                                     "vs_pure": [len(v["stopping_vs_pure"]["a_only"]), len(v["stopping_vs_pure"]["b_only"])],
                                     "vs_learned": [len(v["stopping_vs_learned"]["a_only"]), len(v["stopping_vs_learned"]["b_only"])]}
                                 for m, v in result["models"].items()}}, indent=1))


if __name__ == "__main__":
    main()
