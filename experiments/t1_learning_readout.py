"""Frozen learning-efficiency readout for the M0/M1 training round (protocol of 2026-10-12, sec. 7.1).

Inputs: the training run directories (each with ``dev_eval/<decisions>/`` written by
``experiments.t1_dev_eval --watch``), paired by seed: ``--m0 RUN_s0 RUN_s1 RUN_s2 --m1 ...``.

Frozen definitions (do not edit after the round starts):
* development block: ``experiments/t1_dev_openings_295000.json`` (checked by SHA-256);
* checkpoints 5k, 10k, 15k, 20k decisions; deterministic policy, plain environment;
* primary scalar: mean discounted return (gamma 0.999 per decision) over the block;
* AUC = trapezoid over the four checkpoints / 15 000 decisions (a mean level);
* rule level = earliest-feasible G* rule on the same block (evidence/T1_dev_rule_295000,
  81.0895); decisions-to-rule = first checkpoint whose mean >= rule level, else "not reached";
* pairing: M0 and M1 with the same seed index (same initial weights and training
  openings); per seed AUC and final-checkpoint differences, then mean and sample s.d.
  over the seeds; per-opening paired differences at 20k; no seed is dropped;
* reported beside it: clean captures, mean time (failures at 300 s), mean delta-v,
  failures by kind, end phase of failures.
A missing checkpoint is reported as missing, never interpolated.

    python -B -m experiments.t1_learning_readout --m0 logs/t1/t1_m0_s0 ... --m1 logs/t1/t1_m1_s0 ... --output eval/t1/learning_readout.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CHECKPOINTS = (5_000, 10_000, 15_000, 20_000)
OPENINGS = Path("experiments/t1_dev_openings_295000.json")
OPENINGS_SHA256 = "97f4d8f1f2ffdc952f0d90cb6fe95c5ae607d8281f0969935a97cfa2c9230b5f"
RULE_LEVEL = 81.08945807364475


def load_run(run: Path) -> dict:
    out = {"run": str(run), "checkpoints": {}}
    for n in CHECKPOINTS:
        summary = run / "dev_eval" / str(n) / "summary.json"
        if not summary.exists():
            out["checkpoints"][n] = None
            continue
        rows = {int(p.stem): json.loads(p.read_text()) for p in (run / "dev_eval" / str(n)).glob("2*.json")}
        out["checkpoints"][n] = {"summary": json.loads(summary.read_text()), "rows": rows}
    return out


def curve(run: dict) -> list[float | None]:
    return [None if run["checkpoints"][n] is None else run["checkpoints"][n]["summary"]["mean_discounted_return"]
            for n in CHECKPOINTS]


def auc(values: list[float | None]) -> float | None:
    if any(v is None for v in values):
        return None
    x = np.array(CHECKPOINTS, dtype=float)
    y = np.array(values, dtype=float)
    area = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))  # trapezoid, numpy-version independent
    return area / float(x[-1] - x[0])


def to_rule(values: list[float | None]) -> int | str:
    for n, v in zip(CHECKPOINTS, values):
        if v is not None and v >= RULE_LEVEL:
            return n
    return "not reached"


def readout(m0: list[Path], m1: list[Path]) -> dict:
    digest = hashlib.sha256(OPENINGS.read_bytes()).hexdigest()
    if digest != OPENINGS_SHA256:
        raise SystemExit(f"development openings changed: {digest}")
    result: dict = {"rule_level": RULE_LEVEL, "checkpoints": CHECKPOINTS, "methods": {}, "paired": []}
    runs = {"m0": [load_run(r) for r in m0], "m1": [load_run(r) for r in m1]}
    for method, items in runs.items():
        result["methods"][method] = []
        for run in items:
            values = curve(run)
            final = run["checkpoints"][CHECKPOINTS[-1]]
            result["methods"][method].append({
                "run": run["run"], "curve": values, "auc": auc(values), "decisions_to_rule": to_rule(values),
                "final": None if final is None else final["summary"],
            })
        aucs = [r["auc"] for r in result["methods"][method] if r["auc"] is not None]
        result["methods"][method + "_auc_mean_sd"] = (
            [float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None] if aucs else None)
    for a, b in zip(runs["m0"], runs["m1"]):
        ca, cb = curve(a), curve(b)
        fa, fb = a["checkpoints"][CHECKPOINTS[-1]], b["checkpoints"][CHECKPOINTS[-1]]
        pair = {"m0": a["run"], "m1": b["run"],
                "auc_m1_minus_m0": None if auc(ca) is None or auc(cb) is None else auc(cb) - auc(ca),
                "final_m1_minus_m0": None if ca[-1] is None or cb[-1] is None else cb[-1] - ca[-1]}
        if fa is not None and fb is not None:
            common = sorted(set(fa["rows"]) & set(fb["rows"]))
            pair["per_opening_final_return_diff"] = {
                s: fb["rows"][s]["discounted_return"] - fa["rows"][s]["discounted_return"] for s in common}
        result["paired"].append(pair)
    diffs = [p["auc_m1_minus_m0"] for p in result["paired"] if p["auc_m1_minus_m0"] is not None]
    result["auc_diff_mean_sd"] = [float(np.mean(diffs)), float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None] if diffs else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0", type=Path, nargs="+", required=True)
    parser.add_argument("--m1", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.m0) != len(args.m1):
        raise SystemExit("M0 and M1 runs must be paired by seed (same count, same order)")
    result = readout(args.m0, args.m1)
    args.output.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "paired"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
