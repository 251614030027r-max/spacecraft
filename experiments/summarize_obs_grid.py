"""Summarize the observation-error x margin confirm grid into one table.

Reads every ``evaluate_mpc`` JSON matching a glob (default the grid's
``logs/obs_*.json``), parses the error level / speed fraction / seed from each
filename, and prints one compact row per file: completion, violation rate,
worst real-geometry margin, and the compute p95 budget ratio. Tolerant of a
partial grid (interrupted run) -- it summarizes whatever is present.

Filename convention (from the confirm-run batch):
``logs/obs_<err>_f<frac>_s<seed>.json`` e.g. ``logs/obs_big_f0.4_s970000.json``.

    python -B -m experiments.summarize_obs_grid --glob "logs/obs_*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the obs-error grid")
    parser.add_argument("--glob", default="logs/obs_*.json")
    return parser.parse_args()


_NAME = re.compile(r"obs_(?P<err>[a-z0-9]+)_f(?P<frac>[0-9.]+)_s(?P<seed>\d+)")

_ERR_ORDER = {"e0": 0, "small": 1, "mid": 2, "big": 3}


def _row(path: Path) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get("main_table")
    records = payload.get("episode_records", [])
    if table is None:
        return None
    match = _NAME.search(path.name)
    err = match.group("err") if match else "?"
    frac = float(match.group("frac")) if match else float("nan")
    seed = int(match.group("seed")) if match else -1
    episodes = table.get("episodes", len(records))
    completed = table.get("completed_episodes", 0)
    violated = sum(1 for r in records if r.get("first_violation"))
    worst = table.get("worst_constraint_margin") or {}
    worst_val = min(worst.values()) if worst else float("nan")
    compute = table.get("per_step_compute_s", {})
    p95 = compute.get("controller_p95_over_budget")
    times = [
        r["first_violation"]["time_s"]
        for r in records
        if r.get("first_violation")
    ]
    return {
        "err": err, "frac": frac, "seed": seed,
        "completed": f"{completed}/{episodes}",
        "violated": f"{violated}/{episodes}",
        "worst": worst_val,
        "p95x": p95,
        "viol_t_mean": mean(times) if times else None,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.glob))
    rows = [r for r in (_row(Path(p)) for p in paths) if r is not None]
    if not rows:
        print(f"no grid files matched {args.glob!r}")
        return
    rows.sort(key=lambda r: (_ERR_ORDER.get(r["err"], 9), r["seed"], -r["frac"]))
    header = f"{'err':<6}{'frac':>5}{'seed':>9}{'completed':>11}{'violated':>10}{'worst':>9}{'p95x':>7}{'viol_t':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        worst = f"{r['worst']:+.4f}" if r["worst"] == r["worst"] else "--"
        p95 = f"{r['p95x']:.2f}x" if r["p95x"] is not None else "--"
        vt = f"{r['viol_t_mean']:.1f}" if r["viol_t_mean"] is not None else "--"
        print(
            f"{r['err']:<6}{r['frac']:>5.2f}{r['seed']:>9}{r['completed']:>11}"
            f"{r['violated']:>10}{worst:>9}{p95:>7}{vt:>8}"
        )
    print(f"\n{len(rows)} files summarized. Missing rows = not yet run (resume to fill).")


if __name__ == "__main__":
    main()
