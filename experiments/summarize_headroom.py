"""Aggregate the commit-time sweep into the headroom answer.

Takes the ``commit_now`` baseline and the ``commit_at`` runs produced by
``experiments.evaluate_arrival_condition_script`` and reports the four things
``docs/T10_EXECUTION_ORDER.md`` section 5 asks for:

1. the rescue rate -- of the seeds the fixed setpoint fails, how many complete
   under *some* commit time;
2. per rescued seed, the commit times that work, the cheapest one, and what it
   costs in time, fuel and illegal crossings;
3. the shape of the feasible commit set -- one interval or several, which is
   what says whether the decision is a smooth function of the state;
4. the reverse cost -- seeds the fixed setpoint completes that a non-zero
   commit time breaks.

Gate B in the execution order turns on the first and third of those: no
rescues, or every seed rescued by the *same* commit time, both mean no learned
layer is needed. A constant is not a policy.

Usage::

    python -B -m experiments.summarize_headroom \\
        --baseline logs/t10/pure_mpc_h35_48.json \\
        --sweep logs/t10/commit_at_*.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        records.extend(payload if isinstance(payload, list) else [payload])
    return records


def contiguity(times: list[float], grid: list[float]) -> str:
    """Is the successful set one run of the sampled grid, or several?"""

    if not times:
        return "empty"
    index = sorted(grid.index(t) for t in times)
    runs = 1
    for previous, current in zip(index, index[1:]):
        if current != previous + 1:
            runs += 1
    if runs == 1:
        return "contiguous" if len(index) > 1 else "single"
    return f"{runs} fragments"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = {r["seed"]: r for r in load([args.baseline])}
    sweep = load(list(args.sweep))
    by_seed: dict[int, dict[float, dict[str, Any]]] = defaultdict(dict)
    for record in sweep:
        by_seed[record["seed"]][float(record["commit_time_s"])] = record
    grid = sorted({float(r["commit_time_s"]) for r in sweep})

    for record in sweep:
        if record["policy"] != "commit_at":
            raise ValueError("sweep files must all be commit_at runs")
    for record in baseline.values():
        if record["policy"] != "commit_now":
            raise ValueError("the baseline must be a commit_now run")

    failed = sorted(s for s, r in baseline.items() if not r["completed"])
    completed = sorted(s for s, r in baseline.items() if r["completed"])
    swept_failed = [s for s in failed if s in by_seed]
    rescued: list[int] = []
    lines: list[str] = []

    lines.append(f"commit grid: {grid}")
    lines.append(
        f"baseline (commit_now): {len(completed)}/{len(baseline)} completed; "
        f"{len(failed)} failed, {len(swept_failed)} of those swept"
    )
    lines.append("")
    lines.append("RESCUE -- seeds the fixed setpoint fails")
    lines.append(
        f"{'seed':>7s} {'works at':<28s} {'shape':<14s} "
        f"{'best T':>7s} {'t_s':>7s} {'fuel':>8s} {'illegal':>7s}"
    )
    for seed in swept_failed:
        runs = by_seed[seed]
        good = sorted(t for t, r in runs.items() if r["completed"])
        if not good:
            lines.append(f"{seed:>7d} {'-- none --':<28s} {'':<14s}")
            continue
        rescued.append(seed)
        best = min(good, key=lambda t: runs[t]["force_impulse_n_s"])
        record = runs[best]
        lines.append(
            f"{seed:>7d} {str(good):<28s} {contiguity(good, grid):<14s} "
            f"{best:>7.0f} {record['end_time_s']:>7.1f} "
            f"{record['force_impulse_n_s']:>8.1f} "
            f"{record['illegal_terminal_entry_count']:>7d}"
        )
    rate = len(rescued) / len(swept_failed) if swept_failed else 0.0
    lines.append("")
    lines.append(f"rescue rate: {len(rescued)}/{len(swept_failed)} = {rate:.1%}")

    if rescued:
        best_times = {
            seed: min(
                (t for t, r in by_seed[seed].items() if r["completed"]),
                key=lambda t: by_seed[seed][t]["force_impulse_n_s"],
            )
            for seed in rescued
        }
        distinct = sorted(set(best_times.values()))
        shared = [
            t for t in grid
            if all(by_seed[s].get(t, {}).get("completed") for s in rescued)
        ]
        lines.append(f"cheapest commit time per rescued seed: {best_times}")
        lines.append(f"distinct best commit times: {distinct}")
        lines.append(
            f"commit times that rescue EVERY rescued seed: {shared}"
            + ("  <-- a constant would do; that is not a policy" if shared else "")
        )

    reverse = []
    for seed in completed:
        runs = by_seed.get(seed, {})
        broken = sorted(t for t, r in runs.items() if not r["completed"])
        if broken:
            reverse.append((seed, broken))
    lines.append("")
    lines.append("REVERSE COST -- seeds the fixed setpoint completes")
    if not by_seed.keys() & set(completed):
        lines.append("  (no completing seed was swept)")
    elif not reverse:
        lines.append("  none broken by any swept commit time")
    else:
        for seed, broken in reverse:
            lines.append(f"  {seed}: broken at {broken}")

    report = "\n".join(lines)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
