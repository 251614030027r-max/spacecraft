"""R1 opening block: 32 stratified, pre-registered T1 openings.

Rule (fixed before any R1 row is run): scan seeds upward from ``--start``;
for each, read the reset features only (no rollout); cell = (latitude bin,
phase bin) with four equal-probability bins of |sin latitude| (latitude of
the line of sight against the target angular momentum axis H) and four 90 deg
bins of the port phase (angle the port still has to sweep about H to reach
the line of sight). Keep the first ``--per-cell`` seeds of every cell in seed
order. Disjoint from R0 (290000-290003) and from any formal block.

    python -B -m experiments.t1_r1_openings --start 291000 --output eval/t1/r1/openings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.t1_env import T1Env, T1EnvConfig


def select(start: int, per_cell: int = 2, bins: int = 4, limit: int = 5000) -> list[dict]:
    env = T1Env(T1EnvConfig())
    cells: dict[tuple[int, int], list[dict]] = {}
    for seed in range(start, start + limit):
        f = env.reset(seed)["features"]
        lat_bin = min(int(abs(np.sin(np.deg2rad(f["latitude_deg"]))) * bins), bins - 1)
        phase_bin = min(int(f["phase_deg"] / (360.0 / bins)), bins - 1)
        cell = cells.setdefault((lat_bin, phase_bin), [])
        if len(cell) < per_cell:
            cell.append({"seed": seed, "latitude_bin": lat_bin, "phase_bin": phase_bin, "features": f})
        if len(cells) == bins * bins and all(len(c) == per_cell for c in cells.values()):
            break
    else:
        raise RuntimeError("not every cell filled")
    return sorted((o for c in cells.values() for o in c), key=lambda o: o["seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=291000)
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    openings = select(args.start, args.per_cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "rule": __doc__.split("\n\n")[1],
        "start": args.start,
        "per_cell": args.per_cell,
        "seeds": [o["seed"] for o in openings],
        "openings": openings,
    }, indent=1))
    print(len(openings), "openings, seeds", openings[0]["seed"], "...", openings[-1]["seed"])
    print(",".join(str(o["seed"]) for o in openings))


if __name__ == "__main__":
    main()
