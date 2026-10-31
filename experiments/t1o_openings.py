"""T1-O Gate opening sets (research decision 2026-10-17), frozen from initial geometry only.

4 x 4 cells (|sin LOS latitude| in four equal-probability bins x port phase in four 90 deg
bins), the first seeds of every cell in seed order within the set's own window; no rollout, no
outcome, no seed excluded afterwards. A window that cannot fill every cell is an error, never
widened into the next set's window.

| set          | openings | per cell | window          |
|--------------|----------|----------|-----------------|
| discovery    | 16       | 1        | 298300-298399   |
| confirmation | 32       | 2        | 298400-298599   |
| development  | 64       | 4        | 298600-298999   |

Disjoint from T1-L (297xxx, 298000-298129), the T1-O screens (299500-299899), T1-R
(299000-299195), the exploratory seeds (299900-) and the formal block (>= 300000).

    python -B -m experiments.t1o_openings --profile p1 --set discovery
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from env.t1_o_task import T1O_PROFILES
from experiments.t1o_screen import select_openings

SETS = {"discovery": (298_300, 298_399, 1), "confirmation": (298_400, 298_599, 2),
        "development": (298_600, 298_999, 4)}


def build(profile: str, name: str) -> dict:
    first, last, per_cell = SETS[name]
    openings = select_openings(profile, first, per_cell=per_cell, limit=last - first)
    return {"rule": __doc__.split("\n\n")[1], "profile": profile, "profile_spec": T1O_PROFILES[profile],
            "set": name, "window": [first, last], "per_cell": per_cell,
            "seeds": [o["seed"] for o in openings], "openings": openings}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("p1", "p2"), required=True)
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    doc = build(args.profile, args.set)
    out = args.output or Path(f"experiments/t1o_{args.profile}_{args.set}_openings.json")
    out.write_text(json.dumps(doc, indent=1))
    print(len(doc["seeds"]), "openings", doc["seeds"][0], "...", doc["seeds"][-1],
          hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
